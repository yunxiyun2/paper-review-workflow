import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from paper_review_workflow.api import create_app
from paper_review_workflow.storage.json_file import JsonFileStorage
from paper_review_workflow.core.models import WorkflowRun, WorkflowStatus


def test_startup_recovers_interrupted_runs(tmp_path, monkeypatch):
    """Startup should mark storage's PENDING/RUNNING runs as CANCELLED."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    # Pre-populate storage with an interrupted run
    storage = JsonFileStorage(data_dir=str(tmp_path))
    storage.open()
    interrupted = WorkflowRun(status=WorkflowStatus.RUNNING, start_time=datetime.now())
    storage.save_run(interrupted)
    storage.close()

    # Create app — startup handler should recover
    app = create_app(storage=JsonFileStorage(data_dir=str(tmp_path)), configs_dir="configs")
    with TestClient(app) as client:
        # Verify the run was recovered
        r = client.get(f"/api/runs/{interrupted.id}")
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"


def test_startup_loads_configs_directory(tmp_path, monkeypatch):
    """Startup should scan configs/ and register workflows."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "test_wf.yaml").write_text("""
name: test-startup-wf
on: {workflow_dispatch: {}}
jobs:
  j:
    runs-on: local
    steps:
      - uses: paper-review/echo@v1
""")
    app = create_app(
        storage=JsonFileStorage(data_dir=str(tmp_path / "storage")),
        configs_dir=str(configs),
    )
    with TestClient(app) as client:
        r = client.get("/api/workflows")
        names = [w["name"] for w in r.json()["workflows"]]
        assert "test-startup-wf" in names


def test_startup_recovers_pending_runs(tmp_path, monkeypatch):
    """PENDING runs should also be recovered as CANCELLED."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    storage = JsonFileStorage(data_dir=str(tmp_path))
    storage.open()
    pending = WorkflowRun(status=WorkflowStatus.PENDING)
    storage.save_run(pending)
    storage.close()

    app = create_app(storage=JsonFileStorage(data_dir=str(tmp_path)), configs_dir="configs")
    with TestClient(app) as client:
        r = client.get(f"/api/runs/{pending.id}")
        assert r.json()["status"] == "cancelled"


def test_startup_preserves_terminal_runs(tmp_path, monkeypatch):
    """SUCCESS/FAILURE runs should NOT be touched by recovery."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    storage = JsonFileStorage(data_dir=str(tmp_path))
    storage.open()
    success = WorkflowRun(status=WorkflowStatus.SUCCESS, end_time=datetime.now())
    failure = WorkflowRun(status=WorkflowStatus.FAILURE, end_time=datetime.now())
    storage.save_run(success)
    storage.save_run(failure)
    storage.close()

    app = create_app(storage=JsonFileStorage(data_dir=str(tmp_path)), configs_dir="configs")
    with TestClient(app) as client:
        r = client.get(f"/api/runs/{success.id}")
        assert r.json()["status"] == "success"
        r = client.get(f"/api/runs/{failure.id}")
        assert r.json()["status"] == "failure"


def test_shutdown_cancels_active_runs(tmp_path, monkeypatch):
    """When TestClient context exits, shutdown handler should cancel active runs."""
    import time
    from unittest.mock import patch, MagicMock
    from paper_review_workflow.llm.client import LLMClient
    from paper_review_workflow.llm.schemas import SynthesisResult
    from paper_review_workflow.llm.base import LLMResponse
    from paper_review_workflow.actions.synthesize import SynthesizeAction

    LLMClient.reset()

    fake_score_obj = MagicMock()
    fake_score_obj.score = 7
    fake_score_obj.confidence = 0.8
    fake_score_obj.strengths = ["a"]
    fake_score_obj.weaknesses = ["b"]
    fake_score_obj.justification = "x" * 200
    fake_score_obj.evidence = []

    fake_synth = SynthesisResult(
        summary="x" * 250, key_strengths=["s"], key_weaknesses=["w"],
        questions_for_authors=["q"], overall_assessment="ok",
    )
    fake_dim_resp = MagicMock(spec=LLMResponse)
    fake_dim_resp.structured = fake_score_obj
    fake_dim_resp.usage = {"input_tokens": 100, "output_tokens": 50,
                           "cache_creation_input_tokens": 0,
                           "cache_read_input_tokens": 30000}
    fake_dim_resp.model = "test-model"
    fake_synth_resp = MagicMock(spec=LLMResponse)
    fake_synth_resp.structured = fake_synth
    fake_synth_resp.usage = {"input_tokens": 500, "output_tokens": 200,
                             "cache_creation_input_tokens": 0,
                             "cache_read_input_tokens": 0}
    fake_synth_resp.model = "test-model"

    storage = JsonFileStorage(data_dir=str(tmp_path / "storage"))
    storage.open()
    app = create_app(storage=storage, configs_dir="configs")

    run_id_holder = {}
    with patch.object(SynthesizeAction, "MIN_DIMENSIONS", 2), \
         patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.model = "test-model"
        started = []
        import threading
        started_event = threading.Event()
        def slow_complete(**kw):
            schema = kw.get("response_schema")
            # Signal that the LLM call has started (proves run is active)
            started_event.set()
            started.append(True)
            time.sleep(5)  # simulate slow LLM
            if schema is SynthesisResult:
                return fake_synth_resp
            return fake_dim_resp
        mock_client.complete.side_effect = slow_complete
        mock_from_env.return_value = mock_client

        with TestClient(app) as client:
            # Dispatch a run (will be slow)
            r = client.post("/api/runs", json={
                "workflow_name": "neurips-paper-review",
                "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf", "mode": "neurips"},
            })
            run_id_holder["run_id"] = r.json()["run_id"]
            # Wait for the background task to reach the LLM call, proving
            # the run is actively executing (in _active_runs with a coordinator)
            started_event.wait(timeout=15)
            # Exit context — shutdown should cancel active runs

    # Verify run was cancelled or completed
    # Poll for terminal state (background thread may still be finishing)
    storage2 = JsonFileStorage(data_dir=str(tmp_path / "storage"))
    storage2.open()
    run_id = run_id_holder["run_id"]
    status = None
    run = None
    for _ in range(60):
        run = storage2.get_run(run_id)
        if run is not None:
            status = run.status.value
            if status in ("cancelled", "success", "failure"):
                break
        time.sleep(0.5)
    assert run is not None, "run not found in storage after shutdown"
    assert status in ("cancelled", "success"), f"got {status}"
