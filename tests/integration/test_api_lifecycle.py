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
