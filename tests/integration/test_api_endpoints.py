import pytest
from fastapi.testclient import TestClient
from paper_review_workflow.api import create_app
from paper_review_workflow.storage.memory import MemoryStorage


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    app = create_app(
        storage=MemoryStorage(),
        configs_dir="configs",
    )
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "active_runs" in data
    assert "total_runs" in data


def test_root_404_or_redirect(client):
    # / should either 404 or redirect to /docs
    r = client.get("/")
    assert r.status_code in (404, 200, 307)


def test_openapi_docs_available(client):
    r = client.get("/docs")
    assert r.status_code == 200


def test_list_workflows_includes_normal_review(client):
    r = client.get("/api/workflows")
    assert r.status_code == 200
    data = r.json()
    assert "total" in data
    assert "workflows" in data
    names = [w["name"] for w in data["workflows"]]
    assert "normal-paper-review" in names


def test_get_workflow_summary_has_jobs(client):
    r = client.get("/api/workflows")
    workflows = r.json()["workflows"]
    normal = [w for w in workflows if w["name"] == "normal-paper-review"][0]
    assert "extract" in normal["jobs"]
    assert "dimensions" in normal["jobs"]
    assert "synthesize" in normal["jobs"]
    assert "decide" in normal["jobs"]


def test_register_workflow_via_post(client):
    yaml_content = """
name: test-registered
on: {workflow_dispatch: {}}
jobs:
  j:
    runs-on: local
    steps:
      - uses: paper-review/echo@v1
"""
    r = client.post("/api/workflows/register", json={"yaml_content": yaml_content})
    assert r.status_code == 200
    assert r.json()["name"] == "test-registered"

    # Verify it appears in list
    r = client.get("/api/workflows")
    names = [w["name"] for w in r.json()["workflows"]]
    assert "test-registered" in names


def test_register_workflow_invalid_yaml_returns_400(client):
    r = client.post("/api/workflows/register", json={"yaml_content": "not: valid: yaml: ["})
    assert r.status_code == 400


# ── POST /api/runs (dispatch) tests (M4.5) ───────────────────────────

from unittest.mock import patch, MagicMock


def test_dispatch_returns_202_with_run_id(client, mock_llm):
    r = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    })
    assert r.status_code == 202
    data = r.json()
    assert "run_id" in data
    assert data["status"] == "pending"
    assert data["workflow_name"] == "normal-paper-review"


def test_dispatch_unknown_workflow_returns_404(client):
    r = client.post("/api/runs", json={
        "workflow_name": "nonexistent",
        "inputs": {},
    })
    assert r.status_code == 404


def test_dispatch_no_workflow_or_yaml_returns_422(client):
    r = client.post("/api/runs", json={"inputs": {}})
    assert r.status_code == 422  # pydantic validation error


def test_dispatch_with_inline_yaml(client, mock_llm):
    yaml_content = """
name: inline-test
on: {workflow_dispatch: {}}
jobs:
  j:
    runs-on: local
    steps:
      - uses: paper-review/echo@v1
"""
    r = client.post("/api/runs", json={
        "yaml_content": yaml_content,
        "inputs": {},
    })
    assert r.status_code == 202
    assert r.json()["workflow_name"] == "inline-test"


# Mock LLM fixture for tests that need actual execution
@pytest.fixture
def mock_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from paper_review_workflow.llm.schemas import DimensionScore, SynthesisResult
    from paper_review_workflow.llm.base import LLMResponse
    from paper_review_workflow.llm.client import LLMClient
    LLMClient.reset()

    fake_dim = DimensionScore(
        score=4, confidence=0.8, strengths=["a"], weaknesses=["b"],
        justification="x" * 200, evidence=[],
    )
    fake_synth = SynthesisResult(
        summary="x" * 250, key_strengths=["s"], key_weaknesses=["w"],
        questions_for_authors=["q"], overall_assessment="ok",
    )
    fake_dim_resp = MagicMock(spec=LLMResponse)
    fake_dim_resp.structured = fake_dim
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

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.model = "test-model"
        def side_effect(**kw):
            schema = kw.get("response_schema")
            if schema is SynthesisResult:
                return fake_synth_resp
            return fake_dim_resp
        mock_client.complete.side_effect = side_effect
        mock_from_env.return_value = mock_client
        yield mock_client


# ── GET /api/runs (list) + GET /api/runs/{id} tests (M4.6) ──────────

import time


def test_list_runs_empty(client):
    r = client.get("/api/runs")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 0
    assert data["runs"] == []


def test_list_runs_after_dispatch(client, mock_llm):
    # Dispatch a run
    r = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    })
    run_id = r.json()["run_id"]
    # Wait for it to complete (or fail)
    for _ in range(60):
        r = client.get(f"/api/runs/{run_id}")
        if r.json()["status"] in ("success", "failure", "cancelled"):
            break
        time.sleep(0.5)
    # List runs
    r = client.get("/api/runs")
    data = r.json()
    assert data["total"] >= 1
    assert any(r["run_id"] == run_id for r in data["runs"])


def test_get_run_returns_jobs_status(client, mock_llm):
    r = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    })
    run_id = r.json()["run_id"]
    # Wait for completion
    for _ in range(60):
        r = client.get(f"/api/runs/{run_id}")
        if r.json()["status"] in ("success", "failure", "cancelled"):
            break
        time.sleep(0.5)
    r = client.get(f"/api/runs/{run_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["run_id"] == run_id
    assert data["status"] == "success"
    assert "extract" in data["jobs"]
    assert "decide" in data["jobs"]


def test_get_run_nonexistent_returns_404(client):
    r = client.get("/api/runs/nonexistent-id")
    assert r.status_code == 404


def test_list_runs_filter_by_status(client, mock_llm):
    # Dispatch a run that will succeed
    r = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    })
    run_id = r.json()["run_id"]
    for _ in range(60):
        r = client.get(f"/api/runs/{run_id}")
        if r.json()["status"] in ("success", "failure", "cancelled"):
            break
        time.sleep(0.5)
    # Filter by success
    r = client.get("/api/runs?status=success")
    data = r.json()
    assert all(r["status"] == "success" for r in data["runs"])


def test_list_runs_pagination(client, mock_llm):
    # Dispatch 3 runs
    run_ids = []
    for _ in range(3):
        r = client.post("/api/runs", json={
            "workflow_name": "normal-paper-review",
            "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
        })
        run_ids.append(r.json()["run_id"])
    # Wait for all to complete
    for rid in run_ids:
        for _ in range(60):
            r = client.get(f"/api/runs/{rid}")
            if r.json()["status"] in ("success", "failure", "cancelled"):
                break
            time.sleep(0.5)
    # Test pagination
    r = client.get("/api/runs?limit=2")
    data = r.json()
    assert len(data["runs"]) <= 2


# ── POST /api/runs/{id}/cancel + resume tests (M4.7) ───────────────


def test_cancel_run_returns_200(client, mock_llm):
    r = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    })
    run_id = r.json()["run_id"]
    # Wait briefly to ensure it's running or queued
    time.sleep(0.3)
    r = client.post(f"/api/runs/{run_id}/cancel")
    assert r.status_code == 200
    assert r.json()["run_id"] == run_id


def test_cancel_nonexistent_returns_404(client):
    r = client.post("/api/runs/nonexistent/cancel")
    assert r.status_code == 404


def test_resume_run_after_completion(client, mock_llm):
    # First dispatch
    r = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    })
    run_id = r.json()["run_id"]
    # Wait for it to finish
    for _ in range(60):
        r = client.get(f"/api/runs/{run_id}")
        if r.json()["status"] in ("success", "failure", "cancelled"):
            break
        time.sleep(0.5)
    # Resume it (should be safe even if already done)
    r = client.post(f"/api/runs/{run_id}/resume", json={})
    assert r.status_code == 200
    assert r.json()["run_id"] == run_id


def test_resume_with_rerun_components(client, mock_llm):
    r = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    })
    run_id = r.json()["run_id"]
    for _ in range(60):
        r = client.get(f"/api/runs/{run_id}")
        if r.json()["status"] in ("success", "failure", "cancelled"):
            break
        time.sleep(0.5)
    r = client.post(f"/api/runs/{run_id}/resume", json={
        "rerun_components": ["dimensions_novelty"]
    })
    assert r.status_code == 200


def test_resume_nonexistent_returns_404(client):
    r = client.post("/api/runs/nonexistent/resume", json={})
    assert r.status_code == 404


def test_resume_rerun_all(client, mock_llm):
    r = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    })
    run_id = r.json()["run_id"]
    for _ in range(60):
        r = client.get(f"/api/runs/{run_id}")
        if r.json()["status"] in ("success", "failure", "cancelled"):
            break
        time.sleep(0.5)
    r = client.post(f"/api/runs/{run_id}/resume", json={"rerun_all": True})
    assert r.status_code == 200
