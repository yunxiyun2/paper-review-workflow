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
