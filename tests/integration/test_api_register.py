import pytest
import time
from fastapi.testclient import TestClient
from paper_review_workflow.api import create_app
from paper_review_workflow.storage.memory import MemoryStorage


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    app = create_app(storage=MemoryStorage(), configs_dir="configs")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_llm_simple(monkeypatch):
    from unittest.mock import patch, MagicMock
    from paper_review_workflow.llm.client import LLMClient
    from paper_review_workflow.llm.schemas import SynthesisResult
    from paper_review_workflow.llm.base import LLMResponse
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


def test_register_workflow_then_dispatch(client, mock_llm_simple):
    """Register a YAML via POST, then dispatch a run with that name."""
    yaml_content = """
name: registered-via-api
on: {workflow_dispatch: {}}
jobs:
  j:
    runs-on: local
    steps:
      - id: echo
        uses: paper-review/echo@v1
        with:
          message: hello from API
"""
    r = client.post("/api/workflows/register", json={"yaml_content": yaml_content})
    assert r.status_code == 200
    assert r.json()["name"] == "registered-via-api"

    # Dispatch
    r = client.post("/api/runs", json={
        "workflow_name": "registered-via-api",
        "inputs": {},
    })
    assert r.status_code == 202
    run_id = r.json()["run_id"]

    # Wait for completion
    for _ in range(60):
        r = client.get(f"/api/runs/{run_id}")
        if r.json()["status"] in ("success", "failure"):
            break
        time.sleep(0.5)
    assert r.json()["status"] == "success"


def test_register_invalid_yaml_returns_400(client):
    r = client.post("/api/workflows/register", json={"yaml_content": "not: valid: yaml: ["})
    assert r.status_code == 400
    assert "YAML parse failed" in r.json()["detail"]


def test_register_with_explicit_name_override(client):
    yaml_content = """
name: original-name
on: {workflow_dispatch: {}}
jobs:
  j:
    runs-on: local
    steps: []
"""
    r = client.post("/api/workflows/register", json={
        "yaml_content": yaml_content,
        "name": "overridden-name",
    })
    assert r.status_code == 200
    assert r.json()["name"] == "overridden-name"

    # Verify it appears in list under overridden name
    r = client.get("/api/workflows")
    names = [w["name"] for w in r.json()["workflows"]]
    assert "overridden-name" in names
