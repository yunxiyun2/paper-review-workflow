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
