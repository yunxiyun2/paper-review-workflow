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
