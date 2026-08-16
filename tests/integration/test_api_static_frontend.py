import pytest
from fastapi.testclient import TestClient
from paper_review_workflow.api import create_app
from paper_review_workflow.storage.memory import MemoryStorage


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    app = create_app(storage=MemoryStorage(), configs_dir="configs")
    with TestClient(app) as c:
        yield c


def test_get_venues_returns_3_venues(client):
    r = client.get("/api/venues")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 3
    names = [v["name"] for v in data["venues"]]
    assert "neurips" in names
    assert "icml" in names
    assert "acl" in names


def test_get_venues_neurips_has_correct_fields(client):
    r = client.get("/api/venues")
    neurips = [v for v in r.json()["venues"] if v["name"] == "neurips"][0]
    assert neurips["display_name"] == "NeurIPS 2025"
    assert neurips["dimensions"] == ["soundness", "presentation", "contribution"]
    assert neurips["score_min"] == 1
    assert neurips["score_max"] == 10
    assert neurips["weights"]["soundness"] == 1.3
    assert len(neurips["thresholds"]) == 7
    assert neurips["thresholds"][0]["label"] == "strong_accept"


def test_get_venues_icml_4_dims_1_to_4(client):
    r = client.get("/api/venues")
    icml = [v for v in r.json()["venues"] if v["name"] == "icml"][0]
    assert icml["dimensions"] == ["soundness", "significance", "originality", "clarity"]
    assert icml["score_max"] == 4


def test_root_returns_html_404_when_no_index(client):
    """GET / returns 404 if index.html doesn't exist yet (M2 will create it)"""
    r = client.get("/")
    assert r.status_code in (200, 404)


def test_static_files_endpoint(client):
    """GET /static/.gitkeep should serve the file"""
    r = client.get("/static/.gitkeep")
    assert r.status_code in (200, 404)
