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


def test_export_nonexistent_returns_404(client):
    r = client.get("/api/runs/nonexistent/export?format=xml")
    assert r.status_code == 404


def test_export_returns_xml_content_type():
    """Test with a mock run that has decision + review"""
    from paper_review_workflow.core.models import WorkflowRun, WorkflowStatus, JobInstance, JobStatus
    from paper_review_workflow.storage.memory import MemoryStorage
    import json
    from pathlib import Path

    storage = MemoryStorage()
    storage.open()

    # Create a mock completed run with decision
    run = WorkflowRun(status=WorkflowStatus.SUCCESS)
    run.env["__paper_id__"] = "test1234"

    decide_job = JobInstance(status=JobStatus.SUCCESS)
    decide_job.outputs = {"decision_path": "/tmp/test_decision.json"}
    synth_job = JobInstance(status=JobStatus.SUCCESS)
    synth_job.outputs = {"review_path": "/tmp/test_review.md"}
    run.jobs = {"decide": decide_job, "synthesize": synth_job}

    # Create the files
    decision = {
        "venue": "neurips", "recommendation": "weak_accept",
        "per_dimension": {"soundness": {"score": 7, "confidence": 0.85},
                         "contribution": {"score": 8, "confidence": 0.9},
                         "presentation": {"score": 5, "confidence": 0.7}},
    }
    Path("/tmp/test_decision.json").write_text(json.dumps(decision))
    Path("/tmp/test_review.md").write_text("# Review\n\nGood paper.")

    storage.save_run(run)

    # Use a fresh app with this storage
    app = create_app(storage=storage, configs_dir="configs")
    with TestClient(app) as c:
        r = c.get(f"/api/runs/{run.id}/export?format=xml")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/xml"
        assert "weak_accept" in r.text
        assert "<note>" in r.text
        assert "Good paper." in r.text

    # Cleanup
    Path("/tmp/test_decision.json").unlink(missing_ok=True)
    Path("/tmp/test_review.md").unlink(missing_ok=True)
