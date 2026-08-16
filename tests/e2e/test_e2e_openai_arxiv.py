import os
import time
import json
import pytest
from fastapi.testclient import TestClient
from paper_review_workflow.api import create_app
from paper_review_workflow.storage.memory import MemoryStorage


@pytest.mark.e2e
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"),
                    reason="requires OPENAI_API_KEY")
def test_e2e_real_openai_arxiv_review():
    """E2E: Real OpenAI API + real arXiv paper + NeurIPS venue."""
    app = create_app(storage=MemoryStorage(), configs_dir="configs")
    with TestClient(app) as client:
        r = client.post("/api/runs", json={
            "workflow_name": "neurips-paper-review",
            "inputs": {"paper_source": "2402.12098", "mode": "neurips"},
        })
        assert r.status_code == 202
        run_id = r.json()["run_id"]

        for _ in range(300):
            r = client.get(f"/api/runs/{run_id}")
            data = r.json()
            if data["status"] in ("success", "failure", "cancelled"):
                break
            time.sleep(1)
        else:
            assert False, "run did not complete within 5 minutes"

        assert data["status"] == "success", f"expected success, got {data['status']}"
        dim_jobs = [k for k in data["jobs"] if k.startswith("dimensions_")]
        assert len(dim_jobs) == 3  # NeurIPS has 3 dims
        assert data["jobs"]["decide"]["status"] == "success"
