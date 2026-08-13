import os
import time
import json
import queue
import threading

import pytest
from fastapi.testclient import TestClient

from paper_review_workflow.api import create_app
from paper_review_workflow.storage.memory import MemoryStorage


@pytest.mark.e2e
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"),
                    reason="requires ANTHROPIC_API_KEY")
def test_e2e_real_arxiv_review_via_api():
    """E2E: Dispatch a real arXiv review via API, poll until complete, verify decision."""
    app = create_app(storage=MemoryStorage(), configs_dir="configs")
    with TestClient(app) as client:
        # Dispatch
        r = client.post("/api/runs", json={
            "workflow_name": "normal-paper-review",
            "inputs": {"paper_source": "2402.12098"},
        })
        assert r.status_code == 202
        run_id = r.json()["run_id"]

        # Poll until success/failure (up to 5 minutes)
        data = None
        for _ in range(300):
            r = client.get(f"/api/runs/{run_id}")
            data = r.json()
            if data["status"] in ("success", "failure", "cancelled"):
                break
            time.sleep(1)
        else:
            assert False, "run did not complete within 5 minutes"

        assert data["status"] == "success", f"expected success, got {data['status']}"

        # Verify all 8 dim jobs present
        dim_jobs = [k for k in data["jobs"] if k.startswith("dimensions_")]
        assert len(dim_jobs) == 8

        # Verify decide job succeeded
        assert data["jobs"]["decide"]["status"] == "success"


@pytest.mark.e2e
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"),
                    reason="requires ANTHROPIC_API_KEY")
def test_e2e_websocket_receives_real_events():
    """E2E: Connect to /ws, dispatch real arXiv review, receive workflow.started event."""
    app = create_app(storage=MemoryStorage(), configs_dir="configs")
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            # Dispatch
            client.post("/api/runs", json={
                "workflow_name": "normal-paper-review",
                "inputs": {"paper_source": "2402.12098"},
            })
            # Should receive workflow.started within 60s
            q = queue.Queue()

            def reader():
                try:
                    while True:
                        msg = ws.receive_text()
                        q.put(msg)
                except Exception:
                    pass

            t = threading.Thread(target=reader, daemon=True)
            t.start()

            events = []
            for _ in range(120):
                try:
                    msg = q.get(timeout=0.5)
                    data = json.loads(msg)
                    events.append(data.get("event"))
                    if data.get("event") == "workflow.started":
                        break
                except queue.Empty:
                    pass
            assert "workflow.started" in events
