import json
import queue
import pytest
import threading
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


def test_websocket_global_connection_receives_heartbeat_or_pong(client):
    """Connect to /ws, send ping, receive pong or heartbeat."""
    with client.websocket_connect("/ws") as ws:
        ws.send_text("ping")
        # Should receive something (pong, heartbeat, or buffered historical events)
        msg = ws.receive_text()
        data = json.loads(msg)
        assert "event" in data


def test_websocket_receives_workflow_events(client, mock_llm_ws):
    """After dispatching a run, /ws should receive workflow.started event."""
    with client.websocket_connect("/ws") as ws:
        # Give WS a moment to subscribe
        time.sleep(0.3)
        # Dispatch a run
        client.post("/api/runs", json={
            "workflow_name": "neurips-paper-review",
            "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf", "mode": "neurips"},
        })
        # Collect messages via a background reader thread (Starlette's test client
        # receive_text() does not support a timeout argument)
        msg_queue: queue.Queue = queue.Queue()

        def reader():
            for _ in range(100):
                try:
                    msg = ws.receive_text()
                    msg_queue.put(msg)
                except Exception as e:
                    msg_queue.put(("__error__", e))
                    return

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        events = []
        for _ in range(60):
            try:
                item = msg_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if isinstance(item, tuple) and item[0] == "__error__":
                break
            data = json.loads(item)
            events.append(data.get("event"))
            if data.get("event") == "workflow.started":
                break
        assert "workflow.started" in events, f"expected workflow.started, got: {events}"


# Mock LLM fixture for WS tests
@pytest.fixture
def mock_llm_ws(monkeypatch):
    from unittest.mock import patch, MagicMock
    from paper_review_workflow.llm.schemas import SynthesisResult
    from paper_review_workflow.llm.base import LLMResponse
    from paper_review_workflow.llm.client import LLMClient
    from paper_review_workflow.actions.synthesize import SynthesizeAction
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

    # NeurIPS has only 3 dims; MIN_DIMENSIONS defaults to 6.
    with patch.object(SynthesizeAction, "MIN_DIMENSIONS", 2), \
         patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
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


def test_websocket_filtered_by_run_id(client, mock_llm_ws):
    """Connect to /ws/runs/{run_id} — should only receive events for that run."""
    # First dispatch a run
    r = client.post("/api/runs", json={
        "workflow_name": "neurips-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf", "mode": "neurips"},
    })
    run_id = r.json()["run_id"]

    # Connect to filtered WS for that run
    with client.websocket_connect(f"/ws/runs/{run_id}") as ws:
        # Collect messages using background reader thread
        import queue
        import threading
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
        for _ in range(60):
            try:
                msg = q.get(timeout=0.5)
                data = json.loads(msg)
                events.append(data)
            except queue.Empty:
                if events:
                    break
        # All events should be for this run_id (or heartbeat/pong)
        for evt in events:
            if evt.get("event") in ("heartbeat", "pong"):
                continue
            assert evt.get("run_id") == run_id, f"got event for {evt.get('run_id')}, expected {run_id}"


def test_websocket_filtered_excludes_other_runs(client, mock_llm_ws):
    """Dispatch 2 runs, connect to /ws/runs/{r1}, should NOT receive r2 events."""
    r1 = client.post("/api/runs", json={
        "workflow_name": "neurips-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf", "mode": "neurips"},
    }).json()
    r2 = client.post("/api/runs", json={
        "workflow_name": "neurips-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf", "mode": "neurips"},
    }).json()

    with client.websocket_connect(f"/ws/runs/{r1['run_id']}") as ws:
        import queue
        import threading
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
        for _ in range(60):
            try:
                msg = q.get(timeout=0.5)
                data = json.loads(msg)
                events.append(data)
            except queue.Empty:
                if events:
                    break
        # No event should mention r2's run_id
        for evt in events:
            assert evt.get("run_id") != r2["run_id"], f"got event for r2 on r1 filter"
