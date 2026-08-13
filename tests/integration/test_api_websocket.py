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
            "workflow_name": "normal-paper-review",
            "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
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
