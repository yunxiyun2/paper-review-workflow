import asyncio
import json
import pytest
from unittest.mock import MagicMock, AsyncMock
from paper_review_workflow.api.ws_manager import WSManager
from paper_review_workflow.core.event_bus import EventBus, WorkflowEvent, EventType


def test_wsmanager_init():
    bus = EventBus()
    mgr = WSManager(bus)
    assert mgr.event_bus is bus
    assert mgr._connections == set()
    assert mgr._history == []
    assert mgr._history_limit == 1000


def test_wsmanager_attach_subscribes_to_event_bus():
    bus = EventBus()
    mgr = WSManager(bus)
    loop = asyncio.new_event_loop()
    mgr.attach(loop)
    # publish event
    event = WorkflowEvent(event_type=EventType.WORKFLOW_STARTED, run_id="r1")
    bus.publish(event)
    # history should have 1 event
    assert len(mgr._history) == 1
    assert mgr._history[0].run_id == "r1"
    mgr.detach()
    loop.close()


def test_wsmanager_history_limit_trims():
    bus = EventBus()
    mgr = WSManager(bus)
    mgr._history_limit = 5
    loop = asyncio.new_event_loop()
    mgr.attach(loop)
    for i in range(10):
        bus.publish(WorkflowEvent(event_type=EventType.STEP_LOG, run_id=f"r{i}"))
    assert len(mgr._history) == 5
    # should keep most recent 5 (r5..r9)
    run_ids = [e.run_id for e in mgr._history]
    assert "r5" in run_ids
    assert "r9" in run_ids
    assert "r4" not in run_ids
    mgr.detach()
    loop.close()


def test_wsmanager_detach_unsubscribes():
    bus = EventBus()
    mgr = WSManager(bus)
    loop = asyncio.new_event_loop()
    mgr.attach(loop)
    mgr.detach()
    # publish after detach should not affect history
    bus.publish(WorkflowEvent(event_type=EventType.WORKFLOW_STARTED, run_id="r-after"))
    assert len(mgr._history) == 0
    loop.close()


def test_wsmanager_add_and_remove_connection():
    bus = EventBus()
    mgr = WSManager(bus)
    ws = AsyncMock()
    mgr._loop = asyncio.new_event_loop()
    asyncio.run(mgr.add_connection(ws))
    assert ws in mgr._connections
    mgr.remove_connection(ws)
    assert ws not in mgr._connections
    mgr._loop.close()


def test_wsmanager_broadcast_sends_to_all_connections():
    bus = EventBus()
    mgr = WSManager(bus)
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    mgr._connections.add(ws1)
    mgr._connections.add(ws2)
    event = WorkflowEvent(event_type=EventType.WORKFLOW_STARTED, run_id="r1")
    asyncio.run(mgr._broadcast(event))
    ws1.send_text.assert_called_once()
    ws2.send_text.assert_called_once()
    # message should be JSON
    msg1 = ws1.send_text.call_args[0][0]
    data = json.loads(msg1)
    assert data["event"] == "workflow.started"
    assert data["run_id"] == "r1"


def test_wsmanager_broadcast_removes_dead_connections():
    bus = EventBus()
    mgr = WSManager(bus)
    ws_alive = AsyncMock()
    ws_dead = AsyncMock()
    ws_dead.send_text.side_effect = Exception("connection closed")
    mgr._connections.add(ws_alive)
    mgr._connections.add(ws_dead)
    event = WorkflowEvent(event_type=EventType.WORKFLOW_STARTED, run_id="r1")
    asyncio.run(mgr._broadcast(event))
    # dead ws should be removed
    assert ws_dead not in mgr._connections
    assert ws_alive in mgr._connections


def test_wsmanager_send_history_no_filter():
    bus = EventBus()
    mgr = WSManager(bus)
    # preload history
    for i in range(3):
        mgr._history.append(WorkflowEvent(event_type=EventType.STEP_LOG, run_id=f"r{i}"))
    ws = AsyncMock()
    asyncio.run(mgr.send_history(ws))
    assert ws.send_text.call_count == 3


def test_wsmanager_send_history_filtered_by_run_id():
    bus = EventBus()
    mgr = WSManager(bus)
    mgr._history.append(WorkflowEvent(event_type=EventType.STEP_LOG, run_id="r1"))
    mgr._history.append(WorkflowEvent(event_type=EventType.STEP_LOG, run_id="r2"))
    mgr._history.append(WorkflowEvent(event_type=EventType.STEP_LOG, run_id="r1"))
    ws = AsyncMock()
    asyncio.run(mgr.send_history(ws, run_id="r1"))
    # only r1 events sent (2 of them)
    assert ws.send_text.call_count == 2
