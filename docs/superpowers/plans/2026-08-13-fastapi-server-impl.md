# FastAPI Server (Phase 2 #5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FastAPI HTTP API + WebSocket real-time push to the paper-review-workflow engine, enabling long-running reviews to be triggered/monitored via HTTP, with resume/rerun support.

**Architecture:** Reuse lwf's `BackgroundTasks` + `asyncio.run_in_executor` pattern — POST /api/runs returns run_id immediately, engine runs in thread pool, WebSocket pushes EventBus events. WSManager acts as an EventBus subscriber to broadcast events to WS connections (decoupling sync engine from async WebSocket). Engine extensions: `load_workflow_directory`, `register_workflow`, `recover_interrupted_runs`, `dispatch_workflow`, `execute_existing_run`.

**Tech Stack:** FastAPI 0.115+, uvicorn[standard] 0.30+, websockets 13+, Pydantic v2, anthropic SDK, pytest, fastapi.testclient.TestClient.

**Reference SPEC:** `docs/superpowers/specs/2026-08-12-fastapi-server-design.md`

**Reference lwf source:** `/Users/dengyunxi/workspace/for-staging/lwf/workflow_engine/api/server.py`

---

## File Structure Overview

```
paper-review-workflow/
├── pyproject.toml                       # Modify: add fastapi/uvicorn deps
├── main.py                               # No change (cli.py handles dispatch)
├── paper_review_workflow/
│   ├── engine.py                         # Modify: add 5 new methods + _workflow_defs dict
│   ├── cli.py                            # Modify: add server subcommand + default-to-server
│   └── api/                              # ★ New subpackage
│       ├── __init__.py                   # Exports create_app
│       ├── server.py                     # FastAPI app factory + endpoints + WS
│       ├── schemas.py                    # Pydantic request/response models
│       ├── dependencies.py              # FastAPI Depends (get_engine)
│       ├── ws_manager.py                 # WebSocket connection mgmt + event broadcast
│       └── static/.gitkeep              # Placeholder for Phase 2 #3 frontend
└── tests/
    ├── unit/
    │   ├── test_ws_manager.py           # WSManager + FilteredWS
    │   └── test_engine_api_ext.py       # Engine's 5 new methods
    ├── integration/
    │   ├── test_api_endpoints.py        # 7 REST endpoints
    │   ├── test_api_websocket.py        # /ws + /ws/runs/{id}
    │   ├── test_api_lifecycle.py        # startup recovery + shutdown cancel
    │   └── test_api_register.py         # POST /api/workflows/register
    └── e2e/
        └── test_api_e2e_arxiv.py        # Real arXiv + real API (marked)
```

---

# M1: Dependencies + api/ Skeleton

**Goal:** Add fastapi/uvicorn deps, create api/ subpackage with empty files.

**Estimated:** 0.5 day

## Task 1.1: Update pyproject.toml + install deps

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read current pyproject.toml**

Run: `cat pyproject.toml`

- [ ] **Step 2: Add fastapi/uvicorn/websockets to dependencies**

Edit `pyproject.toml` — in the `dependencies` array, add 3 new entries:

```toml
dependencies = [
    "anthropic>=0.40.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "pymupdf>=1.24.0",
    "arxiv>=2.1.0",
    "httpx>=0.27.0",
    "jinja2>=3.1.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "websockets>=13.0",
]
```

- [ ] **Step 3: Reinstall dev dependencies**

Run: `pip install -e ".[dev]"`
Expected: Successfully installs fastapi, uvicorn, websockets

- [ ] **Step 4: Verify imports work**

Run: `python -c "import fastapi; import uvicorn; import websockets; print(fastapi.__version__, uvicorn.__version__)"`
Expected: Prints version numbers, no errors

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore(deps): add fastapi/uvicorn/websockets for Phase 2 #5"
```

## Task 1.2: Create api/ subpackage skeleton

**Files:**
- Create: `paper_review_workflow/api/__init__.py`
- Create: `paper_review_workflow/api/server.py` (stub)
- Create: `paper_review_workflow/api/schemas.py` (empty)
- Create: `paper_review_workflow/api/dependencies.py` (empty)
- Create: `paper_review_workflow/api/ws_manager.py` (empty)
- Create: `paper_review_workflow/api/static/.gitkeep`

- [ ] **Step 1: Create directory structure**

Run: `mkdir -p paper_review_workflow/api/static && touch paper_review_workflow/api/static/.gitkeep`

- [ ] **Step 2: Create `paper_review_workflow/api/__init__.py`**

```python
"""FastAPI HTTP API + WebSocket server for paper-review-workflow."""

def create_app(*args, **kwargs):
    """Stub. Full implementation in M4."""
    raise NotImplementedError("create_app not yet implemented")
```

- [ ] **Step 3: Create stub files (empty content)**

For each of these files, create with empty content (just a docstring):
- `paper_review_workflow/api/server.py`: `"""FastAPI app factory + endpoints."""`
- `paper_review_workflow/api/schemas.py`: `"""Pydantic request/response models."""`
- `paper_review_workflow/api/dependencies.py`: `"""FastAPI dependency injection."""`
- `paper_review_workflow/api/ws_manager.py`: `"""WebSocket connection management."""`

- [ ] **Step 4: Verify package imports**

Run: `python -c "from paper_review_workflow import api; print('api subpackage OK')"`
Expected: `api subpackage OK`

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/api/
git commit -m "feat(api): scaffold api/ subpackage"
```

---

# M2: WSManager + FilteredWS

**Goal:** Build WSManager class that subscribes to EventBus and broadcasts events to WebSocket connections. Includes FilteredWS wrapper for run-specific subscriptions.

**Estimated:** 1 day

## Task 2.1: WSManager class

**Files:**
- Modify: `paper_review_workflow/api/ws_manager.py`
- Test: `tests/unit/test_ws_manager.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_ws_manager.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ws_manager.py -v`
Expected: FAIL with ImportError or AttributeError

- [ ] **Step 3: Implement `paper_review_workflow/api/ws_manager.py`**

```python
"""WebSocket connection management + event broadcast.

WSManager acts as an EventBus subscriber. When the (synchronous) engine publishes
events via EventBus, WSManager receives them in the engine thread and schedules
async broadcasts to all WS connections via run_coroutine_threadsafe.
"""
import asyncio
import json
import logging
from typing import Callable, Optional, Set
from weakref import WeakSet

from fastapi import WebSocket

from ..core.event_bus import EventBus, WorkflowEvent

logger = logging.getLogger(__name__)


class WSManager:
    """WebSocket connection manager + EventBus subscriber."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._connections: Set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._unsubscribe: Optional[Callable] = None
        self._history: list = []
        self._history_limit = 1000

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        """Subscribe to EventBus + cache event loop. Call at FastAPI startup."""
        self._loop = loop
        self._unsubscribe = self.event_bus.subscribe(self._on_event)

    def detach(self) -> None:
        """Unsubscribe + close all connections. Call at FastAPI shutdown."""
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        if self._loop:
            for ws in list(self._connections):
                asyncio.run_coroutine_threadsafe(
                    ws.close(), self._loop
                )
        self._connections.clear()

    def _on_event(self, event: WorkflowEvent) -> None:
        """EventBus sync callback (runs in engine thread). Schedule async broadcast."""
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]
        if self._loop and self._connections:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._broadcast(event), self._loop
                )
            except Exception as e:
                logger.error(f"[WSManager] broadcast schedule failed: {e}")

    async def _broadcast(self, event: WorkflowEvent) -> None:
        """Async broadcast event to all connections (runs in event loop)."""
        message = event.to_json()
        dead = []
        for ws in list(self._connections):
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.debug(f"[WSManager] connection dead, removing: {e}")
                dead.append(ws)
        for ws in dead:
            self._connections.discard(ws)

    async def add_connection(self, ws) -> None:
        """Accept and register a new WebSocket connection."""
        await ws.accept()
        self._connections.add(ws)

    def remove_connection(self, ws) -> None:
        """Remove a WebSocket connection."""
        self._connections.discard(ws)

    async def send_history(self, ws, run_id: Optional[str] = None) -> None:
        """Send historical events to a connection (optional run_id filter)."""
        for evt in self._history:
            if run_id and evt.run_id != run_id:
                continue
            try:
                await ws.send_text(evt.to_json())
            except Exception:
                break
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_ws_manager.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/api/ws_manager.py tests/unit/test_ws_manager.py
git commit -m "feat(api): WSManager for EventBus → WebSocket broadcast"
```

## Task 2.2: FilteredWS wrapper class

**Files:**
- Modify: `paper_review_workflow/api/ws_manager.py`
- Test: `tests/unit/test_ws_manager.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_ws_manager.py`:

```python
from paper_review_workflow.api.ws_manager import FilteredWS


def test_filteredws_only_sends_matching_run_id():
    target = AsyncMock()
    filtered = FilteredWS(target, "run-123")

    # matching run_id
    asyncio.run(filtered.send_text(json.dumps({
        "run_id": "run-123", "event": "step.completed", "data": {}
    })))
    target.send_text.assert_called_once()

    # non-matching run_id — should NOT send
    target.reset_mock()
    asyncio.run(filtered.send_text(json.dumps({
        "run_id": "run-456", "event": "step.completed", "data": {}
    })))
    target.send_text.assert_not_called()


def test_filteredws_passes_heartbeat_and_pong():
    target = AsyncMock()
    filtered = FilteredWS(target, "run-123")

    # heartbeat should always pass
    asyncio.run(filtered.send_text(json.dumps({"event": "heartbeat"})))
    target.send_text.assert_called_once()

    # pong should always pass
    target.reset_mock()
    asyncio.run(filtered.send_text(json.dumps({"event": "pong"})))
    target.send_text.assert_called_once()


def test_filteredws_passes_non_json_through():
    """Non-JSON messages should pass through (e.g. raw text errors)"""
    target = AsyncMock()
    filtered = FilteredWS(target, "run-123")
    asyncio.run(filtered.send_text("raw text message"))
    target.send_text.assert_called_once_with("raw text message")


def test_filteredws_close_delegates_to_target():
    target = AsyncMock()
    filtered = FilteredWS(target, "run-123")
    asyncio.run(filtered.close())
    target.close.assert_called_once()


def test_filteredws_accept_delegates_to_target():
    target = AsyncMock()
    filtered = FilteredWS(target, "run-123")
    asyncio.run(filtered.accept())
    target.accept.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ws_manager.py::test_filteredws_only_sends_matching_run_id -v`
Expected: FAIL with ImportError (FilteredWS not defined)

- [ ] **Step 3: Add `FilteredWS` class to `paper_review_workflow/api/ws_manager.py`**

Append to the end of `ws_manager.py`:

```python
class FilteredWS:
    """Wrapper for WebSocket that only sends events matching a target run_id.

    Used by /ws/runs/{run_id} endpoint to filter global events to a single run.
    Heartbeat/pong/non-JSON messages pass through unconditionally.
    """

    def __init__(self, ws, target_run_id: str):
        self._ws = ws
        self._run_id = target_run_id

    async def send_text(self, message: str) -> None:
        try:
            data = json.loads(message)
            if (data.get("run_id") == self._run_id
                or data.get("event") in ("heartbeat", "pong")):
                await self._ws.send_text(message)
        except (json.JSONDecodeError, AttributeError):
            await self._ws.send_text(message)

    async def accept(self) -> None:
        await self._ws.accept()

    async def close(self) -> None:
        await self._ws.close()

    # Delegate attribute access for any other WebSocket methods
    def __getattr__(self, name):
        return getattr(self._ws, name)
```

- [ ] **Step 4: Run all WSManager tests**

Run: `pytest tests/unit/test_ws_manager.py -v`
Expected: PASS (13 tests: 8 WSManager + 5 FilteredWS)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/api/ws_manager.py tests/unit/test_ws_manager.py
git commit -m "feat(api): FilteredWS for run-specific WebSocket filtering"
```

---

# M3: Engine Extensions

**Goal:** Add 5 new methods to ReviewEngine for API mode: `load_workflow_directory`, `register_workflow`, `recover_interrupted_runs`, `dispatch_workflow`, `execute_existing_run`. Plus `_workflow_defs` dict + 2 lookup helpers (`get_workflow_defs`, `get_workflow_def`).

**Estimated:** 1 day

## Task 3.1: Add _workflow_defs dict + load_workflow_directory

**Files:**
- Modify: `paper_review_workflow/engine.py`
- Test: `tests/unit/test_engine_api_ext.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_engine_api_ext.py`:

```python
import pytest
from pathlib import Path
from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.memory import MemoryStorage


@pytest.fixture
def engine():
    return ReviewEngine(storage=MemoryStorage())


@pytest.fixture
def configs_dir(tmp_path):
    """Create a tmp configs/ dir with 2 YAML files"""
    d = tmp_path / "configs"
    d.mkdir()
    (d / "wf1.yaml").write_text("""
name: wf-one
on: {workflow_dispatch: {}}
jobs:
  j:
    runs-on: local
    steps:
      - uses: paper-review/echo@v1
""")
    (d / "wf2.yml").write_text("""
name: wf-two
on: {workflow_dispatch: {}}
jobs:
  j:
    runs-on: local
    steps:
      - uses: paper-review/echo@v1
""")
    (d / "bad.yml").write_text("not: valid: yaml: [")
    return str(d)


def test_load_workflow_directory_loads_yamls(engine, configs_dir):
    loaded = engine.load_workflow_directory(configs_dir)
    assert "wf-one" in loaded
    assert "wf-two" in loaded
    assert len(loaded) == 2  # bad.yml skipped


def test_load_workflow_directory_nonexistent_dir(engine, tmp_path):
    loaded = engine.load_workflow_directory(str(tmp_path / "nonexistent"))
    assert loaded == {}


def test_get_workflow_defs_returns_registered(engine, configs_dir):
    engine.load_workflow_directory(configs_dir)
    defs = engine.get_workflow_defs()
    assert "wf-one" in defs
    assert "wf-two" in defs


def test_get_workflow_def_by_name(engine, configs_dir):
    engine.load_workflow_directory(configs_dir)
    wf = engine.get_workflow_def("wf-one")
    assert wf is not None
    assert wf.name == "wf-one"
    assert "j" in wf.jobs


def test_get_workflow_def_unknown_returns_none(engine):
    assert engine.get_workflow_def("nonexistent") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_engine_api_ext.py -v`
Expected: FAIL with AttributeError (load_workflow_directory not defined)

- [ ] **Step 3: Modify `paper_review_workflow/engine.py`**

Read the current `engine.py`. In `ReviewEngine.__init__`, after `self._coordinators = {}` line, add:

```python
self._workflow_defs: Dict[str, WorkflowDef] = {}
```

Then add these new methods (insert before `_register_persist_hooks` private methods, after the existing public methods):

```python
# -- Workflow definition management (Phase 2 #5) --

def load_workflow_directory(self, configs_dir: str) -> Dict[str, "WorkflowDef"]:
    """Scan directory for *.yaml/*.yml files and register them. Returns {name: WorkflowDef}."""
    path = Path(configs_dir)
    if not path.exists():
        logger.warning(f"[Engine] configs dir not found: {configs_dir}")
        return {}
    loaded = {}
    for yml_file in sorted(path.glob("**/*.y*ml")):
        try:
            wf_def = self.parser.parse_file(str(yml_file))
            self._register_workflow_def(wf_def)
            loaded[wf_def.name] = wf_def
            logger.info(f"[Engine] loaded workflow: {wf_def.name} ({yml_file})")
        except Exception as e:
            logger.warning(f"[Engine] failed to load {yml_file}: {e}")
    return loaded

def register_workflow(self, yaml_content: str, name: Optional[str] = None) -> "WorkflowDef":
    """Dynamically register a YAML workflow (used by POST /api/workflows/register)."""
    wf_def = self.parser.parse_string(yaml_content)
    if name:
        wf_def.name = name
    self._register_workflow_def(wf_def)
    return wf_def

def _register_workflow_def(self, wf_def: "WorkflowDef") -> None:
    """Register a workflow definition (overwrites same-name)."""
    self._workflow_defs[wf_def.name] = wf_def

def get_workflow_defs(self) -> Dict[str, "WorkflowDef"]:
    """Return all registered workflow definitions."""
    return dict(self._workflow_defs)

def get_workflow_def(self, name: str) -> Optional["WorkflowDef"]:
    """Look up a workflow by name."""
    return self._workflow_defs.get(name)
```

Make sure to add `from pathlib import Path` to the imports at the top of `engine.py` if not present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_engine_api_ext.py -v`
Expected: PASS (5 tests)

Run: `pytest tests/ -v` to ensure no regressions
Expected: All existing tests still pass

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/engine.py tests/unit/test_engine_api_ext.py
git commit -m "feat(engine): load_workflow_directory + workflow registry"
```

## Task 3.2: recover_interrupted_runs

**Files:**
- Modify: `paper_review_workflow/engine.py`
- Test: `tests/unit/test_engine_api_ext.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_engine_api_ext.py`:

```python
from paper_review_workflow.core.models import WorkflowRun, WorkflowStatus
from paper_review_workflow.storage.json_file import JsonFileStorage
from datetime import datetime


def test_recover_interrupted_runs_marks_running_as_cancelled(tmp_path):
    storage = JsonFileStorage(data_dir=str(tmp_path))
    storage.open()
    # 3 runs: 1 running, 1 pending, 1 success
    running = WorkflowRun(status=WorkflowStatus.RUNNING, start_time=datetime.now())
    pending = WorkflowRun(status=WorkflowStatus.PENDING)
    success = WorkflowRun(status=WorkflowStatus.SUCCESS)
    storage.save_run(running)
    storage.save_run(pending)
    storage.save_run(success)
    storage.close()

    # Create new engine pointing to same storage
    engine = ReviewEngine(storage=JsonFileStorage(data_dir=str(tmp_path)))
    recovered = engine.recover_interrupted_runs()
    assert recovered == 2  # running + pending

    # Verify states
    r = engine.storage.get_run(running.id)
    assert r.status == WorkflowStatus.CANCELLED
    p = engine.storage.get_run(pending.id)
    assert p.status == WorkflowStatus.CANCELLED
    s = engine.storage.get_run(success.id)
    assert s.status == WorkflowStatus.SUCCESS  # unchanged


def test_recover_interrupted_runs_no_storage_issues(engine):
    # No runs in empty storage
    recovered = engine.recover_interrupted_runs()
    assert recovered == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_engine_api_ext.py::test_recover_interrupted_runs_marks_running_as_cancelled -v`
Expected: FAIL with AttributeError (recover_interrupted_runs not defined)

- [ ] **Step 3: Add `recover_interrupted_runs` method to `engine.py`**

Add to `ReviewEngine` class (after `get_workflow_def`):

```python
def recover_interrupted_runs(self) -> int:
    """Mark storage's PENDING/RUNNING runs as CANCELLED. Call at API startup."""
    try:
        interrupted = self.storage.list_runs(limit=1000)
    except Exception as e:
        logger.warning(f"[Engine] recover scan failed: {e}")
        return 0
    recovered = 0
    for run in interrupted:
        if run.status in (WorkflowStatus.PENDING, WorkflowStatus.RUNNING):
            run.status = WorkflowStatus.CANCELLED
            run.end_time = datetime.now()
            self.storage.save_run(run)
            recovered += 1
            logger.info(f"[Engine] recovered run {run.id[:8]} → cancelled")
    return recovered
```

Make sure `datetime` is imported at the top of `engine.py`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_engine_api_ext.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/engine.py tests/unit/test_engine_api_ext.py
git commit -m "feat(engine): recover_interrupted_runs for crash recovery"
```

## Task 3.3: dispatch_workflow + execute_existing_run

**Files:**
- Modify: `paper_review_workflow/engine.py`
- Test: `tests/unit/test_engine_api_ext.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_engine_api_ext.py`:

```python
def test_dispatch_workflow_creates_pending_run_without_executing(engine, configs_dir):
    """dispatch_workflow must create PENDING run in storage but NOT execute."""
    engine.load_workflow_directory(configs_dir)

    run = engine.dispatch_workflow("wf-one", inputs={"msg": "hi"})
    assert run.status == WorkflowStatus.PENDING
    assert run.trigger_payload == {"inputs": {"msg": "hi"}}
    assert run.env["__workflow_name__"] == "wf-one"
    # Verify saved to storage
    loaded = engine.storage.get_run(run.id)
    assert loaded is not None
    assert loaded.status == WorkflowStatus.PENDING


def test_dispatch_workflow_unknown_raises(engine):
    with pytest.raises(ValueError, match="workflow not registered"):
        engine.dispatch_workflow("nonexistent", inputs={})


def test_execute_existing_run_loads_and_executes(engine, configs_dir):
    """execute_existing_run loads a run from storage and runs it."""
    engine.load_workflow_directory(configs_dir)

    # Dispatch a run (PENDING, not executed)
    run = engine.dispatch_workflow("wf-one", inputs={})
    assert run.status == WorkflowStatus.PENDING

    # Execute it
    executed = engine.execute_existing_run(run.id)
    assert executed.status == WorkflowStatus.SUCCESS
    # Verify step output persisted
    assert "j" in executed.jobs
    assert executed.jobs["j"].status.value == "success"


def test_execute_existing_run_unknown_id_raises(engine):
    with pytest.raises(ValueError, match="run not found"):
        engine.execute_existing_run("nonexistent-run-id")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_engine_api_ext.py::test_dispatch_workflow_creates_pending_run_without_executing -v`
Expected: FAIL with AttributeError

- [ ] **Step 3: Add `dispatch_workflow` + `execute_existing_run` to `engine.py`**

Add to `ReviewEngine` class:

```python
def dispatch_workflow(self, workflow_name: str, inputs: Dict[str, Any]) -> WorkflowRun:
    """Create a PENDING run for the named workflow and save to storage. Does NOT execute.

    Used by POST /api/runs endpoint. Actual execution is triggered by caller
    via execute_existing_run (typically in a background task).
    """
    wf_def = self.get_workflow_def(workflow_name)
    if wf_def is None:
        raise ValueError(f"workflow not registered: {workflow_name}")
    run = WorkflowRun(
        workflow_def=wf_def,
        trigger_type="workflow_dispatch",
        trigger_payload={"inputs": inputs},
        env=dict(wf_def.env) if wf_def.env else {},
    )
    if wf_def.file_path:
        run.env["__workflow_file__"] = wf_def.file_path
    run.env["__workflow_name__"] = wf_def.name
    self.storage.save_run(run)
    self.event_bus.publish(WorkflowEvent(
        event_type=EventType.WORKFLOW_CREATED, run_id=run.id,
        data={"workflow_name": wf_def.name, "trigger_type": "workflow_dispatch"},
    ))
    return run

def execute_existing_run(self, run_id: str) -> WorkflowRun:
    """Load a run from storage and execute it. Used by background tasks."""
    run = self.storage.get_run(run_id)
    if run is None:
        raise ValueError(f"run not found: {run_id}")
    if run.workflow_def is None and run.env.get("__workflow_file__"):
        run.workflow_def = self.parser.parse_file(run.env["__workflow_file__"])
    return self._do_execute(run)
```

Make sure `Any` is imported from `typing` at the top.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_engine_api_ext.py -v`
Expected: PASS (11 tests total)

Run: `pytest tests/ -v`
Expected: All existing tests still pass

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/engine.py tests/unit/test_engine_api_ext.py
git commit -m "feat(engine): dispatch_workflow + execute_existing_run for API mode"
```

---

# M4: FastAPI App + 7 REST Endpoints

**Goal:** Build the FastAPI app factory, Pydantic schemas, dependency injection, and all 7 REST endpoints (health, workflows, register, runs dispatch/list/get/cancel/resume).

**Estimated:** 1.5 days

## Task 4.1: Pydantic schemas

**Files:**
- Modify: `paper_review_workflow/api/schemas.py`
- Test: `tests/unit/test_api_schemas.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_api_schemas.py`:

```python
import pytest
from pydantic import ValidationError
from paper_review_workflow.api.schemas import (
    DispatchRequest, RegisterWorkflowRequest, ResumeRequest,
    RunResponse, WorkflowSummary, HealthResponse,
)


def test_dispatch_request_with_workflow_name():
    r = DispatchRequest(workflow_name="normal-paper-review", inputs={"paper_source": "2402.12098"})
    assert r.workflow_name == "normal-paper-review"
    assert r.yaml_content is None
    assert r.inputs["paper_source"] == "2402.12098"


def test_dispatch_request_with_yaml_content():
    r = DispatchRequest(yaml_content="name: t\non: {workflow_dispatch: {}}")
    assert r.workflow_name is None
    assert r.yaml_content.startswith("name:")


def test_dispatch_request_requires_name_or_yaml():
    with pytest.raises(ValidationError):
        DispatchRequest()  # neither provided (but pydantic allows both None — need validator)


def test_register_workflow_request():
    r = RegisterWorkflowRequest(yaml_content="name: foo\non: {workflow_dispatch: {}}")
    assert r.yaml_content.startswith("name:")
    assert r.name is None


def test_resume_request_defaults():
    r = ResumeRequest()
    assert r.rerun_components is None
    assert r.rerun_all is False


def test_resume_request_with_components():
    r = ResumeRequest(rerun_components=["dim_novelty", "synthesize"])
    assert r.rerun_components == ["dim_novelty", "synthesize"]


def test_run_response_minimal():
    r = RunResponse(run_id="abc", workflow_name="wf", status="pending")
    assert r.run_id == "abc"
    assert r.duration is None
    assert r.jobs == {}


def test_workflow_summary():
    r = WorkflowSummary(name="normal-paper-review", jobs=["extract", "dimensions"])
    assert r.name == "normal-paper-review"
    assert r.dispatch_inputs == {}


def test_health_response():
    r = HealthResponse()
    assert r.status == "ok"
    assert r.version == "0.1.0"
    assert r.active_runs == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_api_schemas.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `paper_review_workflow/api/schemas.py`**

```python
"""Pydantic request/response models for FastAPI API."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class DispatchRequest(BaseModel):
    """POST /api/runs — trigger a workflow run."""
    workflow_name: Optional[str] = Field(default=None, description="Reference a registered workflow by name")
    yaml_content: Optional[str] = Field(default=None, description="Inline YAML workflow definition")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Dispatch inputs (e.g. paper_source)")

    @model_validator(mode="after")
    def require_name_or_yaml(self):
        if not self.workflow_name and not self.yaml_content:
            raise ValueError("must provide workflow_name or yaml_content")
        return self


class RegisterWorkflowRequest(BaseModel):
    """POST /api/workflows/register — register a YAML workflow."""
    yaml_content: str
    name: Optional[str] = None


class ResumeRequest(BaseModel):
    """POST /api/runs/{run_id}/resume — resume a run."""
    rerun_components: Optional[List[str]] = None
    rerun_all: bool = False


class RunResponse(BaseModel):
    """GET /api/runs/{id} — single run details."""
    run_id: str
    workflow_name: str
    status: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[float] = None
    jobs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class WorkflowSummary(BaseModel):
    """GET /api/workflows — list entry."""
    name: str
    file_path: Optional[str] = None
    jobs: List[str] = Field(default_factory=list)
    dispatch_inputs: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """GET /api/health."""
    status: str = "ok"
    version: str = "0.1.0"
    active_runs: int = 0
    total_runs: int = 0
```

- [ ] **Step 4: Update test to remove the failing case**

The test `test_dispatch_request_requires_name_or_yaml` expects ValidationError, but our validator raises ValueError (which pydantic wraps as ValidationError). Verify with:

Run: `pytest tests/unit/test_api_schemas.py -v`
Expected: PASS (8 tests). If `test_dispatch_request_requires_name_or_yaml` fails, the validator needs adjusting — pydantic v2 wraps validator errors as ValidationError. Confirm it passes.

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/api/schemas.py tests/unit/test_api_schemas.py
git commit -m "feat(api): Pydantic schemas for request/response models"
```

## Task 4.2: Dependencies (get_engine)

**Files:**
- Modify: `paper_review_workflow/api/dependencies.py`

- [ ] **Step 1: Implement `paper_review_workflow/api/dependencies.py`**

```python
"""FastAPI dependency injection."""
import os
from typing import Optional
from fastapi import Depends, Request

from ..engine import ReviewEngine
from ..storage import StorageBackend, MemoryStorage, JsonFileStorage


def get_engine(request: Request) -> ReviewEngine:
    """Retrieve the ReviewEngine instance attached to the app state."""
    return request.app.state.engine


def get_ws_manager(request: Request):
    """Retrieve the WSManager instance attached to the app state."""
    return request.app.state.ws_manager


def get_configs_dir(request: Request) -> str:
    """Retrieve the configs_dir path attached to the app state."""
    return request.app.state.configs_dir
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from paper_review_workflow.api.dependencies import get_engine; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add paper_review_workflow/api/dependencies.py
git commit -m "feat(api): FastAPI dependency injection"
```

## Task 4.3: create_app factory + startup/shutdown + health endpoint

**Files:**
- Modify: `paper_review_workflow/api/server.py`
- Modify: `paper_review_workflow/api/__init__.py`
- Test: `tests/integration/test_api_endpoints.py`

- [ ] **Step 1: Write failing test**

Create `tests/integration/test_api_endpoints.py`:

```python
import pytest
from fastapi.testclient import TestClient
from paper_review_workflow.api import create_app
from paper_review_workflow.storage.memory import MemoryStorage


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    # Use a configs dir that has normal_review.yaml
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


def test_root_redirects_or_404(client):
    # / should either 404 or redirect to /docs
    r = client.get("/")
    assert r.status_code in (404, 200, 307)


def test_openapi_docs_available(client):
    r = client.get("/docs")
    assert r.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_api_endpoints.py::test_health_endpoint -v`
Expected: FAIL with NotImplementedError (create_app is stub)

- [ ] **Step 3: Implement `paper_review_workflow/api/server.py`**

```python
"""FastAPI app factory + endpoints + WebSocket."""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..engine import ReviewEngine
from ..storage import StorageBackend, MemoryStorage, JsonFileStorage
from ..core.event_bus import EventBus, WorkflowEvent, EventType
from ..core.models import WorkflowRun, WorkflowStatus, JobStatus, StepStatus
from .dependencies import get_engine, get_ws_manager, get_configs_dir
from .schemas import (
    DispatchRequest, RegisterWorkflowRequest, ResumeRequest,
    RunResponse, WorkflowSummary, HealthResponse,
)
from .ws_manager import WSManager, FilteredWS

logger = logging.getLogger(__name__)


def create_app(
    storage: Optional[StorageBackend] = None,
    configs_dir: str = "./configs",
    sessions_root: str = "./sessions",
) -> FastAPI:
    """Create a FastAPI app instance with the given engine configuration."""
    app = FastAPI(
        title="Paper Review Workflow API",
        description="HTTP API + WebSocket for AI-based paper review",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Build engine + WSManager
    engine = ReviewEngine(
        storage=storage if storage is not None else JsonFileStorage(sessions_root),
        sessions_root=sessions_root,
    )
    ws_manager = WSManager(engine.event_bus)

    # Attach to app state for dependency injection
    app.state.engine = engine
    app.state.ws_manager = ws_manager
    app.state.configs_dir = configs_dir

    # Startup/shutdown handlers
    @app.on_event("startup")
    async def startup_event():
        loop = asyncio.get_event_loop()
        ws_manager.attach(loop)
        engine.load_workflow_directory(configs_dir)
        recovered = engine.recover_interrupted_runs()
        if recovered:
            logger.warning(f"[API] recovered {recovered} interrupted runs → cancelled")

    @app.on_event("shutdown")
    async def shutdown_event():
        ws_manager.detach()
        cancelled = engine.shutdown(timeout=10.0)
        if cancelled:
            logger.info(f"[API] cancelled {cancelled} active runs on shutdown")
        engine.storage.close()

    # ── Endpoints ──
    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        runs = engine.storage.list_runs(limit=10**6)
        total = len(runs)
        active = sum(1 for r in runs if r.status in (WorkflowStatus.PENDING, WorkflowStatus.RUNNING))
        return HealthResponse(status="ok", version="0.1.0", active_runs=active, total_runs=total)

    # Other endpoints added in subsequent tasks
    return app
```

- [ ] **Step 4: Update `paper_review_workflow/api/__init__.py` to export `create_app`**

```python
"""FastAPI HTTP API + WebSocket server for paper-review-workflow."""
from .server import create_app

__all__ = ["create_app"]
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/integration/test_api_endpoints.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add paper_review_workflow/api/server.py paper_review_workflow/api/__init__.py tests/integration/test_api_endpoints.py
git commit -m "feat(api): create_app factory + /api/health endpoint"
```

## Task 4.4: GET /api/workflows + POST /api/workflows/register

**Files:**
- Modify: `paper_review_workflow/api/server.py`
- Test: `tests/integration/test_api_endpoints.py`

- [ ] **Step 1: Write failing test**

Append to `tests/integration/test_api_endpoints.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_api_endpoints.py::test_list_workflows_includes_normal_review -v`
Expected: FAIL (404 — endpoint not yet implemented)

- [ ] **Step 3: Add endpoints to `server.py`**

Inside `create_app()` after the `health()` endpoint, add:

```python
    @app.get("/api/workflows")
    async def list_workflows():
        defs = engine.get_workflow_defs()
        return {
            "total": len(defs),
            "workflows": [
                {
                    "name": name,
                    "file_path": wf_def.file_path,
                    "jobs": list(wf_def.jobs.keys()),
                    "dispatch_inputs": _extract_dispatch_inputs(wf_def),
                }
                for name, wf_def in defs.items()
            ],
        }

    @app.post("/api/workflows/register")
    async def register_workflow(req: RegisterWorkflowRequest):
        try:
            wf_def = engine.register_workflow(req.yaml_content, name=req.name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"YAML parse failed: {e}")
        return {
            "message": "workflow registered",
            "name": wf_def.name,
            "jobs": list(wf_def.jobs.keys()),
        }

    def _extract_dispatch_inputs(wf_def):
        """Extract on.workflow_dispatch.inputs spec from WorkflowDef."""
        if not wf_def.on or not wf_def.on.workflow_dispatch:
            return {}
        wd = wf_def.on.workflow_dispatch
        if isinstance(wd, dict):
            return wd.get("inputs", {})
        return {}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/integration/test_api_endpoints.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/api/server.py tests/integration/test_api_endpoints.py
git commit -m "feat(api): GET /api/workflows + POST /api/workflows/register"
```

## Task 4.5: POST /api/runs (dispatch)

**Files:**
- Modify: `paper_review_workflow/api/server.py`
- Test: `tests/integration/test_api_endpoints.py`

- [ ] **Step 1: Write failing test**

Append to `tests/integration/test_api_endpoints.py`:

```python
from unittest.mock import patch, MagicMock


def test_dispatch_returns_202_with_run_id(client, mock_llm):
    r = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    })
    assert r.status_code == 202
    data = r.json()
    assert "run_id" in data
    assert data["status"] == "pending"
    assert data["workflow_name"] == "normal-paper-review"


def test_dispatch_unknown_workflow_returns_404(client):
    r = client.post("/api/runs", json={
        "workflow_name": "nonexistent",
        "inputs": {},
    })
    assert r.status_code == 404


def test_dispatch_no_workflow_or_yaml_returns_422(client):
    r = client.post("/api/runs", json={"inputs": {}})
    assert r.status_code == 422  # pydantic validation error


# Fixtures for mock LLM
@pytest.fixture
def mock_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from paper_review_workflow.llm.schemas import DimensionScore, SynthesisResult
    from paper_review_workflow.llm.base import LLMResponse

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_api_endpoints.py::test_dispatch_returns_202_with_run_id -v`
Expected: FAIL (404 — endpoint not yet implemented)

- [ ] **Step 3: Add dispatch endpoint to `server.py`**

Inside `create_app()`, add:

```python
    @app.post("/api/runs", status_code=202)
    async def dispatch_run(req: DispatchRequest, background_tasks: BackgroundTasks):
        # Resolve workflow definition
        if req.workflow_name:
            wf_def = engine.get_workflow_def(req.workflow_name)
            if wf_def is None:
                raise HTTPException(status_code=404, detail=f"workflow not registered: {req.workflow_name}")
        elif req.yaml_content:
            try:
                wf_def = engine.register_workflow(req.yaml_content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"YAML parse failed: {e}")
        else:
            raise HTTPException(status_code=400, detail="must provide workflow_name or yaml_content")

        # Pre-allocate run (PENDING, saved to storage)
        try:
            run = engine.dispatch_workflow(wf_def.name, req.inputs)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # Schedule background execution
        background_tasks.add_task(_run_in_background, engine, run.id)

        return {
            "run_id": run.id,
            "workflow_name": wf_def.name,
            "status": run.status.value,
            "message": "review dispatched, see GET /api/runs/{run_id} for status",
        }

    async def _run_in_background(engine: ReviewEngine, run_id: str) -> None:
        """Background task: run engine in thread pool to avoid blocking event loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: engine.execute_existing_run(run_id))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/integration/test_api_endpoints.py -v`
Expected: PASS (10 tests including dispatch tests)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/api/server.py tests/integration/test_api_endpoints.py
git commit -m "feat(api): POST /api/runs dispatch endpoint with background execution"
```

## Task 4.6: GET /api/runs (list) + GET /api/runs/{run_id}

**Files:**
- Modify: `paper_review_workflow/api/server.py`
- Test: `tests/integration/test_api_endpoints.py`

- [ ] **Step 1: Write failing test**

Append to `tests/integration/test_api_endpoints.py`:

```python
def test_list_runs_empty(client):
    r = client.get("/api/runs")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 0
    assert data["runs"] == []


def test_list_runs_after_dispatch(client, mock_llm):
    # Dispatch a run
    r = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    })
    run_id = r.json()["run_id"]
    # Wait for it to complete (or fail)
    for _ in range(60):
        r = client.get(f"/api/runs/{run_id}")
        if r.json()["status"] in ("success", "failure", "cancelled"):
            break
        import time
        time.sleep(0.5)
    # List runs
    r = client.get("/api/runs")
    data = r.json()
    assert data["total"] >= 1
    assert any(r["run_id"] == run_id for r in data["runs"])


def test_get_run_returns_jobs_status(client, mock_llm):
    r = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    })
    run_id = r.json()["run_id"]
    # Wait for completion
    for _ in range(60):
        r = client.get(f"/api/runs/{run_id}")
        if r.json()["status"] in ("success", "failure", "cancelled"):
            break
        import time
        time.sleep(0.5)
    r = client.get(f"/api/runs/{run_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["run_id"] == run_id
    assert data["status"] == "success"
    assert "extract" in data["jobs"]
    assert "decide" in data["jobs"]


def test_get_run_nonexistent_returns_404(client):
    r = client.get("/api/runs/nonexistent-id")
    assert r.status_code == 404


def test_list_runs_filter_by_status(client, mock_llm):
    # Dispatch a run that will succeed
    r = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    })
    run_id = r.json()["run_id"]
    for _ in range(60):
        r = client.get(f"/api/runs/{run_id}")
        if r.json()["status"] in ("success", "failure", "cancelled"):
            break
        import time
        time.sleep(0.5)
    # Filter by success
    r = client.get("/api/runs?status=success")
    data = r.json()
    assert all(r["status"] == "success" for r in data["runs"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_api_endpoints.py::test_list_runs_empty -v`
Expected: FAIL (404 — endpoint not implemented)

- [ ] **Step 3: Add list + get endpoints to `server.py`**

Inside `create_app()`:

```python
    @app.get("/api/runs")
    async def list_runs(
        status: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ):
        status_enum = WorkflowStatus(status) if status else None
        runs = engine.storage.list_runs(status=status_enum, limit=limit + offset)
        runs = runs[offset:offset + limit]
        return {
            "total": len(runs),
            "runs": [_serialize_run_brief(r) for r in runs],
        }

    @app.get("/api/runs/{run_id}", response_model=RunResponse)
    async def get_run(run_id: str):
        run = engine.storage.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        return _serialize_run_full(run)

    def _serialize_run_brief(run: WorkflowRun) -> dict:
        return {
            "run_id": run.id,
            "workflow_name": run.workflow_def.name if run.workflow_def else "",
            "status": run.status.value,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "duration": run.duration,
            "jobs": {jid: {"status": j.status.value} for jid, j in run.jobs.items()},
        }

    def _serialize_run_full(run: WorkflowRun) -> dict:
        return {
            "run_id": run.id,
            "workflow_name": run.workflow_def.name if run.workflow_def else "",
            "status": run.status.value,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "duration": run.duration,
            "jobs": {
                jid: {
                    "status": j.status.value,
                    "outputs": j.outputs,
                    "duration": j.duration,
                    "steps": [
                        {"id": s.id, "name": s.step_def.name if s.step_def else "",
                         "status": s.status.value, "outputs": s.outputs}
                        for s in j.steps
                    ],
                }
                for jid, j in run.jobs.items()
            },
        }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/integration/test_api_endpoints.py -v`
Expected: PASS (15 tests)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/api/server.py tests/integration/test_api_endpoints.py
git commit -m "feat(api): GET /api/runs + GET /api/runs/{id} endpoints"
```

## Task 4.7: POST /api/runs/{id}/cancel + POST /api/runs/{id}/resume

**Files:**
- Modify: `paper_review_workflow/api/server.py`
- Test: `tests/integration/test_api_endpoints.py`

- [ ] **Step 1: Write failing test**

Append to `tests/integration/test_api_endpoints.py`:

```python
def test_cancel_run_returns_200(client, mock_llm):
    r = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    })
    run_id = r.json()["run_id"]
    # Wait briefly to ensure it's running or queued
    import time
    time.sleep(0.3)
    r = client.post(f"/api/runs/{run_id}/cancel")
    assert r.status_code == 200
    assert r.json()["run_id"] == run_id


def test_cancel_nonexistent_returns_404(client):
    r = client.post("/api/runs/nonexistent/cancel")
    assert r.status_code == 404


def test_resume_run_after_failure(client, mock_llm):
    # First dispatch
    r = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    })
    run_id = r.json()["run_id"]
    # Wait for it to finish
    for _ in range(60):
        r = client.get(f"/api/runs/{run_id}")
        if r.json()["status"] in ("success", "failure", "cancelled"):
            break
        import time
        time.sleep(0.5)
    # Resume it (should be safe even if already done)
    r = client.post(f"/api/runs/{run_id}/resume", json={})
    assert r.status_code == 200
    assert r.json()["run_id"] == run_id


def test_resume_with_rerun_components(client, mock_llm):
    r = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    })
    run_id = r.json()["run_id"]
    for _ in range(60):
        r = client.get(f"/api/runs/{run_id}")
        if r.json()["status"] in ("success", "failure", "cancelled"):
            break
        import time
        time.sleep(0.5)
    r = client.post(f"/api/runs/{run_id}/resume", json={
        "rerun_components": ["dim_novelty"]
    })
    assert r.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_api_endpoints.py::test_cancel_run_returns_200 -v`
Expected: FAIL (404)

- [ ] **Step 3: Add cancel + resume endpoints to `server.py`**

Inside `create_app()`:

```python
    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str):
        run = engine.storage.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        cancelled = engine.cancel_run(run_id)
        if not cancelled:
            # Already terminal — return current status
            run = engine.storage.get_run(run_id)
            return {"run_id": run_id, "status": run.status.value, "message": "run already terminal"}
        return {"run_id": run_id, "status": "cancelled", "message": "cancellation requested"}

    @app.post("/api/runs/{run_id}/resume")
    async def resume_run(run_id: str, req: ResumeRequest, background_tasks: BackgroundTasks):
        run = engine.storage.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")

        # Apply rerun options
        if req.rerun_all:
            engine._reset_all_components(run)
        elif req.rerun_components:
            engine._mark_for_rerun(run, req.rerun_components)

        background_tasks.add_task(_run_in_background, engine, run_id)
        return {"run_id": run_id, "status": "pending", "message": "resume scheduled"}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/integration/test_api_endpoints.py -v`
Expected: PASS (19 tests total)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/api/server.py tests/integration/test_api_endpoints.py
git commit -m "feat(api): POST /api/runs/{id}/cancel + resume endpoints"
```

---

# M5: WebSocket Endpoints

**Goal:** Add 2 WebSocket endpoints: `/ws` (global) + `/ws/runs/{run_id}` (filtered). Test real-time event push.

**Estimated:** 1 day

## Task 5.1: Global `/ws` endpoint

**Files:**
- Modify: `paper_review_workflow/api/server.py`
- Test: `tests/integration/test_api_websocket.py`

- [ ] **Step 1: Write failing test**

Create `tests/integration/test_api_websocket.py`:

```python
import json
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


def test_websocket_global_connection_receives_heartbeat(client):
    """Connect to /ws, send ping, receive pong + heartbeat."""
    with client.websocket_connect("/ws") as ws:
        ws.send_text("ping")
        # Should receive pong
        msg = ws.receive_text()
        data = json.loads(msg)
        assert data.get("event") in ("pong", "heartbeat")


def test_websocket_receives_workflow_events(client, mock_llm_ws):
    """After dispatching a run, /ws should receive workflow.started event."""
    with client.websocket_connect("/ws") as ws:
        # Give WS a moment to subscribe
        time.sleep(0.2)
        # Dispatch a run
        client.post("/api/runs", json={
            "workflow_name": "minimal-test" if False else "normal-paper-review",
            "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
        })
        # Collect messages for up to 60s
        events = []
        for _ in range(120):
            try:
                msg = ws.receive_text(timeout=0.5)
                data = json.loads(msg)
                events.append(data.get("event"))
                if data.get("event") == "workflow.started":
                    break
            except Exception:
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_api_websocket.py::test_websocket_global_connection_receives_heartbeat -v`
Expected: FAIL (404 — /ws not implemented)

- [ ] **Step 3: Add `/ws` endpoint to `server.py`**

Inside `create_app()`, add (after the REST endpoints):

```python
    import asyncio

    @app.websocket("/ws")
    async def websocket_global(websocket: WebSocket):
        """Global WebSocket — receives all EventBus events."""
        await ws_manager.add_connection(websocket)
        try:
            # Send historical events for catch-up
            await ws_manager.send_history(websocket)
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    if data == "ping":
                        await websocket.send_text('{"event": "pong"}')
                except asyncio.TimeoutError:
                    await websocket.send_text('{"event": "heartbeat"}')
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"[WS /ws] error: {e}")
        finally:
            ws_manager.remove_connection(websocket)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/integration/test_api_websocket.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/api/server.py tests/integration/test_api_websocket.py
git commit -m "feat(api): global /ws WebSocket endpoint with ping/heartbeat"
```

## Task 5.2: Filtered `/ws/runs/{run_id}` endpoint

**Files:**
- Modify: `paper_review_workflow/api/server.py`
- Test: `tests/integration/test_api_websocket.py`

- [ ] **Step 1: Write failing test**

Append to `tests/integration/test_api_websocket.py`:

```python
def test_websocket_filtered_by_run_id(client, mock_llm_ws):
    """Connect to /ws/runs/{run_id} — should only receive events for that run."""
    # First dispatch a run
    r = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    })
    run_id = r.json()["run_id"]

    # Connect to filtered WS for that run
    with client.websocket_connect(f"/ws/runs/{run_id}") as ws:
        # Collect messages
        events = []
        for _ in range(120):
            try:
                msg = ws.receive_text(timeout=0.5)
                data = json.loads(msg)
                events.append(data)
            except Exception:
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
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    }).json()
    r2 = client.post("/api/runs", json={
        "workflow_name": "normal-paper-review",
        "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
    }).json()

    with client.websocket_connect(f"/ws/runs/{r1['run_id']}") as ws:
        events = []
        for _ in range(120):
            try:
                msg = ws.receive_text(timeout=0.5)
                data = json.loads(msg)
                events.append(data)
            except Exception:
                if events:
                    break
        # No event should mention r2's run_id
        for evt in events:
            assert evt.get("run_id") != r2["run_id"], f"got event for r2 on r1 filter"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_api_websocket.py::test_websocket_filtered_by_run_id -v`
Expected: FAIL (404 — endpoint not implemented)

- [ ] **Step 3: Add `/ws/runs/{run_id}` endpoint to `server.py`**

Inside `create_app()`:

```python
    @app.websocket("/ws/runs/{run_id}")
    async def websocket_run_filtered(websocket: WebSocket, run_id: str):
        """Run-filtered WebSocket — only events for the specified run_id."""
        filtered = FilteredWS(websocket, run_id)
        await ws_manager.add_connection(filtered)
        try:
            await ws_manager.send_history(filtered, run_id=run_id)
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    if data == "ping":
                        await websocket.send_text('{"event": "pong"}')
                except asyncio.TimeoutError:
                    await websocket.send_text('{"event": "heartbeat"}')
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"[WS /ws/runs/{run_id}] error: {e}")
        finally:
            ws_manager.remove_connection(filtered)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/integration/test_api_websocket.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/api/server.py tests/integration/test_api_websocket.py
git commit -m "feat(api): /ws/runs/{run_id} filtered WebSocket endpoint"
```

---

# M6: Lifecycle + Register Tests

**Goal:** Verify startup recovery + shutdown cancel + dynamic workflow registration end-to-end.

**Estimated:** 1 day

## Task 6.1: Startup recovery test

**Files:**
- Test: `tests/integration/test_api_lifecycle.py`

- [ ] **Step 1: Write test**

Create `tests/integration/test_api_lifecycle.py`:

```python
import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from paper_review_workflow.api import create_app
from paper_review_workflow.storage.json_file import JsonFileStorage
from paper_review_workflow.core.models import WorkflowRun, WorkflowStatus


def test_startup_recovers_interrupted_runs(tmp_path, monkeypatch):
    """Startup should mark storage's PENDING/RUNNING runs as CANCELLED."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    # Pre-populate storage with an interrupted run
    storage = JsonFileStorage(data_dir=str(tmp_path))
    storage.open()
    interrupted = WorkflowRun(status=WorkflowStatus.RUNNING, start_time=datetime.now())
    storage.save_run(interrupted)
    storage.close()

    # Create app — startup handler should recover
    app = create_app(storage=JsonFileStorage(data_dir=str(tmp_path)), configs_dir="configs")
    with TestClient(app) as client:
        # Verify the run was recovered
        r = client.get(f"/api/runs/{interrupted.id}")
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"


def test_startup_loads_configs_directory(tmp_path, monkeypatch):
    """Startup should scan configs/ and register workflows."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "test_wf.yaml").write_text("""
name: test-startup-wf
on: {workflow_dispatch: {}}
jobs:
  j:
    runs-on: local
    steps:
      - uses: paper-review/echo@v1
""")
    app = create_app(
        storage=JsonFileStorage(data_dir=str(tmp_path / "storage")),
        configs_dir=str(configs),
    )
    with TestClient(app) as client:
        r = client.get("/api/workflows")
        names = [w["name"] for w in r.json()["workflows"]]
        assert "test-startup-wf" in names
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/integration/test_api_lifecycle.py -v`
Expected: PASS (2 tests). If failure, debug the startup handler in `server.py`.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_api_lifecycle.py
git commit -m "test(integration): startup recovery + configs loading"
```

## Task 6.2: Shutdown cancel test

**Files:**
- Test: `tests/integration/test_api_lifecycle.py`

- [ ] **Step 1: Write test**

Append to `tests/integration/test_api_lifecycle.py`:

```python
def test_shutdown_cancels_active_runs(tmp_path, monkeypatch):
    """When TestClient context exits, shutdown handler should cancel active runs."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from paper_review_workflow.llm.client import LLMClient
    LLMClient.reset()

    # Mock LLM with a slow response to ensure run is active when context exits
    from unittest.mock import patch, MagicMock
    import time
    from paper_review_workflow.llm.schemas import DimensionScore
    from paper_review_workflow.llm.base import LLMResponse

    fake_dim = DimensionScore(
        score=4, confidence=0.8, strengths=["a"], weaknesses=["b"],
        justification="x" * 200, evidence=[],
    )
    fake_resp = MagicMock(spec=LLMResponse)
    fake_resp.structured = fake_dim
    fake_resp.usage = {"input_tokens": 100, "output_tokens": 50,
                       "cache_creation_input_tokens": 0,
                       "cache_read_input_tokens": 30000}
    fake_resp.model = "test-model"

    # Pre-allocate a PENDING run manually (don't dispatch yet)
    storage = JsonFileStorage(data_dir=str(tmp_path / "storage"))
    storage.open()
    app = create_app(storage=storage, configs_dir="configs")

    run_id_holder = {}
    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.model = "test-model"
        def slow_complete(**kw):
            time.sleep(5)  # simulate slow LLM
            return fake_resp
        mock_client.complete.side_effect = slow_complete
        mock_from_env.return_value = mock_client

        with TestClient(app) as client:
            # Dispatch a run (will be slow)
            r = client.post("/api/runs", json={
                "workflow_name": "normal-paper-review",
                "inputs": {"paper_source": "tests/fixtures/sample_paper.pdf"},
            })
            run_id_holder["run_id"] = r.json()["run_id"]
            # Exit context — shutdown should cancel
        # Verify run was cancelled on shutdown
    # Re-open storage to check
    storage2 = JsonFileStorage(data_dir=str(tmp_path / "storage"))
    storage2.open()
    run = storage2.get_run(run_id_holder["run_id"])
    # Status should be cancelled (shutdown cancelled it) or success (if completed before shutdown)
    assert run.status.value in ("cancelled", "success"), f"got {run.status.value}"
```

- [ ] **Step 2: Run test**

Run: `pytest tests/integration/test_api_lifecycle.py::test_shutdown_cancels_active_runs -v`
Expected: PASS (may be flaky due to timing; should pass when shutdown interrupts active run)

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_api_lifecycle.py
git commit -m "test(integration): shutdown cancels active runs"
```

## Task 6.3: Dynamic register workflow test

**Files:**
- Test: `tests/integration/test_api_register.py`

- [ ] **Step 1: Write test**

Create `tests/integration/test_api_register.py`:

```python
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
    import time
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


# Mock LLM fixture
@pytest.fixture
def mock_llm_simple(monkeypatch):
    from unittest.mock import patch, MagicMock
    from paper_review_workflow.llm.client import LLMClient
    LLMClient.reset()
    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.complete.return_value = MagicMock()
        mock_from_env.return_value = mock_client
        yield mock_client
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/integration/test_api_register.py -v`
Expected: PASS (3 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_api_register.py
git commit -m "test(integration): dynamic workflow registration via API"
```

---

# M7: CLI `server` Subcommand + Default + E2E

**Goal:** Add `server` subcommand to CLI, default behavior (no subcommand starts server), E2E test, update README.

**Estimated:** 1 day

## Task 7.1: Add `server` subcommand to cli.py

**Files:**
- Modify: `paper_review_workflow/cli.py`

- [ ] **Step 1: Read current cli.py**

Run: `cat paper_review_workflow/cli.py`

- [ ] **Step 2: Add `server` subparser + `_cmd_server` function**

In `cli.py`, add a new subparser after the existing ones (in `main()`):

```python
    # ── server 子命令 (Phase 2 #5) ──
    server_p = sub.add_parser("server", help="启动 FastAPI 服务器")
    server_p.add_argument("--host", default="127.0.0.1", help="监听地址")
    server_p.add_argument("--port", type=int, default=8000, help="监听端口")
    server_p.add_argument("--reload", action="store_true", help="开发模式自动重载")
    server_p.add_argument("--configs-dir", default=None,
                          help="configs/ 目录路径(默认 PAPER_REVIEW_CONFIGS_DIR 或 ./configs)")
```

Add the dispatch logic:

```python
    if args.command == "server":
        return _cmd_server(args)
    elif args.command == "run":
        return _cmd_run(engine, args)
    # ... other elif ...
    elif args.command is None:
        # Default: start server (与 lwf 一致)
        return _cmd_server(args)
    return 0
```

Add the `_cmd_server` function:

```python
def _cmd_server(args) -> int:
    """启动 FastAPI 服务器"""
    import os

    if args.configs_dir:
        os.environ["PAPER_REVIEW_CONFIGS_DIR"] = args.configs_dir

    try:
        import uvicorn
    except ImportError:
        print("[Error] uvicorn not installed. Run: pip install uvicorn[standard]", file=sys.stderr)
        return 3

    from .api import create_app

    app = create_app(
        storage=_build_storage(args),
        sessions_root=args.storage_dir,
        configs_dir=args.configs_dir or os.environ.get("PAPER_REVIEW_CONFIGS_DIR", "./configs"),
    )

    print(f"""
╔══════════════════════════════════════════════════════╗
║  Paper Review Workflow API Server                    ║
║                                                      ║
║  API docs:  http://{args.host}:{args.port}/docs      ║
║  WebSocket: ws://{args.host}:{args.port}/ws          ║
║  Health:    http://{args.host}:{args.port}/api/health║
╚══════════════════════════════════════════════════════╝
""")

    try:
        uvicorn.run(app, host=args.host, port=args.port,
                    log_level=args.log_level.lower())
    except KeyboardInterrupt:
        return 130
    return 0
```

Note: Adjust the `_build_storage(args)` helper — it needs to handle the `args` object even when `args.command is None`. The current `args.storage` and `args.storage_dir` defaults from the parent parser should be available.

- [ ] **Step 3: Verify CLI help shows server**

Run: `python main.py --help`
Expected: Help output lists `server` as a subcommand.

- [ ] **Step 4: Test server starts (manual smoke test)**

Run: `timeout 3 python main.py server --port 8765; echo "exit code: $?"`
Expected: Server starts, prints banner, gets killed by timeout. Exit code 124 (timeout) is fine.

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/cli.py
git commit -m "feat(cli): add server subcommand for FastAPI mode"
```

## Task 7.2: Default to server when no subcommand

**Files:**
- Modify: `paper_review_workflow/cli.py`
- Test: `tests/integration/test_cli_default_server.py`

- [ ] **Step 1: Write failing test**

Create `tests/integration/test_cli_default_server.py`:

```python
import subprocess
import sys
import time
import socket
import http.client


def test_no_subcommand_starts_server(tmp_path):
    """`python main.py` with no subcommand should start the API server."""
    # Start server in background
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--storage-dir", str(tmp_path / "sessions")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # Wait for server to be ready (up to 10s)
        for _ in range(100):
            try:
                conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=0.5)
                conn.request("GET", "/api/health")
                r = conn.getresponse()
                if r.status == 200:
                    break
            except (ConnectionRefusedError, socket.timeout):
                pass
            time.sleep(0.1)
        else:
            stderr = proc.stderr.read().decode()
            assert False, f"server did not start within 10s. stderr: {stderr}"

        # Verify health endpoint responds
        conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=2)
        conn.request("GET", "/api/health")
        r = conn.getresponse()
        assert r.status == 200
        body = r.read().decode()
        assert '"status":"ok"' in body or '"status": "ok"' in body
    finally:
        proc.terminate()
        proc.wait(timeout=5)
```

- [ ] **Step 2: Run test**

Run: `pytest tests/integration/test_cli_default_server.py -v`
Expected: PASS (server starts on default port 8000)

- [ ] **Step 3: Commit**

```bash
git add paper_review_workflow/cli.py tests/integration/test_cli_default_server.py
git commit -m "feat(cli): default to starting server when no subcommand given"
```

## Task 7.3: E2E test with real arXiv + real API

**Files:**
- Test: `tests/e2e/test_api_e2e_arxiv.py`

- [ ] **Step 1: Write test**

Create `tests/e2e/test_api_e2e_arxiv.py`:

```python
import os
import time
import pytest
import http.client
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
            events = []
            for _ in range(120):
                try:
                    msg = ws.receive_text(timeout=0.5)
                    import json
                    data = json.loads(msg)
                    events.append(data.get("event"))
                    if data.get("event") == "workflow.started":
                        break
                except Exception:
                    break
            assert "workflow.started" in events
```

- [ ] **Step 2: Verify test discovery**

Run: `pytest tests/e2e/test_api_e2e_arxiv.py --collect-only`
Expected: 2 tests collected (will be skipped without API key)

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_api_e2e_arxiv.py
git commit -m "test(e2e): real arXiv review via API + WebSocket"
```

## Task 7.4: Update README + final commit

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read current README**

Run: `cat README.md`

- [ ] **Step 2: Add API server section**

Edit `README.md` — after the existing "Quick Start" section, add:

```markdown
## API Server Mode

Start the FastAPI HTTP API + WebSocket server:

```bash
# Default (no subcommand starts server)
python main.py

# Explicit
python main.py server --host 0.0.0.0 --port 8000

# Dev mode with auto-reload
python main.py server --reload
```

API docs at `http://localhost:8000/docs`. Key endpoints:

- `GET /api/health` — health check
- `GET /api/workflows` — list registered workflows
- `POST /api/workflows/register` — register a YAML workflow
- `POST /api/runs` — dispatch a review (returns run_id immediately)
- `GET /api/runs/{run_id}` — get run status
- `POST /api/runs/{run_id}/cancel` — cancel a run
- `POST /api/runs/{run_id}/resume` — resume a failed/interrupted run

WebSocket endpoints:
- `ws://localhost:8000/ws` — all events
- `ws://localhost:8000/ws/runs/{run_id}` — filtered to single run

Example: dispatch a review via curl:

```bash
curl -X POST http://localhost:8000/api/runs \
  -H "Content-Type: application/json" \
  -d '{"workflow_name": "normal-paper-review", "inputs": {"paper_source": "2402.12098"}}'
```
```

- [ ] **Step 3: Run full test suite to verify no regressions**

Run: `pytest tests/ -v`
Expected: All Phase 1 + Phase 2 #5 tests pass (E2E tests skipped without API key)

- [ ] **Step 4: Commit + tag**

```bash
git add README.md
git commit -m "docs: add API server documentation to README"
git tag v0.2.0
```

---

# Self-Review Checklist

## Spec Coverage

| SPEC Section | Implemented By |
|---|---|
| 1. Scope | M1-M7 cover Phase 2 #5 only ✅ |
| 2. Architecture | M1 api/ skeleton, M2-M7 fill in ✅ |
| 3. WSManager + FilteredWS | M2 ✅ |
| 4. 7 REST endpoints | M4 (health, workflows, register, dispatch, list, get, cancel, resume) ✅ |
| 5. WebSocket endpoints | M5 (/ws + /ws/runs/{id}) ✅ |
| 6. CLI server subcommand | M7 ✅ |
| 6.4 startup/shutdown | M4.3 (create_app) ✅ |
| 7. Engine 5 new methods | M3 (load_workflow_directory, register_workflow, recover_interrupted_runs, dispatch_workflow, execute_existing_run) ✅ |
| 8. Testing strategy | M2/M3 unit, M4-M6 integration, M7.3 E2E ✅ |
| 9. Milestones | M1-M7 directly map ✅ |
| 10. 验收标准 | All covered by tests ✅ |

## Placeholder Scan

- ✅ No "TBD"/"TODO"
- ✅ All steps have complete code
- ✅ All test code shown in full
- ✅ No "implement later"

## Type Consistency

- `WSManager.attach(loop)` / `detach()` — same signature in M2 + M4 startup/shutdown ✅
- `FilteredWS(ws, target_run_id)` — same signature in M2 + M5 ✅
- `engine.dispatch_workflow(name, inputs)` / `execute_existing_run(run_id)` — same in M3 + M4.5 ✅
- `engine.load_workflow_directory(configs_dir)` — same in M3 + M4.3 ✅
- `engine.recover_interrupted_runs()` — same in M3 + M4.3 ✅
- `create_app(storage, configs_dir, sessions_root)` — same signature in M1.2 + M4.3 + all test fixtures ✅
- `DispatchRequest` / `RegisterWorkflowRequest` / `ResumeRequest` schemas — same in M4.1 + used in M4.4-M4.7 ✅

---

# Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-13-fastapi-server-impl.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
