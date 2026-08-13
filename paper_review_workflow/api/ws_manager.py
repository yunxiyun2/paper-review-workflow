"""WebSocket connection management + event broadcast.

WSManager acts as an EventBus subscriber. When the (synchronous) engine publishes
events via EventBus, WSManager receives them in the engine thread and schedules
async broadcasts to all WS connections via run_coroutine_threadsafe.
"""
import asyncio
import json
import logging
from typing import Callable, Optional, Set

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
                try:
                    asyncio.run_coroutine_threadsafe(
                        ws.close(), self._loop
                    )
                except Exception as e:
                    logger.debug(f"[WSManager] close failed: {e}")
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
