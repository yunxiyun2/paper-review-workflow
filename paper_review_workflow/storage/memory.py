"""
In-memory storage backend.
All data stored in Python dicts, lost on process exit.
Suitable for: development, testing, embedded use cases.
Thread-safe: uses threading.RLock for concurrent write protection.

Race condition protection:
  save_run / get_run both use copy.deepcopy to store/return snapshots,
  preventing callers' in-memory references (e.g. background exec threads)
  from mutating stored objects after persistence, which would cause
  "state polluted after save" race condition bugs.
"""
from __future__ import annotations

import copy
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional

from ..core.models import WorkflowRun, WorkflowStatus
from .backend import StorageBackend

logger = logging.getLogger(__name__)


class MemoryStorage(StorageBackend):
    """
    In-memory dict-based storage backend (thread-safe).
    """

    def __init__(self):
        # { run_id: WorkflowRun }
        self._store: Dict[str, WorkflowRun] = {}
        self._lock  = threading.RLock()

    # -- Write operations --

    def save_run(self, run: WorkflowRun) -> None:
        # Store deep-copy snapshot to prevent caller mutations polluting persisted state
        snapshot = copy.deepcopy(run)
        with self._lock:
            self._store[run.id] = snapshot
            logger.debug(f"[MemoryStorage] save run {run.id[:8]} status={run.status.value}")

    def delete_run(self, run_id: str) -> bool:
        with self._lock:
            if run_id in self._store:
                del self._store[run_id]
                logger.debug(f"[MemoryStorage] delete run {run_id[:8]}")
                return True
            return False

    # -- Read operations --

    def get_run(self, run_id: str) -> Optional[WorkflowRun]:
        # Return deep copy to prevent caller mutations affecting stored snapshot
        with self._lock:
            snapshot = self._store.get(run_id)
            return copy.deepcopy(snapshot) if snapshot is not None else None

    def list_runs(
        self,
        workflow_name: Optional[str]            = None,
        status:        Optional[WorkflowStatus] = None,
        limit:         int                      = 100,
        offset:        int                      = 0,
    ) -> List[WorkflowRun]:
        with self._lock:
            runs = list(self._store.values())

        # Filter
        if workflow_name:
            runs = [
                r for r in runs
                if r.workflow_def and r.workflow_def.name == workflow_name
            ]
        if status:
            runs = [r for r in runs if r.status == status]

        # Sort by start_time descending (None sorts last)
        runs.sort(
            key=lambda r: r.start_time or datetime.min,
            reverse=True,
        )
        return runs[offset: offset + limit]

    def count_runs(
        self,
        workflow_name: Optional[str]            = None,
        status:        Optional[WorkflowStatus] = None,
    ) -> int:
        return len(self.list_runs(workflow_name=workflow_name, status=status, limit=10**9))

    # -- Bulk / utilities --

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
        logger.info("[MemoryStorage] cleared all records")

    def bulk_save(self, runs: List[WorkflowRun]) -> None:
        with self._lock:
            for run in runs:
                self._store[run.id] = run

    def all_run_ids(self) -> List[str]:
        """Return all run_ids (for debugging)."""
        with self._lock:
            return list(self._store.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        return f"<MemoryStorage runs={len(self)}>"
