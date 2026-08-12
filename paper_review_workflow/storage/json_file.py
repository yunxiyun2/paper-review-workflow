"""
JSON file storage backend.
Persists WorkflowRun as JSON files to disk.

Directory structure:
  <data_dir>/
    runs/
      <run_id>.json      <- one file per WorkflowRun
    index.json           <- quick index (run_id -> {status, wf_name, start_time})

The workflow_def field of WorkflowRun (which contains the full YAML definition
object) is not serialized. On restart, the engine re-parses the workflow
definition; runtime info (jobs/steps/status/times) is fully saved.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.models import (
    WorkflowRun, WorkflowStatus,
    JobInstance, JobStatus,
    StepInstance, StepStatus,
    StepDef, JobDef,
)
from .backend import StorageBackend

logger = logging.getLogger(__name__)


# -----------------------------------------------
# Serialization / deserialization helpers
# -----------------------------------------------

def _serialize_run(run: WorkflowRun) -> Dict[str, Any]:
    """Convert WorkflowRun to a JSON-serializable dict."""
    return {
        "schema_version":   1,
        "id":               run.id,
        "workflow_name":    run.workflow_def.name if run.workflow_def else "",
        "workflow_file":    run.workflow_def.file_path if run.workflow_def else None,
        "status":           run.status.value,
        "trigger_type":     run.trigger_type,
        "trigger_payload":  run.trigger_payload,
        "start_time":       run.start_time.isoformat() if run.start_time else None,
        "end_time":         run.end_time.isoformat() if run.end_time else None,
        "run_number":       run.run_number,
        "env":              run.env,
        "jobs": {
            job_id: _serialize_job(ji)
            for job_id, ji in run.jobs.items()
        },
    }


def _serialize_job(ji: JobInstance) -> Dict[str, Any]:
    return {
        "id":          ji.id,
        "job_id":      ji.job_def.id if ji.job_def else "",
        "job_name":    ji.job_def.name if ji.job_def else "",
        "status":      ji.status.value,
        "start_time":  ji.start_time.isoformat() if ji.start_time else None,
        "end_time":    ji.end_time.isoformat() if ji.end_time else None,
        "outputs":     ji.outputs,
        "runner_info": ji.runner_info,
        "steps":       [_serialize_step(si) for si in ji.steps],
        "needs":       ji.job_def.needs if ji.job_def else [],
    }


def _serialize_step(si: StepInstance) -> Dict[str, Any]:
    step_def = si.step_def
    return {
        "id":           si.id,
        "step_id":      step_def.id if step_def else "",
        "step_name":    step_def.name if step_def else "",
        "action_type":  step_def.uses if step_def else None,
        "status":       si.status.value,
        "start_time":   si.start_time.isoformat() if si.start_time else None,
        "end_time":     si.end_time.isoformat() if si.end_time else None,
        "exit_code":    si.exit_code,
        "error_msg":    si.error_msg,
        "outputs":      si.outputs,
        "log":          si.log,
    }


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _deserialize_run(data: Dict[str, Any]) -> WorkflowRun:
    """
    Restore a WorkflowRun runtime object from a JSON dict.
    The workflow_def field is not restored (engine re-associates on reload);
    only name/path info is preserved in env for the engine to re-parse.
    """
    run = WorkflowRun(
        id=data["id"],
        status=WorkflowStatus(data["status"]),
        trigger_type=data.get("trigger_type", "manual"),
        trigger_payload=data.get("trigger_payload", {}),
        start_time=_parse_dt(data.get("start_time")),
        end_time=_parse_dt(data.get("end_time")),
        run_number=data.get("run_number", 1),
        env=data.get("env", {}),
    )
    # Store workflow name/file in env for engine to re-parse (workflow_def not deserialized)
    run.env.setdefault("__workflow_name__", data.get("workflow_name", ""))
    run.env.setdefault("__workflow_file__", data.get("workflow_file") or "")

    # Restore Job instances (runtime data only, with minimal JobDef for to_dict())
    for job_id, jd in data.get("jobs", {}).items():
        # Build minimal JobDef from persisted data so to_dict() returns correct name/id/needs
        minimal_job_def = JobDef(
            id=jd.get("job_id", job_id),
            name=jd.get("job_name", jd.get("name", job_id)),
            runs_on=jd.get("runs_on", ""),
            needs=jd.get("needs", []),
        )
        ji = JobInstance(
            id=jd["id"],
            job_def=minimal_job_def,
            status=JobStatus(jd["status"]),
            start_time=_parse_dt(jd.get("start_time")),
            end_time=_parse_dt(jd.get("end_time")),
            outputs=jd.get("outputs", {}),
            runner_info=jd.get("runner_info", {}),
        )
        for sd in jd.get("steps", []):
            # Build minimal StepDef from persisted data so to_dict() returns correct name/id
            minimal_step_def = StepDef(
                id=sd.get("step_id", ""),
                name=sd.get("step_name", sd.get("name", "")),
                uses=sd.get("action_type"),
            )
            si = StepInstance(
                id=sd["id"],
                step_def=minimal_step_def,
                status=StepStatus(sd["status"]),
                start_time=_parse_dt(sd.get("start_time")),
                end_time=_parse_dt(sd.get("end_time")),
                exit_code=sd.get("exit_code", 0),
                error_msg=sd.get("error_msg", ""),
                outputs=sd.get("outputs", {}),
                log=sd.get("log", []),
            )
            ji.steps.append(si)
        run.jobs[job_id] = ji

    return run


# -----------------------------------------------
# JsonFileStorage
# -----------------------------------------------

class JsonFileStorage(StorageBackend):
    """
    JSON file storage backend.
    Each WorkflowRun persisted as an independent JSON file, with an in-memory
    index for fast queries.
    Thread-safe: uses RLock for all write operations.
    """

    def __init__(self, data_dir: str = "./workflow_data"):
        self._data_dir  = Path(data_dir)
        self._runs_dir  = self._data_dir / "runs"
        self._index_file = self._data_dir / "index.json"
        self._lock      = threading.RLock()
        # In-memory index { run_id: {status, wf_name, start_time, end_time} }
        self._index: Dict[str, Dict] = {}

    # -- Lifecycle --

    def open(self) -> None:
        """Create directories, load existing index."""
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._load_index()
        logger.info(f"[JsonFileStorage] opened: {self._data_dir}, {len(self._index)} records")

    def close(self) -> None:
        self._flush_index()

    # -- Write operations --

    def save_run(self, run: WorkflowRun) -> None:
        data = _serialize_run(run)
        path = self._runs_dir / f"{run.id}.json"
        with self._lock:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            self._index[run.id] = self._make_index_entry(run)
            self._flush_index()
        logger.debug(f"[JsonFileStorage] saved {run.id[:8]} -> {path.name}")

    def delete_run(self, run_id: str) -> bool:
        path = self._runs_dir / f"{run_id}.json"
        with self._lock:
            if path.exists():
                path.unlink()
            if run_id in self._index:
                del self._index[run_id]
                self._flush_index()
                return True
        return False

    # -- Read operations --

    def get_run(self, run_id: str) -> Optional[WorkflowRun]:
        path = self._runs_dir / f"{run_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _deserialize_run(data)
        except Exception as e:
            logger.error(f"[JsonFileStorage] failed to read {run_id}: {e}")
            return None

    def list_runs(
        self,
        workflow_name: Optional[str]            = None,
        status:        Optional[WorkflowStatus] = None,
        limit:         int                      = 100,
        offset:        int                      = 0,
    ) -> List[WorkflowRun]:
        with self._lock:
            entries = list(self._index.items())

        # Fast filter via index
        if workflow_name:
            entries = [(rid, e) for rid, e in entries if e.get("wf_name") == workflow_name]
        if status:
            entries = [(rid, e) for rid, e in entries if e.get("status") == status.value]

        # Sort by start_time descending
        entries.sort(
            key=lambda x: x[1].get("start_time") or "",
            reverse=True,
        )
        page = entries[offset: offset + limit]

        runs = []
        for run_id, _ in page:
            run = self.get_run(run_id)
            if run:
                runs.append(run)
        return runs

    def count_runs(
        self,
        workflow_name: Optional[str]            = None,
        status:        Optional[WorkflowStatus] = None,
    ) -> int:
        with self._lock:
            entries = list(self._index.values())
        if workflow_name:
            entries = [e for e in entries if e.get("wf_name") == workflow_name]
        if status:
            entries = [e for e in entries if e.get("status") == status.value]
        return len(entries)

    # -- Clear --

    def clear(self) -> None:
        with self._lock:
            for path in self._runs_dir.glob("*.json"):
                path.unlink()
            self._index.clear()
            self._flush_index()
        logger.info("[JsonFileStorage] cleared all records")

    # -- Internal utilities --

    def _make_index_entry(self, run: WorkflowRun) -> Dict:
        return {
            "wf_name":   run.workflow_def.name if run.workflow_def else run.env.get("__workflow_name__", ""),
            "status":    run.status.value,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time":  run.end_time.isoformat() if run.end_time else None,
            "run_number": run.run_number,
        }

    def _flush_index(self) -> None:
        try:
            self._index_file.write_text(
                json.dumps(self._index, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"[JsonFileStorage] failed to write index: {e}")

    def _load_index(self) -> None:
        if self._index_file.exists():
            try:
                self._index = json.loads(self._index_file.read_text(encoding="utf-8"))
                logger.debug(f"[JsonFileStorage] loaded index: {len(self._index)} records")
            except Exception as e:
                logger.error(f"[JsonFileStorage] failed to load index: {e}")
                self._index = {}
        else:
            self._index = {}
            # Scan existing files to rebuild index
            for f in self._runs_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    self._index[data["id"]] = {
                        "wf_name":    data.get("workflow_name", ""),
                        "status":     data.get("status", "unknown"),
                        "start_time": data.get("start_time"),
                        "end_time":   data.get("end_time"),
                        "run_number": data.get("run_number", 1),
                    }
                except Exception:
                    pass
            if self._index:
                self._flush_index()

    def __repr__(self) -> str:
        return f"<JsonFileStorage dir={self._data_dir} runs={len(self._index)}>"
