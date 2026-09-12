"""Event bus for workflow state changes.

Ported from lwf, removed WAITING/RESUMED/RETRY events (no human approval).
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)


class EventType(Enum):
    WORKFLOW_CREATED   = "workflow.created"
    WORKFLOW_STARTED   = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_CANCELLED = "workflow.cancelled"

    JOB_QUEUED    = "job.queued"
    JOB_STARTED   = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_SKIPPED   = "job.skipped"

    STEP_STARTED   = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_SKIPPED   = "step.skipped"
    STEP_LOG       = "step.log"

    ENGINE_READY = "engine.ready"
    ERROR        = "error"


@dataclass
class WorkflowEvent:
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    run_id: Optional[str] = None
    job_id: Optional[str] = None
    step_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
            "job_id": self.job_id,
            "step_id": self.step_id,
            "data": self.data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def make_workflow_started_event(run) -> WorkflowEvent:
    env = run.env or {}
    return WorkflowEvent(
        event_type=EventType.WORKFLOW_STARTED, run_id=run.id,
        data={"workflow_name": run.workflow_def.name if run.workflow_def else "",
              "trigger_type": run.trigger_type, "status": run.status.value,
              "task_name": (run.trigger_payload or {}).get("task_name", ""),
              "venue": env.get("VENUE", ""),
              "provider": env.get("LLM_PROVIDER", ""),
              "model": env.get("LLM_MODEL", "")},
    )


def make_workflow_completed_event(run) -> WorkflowEvent:
    return WorkflowEvent(
        event_type=EventType.WORKFLOW_COMPLETED, run_id=run.id,
        data={"status": run.status.value, "duration": run.duration},
    )


def make_job_started_event(run_id, job) -> WorkflowEvent:
    return WorkflowEvent(
        event_type=EventType.JOB_STARTED, run_id=run_id, job_id=job.job_def.id if job.job_def else None,
        data={"name": job.job_def.name if job.job_def else ""},
    )


def make_job_completed_event(run_id, job) -> WorkflowEvent:
    return WorkflowEvent(
        event_type=EventType.JOB_COMPLETED, run_id=run_id, job_id=job.job_def.id if job.job_def else None,
        data={"status": job.status.value, "duration": job.duration},
    )


def make_step_started_event(run_id, job_id, step) -> WorkflowEvent:
    return WorkflowEvent(
        event_type=EventType.STEP_STARTED, run_id=run_id, job_id=job_id, step_id=step.step_def.id if step.step_def else None,
        data={"name": step.step_def.name if step.step_def else ""},
    )


def make_step_completed_event(run_id, job_id, step) -> WorkflowEvent:
    return WorkflowEvent(
        event_type=EventType.STEP_COMPLETED, run_id=run_id, job_id=job_id, step_id=step.step_def.id if step.step_def else None,
        data={"status": step.status.value, "duration": step.duration, "outputs": step.outputs,
              "error": step.error_msg},
    )


def make_step_log_event(run_id, job_id, step_id, line) -> WorkflowEvent:
    return WorkflowEvent(
        event_type=EventType.STEP_LOG, run_id=run_id, job_id=job_id, step_id=step_id,
        data={"line": line},
    )


class EventBus:
    def __init__(self):
        self._subscribers: Set[Callable[[WorkflowEvent], None]] = set()

    def subscribe(self, callback: Callable[[WorkflowEvent], None]) -> Callable[[], None]:
        self._subscribers.add(callback)
        def unsubscribe():
            self._subscribers.discard(callback)
        return unsubscribe

    def publish(self, event: WorkflowEvent) -> None:
        for cb in list(self._subscribers):
            try:
                cb(event)
            except Exception as e:
                logger.error(f"[EventBus] subscriber error: {e}", exc_info=True)

