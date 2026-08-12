"""Core data models for the paper review workflow engine.

Ported from lwf/workflow_engine/core/models.py with these removals:
- WorkflowDefRecord / DefStatus (no version management)
- WaitFormField / StepWaitInfo (no human approval)
- ActionType.RUN / SCRIPT (only uses)
- step fields: script, script_args, run, fail_on_error, max_retries, working_directory
- step instance fields: wait_info, resumed_data, retry_info, retry_count, claude_session
- WorkflowRun fields: workflow_version, workflow_def_id, created_by
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


# ── Status enums ──────────────────────────────────────

class WorkflowStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILURE   = "failure"
    CANCELLED = "cancelled"
    SKIPPED   = "skipped"


class JobStatus(Enum):
    PENDING   = "pending"
    QUEUED    = "queued"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILURE   = "failure"
    CANCELLED = "cancelled"
    SKIPPED   = "skipped"


class StepStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILURE   = "failure"
    SKIPPED   = "skipped"
    CANCELLED = "cancelled"


# ── Definitions (static, from YAML) ───────────────────

@dataclass
class StepDef:
    id: str
    name: str
    uses: Optional[str] = None           # Action name like paper-review/extract@v1
    with_params: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    condition: Optional[str] = None
    continue_on_error: bool = False
    timeout_minutes: Optional[int] = None


@dataclass
class JobDef:
    id: str
    name: str
    runs_on: str
    steps: List[StepDef] = field(default_factory=list)
    needs: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    condition: Optional[str] = None
    strategy: Optional[Dict[str, Any]] = None
    timeout_minutes: Optional[int] = None
    continue_on_error: bool = False


@dataclass
class TriggerDef:
    workflow_dispatch: Optional[Dict] = None


@dataclass
class WorkflowDef:
    name: str
    on: TriggerDef
    jobs: Dict[str, JobDef] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    defaults: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None


# ── Runtime instances ─────────────────────────────────

@dataclass
class StepInstance:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_def: Optional[StepDef] = None
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)
    exit_code: int = 0
    error_msg: str = ""

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.step_def.name if self.step_def else "",
            "step_id": self.step_def.id if self.step_def else "",
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "exit_code": self.exit_code,
            "error_msg": self.error_msg,
            "outputs": self.outputs,
            "log": self.log,
        }


@dataclass
class JobInstance:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_def: Optional[JobDef] = None
    status: JobStatus = JobStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    steps: List[StepInstance] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)
    runner_info: Dict[str, str] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.job_def.name if self.job_def else "",
            "job_id": self.job_def.id if self.job_def else "",
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "steps": [s.to_dict() for s in self.steps],
            "outputs": self.outputs,
            "needs": self.job_def.needs if self.job_def else [],
        }


@dataclass
class WorkflowRun:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_def: Optional[WorkflowDef] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    trigger_type: str = "manual"
    trigger_payload: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    jobs: Dict[str, JobInstance] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    run_number: int = 1

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workflow_name": self.workflow_def.name if self.workflow_def else "",
            "status": self.status.value,
            "trigger_type": self.trigger_type,
            "trigger_payload": self.trigger_payload,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "run_number": self.run_number,
            "env": self.env,
            "jobs": {k: v.to_dict() for k, v in self.jobs.items()},
        }
