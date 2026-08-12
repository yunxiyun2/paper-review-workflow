"""Three-layer state machine: Workflow / Job / Step.

Ported from lwf, removed WAITING and WAITING_RETRY states.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Dict, FrozenSet, Optional

from .models import (
    WorkflowStatus, JobStatus, StepStatus,
    WorkflowRun, JobInstance, StepInstance,
)
from .event_bus import (
    EventBus, WorkflowEvent, EventType,
    make_workflow_started_event, make_workflow_completed_event,
    make_job_started_event, make_job_completed_event,
    make_step_started_event, make_step_completed_event,
)

logger = logging.getLogger(__name__)


class InvalidTransitionError(Exception):
    def __init__(self, entity: str, from_state, to_state):
        super().__init__(
            f"[StateMachine] {entity} cannot transition {from_state.value!r} -> {to_state.value!r}"
        )
        self.entity = entity
        self.from_state = from_state
        self.to_state = to_state


# ── Transition tables ─────────────────────────────────

WORKFLOW_TRANSITIONS: Dict[WorkflowStatus, FrozenSet[WorkflowStatus]] = {
    WorkflowStatus.PENDING:   frozenset({WorkflowStatus.RUNNING, WorkflowStatus.CANCELLED}),
    WorkflowStatus.RUNNING:   frozenset({WorkflowStatus.SUCCESS, WorkflowStatus.FAILURE, WorkflowStatus.CANCELLED}),
    WorkflowStatus.SUCCESS:   frozenset(),
    WorkflowStatus.FAILURE:   frozenset(),
    WorkflowStatus.CANCELLED: frozenset(),
    WorkflowStatus.SKIPPED:   frozenset(),
}

JOB_TRANSITIONS: Dict[JobStatus, FrozenSet[JobStatus]] = {
    JobStatus.PENDING:   frozenset({JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.SKIPPED, JobStatus.CANCELLED}),
    JobStatus.QUEUED:    frozenset({JobStatus.RUNNING, JobStatus.CANCELLED, JobStatus.SKIPPED}),
    JobStatus.RUNNING:   frozenset({JobStatus.SUCCESS, JobStatus.FAILURE, JobStatus.CANCELLED}),
    JobStatus.SUCCESS:   frozenset(),
    JobStatus.FAILURE:   frozenset(),
    JobStatus.CANCELLED: frozenset(),
    JobStatus.SKIPPED:   frozenset(),
}

STEP_TRANSITIONS: Dict[StepStatus, FrozenSet[StepStatus]] = {
    StepStatus.PENDING:   frozenset({StepStatus.RUNNING, StepStatus.SKIPPED, StepStatus.CANCELLED}),
    StepStatus.RUNNING:   frozenset({StepStatus.SUCCESS, StepStatus.FAILURE, StepStatus.CANCELLED}),
    StepStatus.SUCCESS:   frozenset(),
    StepStatus.FAILURE:   frozenset(),
    StepStatus.SKIPPED:   frozenset(),
    StepStatus.CANCELLED: frozenset(),
}

WORKFLOW_TERMINAL = frozenset({WorkflowStatus.SUCCESS, WorkflowStatus.FAILURE,
                               WorkflowStatus.CANCELLED, WorkflowStatus.SKIPPED})
JOB_TERMINAL = frozenset({JobStatus.SUCCESS, JobStatus.FAILURE,
                          JobStatus.CANCELLED, JobStatus.SKIPPED})
STEP_TERMINAL = frozenset({StepStatus.SUCCESS, StepStatus.FAILURE,
                           StepStatus.CANCELLED, StepStatus.SKIPPED})


# ── Generic state machine ─────────────────────────────

class _StateMachine:
    def __init__(self, initial, transitions, name: str):
        self.current = initial
        self._transitions = transitions
        self._name = name
        self._history = [initial]

    def transition(self, to_state) -> None:
        allowed = self._transitions.get(self.current, frozenset())
        if to_state not in allowed:
            raise InvalidTransitionError(self._name, self.current, to_state)
        self.current = to_state
        self._history.append(to_state)

    @property
    def is_terminal(self) -> bool:
        return len(self._transitions.get(self.current, frozenset())) == 0

    @property
    def history(self):
        return list(self._history)


class WorkflowStateMachine(_StateMachine):
    def __init__(self, initial=WorkflowStatus.PENDING):
        super().__init__(initial, WORKFLOW_TRANSITIONS, "Workflow")


class JobStateMachine(_StateMachine):
    def __init__(self, initial=JobStatus.PENDING):
        super().__init__(initial, JOB_TRANSITIONS, "Job")


class StepStateMachine(_StateMachine):
    def __init__(self, initial=StepStatus.PENDING):
        super().__init__(initial, STEP_TRANSITIONS, "Step")


# ── Coordinator ───────────────────────────────────────

class WorkflowStateMachineCoordinator:
    """Coordinates state transitions across the three layers and publishes events."""

    def __init__(self, run: WorkflowRun, event_bus: EventBus):
        self.run = run
        self.event_bus = event_bus
        self.workflow_sm = WorkflowStateMachine(run.status)
        self._job_sms: Dict[str, JobStateMachine] = {}
        self._step_sms: Dict[str, StepStateMachine] = {}

    # Workflow
    def start_workflow(self) -> None:
        self.workflow_sm.transition(WorkflowStatus.RUNNING)
        self.run.status = WorkflowStatus.RUNNING
        self.run.start_time = datetime.now()
        self.event_bus.publish(make_workflow_started_event(self.run))

    def complete_workflow(self, success: bool) -> None:
        target = WorkflowStatus.SUCCESS if success else WorkflowStatus.FAILURE
        self.workflow_sm.transition(target)
        self.run.status = target
        self.run.end_time = datetime.now()
        self.event_bus.publish(make_workflow_completed_event(self.run))

    def cancel_workflow(self) -> None:
        if self.workflow_sm.is_terminal:
            return
        self.workflow_sm.transition(WorkflowStatus.CANCELLED)
        self.run.status = WorkflowStatus.CANCELLED
        self.run.end_time = datetime.now()
        self.event_bus.publish(WorkflowEvent(
            event_type=EventType.WORKFLOW_CANCELLED, run_id=self.run.id,
        ))

    # Job
    def start_job(self, job_id: str, job: JobInstance) -> None:
        sm = JobStateMachine(job.status)
        self._job_sms[job_id] = sm
        sm.transition(JobStatus.RUNNING)
        job.status = JobStatus.RUNNING
        job.start_time = datetime.now()
        self.event_bus.publish(make_job_started_event(self.run.id, job))

    def complete_job(self, job_id: str, job: JobInstance, success: bool) -> None:
        sm = self._job_sms.get(job_id)
        if sm is None:
            sm = JobStateMachine(job.status)
            self._job_sms[job_id] = sm
        target = JobStatus.SUCCESS if success else JobStatus.FAILURE
        sm.transition(target)
        job.status = target
        job.end_time = datetime.now()
        self.event_bus.publish(make_job_completed_event(self.run.id, job))

    def skip_job(self, job_id: str, job: JobInstance) -> None:
        sm = JobStateMachine(job.status)
        self._job_sms[job_id] = sm
        sm.transition(JobStatus.SKIPPED)
        job.status = JobStatus.SKIPPED
        self.event_bus.publish(WorkflowEvent(
            event_type=EventType.JOB_SKIPPED, run_id=self.run.id, job_id=job_id,
        ))

    def cancel_job(self, job_id: str, job: JobInstance) -> None:
        sm = self._job_sms.get(job_id) or JobStateMachine(job.status)
        self._job_sms[job_id] = sm
        if not sm.is_terminal:
            sm.transition(JobStatus.CANCELLED)
            job.status = JobStatus.CANCELLED
            job.end_time = datetime.now()

    # Step
    def start_step(self, job_id: str, step: StepInstance) -> None:
        sm = StepStateMachine(step.status)
        self._step_sms[step.id] = sm
        sm.transition(StepStatus.RUNNING)
        step.status = StepStatus.RUNNING
        step.start_time = datetime.now()
        self.event_bus.publish(make_step_started_event(self.run.id, job_id, step))

    def complete_step(self, job_id: str, step: StepInstance, success: bool) -> None:
        sm = self._step_sms.get(step.id) or StepStateMachine(step.status)
        self._step_sms[step.id] = sm
        target = StepStatus.SUCCESS if success else StepStatus.FAILURE
        sm.transition(target)
        step.status = target
        step.end_time = datetime.now()
        self.event_bus.publish(make_step_completed_event(self.run.id, job_id, step))

    def skip_step(self, job_id: str, step: StepInstance) -> None:
        sm = StepStateMachine(step.status)
        self._step_sms[step.id] = sm
        sm.transition(StepStatus.SKIPPED)
        step.status = StepStatus.SKIPPED
        self.event_bus.publish(WorkflowEvent(
            event_type=EventType.STEP_SKIPPED, run_id=self.run.id, job_id=job_id, step_id=step.id,
        ))

    def cancel_step(self, job_id: str, step: StepInstance) -> None:
        sm = self._step_sms.get(step.id) or StepStateMachine(step.status)
        self._step_sms[step.id] = sm
        if not sm.is_terminal:
            sm.transition(StepStatus.CANCELLED)
            step.status = StepStatus.CANCELLED

    def cancel_all_pending_jobs_and_steps(self) -> None:
        for job_id, job in self.run.jobs.items():
            if not JOB_TERMINAL.__contains__(job.status):
                self.cancel_job(job_id, job)
            for step in job.steps:
                if not STEP_TERMINAL.__contains__(step.status):
                    self.cancel_step(job_id, step)
