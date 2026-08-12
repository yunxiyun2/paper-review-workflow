import pytest
from paper_review_workflow.core.state_machine import (
    WorkflowStateMachine, JobStateMachine, StepStateMachine,
    WorkflowStateMachineCoordinator, InvalidTransitionError,
)
from paper_review_workflow.core.models import (
    WorkflowStatus, JobStatus, StepStatus,
    WorkflowRun, JobInstance, StepInstance,
)
from paper_review_workflow.core.event_bus import EventBus


def test_workflow_pending_to_running():
    sm = WorkflowStateMachine(WorkflowStatus.PENDING)
    sm.transition(WorkflowStatus.RUNNING)
    assert sm.current == WorkflowStatus.RUNNING


def test_workflow_invalid_transition():
    sm = WorkflowStateMachine(WorkflowStatus.PENDING)
    with pytest.raises(InvalidTransitionError):
        sm.transition(WorkflowStatus.SUCCESS)  # must go through RUNNING


def test_workflow_terminal_no_transition():
    sm = WorkflowStateMachine(WorkflowStatus.SUCCESS)
    with pytest.raises(InvalidTransitionError):
        sm.transition(WorkflowStatus.RUNNING)


def test_step_no_waiting_state():
    """WAITING removed from our state machine"""
    sm = StepStateMachine(StepStatus.RUNNING)
    # Verify WAITING is not a valid transition from RUNNING
    # (we can't reference StepStatus.WAITING because it doesn't exist)
    # Instead, verify RUNNING can only go to terminal states
    from paper_review_workflow.core.state_machine import STEP_TRANSITIONS
    allowed = STEP_TRANSITIONS[StepStatus.RUNNING]
    assert StepStatus.SUCCESS in allowed
    assert StepStatus.FAILURE in allowed
    assert StepStatus.CANCELLED in allowed
    # No WAITING in the enum at all
    assert not hasattr(StepStatus, "WAITING")
    assert not hasattr(StepStatus, "WAITING_RETRY")


def test_coordinator_start_and_complete_workflow():
    bus = EventBus()
    run = WorkflowRun()
    coord = WorkflowStateMachineCoordinator(run, bus)
    coord.start_workflow()
    assert run.status == WorkflowStatus.RUNNING
    coord.complete_workflow(success=True)
    assert run.status == WorkflowStatus.SUCCESS


def test_coordinator_start_and_complete_job():
    bus = EventBus()
    run = WorkflowRun()
    coord = WorkflowStateMachineCoordinator(run, bus)
    job = JobInstance()
    coord.start_job("j1", job)
    assert job.status == JobStatus.RUNNING
    coord.complete_job("j1", job, success=True)
    assert job.status == JobStatus.SUCCESS


def test_coordinator_start_and_complete_step():
    bus = EventBus()
    run = WorkflowRun()
    coord = WorkflowStateMachineCoordinator(run, bus)
    step = StepInstance()
    coord.start_step("j1", step)
    assert step.status == StepStatus.RUNNING
    coord.complete_step("j1", step, success=True)
    assert step.status == StepStatus.SUCCESS


def test_complete_job_idempotent():
    """Calling complete_job twice should be a no-op, not raise"""
    from paper_review_workflow.core.event_bus import EventBus
    bus = EventBus()
    run = WorkflowRun()
    coord = WorkflowStateMachineCoordinator(run, bus)
    job = JobInstance()
    coord.start_job("j1", job)
    coord.complete_job("j1", job, success=True)
    # Second call should not raise
    coord.complete_job("j1", job, success=True)
    assert job.status == JobStatus.SUCCESS
