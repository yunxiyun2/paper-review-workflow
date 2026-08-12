from paper_review_workflow.core.models import (
    WorkflowDef, WorkflowRun, WorkflowStatus,
    JobDef, JobInstance, JobStatus,
    StepDef, StepInstance, StepStatus,
    TriggerDef,
)


def test_workflow_run_default_status():
    run = WorkflowRun()
    assert run.status == WorkflowStatus.PENDING
    assert run.id  # auto-generated uuid


def test_step_def_uses_action_type():
    step = StepDef(id="s1", name="step1", uses="paper-review/echo@v1")
    assert step.uses == "paper-review/echo@v1"
    # run/script fields removed (only uses is supported)


def test_workflow_status_transitions():
    assert WorkflowStatus.PENDING != WorkflowStatus.RUNNING


def test_job_def_needs_default_empty():
    job = JobDef(id="j1", name="job1", runs_on="local")
    assert job.needs == []
    assert job.steps == []
