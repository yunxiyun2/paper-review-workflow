import pytest
from pydantic import ValidationError
from paper_review_workflow.api.schemas import (
    DispatchRequest, RegisterWorkflowRequest, ResumeRequest,
    RunResponse, WorkflowSummary, HealthResponse,
)


def test_dispatch_request_with_workflow_name():
    r = DispatchRequest(workflow_name="neurips-paper-review", inputs={"paper_source": "2402.12098"})
    assert r.workflow_name == "neurips-paper-review"
    assert r.yaml_content is None
    assert r.inputs["paper_source"] == "2402.12098"


def test_dispatch_request_with_yaml_content():
    r = DispatchRequest(yaml_content="name: t\non: {workflow_dispatch: {}}")
    assert r.workflow_name is None
    assert r.yaml_content.startswith("name:")


def test_dispatch_request_requires_name_or_yaml():
    with pytest.raises(ValidationError):
        DispatchRequest()


def test_register_workflow_request():
    r = RegisterWorkflowRequest(yaml_content="name: foo\non: {workflow_dispatch: {}}")
    assert r.yaml_content.startswith("name:")
    assert r.name is None


def test_resume_request_defaults():
    r = ResumeRequest()
    assert r.rerun_components is None
    assert r.rerun_all is False


def test_resume_request_with_components():
    r = ResumeRequest(rerun_components=["dimensions_soundness", "synthesize"])
    assert r.rerun_components == ["dimensions_soundness", "synthesize"]


def test_run_response_minimal():
    r = RunResponse(run_id="abc", workflow_name="wf", status="pending")
    assert r.run_id == "abc"
    assert r.duration is None
    assert r.jobs == {}


def test_workflow_summary():
    r = WorkflowSummary(name="neurips-paper-review", jobs=["extract", "dimensions"])
    assert r.name == "neurips-paper-review"
    assert r.dispatch_inputs == {}


def test_health_response():
    r = HealthResponse()
    assert r.status == "ok"
    assert r.version == "0.1.0"
    assert r.active_runs == 0
