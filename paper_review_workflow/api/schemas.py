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
