"""FastAPI app factory + endpoints + WebSocket."""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..engine import ReviewEngine
from ..storage import StorageBackend, MemoryStorage, JsonFileStorage
from ..core.event_bus import EventBus, WorkflowEvent, EventType
from ..core.models import WorkflowRun, WorkflowStatus, JobStatus, StepStatus
from .dependencies import get_engine, get_ws_manager, get_configs_dir
from .schemas import (
    DispatchRequest, RegisterWorkflowRequest, ResumeRequest,
    RunResponse, WorkflowSummary, HealthResponse,
)
from .ws_manager import WSManager, FilteredWS

logger = logging.getLogger(__name__)


def create_app(
    storage: Optional[StorageBackend] = None,
    configs_dir: str = "./configs",
    sessions_root: str = "./sessions",
) -> FastAPI:
    """Create a FastAPI app instance with the given engine configuration."""
    app = FastAPI(
        title="Paper Review Workflow API",
        description="HTTP API + WebSocket for AI-based paper review",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Build engine + WSManager
    engine = ReviewEngine(
        storage=storage if storage is not None else JsonFileStorage(sessions_root),
        sessions_root=sessions_root,
    )
    ws_manager = WSManager(engine.event_bus)

    # Attach to app state for dependency injection
    app.state.engine = engine
    app.state.ws_manager = ws_manager
    app.state.configs_dir = configs_dir

    # Startup/shutdown handlers
    @app.on_event("startup")
    async def startup_event():
        loop = asyncio.get_event_loop()
        ws_manager.attach(loop)
        engine.load_workflow_directory(configs_dir)
        recovered = engine.recover_interrupted_runs()
        if recovered:
            logger.warning(f"[API] recovered {recovered} interrupted runs -> cancelled")

    @app.on_event("shutdown")
    async def shutdown_event():
        ws_manager.detach()
        cancelled = engine.shutdown(timeout=10.0)
        if cancelled:
            logger.info(f"[API] cancelled {cancelled} active runs on shutdown")
        engine.storage.close()

    # -- Endpoints --
    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        runs = engine.storage.list_runs(limit=10**6)
        total = len(runs)
        active = sum(1 for r in runs if r.status in (WorkflowStatus.PENDING, WorkflowStatus.RUNNING))
        return HealthResponse(status="ok", version="0.1.0", active_runs=active, total_runs=total)

    @app.get("/api/workflows")
    async def list_workflows():
        defs = engine.get_workflow_defs()
        return {
            "total": len(defs),
            "workflows": [
                {
                    "name": name,
                    "file_path": wf_def.file_path,
                    "jobs": list(wf_def.jobs.keys()),
                    "dispatch_inputs": _extract_dispatch_inputs(wf_def),
                }
                for name, wf_def in defs.items()
            ],
        }

    @app.post("/api/workflows/register")
    async def register_workflow(req: RegisterWorkflowRequest):
        try:
            wf_def = engine.register_workflow(req.yaml_content, name=req.name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"YAML parse failed: {e}")
        return {
            "message": "workflow registered",
            "name": wf_def.name,
            "jobs": list(wf_def.jobs.keys()),
        }

    def _extract_dispatch_inputs(wf_def):
        """Extract on.workflow_dispatch.inputs spec from WorkflowDef."""
        if not wf_def.on or not wf_def.on.workflow_dispatch:
            return {}
        wd = wf_def.on.workflow_dispatch
        if isinstance(wd, dict):
            return wd.get("inputs", {})
        return {}

    @app.post("/api/runs", status_code=202)
    async def dispatch_run(req: DispatchRequest, background_tasks: BackgroundTasks):
        # Resolve workflow definition
        if req.workflow_name:
            wf_def = engine.get_workflow_def(req.workflow_name)
            if wf_def is None:
                raise HTTPException(status_code=404, detail=f"workflow not registered: {req.workflow_name}")
        elif req.yaml_content:
            try:
                wf_def = engine.register_workflow(req.yaml_content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"YAML parse failed: {e}")
        else:
            raise HTTPException(status_code=400, detail="must provide workflow_name or yaml_content")

        # Pre-allocate run (PENDING, saved to storage)
        try:
            run = engine.dispatch_workflow(wf_def.name, req.inputs)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # Schedule background execution
        background_tasks.add_task(_run_in_background, engine, run.id)

        return {
            "run_id": run.id,
            "workflow_name": wf_def.name,
            "status": run.status.value,
            "message": "review dispatched, see GET /api/runs/{run_id} for status",
        }

    # Other endpoints added in subsequent tasks
    return app


async def _run_in_background(engine: ReviewEngine, run_id: str) -> None:
    """Background task: run engine in thread pool to avoid blocking event loop."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: engine.execute_existing_run(run_id))
