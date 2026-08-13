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

    # Other endpoints added in subsequent tasks
    return app
