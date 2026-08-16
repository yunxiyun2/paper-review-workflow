"""FastAPI app factory + endpoints + WebSocket."""
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

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

    @app.get("/api/venues")
    async def list_venues():
        """Return all venue configs for the frontend assembler."""
        from ..core.venue_config import VenueConfig

        venues_dir = os.environ.get("PAPER_REVIEW_CONFIGS_DIR", "configs/venues")
        venues = []
        for yml_file in sorted(Path(venues_dir).glob("*.yaml")):
            try:
                # Clear cache for dev-mode hot reload
                VenueConfig._cache.pop(yml_file.stem, None)
                config = VenueConfig.from_yaml(str(yml_file))
                venues.append({
                    "name": config.name,
                    "display_name": config.display_name,
                    "dimensions": config.dimensions,
                    "score_min": config.score_min,
                    "score_max": config.score_max,
                    "weights": config.weights,
                    "thresholds": [{"threshold": t, "label": l} for t, l in config.thresholds],
                    "prompts_dir": config.prompts_dir,
                })
            except Exception as e:
                logger.warning(f"[API] failed to load venue {yml_file}: {e}")
        return {"total": len(venues), "venues": venues}

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

    @app.get("/api/runs")
    async def list_runs(
        status: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ):
        status_enum = WorkflowStatus(status) if status else None
        runs = engine.storage.list_runs(status=status_enum, limit=limit + offset)
        runs = runs[offset:offset + limit]
        return {
            "total": len(runs),
            "runs": [_serialize_run_brief(r) for r in runs],
        }

    @app.get("/api/runs/{run_id}", response_model=RunResponse)
    async def get_run(run_id: str):
        run = engine.storage.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        return _serialize_run_full(run)

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str):
        run = engine.storage.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        cancelled = engine.cancel_run(run_id)
        if not cancelled:
            # Already terminal — return current status
            run = engine.storage.get_run(run_id)
            return {"run_id": run_id, "status": run.status.value, "message": "run already terminal"}
        return {"run_id": run_id, "status": "cancelled", "message": "cancellation requested"}

    @app.post("/api/runs/{run_id}/resume")
    async def resume_run(run_id: str, req: ResumeRequest, background_tasks: BackgroundTasks):
        run = engine.storage.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")

        # Apply rerun options
        if req.rerun_all:
            engine._reset_all_components(run)
        elif req.rerun_components:
            engine._mark_for_rerun(run, req.rerun_components)

        background_tasks.add_task(_run_in_background, engine, run_id)
        return {"run_id": run_id, "status": "pending", "message": "resume scheduled"}

    @app.get("/api/runs/{run_id}/export")
    async def export_run(run_id: str, format: str = "xml"):
        run = engine.storage.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        from ..exporters.openreview import OpenReviewExporter
        exporter = OpenReviewExporter()
        try:
            xml = exporter.export(run_id, engine.storage)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        from fastapi.responses import Response
        return Response(content=xml, media_type="application/xml")

    def _serialize_run_brief(run: WorkflowRun) -> dict:
        return {
            "run_id": run.id,
            "workflow_name": run.workflow_def.name if run.workflow_def else "",
            "status": run.status.value,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "duration": run.duration,
            "jobs": {jid: {"status": j.status.value} for jid, j in run.jobs.items()},
        }

    def _serialize_run_full(run: WorkflowRun) -> dict:
        return {
            "run_id": run.id,
            "workflow_name": run.workflow_def.name if run.workflow_def else "",
            "status": run.status.value,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "duration": run.duration,
            "jobs": {
                jid: {
                    "status": j.status.value,
                    "outputs": j.outputs,
                    "duration": j.duration,
                    "steps": [
                        {"id": s.id, "name": s.step_def.name if s.step_def else "",
                         "status": s.status.value, "outputs": s.outputs}
                        for s in j.steps
                    ],
                }
                for jid, j in run.jobs.items()
            },
        }

    # -- WebSocket endpoints --
    @app.websocket("/ws")
    async def websocket_global(websocket: WebSocket):
        """Global WebSocket — receives all EventBus events."""
        await ws_manager.add_connection(websocket)
        try:
            # Send historical events for catch-up
            await ws_manager.send_history(websocket)
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    if data == "ping":
                        await websocket.send_text('{"event": "pong"}')
                except asyncio.TimeoutError:
                    await websocket.send_text('{"event": "heartbeat"}')
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"[WS /ws] error: {e}")
        finally:
            ws_manager.remove_connection(websocket)

    @app.websocket("/ws/runs/{run_id}")
    async def websocket_run_filtered(websocket: WebSocket, run_id: str):
        """Run-filtered WebSocket — only events for the specified run_id."""
        filtered = FilteredWS(websocket, run_id)
        await ws_manager.add_connection(filtered)
        try:
            await ws_manager.send_history(filtered, run_id=run_id)
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    if data == "ping":
                        await websocket.send_text('{"event": "pong"}')
                except asyncio.TimeoutError:
                    await websocket.send_text('{"event": "heartbeat"}')
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"[WS /ws/runs/{run_id}] error: {e}")
        finally:
            ws_manager.remove_connection(filtered)

    # Other endpoints added in subsequent tasks

    # ── Static files + root route (Phase 2 #3) ──
    _static_dir = Path(__file__).parent / "static"
    if _static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root():
        """Serve the frontend index.html."""
        index_path = _static_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path), media_type="text/html")
        return HTMLResponse("<h1>Frontend not built yet</h1>", status_code=404)

    return app


async def _run_in_background(engine: ReviewEngine, run_id: str) -> None:
    """Background task: run engine in thread pool to avoid blocking event loop."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: engine.execute_existing_run(run_id))
