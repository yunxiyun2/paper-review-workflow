"""FastAPI app factory + endpoints + WebSocket."""
import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, Query
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

# Per-run credentials (user-supplied API keys) live in this in-memory map only —
# never written to disk, and stripped from run.env by the engine before any
# persistence. Retained until the user explicitly deletes them via
# POST /api/runs/{run_id}/delete-key (or deletes the whole run).
_pending_secrets: dict = {}
SUPPORTED_VENUES = ("neurips", "icml", "acl")


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
                if wf_def.jobs  # skip venue param configs loaded as empty workflow defs
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

    @app.post("/api/review", status_code=202)
    async def create_review(
        background_tasks: BackgroundTasks,
        pdf: UploadFile = File(..., description="论文 PDF 文件"),
        api_key: str = Form(..., description="用户提供的 LLM API key"),
        model: str = Form(..., description="模型名, 如 glm-5.3"),
        provider: str = Form("zhipu", description="LLM 提供商"),
        venue: str = Form("neurips", description="评审会议: neurips/icml/acl"),
        weights: str = Form("{}", description='维度权重 JSON, 如 {"soundness": 1.5}'),
        task_name: str = Form("", description="任务名称, 显示在监控列表"),
    ):
        """Ephemeral review flow: upload PDF + per-request credentials, execute,
        then auto-delete artifacts and the key. Nothing is persisted."""
        provider = provider.strip().lower()
        from ..llm.registry import ProviderRegistry
        try:
            ProviderRegistry().get(provider)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"unsupported provider: {provider} "
                                                       f"(available: {ProviderRegistry().list_providers()})")
        venue = venue.strip().lower()
        if venue not in SUPPORTED_VENUES:
            raise HTTPException(status_code=400, detail=f"unsupported venue: {venue} (available: {list(SUPPORTED_VENUES)})")
        if not model.strip():
            raise HTTPException(status_code=400, detail="model is required")
        if pdf.filename and not pdf.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="only PDF files are accepted")

        # Parse + validate dimension weights against the venue's dimensions
        import json as _json
        from ..core.venue_config import VenueConfig
        try:
            weight_map = _json.loads(weights) if weights and weights.strip() else {}
        except _json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="weights must be valid JSON")
        if not isinstance(weight_map, dict):
            raise HTTPException(status_code=400, detail="weights must be a JSON object")
        try:
            venue_dims = VenueConfig.load(venue).dimensions
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        weight_env = {}
        for dim, value in weight_map.items():
            if dim not in venue_dims:
                raise HTTPException(status_code=400,
                                    detail=f"dimension '{dim}' not in venue '{venue}' (dims: {venue_dims})")
            try:
                w = float(value)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"weight for '{dim}' must be a number")
            if not (0.1 <= w <= 3.0):
                raise HTTPException(status_code=400, detail=f"weight for '{dim}' out of range [0.1, 3.0]")
            weight_env[f"WEIGHT_{dim.upper()}"] = str(w)

        # Build + register a workflow whose dimension matrix matches the venue,
        # so ICML/ACL run their own 4 dimensions instead of NeurIPS's 3.
        workflow_name = f"review-{venue}"
        engine.register_workflow(_build_review_workflow_yaml(venue), name=workflow_name)

        # Allocate the run first so the upload lands inside its session dir
        run = engine.dispatch_workflow(workflow_name, {"mode": venue})
        upload_dir = Path(engine.sessions_root).resolve() / run.id / "upload"
        upload_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = upload_dir / "paper.pdf"
        content = await pdf.read()
        if not content:
            raise HTTPException(status_code=400, detail="uploaded PDF is empty")
        pdf_path.write_bytes(content)

        # paper_source points at the uploaded file; mode selects the venue;
        # user-confirmed weights override the venue defaults via WEIGHT_* env.
        # provider/model are persisted (non-secret) so resume can rebuild the
        # LLM env even after a server restart.
        run.trigger_payload["paper_source"] = str(pdf_path)
        run.trigger_payload["task_name"] = task_name.strip() or "未命名评审"
        run.trigger_payload["provider"] = provider
        run.trigger_payload["model"] = model.strip()
        run.env.update(weight_env)
        engine.storage.save_run(run)

        # Credentials stay in memory only; they are never persisted and stay
        # available (for resume) until explicitly deleted via the API.
        _pending_secrets[run.id] = {
            "LLM_PROVIDER": provider,
            "LLM_MODEL": model.strip(),
            f"{provider.upper()}_API_KEY": api_key,
        }
        background_tasks.add_task(_run_in_background, engine, run.id)

        return {
            "run_id": run.id,
            "workflow_name": workflow_name,
            "task_name": run.trigger_payload["task_name"],
            "status": run.status.value,
            "provider": provider,
            "model": model.strip(),
            "venue": venue,
            "message": "review dispatched; API key stays in memory until deleted via the API",
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

    @app.post("/api/runs/{run_id}/delete-key")
    async def delete_run_key(run_id: str):
        """Drop the in-memory API key for this run (manual privacy control)."""
        run = engine.storage.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        secret = _pending_secrets.pop(run_id, None)
        if secret:
            return {"run_id": run_id, "deleted": True,
                    "message": "API key 已从内存中删除"}
        return {"run_id": run_id, "deleted": False,
                "message": "该 run 没有保存中的 API key（可能已删除，或未通过上传评审创建）"}

    @app.delete("/api/runs/{run_id}")
    async def delete_run(run_id: str):
        """Delete a run entirely: in-memory key, session dir (uploaded PDF,
        artifacts, reports) and the run record."""
        run = engine.storage.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        if run_id in engine._active_runs:
            raise HTTPException(status_code=409,
                                detail="run 正在执行中，请先取消再删除")
        _pending_secrets.pop(run_id, None)
        session_dir = Path(engine.sessions_root).resolve() / run_id
        shutil.rmtree(session_dir, ignore_errors=True)
        engine.storage.delete_run(run_id)
        logger.info(f"[delete] removed key/session/record for run {run_id}")
        return {"run_id": run_id, "deleted": True, "message": "任务已删除"}

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

        # Per-venue review workflows are registered dynamically; re-register
        # the definition if it was lost (server restart cleared the registry).
        wf_name = (run.env or {}).get("__workflow_name__", "")
        if wf_name.startswith("review-") and engine.get_workflow_def(wf_name) is None:
            venue = wf_name.removeprefix("review-")
            try:
                engine.register_workflow(_build_review_workflow_yaml(venue), name=wf_name)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"cannot rebuild workflow: {e}")

        # Rebuild the LLM execution env: in-memory key first, else the key
        # supplied with the resume request; provider/model from persisted inputs.
        secret_env = dict(_pending_secrets.get(run_id) or {})
        payload = run.trigger_payload or {}
        if req.api_key:
            provider = payload.get("provider", "zhipu")
            secret_env["LLM_PROVIDER"] = provider
            secret_env["LLM_MODEL"] = payload.get("model") or secret_env.get("LLM_MODEL", "")
            secret_env[f"{provider.upper()}_API_KEY"] = req.api_key
            _pending_secrets[run_id] = secret_env

        background_tasks.add_task(
            _resume_in_background, engine, run_id,
            req.rerun_components, req.rerun_all,
        )
        return {"run_id": run_id, "status": "pending", "message": "resume scheduled"}

    def _find_step_output(run: WorkflowRun, key: str):
        """Locate an output value across all jobs/steps of a run (decide/synthesize
        write file paths into their step outputs)."""
        for job in run.jobs.values():
            if key in (job.outputs or {}):
                return job.outputs[key]
            for step in job.steps:
                if key in (step.outputs or {}):
                    return step.outputs[key]
        return None

    @app.get("/api/runs/{run_id}/decision")
    async def get_run_decision(run_id: str):
        """Full decision payload (per-dimension scores, strengths/concerns, rationale)."""
        run = engine.storage.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        decision_path = _find_step_output(run, "decision_path")
        if not decision_path or not Path(decision_path).is_file():
            raise HTTPException(
                status_code=404,
                detail=f"decision not available yet for run {run_id} (decide step has not completed)",
            )
        import json as _json
        return _json.loads(Path(decision_path).read_text(encoding="utf-8"))

    @app.get("/api/runs/{run_id}/review")
    async def get_run_review(run_id: str):
        """Serve the synthesized review markdown for a completed run."""
        run = engine.storage.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        review_path = _find_step_output(run, "review_path")
        candidates = []
        if review_path:
            candidates.append(Path(review_path))
        if candidates and candidates[0].parent.parent.name:
            # sibling final_report.md produced by the decide step
            candidates.append(candidates[0].parent.parent / "final_report.md")
        for p in candidates:
            if p.is_file():
                from fastapi.responses import Response
                return Response(
                    content=p.read_text(encoding="utf-8"),
                    media_type="text/markdown; charset=utf-8",
                )
        raise HTTPException(status_code=404, detail=f"review not available yet for run {run_id}")

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

    def _workflow_name_of(run: WorkflowRun) -> str:
        if run.workflow_def:
            return run.workflow_def.name
        return (run.env or {}).get("__workflow_name__", "")

    def _task_name_of(run: WorkflowRun) -> str:
        return (run.trigger_payload or {}).get("task_name", "")

    def _serialize_run_brief(run: WorkflowRun) -> dict:
        return {
            "run_id": run.id,
            "workflow_name": _workflow_name_of(run),
            "task_name": _task_name_of(run),
            "status": run.status.value,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "duration": run.duration,
            "jobs": {jid: {"status": j.status.value} for jid, j in run.jobs.items()},
        }

    def _serialize_run_full(run: WorkflowRun) -> dict:
        return {
            "run_id": run.id,
            "workflow_name": _workflow_name_of(run),
            "task_name": _task_name_of(run),
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
                         "status": s.status.value, "outputs": s.outputs,
                         "error": s.error_msg, "log": s.log}
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


def _build_review_workflow_yaml(venue: str) -> str:
    """Build a review workflow whose dimension matrix matches the venue's own
    dimensions (NeurIPS 3 / ICML 4 / ACL 4)."""
    from ..core.venue_config import VenueConfig
    dims = VenueConfig.load(venue).dimensions
    matrix = ", ".join(dims)
    session_dir = "${{ env.SESSIONS_ROOT }}/${{ env.RUN_ID }}"
    return (
        f"name: review-{venue}\n\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "    inputs:\n"
        "      paper_source:\n"
        '        description: "Paper PDF path"\n'
        "        required: true\n"
        "        type: string\n\n"
        "env:\n"
        "  SESSIONS_ROOT: ./sessions\n"
        f"  VENUE: {venue}\n\n"
        "jobs:\n"
        "  extract:\n"
        "    runs-on: local\n"
        "    outputs:\n"
        "      full_text_path: ${{ steps.extract.outputs.full_text_path }}\n"
        "      metadata_path: ${{ steps.extract.outputs.metadata_path }}\n"
        "    steps:\n"
        "      - id: extract\n"
        "        uses: paper-review/extract@v1\n"
        "        with:\n"
        "          source: ${{ inputs.paper_source }}\n"
        f"          session_dir: {session_dir}\n"
        "  dimensions:\n"
        "    needs: extract\n"
        "    strategy:\n"
        "      matrix:\n"
        f"        dimension: [{matrix}]\n"
        f"      max-parallel: {len(dims)}\n"
        "    runs-on: local\n"
        "    steps:\n"
        "      - uses: paper-review/dim_score@v1\n"
        "        with:\n"
        "          dimension: ${{ matrix.dimension }}\n"
        f"          session_dir: {session_dir}\n"
        "          full_text_path: ${{ needs.extract.outputs.full_text_path }}\n"
        "          metadata_path: ${{ needs.extract.outputs.metadata_path }}\n"
        "  synthesize:\n"
        "    needs: dimensions\n"
        "    runs-on: local\n"
        "    outputs:\n"
        "      review_path: ${{ steps.synthesize.outputs.review_path }}\n"
        "      scores_path: ${{ steps.synthesize.outputs.scores_path }}\n"
        "    steps:\n"
        "      - id: synthesize\n"
        "        uses: paper-review/synthesize@v1\n"
        "        with:\n"
        f"          session_dir: {session_dir}\n"
        "  decide:\n"
        "    needs: synthesize\n"
        "    runs-on: local\n"
        "    outputs:\n"
        "      recommendation: ${{ steps.decide.outputs.recommendation }}\n"
        "      weighted_score: ${{ steps.decide.outputs.weighted_score }}\n"
        "      decision_path: ${{ steps.decide.outputs.decision_path }}\n"
        "    steps:\n"
        "      - id: decide\n"
        "        uses: paper-review/decide@v1\n"
        "        with:\n"
        f"          session_dir: {session_dir}\n"
        "          scores_path: ${{ needs.synthesize.outputs.scores_path }}\n"
    )


async def _run_in_background(engine: ReviewEngine, run_id: str) -> None:
    """Background task: run engine in thread pool to avoid blocking event loop.

    The per-run API key is READ from _pending_secrets (not popped) — it stays
    in server memory, available for resume, until the user deletes it via
    POST /api/runs/{run_id}/delete-key or deletes the whole run.
    """
    secret_env = _pending_secrets.get(run_id)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: engine.execute_existing_run(run_id, secret_env=secret_env)
    )


async def _resume_in_background(engine: ReviewEngine, run_id: str,
                                rerun_components=None, rerun_all: bool = False) -> None:
    """Resume a failed/interrupted run: reset failed+downstream jobs, keep
    completed ones, then re-execute (in a thread pool)."""
    secret_env = _pending_secrets.get(run_id)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: engine.resume_run(run_id, rerun_components=rerun_components,
                                  rerun_all=rerun_all, secret_env=secret_env),
    )
