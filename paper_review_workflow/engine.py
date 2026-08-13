"""ReviewEngine: facade layer (simplified from lwf WorkflowEngine)."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.models import WorkflowRun, WorkflowStatus, JobStatus, StepStatus, WorkflowDef
from .core.parser import WorkflowParser
from .core.event_bus import EventBus, WorkflowEvent, EventType
from .core.state_machine import WorkflowStateMachineCoordinator
from .actions.registry import ActionRegistry
from .actions.builtin import register_builtin_actions
from .executors import WorkflowExecutor
from .storage import StorageBackend, MemoryStorage, JsonFileStorage

logger = logging.getLogger(__name__)


class ReviewEngine:
    def __init__(self, storage: Optional[StorageBackend] = None,
                 sessions_root: str = "./sessions"):
        self.parser = WorkflowParser()
        self.registry = ActionRegistry()
        self.event_bus = EventBus()
        self.storage = storage if storage is not None else JsonFileStorage(sessions_root)
        self.storage.open()

        self._active_runs: Dict[str, WorkflowRun] = {}
        self._coordinators: Dict[str, WorkflowStateMachineCoordinator] = {}
        self._workflow_defs: Dict[str, WorkflowDef] = {}

        self._workflow_executor = WorkflowExecutor(
            registry=self.registry,
            event_bus=self.event_bus,
            on_coordinator_created=self._on_coordinator_created,
        )

        register_builtin_actions(self.registry)
        self._register_persist_hooks()

    # -- Entry points --

    def run_from_file(self, yaml_path: str,
                      trigger_type: str = "workflow_dispatch",
                      payload: Optional[Dict] = None,
                      extra_env: Optional[Dict[str, str]] = None) -> WorkflowRun:
        wf_def = self.parser.parse_file(yaml_path)
        return self.run_workflow(wf_def, trigger_type, payload, extra_env=extra_env)

    def run_workflow(self, wf_def, trigger_type="workflow_dispatch", payload=None,
                     extra_env: Optional[Dict[str, str]] = None):
        run = WorkflowRun(
            workflow_def=wf_def,
            trigger_type=trigger_type,
            trigger_payload=payload or {},
            env=dict(wf_def.env) if wf_def.env else {},
        )
        # Apply CLI env overrides (take precedence over wf_def.env)
        if extra_env:
            run.env.update(extra_env)
        # Stash workflow file path for resume
        if wf_def.file_path:
            run.env["__workflow_file__"] = wf_def.file_path
        run.env["__workflow_name__"] = wf_def.name

        self.storage.save_run(run)
        self.event_bus.publish(WorkflowEvent(
            event_type=EventType.WORKFLOW_CREATED, run_id=run.id,
            data={"workflow_name": wf_def.name, "trigger_type": trigger_type},
        ))
        return self._do_execute(run)

    def resume_run(self, run_id: str,
                   rerun_components: Optional[List[str]] = None,
                   rerun_all: bool = False) -> WorkflowRun:
        run = self.storage.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        # Reload workflow_def from original YAML
        if run.workflow_def is None and run.env.get("__workflow_file__"):
            run.workflow_def = self.parser.parse_file(run.env["__workflow_file__"])
        if rerun_all:
            self._reset_all_components(run)
        elif rerun_components:
            self._mark_for_rerun(run, rerun_components)
        else:
            # Default resume: reset only failed/skipped jobs (and the workflow
            # status) so already-completed jobs are skipped on re-execution.
            # This is the "resume from failure" path: completed jobs are
            # preserved, failed + downstream jobs are re-run.
            self._reset_failed_and_downstream(run)
        return self._do_execute(run)

    # -- Control --

    def cancel_run(self, run_id: str) -> bool:
        coordinator = self._coordinators.get(run_id)
        if coordinator and not coordinator.workflow_sm.is_terminal:
            coordinator.cancel_all_pending_jobs_and_steps()
            coordinator.cancel_workflow()
            return True
        return False

    def shutdown(self, timeout: float = 30.0) -> int:
        cancelled = 0
        for run_id in list(self._active_runs.keys()):
            if self.cancel_run(run_id):
                cancelled += 1
        return cancelled

    # -- Query --

    def list_runs(self, paper_id: Optional[str] = None,
                  status: Optional[WorkflowStatus] = None,
                  limit: int = 50):
        runs = self.storage.list_runs(status=status, limit=limit * 5)
        if paper_id:
            runs = [r for r in runs if r.env.get("__paper_id__") == paper_id]
        return runs[:limit]

    def get_run(self, run_id: str):
        return self.storage.get_run(run_id)

    # -- Workflow definition management (Phase 2 #5) --

    def load_workflow_directory(self, configs_dir: str) -> Dict[str, WorkflowDef]:
        """Scan directory for *.yaml/*.yml files and register them. Returns {name: WorkflowDef}."""
        path = Path(configs_dir)
        if not path.exists():
            logger.warning(f"[Engine] configs dir not found: {configs_dir}")
            return {}
        loaded = {}
        for yml_file in sorted(path.glob("**/*.y*ml")):
            try:
                wf_def = self.parser.parse_file(str(yml_file))
                self._register_workflow_def(wf_def)
                loaded[wf_def.name] = wf_def
                logger.info(f"[Engine] loaded workflow: {wf_def.name} ({yml_file})")
            except Exception as e:
                logger.warning(f"[Engine] failed to load {yml_file}: {e}")
        return loaded

    def register_workflow(self, yaml_content: str, name: Optional[str] = None) -> WorkflowDef:
        """Dynamically register a YAML workflow (used by POST /api/workflows/register)."""
        wf_def = self.parser.parse_string(yaml_content)
        if name:
            wf_def.name = name
        self._register_workflow_def(wf_def)
        return wf_def

    def _register_workflow_def(self, wf_def: WorkflowDef) -> None:
        """Register a workflow definition (overwrites same-name)."""
        self._workflow_defs[wf_def.name] = wf_def

    def get_workflow_defs(self) -> Dict[str, WorkflowDef]:
        """Return all registered workflow definitions."""
        return dict(self._workflow_defs)

    def get_workflow_def(self, name: str) -> Optional[WorkflowDef]:
        """Look up a workflow by name."""
        return self._workflow_defs.get(name)

    def recover_interrupted_runs(self) -> int:
        """Mark storage's PENDING/RUNNING runs as CANCELLED. Call at API startup."""
        try:
            interrupted = self.storage.list_runs(limit=1000)
        except Exception as e:
            logger.warning(f"[Engine] recover scan failed: {e}")
            return 0
        recovered = 0
        for run in interrupted:
            if run.status in (WorkflowStatus.PENDING, WorkflowStatus.RUNNING):
                run.status = WorkflowStatus.CANCELLED
                run.end_time = datetime.now()
                self.storage.save_run(run)
                recovered += 1
                logger.info(f"[Engine] recovered run {run.id[:8]} -> cancelled")
        return recovered

    def dispatch_workflow(self, workflow_name: str, inputs: Dict[str, Any]) -> WorkflowRun:
        """Create a PENDING run for the named workflow and save to storage. Does NOT execute.

        Used by POST /api/runs endpoint. Actual execution is triggered by caller
        via execute_existing_run (typically in a background task).
        """
        wf_def = self.get_workflow_def(workflow_name)
        if wf_def is None:
            raise ValueError(f"workflow not registered: {workflow_name}")
        run = WorkflowRun(
            workflow_def=wf_def,
            trigger_type="workflow_dispatch",
            trigger_payload=inputs or {},
            env=dict(wf_def.env) if wf_def.env else {},
        )
        if wf_def.file_path:
            run.env["__workflow_file__"] = wf_def.file_path
        run.env["__workflow_name__"] = wf_def.name
        self.storage.save_run(run)
        self.event_bus.publish(WorkflowEvent(
            event_type=EventType.WORKFLOW_CREATED, run_id=run.id,
            data={"workflow_name": wf_def.name, "trigger_type": "workflow_dispatch"},
        ))
        return run

    def execute_existing_run(self, run_id: str) -> WorkflowRun:
        """Load a run from storage and execute it. Used by background tasks."""
        run = self.storage.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        if run.workflow_def is None and run.env.get("__workflow_file__"):
            run.workflow_def = self.parser.parse_file(run.env["__workflow_file__"])
        return self._do_execute(run)

    # -- Internal --

    def _on_coordinator_created(self, run_id: str,
                                coordinator: WorkflowStateMachineCoordinator) -> None:
        """Called by WorkflowExecutor the moment a coordinator is created.
        Allows cancel_run to find the coordinator during execution."""
        self._coordinators[run_id] = coordinator

    def _do_execute(self, run: WorkflowRun) -> WorkflowRun:
        self._active_runs[run.id] = run
        try:
            coordinator = self._workflow_executor.execute(run)
            self._coordinators[run.id] = coordinator
            self.storage.save_run(run)
        finally:
            self._active_runs.pop(run.id, None)
        return run

    def _register_persist_hooks(self) -> None:
        """Subscribe to state events -> auto-save run on every transition."""
        def on_event(event: WorkflowEvent):
            if event.run_id and event.run_id in self._active_runs:
                self.storage.save_run(self._active_runs[event.run_id])
        self.event_bus.subscribe(on_event)

    def _reset_all_components(self, run: WorkflowRun) -> None:
        import shutil
        from pathlib import Path
        for job_id, job in run.jobs.items():
            job.status = JobStatus.PENDING
            job.start_time = None
            job.end_time = None
            job.outputs = {}
            for step in job.steps:
                step.status = StepStatus.PENDING
                step.start_time = None
                step.end_time = None
                step.outputs = {}
                step.log = []
                step.error_msg = ""
        run.status = WorkflowStatus.PENDING
        run.start_time = None
        run.end_time = None
        # Clear product dirs (keep run_manifest.json if exists)
        session_dir = run.env.get("__session_dir__")
        if session_dir:
            for p in Path(session_dir).iterdir():
                if p.name.startswith(("00_", "10_", "50_", "60_")):
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()

    def _mark_for_rerun(self, run: WorkflowRun, components: List[str]) -> None:
        """Reset specified components and their downstream to pending.

        Matrix sub-jobs (e.g. dimensions_novelty) are expanded at runtime by
        JobExecutor._execute_matrix_job, which sets the sub-job ID to
        f"{parent_id}_{combo}".  Downstream jobs (e.g. synthesize) declare
        their `needs` against the PARENT job ID (e.g. dimensions), not the
        sub-job ID, so a naive transitive-closure on sub-job IDs would fail
        to cascade.  To bridge this, when a matrix sub-job is in the rerun
        set we also add its parent job ID to the downstream set.  The
        parent is reset to PENDING so the matrix re-expands on re-execution;
        this means sibling sub-jobs (e.g. dimensions_soundness) are also
        re-run -- a known limitation for matrix rerun.
        """
        all_jobs = list(run.jobs.keys())
        downstream: set = set(components)
        # Matrix parent expansion: if a component is a matrix sub-job (its
        # ID starts with "<parent_id>_"), add the parent to downstream so
        # jobs depending on the parent cascade correctly.
        for c in list(downstream):
            for job_id in all_jobs:
                if job_id == c:
                    continue
                if c.startswith(job_id + "_"):
                    downstream.add(job_id)
                    break
        changed = True
        while changed:
            changed = False
            for job_id in all_jobs:
                if job_id in downstream:
                    continue
                job = run.jobs[job_id]
                if job.job_def and any(d in downstream for d in job.job_def.needs):
                    downstream.add(job_id)
                    changed = True
        for job_id in downstream:
            job = run.jobs[job_id]
            job.status = JobStatus.PENDING
            job.start_time = None
            job.end_time = None
            job.outputs = {}
            for step in job.steps:
                step.status = StepStatus.PENDING
                step.start_time = None
                step.end_time = None
                step.outputs = {}
                step.log = []
                step.error_msg = ""
        run.status = WorkflowStatus.PENDING
        run.start_time = None
        run.end_time = None

    def _reset_failed_and_downstream(self, run: WorkflowRun) -> None:
        """Default resume: reset all non-SUCCESS jobs to PENDING.

        Already-completed (SUCCESS) jobs are preserved so the executor's
        _schedule_jobs can skip them on re-execution.  Failed, skipped,
        and cancelled jobs (and any downstream jobs that depend on them,
        directly or transitively) are reset to PENDING so they re-run.

        Matrix sub-jobs (e.g. dimensions_novelty) are individual JobInstance
        entries in run.jobs; if their parent job is being re-run, the matrix
        executor will overwrite them -- that is a known limitation for M6.2.
        For non-matrix jobs, SUCCESS is preserved and skipped.
        """
        # Identify all non-SUCCESS job IDs that need to be re-run
        to_reset: set = set()
        for job_id, job in run.jobs.items():
            if job.status != JobStatus.SUCCESS:
                to_reset.add(job_id)
        # Propagate to downstream jobs that depend (transitively) on any
        # job being reset -- they must also re-run because their inputs may
        # have changed or they were skipped due to upstream failure.
        all_job_ids = list(run.jobs.keys())
        changed = True
        while changed:
            changed = False
            for job_id in all_job_ids:
                if job_id in to_reset:
                    continue
                job = run.jobs[job_id]
                if job.job_def and any(d in to_reset for d in job.job_def.needs):
                    to_reset.add(job_id)
                    changed = True
        # Reset each job in to_reset to PENDING
        for job_id in to_reset:
            job = run.jobs[job_id]
            job.status = JobStatus.PENDING
            job.start_time = None
            job.end_time = None
            job.outputs = {}
            for step in job.steps:
                step.status = StepStatus.PENDING
                step.start_time = None
                step.end_time = None
                step.outputs = {}
                step.log = []
                step.error_msg = ""
        run.status = WorkflowStatus.PENDING
        run.start_time = None
        run.end_time = None
