"""
Workflow executor: topological job scheduling with parallel execution.

Ported from lwf, stripped of:
- def_record / workflow_version / workflow_def_id (no version management)
- load_secrets_from_env (no cookie/secret loading)

The ThreadPoolExecutor-based _schedule_jobs logic is reusable as-is.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from typing import Callable, Dict, List, Optional, Set

from ..core.models import (
    WorkflowDef, WorkflowRun, WorkflowStatus,
    JobInstance, JobStatus, JobDef,
)
from ..core.context import WorkflowContext
from ..core.event_bus import EventBus
from ..core.state_machine import (
    WorkflowStateMachineCoordinator,
    InvalidTransitionError,
)
from ..actions.registry import ActionRegistry
from .job_executor import JobExecutor

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """
    Workflow executor.

    Responsibilities (single responsibility):
      1. Create and hold a WorkflowStateMachineCoordinator for one WorkflowRun
      2. Build the WorkflowContext (env injection)
      3. Topologically schedule jobs by needs, running ready jobs in parallel
      4. Handle failure propagation: skip jobs that depend on failed jobs
      5. Summarize all job results to derive the workflow's final status
      6. Print execution log summary

    Not responsible for:
      - Creating the WorkflowRun object (done by Engine)
      - Persistence (done by Engine via storage)
      - Triggering logic (done by TriggerRegistry)
    """

    def __init__(
        self,
        registry: ActionRegistry,
        event_bus: EventBus,
        on_step_executor_created: Optional[Callable] = None,
        on_coordinator_created: Optional[Callable] = None,
        on_workflow_started: Optional[Callable] = None,
    ):
        self.registry = registry
        self.event_bus = event_bus
        self._on_step_executor_created = on_step_executor_created
        self._on_coordinator_created = on_coordinator_created
        self._on_workflow_started = on_workflow_started

    def execute(self, run: WorkflowRun) -> WorkflowStateMachineCoordinator:
        """
        Execute a complete WorkflowRun.

        :param run: A WorkflowRun object (in PENDING state)
        :returns:   The WorkflowStateMachineCoordinator for this execution,
                    usable for querying transition history or cancelling
        """
        wf_def = run.workflow_def
        coordinator = WorkflowStateMachineCoordinator(run, self.event_bus)

        # Notify engine immediately so cancel_run can find the coordinator mid-execution
        if self._on_coordinator_created:
            self._on_coordinator_created(run.id, coordinator)

        self._print_header(run)

        # ── PENDING → RUNNING ──
        try:
            coordinator.start_workflow()
        except InvalidTransitionError as e:
            logger.error(f"[WorkflowExecutor] state transition failed: {e}")
            return coordinator

        # ── Incremental persistence: start_workflow done, run.status=RUNNING ──
        if self._on_workflow_started:
            self._on_workflow_started(run)

        ctx = WorkflowContext(run)

        # ── Resolve expressions in workflow-level env ──
        if run.env:
            run.env = {
                k: str(ctx.resolve(v)) if isinstance(v, str) else str(v)
                for k, v in run.env.items()
                if not k.startswith("__")  # preserve internal __workflow_name__ etc.
            }
            if run.workflow_def:
                run.env.setdefault("__workflow_name__", run.workflow_def.name)
                if run.workflow_def.file_path:
                    run.env.setdefault("__workflow_file__", run.workflow_def.file_path)

        try:
            self._schedule_jobs(wf_def, run, ctx, coordinator)
            # If the workflow was cancelled during scheduling, the state machine
            # is already terminal; don't call complete again
            if not coordinator.workflow_sm.is_terminal:
                summary = self._summarize_status(run)
                coordinator.complete_workflow(success=(summary == WorkflowStatus.SUCCESS))
            else:
                logger.info("[WorkflowExecutor] workflow cancelled, skipping complete_workflow")
        except InvalidTransitionError as e:
            logger.error(f"[WorkflowExecutor] state machine transition error: {e}")
            if not coordinator.workflow_sm.is_terminal:
                coordinator.complete_workflow(success=False)
        except Exception as e:
            logger.error(f"[WorkflowExecutor] ❌ execution error: {e}", exc_info=True)
            if not coordinator.workflow_sm.is_terminal:
                coordinator.complete_workflow(success=False)

        self._print_summary(run)
        return coordinator

    # ─────────────────────────────────────────
    # Job scheduling: topological sort + parallel execution
    # ─────────────────────────────────────────

    def _schedule_jobs(
        self,
        workflow_def: WorkflowDef,
        run: WorkflowRun,
        ctx: WorkflowContext,
        coordinator: WorkflowStateMachineCoordinator,
    ) -> None:
        """
        Schedule jobs by needs dependency with topological sort, supporting
        continuous parallel execution.

        Design: event-driven streaming scheduling.
          1. Maintain a global ThreadPoolExecutor for the entire workflow lifecycle.
          2. Whenever a job future completes, add it to the completed set and
             check if any new ready jobs can be submitted.
          3. Stream "complete one, schedule one" via wait(FIRST_COMPLETED).
          4. If a job fails, collect all jobs that depend on it and mark SKIPPED.
        """
        job_executor = JobExecutor(
            registry=self.registry,
            context=ctx,
            event_bus=self.event_bus,
            coordinator=coordinator,
            run_id=run.id,
            on_step_executor_created=self._on_step_executor_created,
        )
        jobs = workflow_def.jobs
        completed: Set[str] = set()
        pending: Set[str] = set(jobs.keys())
        running: Set[str] = set()

        # Global thread pool for the entire workflow scheduling lifecycle
        pool = ThreadPoolExecutor(max_workers=max(len(jobs), 1))
        future_to_jid: Dict = {}

        def _submit_ready_jobs() -> bool:
            """Submit all pending jobs whose dependencies are satisfied."""
            submitted = False
            for jid in list(pending):
                if jid in running:
                    continue
                if coordinator.workflow_sm.is_terminal:
                    break
                if all(dep in completed for dep in jobs[jid].needs):
                    logger.info(f"  ▶ submit Job: {jid}")
                    f = pool.submit(job_executor.execute, jobs[jid], run)
                    future_to_jid[f] = jid
                    running.add(jid)
                    submitted = True
            return submitted

        try:
            # Initial submission: find all dependency-free jobs
            if not _submit_ready_jobs():
                if pending:
                    logger.warning(
                        "[WorkflowExecutor] ⚠️  circular dependency or unsatisfiable dependencies, aborting scheduling"
                    )
                return

            # ── Streaming: after each job completes, try scheduling new ready jobs ──
            while running:
                running_futures = {
                    f for f, jid in future_to_jid.items()
                    if jid in running
                }
                if not running_futures:
                    break

                done, _ = wait(running_futures, return_when=FIRST_COMPLETED)

                for future in done:
                    jid = future_to_jid[future]
                    running.discard(jid)
                    pending.discard(jid)

                    try:
                        future.result()
                    except Exception as exc:
                        logger.error(
                            f"[WorkflowExecutor] Job '{jid}' thread error: {exc}",
                            exc_info=True,
                        )

                    completed.add(jid)

                    # Check if workflow has been cancelled
                    if coordinator.workflow_sm.is_terminal:
                        logger.info("[WorkflowExecutor] workflow cancelled, stop submitting new jobs")
                        break

                    # Check for failure: skip dependent jobs
                    if run.jobs.get(jid) and run.jobs[jid].status == JobStatus.FAILURE:
                        logger.warning(
                            f"[WorkflowExecutor] ⚠️  Job '{jid}' failed, dependent jobs will be skipped"
                        )
                        self._skip_dependent_jobs([jid], pending, jobs, run, coordinator)

                        # Immediately submit always() jobs that were kept in pending
                        if pending:
                            _submit_ready_jobs()

                        if not running:
                            break
                        continue

                    # Job succeeded, check for newly ready jobs
                    if pending:
                        _submit_ready_jobs()
                        if not running and pending:
                            logger.warning(
                                "[WorkflowExecutor] ⚠️  circular dependency or unsatisfiable dependencies, aborting scheduling"
                            )
                            break

                if coordinator.workflow_sm.is_terminal:
                    break

        finally:
            pool.shutdown(wait=False)

        # After cancellation, mark all unstarted pending jobs as CANCELLED
        if coordinator.workflow_sm.is_terminal and pending:
            for job_id in list(pending):
                if job_id not in run.jobs:
                    instance = JobInstance(job_def=jobs[job_id])
                    run.jobs[job_id] = instance
                else:
                    instance = run.jobs[job_id]
                if instance.status == JobStatus.PENDING:
                    coordinator.cancel_job(job_id, instance)
                    logger.info(f"[WorkflowExecutor] Job '{job_id}' marked CANCELLED (unstarted)")

    @staticmethod
    def _is_always_condition(condition: str) -> bool:
        """Check if condition is exactly always() (not via expression evaluation)."""
        import re
        cond = condition.strip()
        if re.fullmatch(r'always\(\)', cond):
            return True
        if re.fullmatch(r'\$\{\{\s*always\(\)\s*\}\}', cond):
            return True
        return False

    def _skip_dependent_jobs(
        self,
        failed_jobs: List[str],
        pending: Set[str],
        jobs: Dict[str, JobDef],
        run: WorkflowRun,
        coordinator: WorkflowStateMachineCoordinator,
    ) -> None:
        """
        Find all jobs (directly or indirectly) depending on failed jobs and
        transition them to SKIPPED via the state machine.

        Exception: jobs with if: always() are not skipped.
        """
        to_skip: Set[str] = set()
        changed = True
        while changed:
            changed = False
            for job_id in list(pending):
                if job_id in to_skip:
                    continue
                if any(dep in failed_jobs or dep in to_skip for dep in jobs[job_id].needs):
                    job_cond = jobs[job_id].condition
                    if job_cond and self._is_always_condition(job_cond):
                        logger.info(
                            f"  ⚠️  Job '{job_id}' depends on failed job but has always(), will still run"
                        )
                        continue
                    to_skip.add(job_id)
                    changed = True

        for job_id in to_skip:
            pending.discard(job_id)
            instance = JobInstance(job_def=jobs[job_id])
            run.jobs[job_id] = instance
            coordinator.skip_job(job_id, instance)
            logger.info(f"  ⏭ skip Job: {job_id}")

    # ─────────────────────────────────────────
    # Status summary
    # ─────────────────────────────────────────

    def _summarize_status(self, run: WorkflowRun) -> WorkflowStatus:
        """Summarize workflow status from all job final states."""
        statuses = [ji.status for ji in run.jobs.values()]
        if not statuses:
            return WorkflowStatus.SUCCESS
        if any(s == JobStatus.FAILURE for s in statuses):
            return WorkflowStatus.FAILURE
        if any(s == JobStatus.CANCELLED for s in statuses):
            return WorkflowStatus.CANCELLED
        if all(s in (JobStatus.SUCCESS, JobStatus.SKIPPED) for s in statuses):
            return WorkflowStatus.SUCCESS
        return WorkflowStatus.FAILURE

    # ─────────────────────────────────────────
    # Log output
    # ─────────────────────────────────────────

    def _print_header(self, run: WorkflowRun) -> None:
        wf = run.workflow_def
        logger.info(f"\n{'█'*60}")
        logger.info(f"  🚀 workflow: {wf.name}")
        logger.info(f"  📋 Run #:  {run.run_number}  ID: {run.id[:8]}")
        logger.info(f"  🎯 trigger:   {run.trigger_type}")
        logger.info(f"  📦 Jobs:   {list(wf.jobs.keys())}")
        logger.info(f"{'█'*60}")

    def _print_summary(self, run: WorkflowRun) -> None:
        icon = {"success": "✅", "failure": "❌", "cancelled": "🚫"}.get(
            run.status.value, "❓"
        )
        logger.info(f"\n{'█'*60}")
        logger.info(f"  {icon} workflow completed: {run.status.value.upper()}")
        dur = f"{run.duration:.2f}s" if run.duration else "?"
        logger.info(f"  ⏱ total duration: {dur}")
        logger.info("\n  Job summary:")
        for job_id, ji in run.jobs.items():
            s_icon = {
                "success": "✅", "failure": "❌",
                "skipped": "⏭", "running": "🔄",
                "cancelled": "🚫",
            }.get(ji.status.value, "❓")
            j_dur = f"{ji.duration:.2f}s" if ji.duration else "—"
            logger.info(f"    {s_icon} {job_id:<20} {ji.status.value:<12} {j_dur}")
        logger.info(f"{'█'*60}\n")
