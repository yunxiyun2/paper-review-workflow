"""
Job executor: runs steps sequentially within a job, supports matrix strategy.

Ported from lwf verbatim (matrix logic is reusable), only import paths changed.
"""
from __future__ import annotations

import copy
import itertools
import logging
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from typing import Callable, Dict, List, Optional

from ..core.models import (
    JobDef, JobInstance, JobStatus, StepInstance, StepStatus, WorkflowRun,
)
from ..core.context import WorkflowContext
from ..core.event_bus import EventBus
from ..core.state_machine import WorkflowStateMachineCoordinator
from ..actions.registry import ActionRegistry
from .step_executor import StepExecutor

logger = logging.getLogger(__name__)


class JobExecutor:
    """
    Job executor.
    - Manages Job / Step state transitions via WorkflowStateMachineCoordinator
    - Executes steps sequentially
    - Supports matrix strategy (parallel execution across parameter combinations)
    """

    def __init__(
        self,
        registry: ActionRegistry,
        context: WorkflowContext,
        event_bus: EventBus,
        coordinator: WorkflowStateMachineCoordinator,
        run_id: str,
        on_step_executor_created: Optional[Callable[[str, str, StepExecutor], None]] = None,
    ):
        self.registry = registry
        self.ctx = context
        self.event_bus = event_bus
        self.coordinator = coordinator
        self.run_id = run_id
        self._on_step_executor_created = on_step_executor_created

    def execute(self, job_def: JobDef, workflow_run: WorkflowRun) -> JobInstance:
        """Execute a single job."""
        instance = JobInstance(job_def=job_def)
        workflow_run.jobs[job_def.id] = instance

        logger.info(f"\n{'='*60}")
        logger.info(f"🔧 Job: {job_def.name}  [runs-on: {job_def.runs_on}]")
        logger.info(f"{'='*60}")

        # ── 1. Condition check (PENDING → SKIPPED) ──
        if job_def.condition:
            should_run = self.ctx.evaluate_condition(job_def.condition)
            if not should_run:
                logger.info(f"⏭ Job skipped (condition false: {job_def.condition})")
                self.coordinator.skip_job(job_def.id, instance)
                return instance

        # ── 2. Start execution (PENDING → RUNNING) ──
        self.coordinator.start_job(job_def.id, instance)

        # ── 3. Handle matrix strategy ──
        if job_def.strategy and "matrix" in job_def.strategy:
            matrix_any_failed = self._execute_matrix(job_def, instance, workflow_run)
        else:
            matrix_any_failed = None  # non-matrix mode, determined by steps
            self._execute_steps(job_def, instance)

        # ── 4. Complete job (RUNNING → SUCCESS / FAILURE / CANCELLED) ──
        if self.coordinator.workflow_sm.is_terminal:
            self.coordinator.cancel_job(job_def.id, instance)
        else:
            if matrix_any_failed is not None:
                job_failed = matrix_any_failed
            else:
                job_failed = any(
                    si.status == StepStatus.FAILURE
                    for si in instance.steps
                )
            self.coordinator.complete_job(job_def.id, instance, success=not job_failed)

        # ── 5. Resolve job-level outputs ──
        if job_def.outputs:
            try:
                resolved_outputs: dict = {}
                for key, expr in job_def.outputs.items():
                    resolved_val = self.ctx.resolve(expr, instance)
                    resolved_outputs[key] = resolved_val
                instance.outputs = resolved_outputs
            except Exception as e:
                logger.warning(f"[JobExecutor] Job '{job_def.id}' outputs resolve failed: {e}")

        dur = f"{instance.duration:.2f}s" if instance.duration else "?"
        logger.info(f"\n{'─'*60}")
        logger.info(f"Job result: {instance.status.value.upper()}  duration: {dur}")
        if instance.outputs:
            logger.info(f"Job outputs: {instance.outputs}")
        return instance

    def _make_step_executor(self, job_id: str) -> StepExecutor:
        """Create a StepExecutor and invoke the registration callback."""
        se = StepExecutor(
            registry=self.registry,
            context=self.ctx,
            event_bus=self.event_bus,
            coordinator=self.coordinator,
            run_id=self.run_id,
            job_id=job_id,
        )
        if self._on_step_executor_created:
            self._on_step_executor_created(self.run_id, job_id, se)
        return se

    def _execute_steps(self, job_def: JobDef, job_instance: JobInstance) -> None:
        """Execute all steps sequentially (state machine driven)."""
        step_executor = self._make_step_executor(job_def.id)
        job_failed = False

        for step_def in job_def.steps:
            # Check if workflow has been cancelled
            if self.coordinator.workflow_sm.is_terminal:
                logger.info(f"[JobExecutor] workflow cancelled, skipping step '{step_def.name}'")
                skip_instance = StepInstance(step_def=step_def)
                job_instance.steps.append(skip_instance)
                self.coordinator.cancel_step(job_def.id, skip_instance)
                continue

            # If a previous step failed, skip subsequent steps (unless always())
            if job_failed and step_def.condition != "always()":
                skip_instance = StepInstance(step_def=step_def)
                job_instance.steps.append(skip_instance)
                self.coordinator.skip_step(job_def.id, skip_instance)
                continue

            # StepExecutor.execute appends the instance to job_instance.steps internally
            step_instance = step_executor.execute(step_def, job_instance)

            if step_instance.status == StepStatus.FAILURE:
                job_failed = True

    def _execute_matrix(
        self,
        job_def: JobDef,
        job_instance: JobInstance,
        workflow_run: WorkflowRun,
    ) -> bool:
        """Execute matrix strategy (parallel execution across combinations)."""
        matrix = job_def.strategy["matrix"]
        combos = self._build_matrix_combinations(matrix)
        max_parallel = job_def.strategy.get("max-parallel", len(combos))

        logger.info(f"📊 Matrix strategy: {len(combos)} combinations, max-parallel: {max_parallel}")
        results: List[JobInstance] = []

        matrix_pool = ThreadPoolExecutor(max_workers=max_parallel)
        try:
            pending_futures = {
                matrix_pool.submit(
                    self._execute_matrix_job, job_def, combo, workflow_run
                )
                for combo in combos
            }

            while pending_futures:
                done, pending_futures = wait(pending_futures, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Matrix job error: {e}")
        finally:
            matrix_pool.shutdown(wait=False)

        any_failed = any(r.status == JobStatus.FAILURE for r in results)
        return any_failed

    def _execute_matrix_job(
        self, job_def: JobDef, combo: Dict, workflow_run: WorkflowRun
    ) -> JobInstance:
        """Execute a single matrix combination."""
        matrix_job_def = copy.deepcopy(job_def)
        matrix_job_def.id = f"{job_def.id}_{'_'.join(str(v) for v in combo.values())}"
        matrix_job_def.name = f"{job_def.name} {combo}"
        matrix_job_def.env.update({
            f"MATRIX_{k.upper()}": str(v) for k, v in combo.items()
        })

        # matrix sub-job uses its own JobInstance and state machine
        sub_instance = JobInstance(job_def=matrix_job_def)
        workflow_run.jobs[matrix_job_def.id] = sub_instance
        self.coordinator.start_job(matrix_job_def.id, sub_instance)

        step_executor = self._make_step_executor(matrix_job_def.id)

        job_failed = False
        for step_def in matrix_job_def.steps:
            # Check if workflow has been cancelled
            if self.coordinator.workflow_sm.is_terminal:
                skip = StepInstance(step_def=step_def)
                sub_instance.steps.append(skip)
                self.coordinator.cancel_step(matrix_job_def.id, skip)
                continue
            if job_failed and step_def.condition != "always()":
                skip = StepInstance(step_def=step_def)
                sub_instance.steps.append(skip)
                self.coordinator.skip_step(matrix_job_def.id, skip)
                continue
            step_instance = step_executor.execute(step_def, sub_instance)
            if step_instance.status == StepStatus.FAILURE:
                job_failed = True

        if self.coordinator.workflow_sm.is_terminal:
            self.coordinator.cancel_job(matrix_job_def.id, sub_instance)
        else:
            self.coordinator.complete_job(matrix_job_def.id, sub_instance, success=not job_failed)

        # Resolve matrix sub-job outputs
        if matrix_job_def.outputs:
            try:
                resolved_outputs: dict = {}
                for key, expr in matrix_job_def.outputs.items():
                    resolved_outputs[key] = self.ctx.resolve(expr, sub_instance)
                sub_instance.outputs = resolved_outputs
            except Exception as e:
                logger.warning(f"[JobExecutor] Matrix Job '{matrix_job_def.id}' outputs resolve failed: {e}")

        return sub_instance

    def _build_matrix_combinations(self, matrix: Dict) -> List[Dict]:
        """Build all parameter combinations for the matrix."""
        keys = list(matrix.keys())
        values = [
            matrix[k] if isinstance(matrix[k], list) else [matrix[k]]
            for k in keys
        ]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
