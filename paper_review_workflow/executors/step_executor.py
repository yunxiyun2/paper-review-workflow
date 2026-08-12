"""
Step executor: dispatches to registered actions via the registry.

Ported from lwf with major removals:
- _execute_run, _execute_script, _ScriptOutputCollector (no shell/script actions)
- WAITING / WAITING_RETRY handling (no human approval)
- WORKFLOW_CONTEXT / WORKFLOW_INPUT_* / WORKFLOW_CTX_* / WORKFLOW_STEP_* env flattening
- STEP_RESUMED_DATA / STEP_WAIT_EXTRA / STEP_RETRY_COUNT / STEP_MAX_RETRIES env vars
- Cookie manager injection
- fail_on_error / max_retries / WAITING_RETRY handling

Only the USES action path remains. State machine transitions:
  PENDING → RUNNING → SUCCESS | FAILURE | CANCELLED | SKIPPED
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from ..core.models import StepDef, StepInstance, JobInstance
from ..core.context import WorkflowContext
from ..core.event_bus import EventBus, make_step_log_event
from ..core.state_machine import WorkflowStateMachineCoordinator
from ..actions.registry import ActionRegistry
from ..actions.base import ActionResult

logger = logging.getLogger(__name__)


class StepExecutor:
    """Step executor: dispatches to a registered action via the registry."""

    def __init__(
        self,
        registry: ActionRegistry,
        context: WorkflowContext,
        event_bus: EventBus,
        coordinator: WorkflowStateMachineCoordinator,
        run_id: str,
        job_id: str,
    ):
        self.registry = registry
        self.ctx = context
        self.event_bus = event_bus
        self.coordinator = coordinator
        self.run_id = run_id
        self.job_id = job_id

    def execute(self, step_def: StepDef, job_instance: JobInstance) -> StepInstance:
        """Execute a single step, driven by the state machine."""
        instance = StepInstance(step_def=step_def)
        # Append early so the coordinator can find it via run.jobs
        job_instance.steps.append(instance)

        logger.info(f"  ▶ Step: {step_def.name}")

        # ── 1. Condition check (PENDING → SKIPPED) ──
        if step_def.condition:
            should_run = self.ctx.evaluate_condition(step_def.condition, job_instance)
            if not should_run:
                logger.info(f"  ⏭ skip (condition false: {step_def.condition})")
                self.coordinator.skip_step(self.job_id, instance)
                return instance

        # ── 2. Start execution (PENDING → RUNNING) ──
        self.coordinator.start_step(self.job_id, instance)

        # Build log callback (writes to instance + broadcasts via EventBus)
        def log_cb(line: str) -> None:
            instance.log.append(line)
            self.event_bus.publish(
                make_step_log_event(self.run_id, self.job_id, instance.id, line)
            )

        # ── 3. Build merged env (workflow < job < step) ──
        env = self.ctx.build_env(job_instance, instance)

        # ── 4. Check if workflow already cancelled ──
        if self.coordinator.workflow_sm.is_terminal:
            log_cb("[INFO] workflow cancelled, skipping step execution")
            self.coordinator.cancel_step(self.job_id, instance)
            return instance

        # cancel_check: long-running actions poll this to terminate early
        def cancel_check() -> bool:
            return self.coordinator.workflow_sm.is_terminal

        # ── 5. Dispatch to action (only USES path) ──
        result = self._execute_uses(step_def, env, job_instance, log_cb, cancel_check)

        # ── 6. Write outputs / exit_code / logs to instance ──
        instance.outputs = result.outputs
        instance.exit_code = result.exit_code
        for line in result.log_lines:
            if line not in instance.log:
                instance.log.append(line)

        # ── 7. If cancelled during action, mark CANCELLED not FAILURE ──
        if not result.success and self.coordinator.workflow_sm.is_terminal:
            log_cb("[INFO] workflow cancelled, step marked CANCELLED")
            self.coordinator.cancel_step(self.job_id, instance)
            return instance

        # ── 8. State transition (RUNNING → SUCCESS / FAILURE) ──
        if result.success:
            self.coordinator.complete_step(self.job_id, instance, success=True)
            logger.info("  ✅ success")
        else:
            instance.error_msg = result.message
            if step_def.continue_on_error:
                self.coordinator.complete_step(self.job_id, instance, success=True)
                logger.warning(f"  ⚠️  failed but continue-on-error: {result.message}")
                log_cb(f"[WARN] failed but continue-on-error=true: {result.message}")
            else:
                self.coordinator.complete_step(self.job_id, instance, success=False)
                logger.warning(f"  ❌ failed: {result.message}")
                log_cb(f"[ERROR] {result.message}")

        dur = f"{instance.duration:.2f}s" if instance.duration else "?"
        logger.info(f"  ⏱ duration: {dur}")
        return instance

    def _execute_uses(
        self,
        step_def: StepDef,
        env: dict,
        job_instance: JobInstance,
        log_cb: Callable,
        cancel_check: Optional[Callable] = None,
    ) -> ActionResult:
        """Execute a uses action."""
        action = self.registry.get(step_def.uses)
        if action is None:
            msg = f"action not found: {step_def.uses}"
            logger.warning(f"  ⚠️  {msg}")
            log_cb(f"[ERROR] {msg}")
            return ActionResult(
                success=False,
                message=msg,
                log_lines=[f"[ERROR] {msg}"],
            )

        # Resolve ${{ }} expressions in with_params values
        resolved_params = {}
        for k, v in (step_def.with_params or {}).items():
            if isinstance(v, str) and "${{" in v:
                resolved_params[k] = self.ctx.resolve(v, job_instance)
            else:
                resolved_params[k] = v

        # Build context snapshot
        ctx_snapshot = self.ctx.build_context(job_instance, None)

        try:
            result = action.run(
                params=resolved_params,
                env=env,
                context=ctx_snapshot,
                log_callback=log_cb,
            )
        except Exception as e:
            logger.exception(f"action {step_def.uses} raised")
            log_cb(f"[ERROR] action raised: {e}")
            return ActionResult(
                success=False,
                message=f"action raised: {e}",
                log_lines=[f"[ERROR] {e}"],
            )

        return result
