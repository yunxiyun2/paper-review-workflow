"""Workflow runtime context with ${{ }} expression evaluation.

Ported from lwf, stripped of cookie/secret loading.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .models import WorkflowRun, JobInstance, StepInstance


_EXPR_PATTERN = re.compile(r"\$\{\{\s*([^}]+?)\s*\}\}")


class WorkflowContext:
    def __init__(self, run: WorkflowRun):
        self.run = run

    def build_context(self, job_instance: Optional[JobInstance] = None,
                      step_instance: Optional[StepInstance] = None) -> Dict[str, Any]:
        jobs_ctx = {jid: {"outputs": j.outputs, "status": j.status.value, "result": j.status.value}
                    for jid, j in self.run.jobs.items()}
        ctx: Dict[str, Any] = {
            "inputs": self.run.trigger_payload or {},
            "env": self.run.env or {},
            "workflow": {"name": self.run.workflow_def.name if self.run.workflow_def else ""},
            "jobs": jobs_ctx,
            # `needs` is an alias for `jobs` — standard GitHub Actions syntax
            # for referencing upstream job outputs (e.g. ${{ needs.extract.outputs.X }}).
            "needs": jobs_ctx,
        }
        # Add matrix context from MATRIX_* env vars (injected by matrix executor
        # into the sub-job's job_def.env, not the workflow-level run.env)
        matrix: Dict[str, str] = {}
        merged_env = dict(self.run.env or {})
        if job_instance and job_instance.job_def:
            merged_env.update(job_instance.job_def.env or {})
        for k, v in merged_env.items():
            if k.startswith("MATRIX_"):
                matrix[k[7:].lower()] = v
        if matrix:
            ctx["matrix"] = matrix
        if job_instance:
            ctx["job"] = {"id": job_instance.job_def.id if job_instance.job_def else ""}
            ctx["steps"] = {s.step_def.id: {"outputs": s.outputs}
                            for s in job_instance.steps if s.step_def}
        if step_instance:
            ctx["step"] = {"id": step_instance.step_def.id if step_instance.step_def else ""}
        return ctx

    def resolve(self, expr: str, job_instance: Optional[JobInstance] = None,
                step_instance: Optional[StepInstance] = None) -> Any:
        if not isinstance(expr, str):
            return expr
        ctx = self.build_context(job_instance, step_instance)

        def replace(m):
            path = m.group(1).strip()
            return str(self._eval_path(path, ctx))

        result = _EXPR_PATTERN.sub(replace, expr)
        # If the whole string was one expression, return the typed value
        full_match = _EXPR_PATTERN.fullmatch(expr.strip())
        if full_match:
            return self._eval_path(full_match.group(1).strip(), ctx)
        return result

    def _eval_path(self, path: str, ctx: Dict[str, Any]) -> Any:
        parts = path.split(".")
        cur: Any = ctx
        for p in parts:
            if cur is None:
                return None
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                cur = getattr(cur, p, None)
        return cur

    def evaluate_condition(self, condition: str,
                           job_instance: Optional[JobInstance] = None) -> bool:
        if not condition:
            return True
        if condition.strip() == "always()":
            return True
        result = self.resolve(condition, job_instance)
        return bool(result)

    def build_env(self, job_instance: JobInstance,
                  step_instance: StepInstance) -> Dict[str, str]:
        env: Dict[str, str] = {}
        env.update(self.run.env or {})
        if job_instance.job_def:
            env.update(job_instance.job_def.env or {})
        if step_instance.step_def:
            env.update(step_instance.step_def.env or {})
        return env
