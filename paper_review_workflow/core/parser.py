"""YAML parser for paper review workflow definitions.

Ported from lwf, stripped of: script/run/working-directory/max-retries fields,
and push/PR/release/schedule triggers.
"""
import yaml
from pathlib import Path
from typing import Dict, List

from .models import WorkflowDef, JobDef, StepDef, TriggerDef


class WorkflowParser:
    def parse_file(self, file_path: str) -> WorkflowDef:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        wf = self._parse_workflow(raw)
        wf.file_path = file_path
        return wf

    def parse_string(self, yaml_content: str) -> WorkflowDef:
        raw = yaml.safe_load(yaml_content)
        return self._parse_workflow(raw)

    def parse_directory(self, workflows_dir: str) -> Dict[str, WorkflowDef]:
        result = {}
        path = Path(workflows_dir)
        for yml_file in sorted(path.glob("**/*.yml")):
            try:
                wf = self.parse_file(str(yml_file))
                result[yml_file.stem] = wf
            except Exception as e:
                print(f"[Parser] failed {yml_file}: {e}")
        for yaml_file in sorted(path.glob("**/*.yaml")):
            try:
                wf = self.parse_file(str(yaml_file))
                result[yaml_file.stem] = wf
            except Exception as e:
                print(f"[Parser] failed {yaml_file}: {e}")
        return result

    # ── private ──

    def _parse_workflow(self, raw: dict) -> WorkflowDef:
        name = raw.get("name", "Unnamed Workflow")
        # YAML 1.1: `on:` parses to bool True
        on_raw = raw.get("on") or raw.get(True) or {}
        trigger = self._parse_trigger(on_raw)
        env = raw.get("env", {}) or {}
        jobs = self._parse_jobs(raw.get("jobs", {}))
        return WorkflowDef(
            name=name, on=trigger, jobs=jobs,
            env={k: str(v) for k, v in env.items()},
        )

    def _parse_trigger(self, on_raw: object) -> TriggerDef:
        # Only workflow_dispatch supported
        if isinstance(on_raw, dict):
            wd = on_raw.get("workflow_dispatch")
            if wd is None:
                wd = {}
            return TriggerDef(workflow_dispatch=wd)
        return TriggerDef(workflow_dispatch={})

    def _parse_jobs(self, jobs_raw: dict) -> Dict[str, JobDef]:
        return {jid: self._parse_job(jid, raw)
                for jid, raw in (jobs_raw or {}).items()}

    def _parse_job(self, job_id: str, raw: dict) -> JobDef:
        steps = [self._parse_step(i, s)
                 for i, s in enumerate(raw.get("steps", []))]
        raw_outputs = raw.get("outputs", {}) or {}
        return JobDef(
            id=job_id,
            name=raw.get("name", job_id),
            runs_on=raw.get("runs-on", "local"),
            steps=steps,
            needs=self._parse_needs(raw.get("needs")),
            env={k: str(v) for k, v in (raw.get("env", {}) or {}).items()},
            outputs={k: str(v) for k, v in raw_outputs.items()},
            condition=raw.get("if"),
            strategy=raw.get("strategy"),
            timeout_minutes=raw.get("timeout-minutes"),
            continue_on_error=raw.get("continue-on-error", False),
        )

    def _parse_step(self, index: int, raw: dict) -> StepDef:
        return StepDef(
            id=raw.get("id", f"step_{index}"),
            name=raw.get("name", f"Step {index + 1}"),
            uses=raw.get("uses"),
            with_params=raw.get("with", {}) or {},
            env=raw.get("env", {}) or {},
            condition=raw.get("if"),
            continue_on_error=raw.get("continue-on-error", False),
            timeout_minutes=raw.get("timeout-minutes"),
        )

    def _parse_needs(self, needs_raw) -> List[str]:
        if needs_raw is None:
            return []
        if isinstance(needs_raw, str):
            return [needs_raw]
        if isinstance(needs_raw, list):
            return needs_raw
        return []
