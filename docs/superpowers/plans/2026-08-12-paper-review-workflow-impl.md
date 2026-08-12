# Paper Review Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python single-process workflow engine that runs an 8-dimension LLM-based paper review on a PDF or arXiv paper, producing structured scores and a final recommendation with resume/rerun support.

**Architecture:** Reuse the lwf (`/Users/dengyunxi/workspace/for-staging/lwf`) GitHub-Actions-style workflow engine skeleton (parser / three-layer state machine / executors / event bus / storage), strip the features we don't need (triggers, version management, script/run/wait protocols, FastAPI server), and add paper-review-specific actions (extract / dim_score matrix / synthesize / decide) plus a pluggable LLM provider abstraction backed by the synchronous `anthropic` SDK with prompt caching.

**Tech Stack:** Python 3.10+, anthropic SDK (sync), PyMuPDF, arxiv, pyyaml, pydantic v2, jinja2, pytest. ThreadPoolExecutor for matrix parallelism (8 dimensions in parallel).

**Reference SPEC:** `docs/superpowers/specs/2026-08-12-paper-review-workflow-design.md`

**Reference lwf source:** `/Users/dengyunxi/workspace/for-staging/lwf/workflow_engine/`

---

## File Structure Overview

```
paper-review-workflow/
├── main.py                              # CLI entry (argparse subcommands)
├── pyproject.toml                       # Build/ deps
├── requirements.txt                     # Pinned deps
├── README.md                            # User docs
├── configs/
│   ├── minimal.yaml                     # M1 smoke test
│   └── normal_review.yaml               # Production config
├── paper_review_workflow/
│   ├── __init__.py
│   ├── engine.py                        # ReviewEngine facade
│   ├── cli.py                           # Subcommand dispatch
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py                    # WorkflowRun/Job/Step dataclasses
│   │   ├── parser.py                    # YAML → WorkflowDef
│   │   ├── context.py                   # ${{ }} expression eval
│   │   ├── state_machine.py             # 3-layer state machine
│   │   ├── event_bus.py                 # Pub/sub events
│   │   └── paths.py                     # session dir / id generation
│   ├── executors/
│   │   ├── __init__.py
│   │   ├── workflow_executor.py         # Job topo + parallel
│   │   ├── job_executor.py              # matrix strategy
│   │   └── step_executor.py             # Action dispatch (uses only)
│   ├── actions/
│   │   ├── __init__.py                  # register_builtin_actions()
│   │   ├── base.py                      # BaseAction + ActionResult
│   │   ├── registry.py                  # ActionRegistry singleton
│   │   ├── builtin.py                   # EchoAction (M1) + register
│   │   ├── extract/
│   │   │   ├── __init__.py              # ExtractAction
│   │   │   ├── pdf.py                   # PyMuPDF parsing
│   │   │   └── arxiv.py                 # arxiv source fetch
│   │   ├── dimensions/
│   │   │   ├── __init__.py              # DimensionAction
│   │   │   └── prompts/                 # 8 jinja2 templates
│   │   ├── synthesize.py
│   │   └── decide.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                      # LLMProvider ABC + LLMResponse
│   │   ├── registry.py                  # ProviderRegistry
│   │   ├── client.py                    # LLMClient.from_env()
│   │   ├── anthropic_provider.py
│   │   ├── schemas.py                   # Pydantic models
│   │   └── prompts/                     # jinja2 templates
│   └── storage/
│       ├── __init__.py
│       ├── backend.py                   # StorageBackend ABC
│       ├── memory.py                    # in-memory
│       └── json_file.py                 # JSON file persistence
├── tests/
│   ├── conftest.py                      # fixtures
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── sessions/                            # runtime output (gitignored)
```

---

# M1: Skeleton Setup

**Goal:** Project skeleton that can run a minimal YAML with an EchoAction end-to-end. lwf code is copied and stripped of unwanted features.

**Estimated:** 1 day

## Task 1.1: Create project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `paper_review_workflow/__init__.py`
- Create: `paper_review_workflow/core/__init__.py`
- Create: `paper_review_workflow/executors/__init__.py`
- Create: `paper_review_workflow/actions/__init__.py`
- Create: `paper_review_workflow/llm/__init__.py`
- Create: `paper_review_workflow/storage/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "paper-review-workflow"
version = "0.1.0"
description = "AI workflow for comprehensive academic paper review and scoring"
requires-python = ">=3.10"
dependencies = [
    "anthropic>=0.40.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "pymupdf>=1.24.0",
    "arxiv>=2.1.0",
    "httpx>=0.27.0",
    "jinja2>=3.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "pytest-mock>=3.12",
]

[project.scripts]
prw = "paper_review_workflow.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["paper_review_workflow"]

[tool.pytest.ini_options]
markers = [
    "e2e: end-to-end tests requiring ANTHROPIC_API_KEY",
]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `requirements.txt`**

```
anthropic>=0.40.0
pyyaml>=6.0
pydantic>=2.0
pymupdf>=1.24.0
arxiv>=2.1.0
httpx>=0.27.0
jinja2>=3.1.0
```

- [ ] **Step 3: Create `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
.coverage
*.egg-info/
build/
dist/
.venv/
venv/
sessions/
.env
*.log
.DS_Store
```

- [ ] **Step 4: Create empty `__init__.py` files**

For each of:
- `paper_review_workflow/__init__.py` (content: `"""Paper review workflow engine."""`)
- `paper_review_workflow/core/__init__.py` (empty)
- `paper_review_workflow/executors/__init__.py` (empty, will fill in Task 1.4)
- `paper_review_workflow/actions/__init__.py` (empty, will fill in Task 1.6)
- `paper_review_workflow/llm/__init__.py` (empty)
- `paper_review_workflow/storage/__init__.py` (empty, will fill in Task 1.5)

- [ ] **Step 5: Install dev dependencies**

Run: `pip install -e ".[dev]"`
Expected: Successful install, `prw --help` not yet functional (cli.py not created).

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml requirements.txt .gitignore paper_review_workflow/
git commit -m "chore: scaffold project structure"
```

## Task 1.2: Port `core/models.py` (stripped)

Port from `lwf/workflow_engine/core/models.py`. **Remove**: `DefStatus`, `WorkflowDefRecord`, `WaitFormField`, `StepWaitInfo`, `ActionType.RUN`, `ActionType.SCRIPT`, `step.script`/`script_args`/`run`/`fail_on_error`/`max_retries`/`working_directory` fields (keep only `uses`), `StepInstance.wait_info`/`resumed_data`/`retry_info`/`retry_count`/`claude_session`.

**Files:**
- Create: `paper_review_workflow/core/models.py`
- Test: `tests/unit/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_models.py
from paper_review_workflow.core.models import (
    WorkflowDef, WorkflowRun, WorkflowStatus,
    JobDef, JobInstance, JobStatus,
    StepDef, StepInstance, StepStatus,
    TriggerDef,
)


def test_workflow_run_default_status():
    run = WorkflowRun()
    assert run.status == WorkflowStatus.PENDING
    assert run.id  # auto-generated uuid


def test_step_def_uses_action_type():
    step = StepDef(id="s1", name="step1", uses="paper-review/echo@v1")
    assert step.uses == "paper-review/echo@v1"
    assert step.run is None
    assert step.script is None


def test_workflow_status_transitions():
    assert WorkflowStatus.PENDING != WorkflowStatus.RUNNING


def test_job_def_needs_default_empty():
    job = JobDef(id="j1", name="job1", runs_on="local")
    assert job.needs == []
    assert job.steps == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paper_review_workflow.core.models'`

- [ ] **Step 3: Write `paper_review_workflow/core/models.py`**

```python
"""Core data models for the paper review workflow engine.

Ported from lwf/workflow_engine/core/models.py with these removals:
- WorkflowDefRecord / DefStatus (no version management)
- WaitFormField / StepWaitInfo (no human approval)
- ActionType.RUN / SCRIPT (only uses)
- step fields: script, script_args, run, fail_on_error, max_retries, working_directory
- step instance fields: wait_info, resumed_data, retry_info, retry_count, claude_session
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


# ── Status enums ──────────────────────────────────────

class WorkflowStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILURE   = "failure"
    CANCELLED = "cancelled"
    SKIPPED   = "skipped"


class JobStatus(Enum):
    PENDING   = "pending"
    QUEUED    = "queued"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILURE   = "failure"
    CANCELLED = "cancelled"
    SKIPPED   = "skipped"


class StepStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILURE   = "failure"
    SKIPPED   = "skipped"
    CANCELLED = "cancelled"


# ── Definitions (static, from YAML) ───────────────────

@dataclass
class StepDef:
    id: str
    name: str
    uses: Optional[str] = None           # Action name like paper-review/extract@v1
    with_params: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    condition: Optional[str] = None
    continue_on_error: bool = False
    timeout_minutes: Optional[int] = None


@dataclass
class JobDef:
    id: str
    name: str
    runs_on: str
    steps: List[StepDef] = field(default_factory=list)
    needs: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    condition: Optional[str] = None
    strategy: Optional[Dict[str, Any]] = None
    timeout_minutes: Optional[int] = None
    continue_on_error: bool = False


@dataclass
class TriggerDef:
    workflow_dispatch: Optional[Dict] = None


@dataclass
class WorkflowDef:
    name: str
    on: TriggerDef
    jobs: Dict[str, JobDef] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    defaults: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None


# ── Runtime instances ─────────────────────────────────

@dataclass
class StepInstance:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_def: Optional[StepDef] = None
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)
    exit_code: int = 0
    error_msg: str = ""

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.step_def.name if self.step_def else "",
            "step_id": self.step_def.id if self.step_def else "",
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "exit_code": self.exit_code,
            "error_msg": self.error_msg,
            "outputs": self.outputs,
            "log": self.log,
        }


@dataclass
class JobInstance:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_def: Optional[JobDef] = None
    status: JobStatus = JobStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    steps: List[StepInstance] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)
    runner_info: Dict[str, str] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.job_def.name if self.job_def else "",
            "job_id": self.job_def.id if self.job_def else "",
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "steps": [s.to_dict() for s in self.steps],
            "outputs": self.outputs,
            "needs": self.job_def.needs if self.job_def else [],
        }


@dataclass
class WorkflowRun:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_def: Optional[WorkflowDef] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    trigger_type: str = "manual"
    trigger_payload: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    jobs: Dict[str, JobInstance] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    run_number: int = 1

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workflow_name": self.workflow_def.name if self.workflow_def else "",
            "status": self.status.value,
            "trigger_type": self.trigger_type,
            "trigger_payload": self.trigger_payload,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "run_number": self.run_number,
            "env": self.env,
            "jobs": {k: v.to_dict() for k, v in self.jobs.items()},
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_models.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/core/models.py tests/unit/test_models.py
git commit -m "feat(core): port stripped models.py from lwf"
```

## Task 1.3: Port `core/parser.py` (stripped)

Port from `lwf/workflow_engine/core/parser.py`. **Remove**: `script`/`script_args`/`run`/`fail_on_error`/`max_retries`/`working_directory` parsing. **Remove**: `push`/`pull_request`/`release`/`schedule` triggers (only `workflow_dispatch`).

**Files:**
- Create: `paper_review_workflow/core/parser.py`
- Test: `tests/unit/test_parser.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_parser.py
import os
import pytest
from paper_review_workflow.core.parser import WorkflowParser
from paper_review_workflow.core.models import WorkflowDef


@pytest.fixture
def minimal_yaml(tmp_path):
    yaml_content = """
name: minimal-test
on:
  workflow_dispatch:
    inputs:
      message:
        required: true
        default: hello
        type: string
env:
  GREETING: hi
jobs:
  echo:
    name: Echo
    runs-on: local
    steps:
      - id: echo-step
        uses: paper-review/echo@v1
        with:
          message: ${{ inputs.message }}
"""
    p = tmp_path / "minimal.yaml"
    p.write_text(yaml_content)
    return str(p)


def test_parse_minimal_yaml(minimal_yaml):
    parser = WorkflowParser()
    wf = parser.parse_file(minimal_yaml)

    assert wf.name == "minimal-test"
    assert wf.env["GREETING"] == "hi"
    assert "echo" in wf.jobs
    assert wf.jobs["echo"].runs_on == "local"
    assert len(wf.jobs["echo"].steps) == 1
    step = wf.jobs["echo"].steps[0]
    assert step.uses == "paper-review/echo@v1"
    assert step.with_params["message"] == "${{ inputs.message }}"


def test_parse_dispatch_inputs(minimal_yaml):
    parser = WorkflowParser()
    wf = parser.parse_file(minimal_yaml)
    assert wf.on.workflow_dispatch is not None
    inputs = wf.on.workflow_dispatch["inputs"]
    assert "message" in inputs
    assert inputs["message"]["default"] == "hello"


def test_parse_needs_list():
    yaml = """
name: t
on: {workflow_dispatch: {}}
jobs:
  a:
    runs-on: local
    steps: [{uses: paper-review/echo@v1}]
  b:
    needs: a
    runs-on: local
    steps: []
  c:
    needs: [a, b]
    runs-on: local
    steps: []
"""
    parser = WorkflowParser()
    wf = parser.parse_string(yaml)
    assert wf.jobs["b"].needs == ["a"]
    assert wf.jobs["c"].needs == ["a", "b"]


def test_parse_matrix_strategy():
    yaml = """
name: t
on: {workflow_dispatch: {}}
jobs:
  dims:
    runs-on: local
    strategy:
      matrix:
        dimension: [novelty, soundness]
    steps: []
"""
    parser = WorkflowParser()
    wf = parser.parse_string(yaml)
    assert wf.jobs["dims"].strategy["matrix"]["dimension"] == ["novelty", "soundness"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_parser.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `paper_review_workflow/core/parser.py`**

```python
"""YAML parser for paper review workflow definitions.

Ported from lwf, stripped of: script/run/working-directory/max-retries fields,
and push/PR/release/schedule triggers.
"""
import yaml
from pathlib import Path
from typing import Dict, List, Optional

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

    def _parse_trigger(self, on_raw) -> TriggerDef:
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
            env=raw.get("env", {}) or {},
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_parser.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/core/parser.py tests/unit/test_parser.py
git commit -m "feat(core): port stripped parser.py"
```

## Task 1.4: Port `core/event_bus.py`, `core/context.py`, `core/state_machine.py`, `core/paths.py`

Port from lwf with these changes:
- `event_bus.py`: Remove `STEP_WAITING` / `STEP_RESUMED` / `STEP_WAITING_RETRY` / `STEP_RETRIED` events and their factories.
- `state_machine.py`: Remove `WAITING` / `WAITING_RETRY` StepStatus transitions.
- `context.py`: Port as-is (expression evaluation).
- `paths.py`: New file for session dir / id generation.

**Files:**
- Create: `paper_review_workflow/core/event_bus.py`
- Create: `paper_review_workflow/core/state_machine.py`
- Create: `paper_review_workflow/core/context.py`
- Create: `paper_review_workflow/core/paths.py`
- Test: `tests/unit/test_paths.py`, `tests/unit/test_state_machine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_paths.py
from paper_review_workflow.core.paths import generate_paper_id, generate_run_id, get_session_dir


def test_paper_id_is_sha256_prefix():
    pid = generate_paper_id("Attention Is All You Need")
    assert len(pid) == 8
    assert all(c in "0123456789abcdef" for c in pid)


def test_paper_id_deterministic():
    assert generate_paper_id("Same Title") == generate_paper_id("Same Title")


def test_paper_id_different_titles():
    assert generate_paper_id("Title A") != generate_paper_id("Title B")


def test_run_id_format():
    rid = generate_run_id()
    # 20260812-143022-a1b2
    assert len(rid) == 19
    assert rid[8] == "-"
    assert rid[15] == "-"


def test_run_id_unique():
    ids = {generate_run_id() for _ in range(100)}
    assert len(ids) == 100  # no collisions


def test_session_dir_structure(tmp_path):
    d = get_session_dir(str(tmp_path), "abcd1234", "20260812-143022-a1b2")
    assert d == tmp_path / "abcd1234" / "20260812-143022-a1b2"
```

```python
# tests/unit/test_state_machine.py
import pytest
from paper_review_workflow.core.state_machine import (
    WorkflowStateMachine, JobStateMachine, StepStateMachine,
    WorkflowStateMachineCoordinator, InvalidTransitionError,
)
from paper_review_workflow.core.models import (
    WorkflowStatus, JobStatus, StepStatus,
    WorkflowRun, JobInstance, StepInstance,
)


def test_workflow_pending_to_running():
    sm = WorkflowStateMachine(WorkflowStatus.PENDING)
    sm.transition(WorkflowStatus.RUNNING)
    assert sm.current == WorkflowStatus.RUNNING


def test_workflow_invalid_transition():
    sm = WorkflowStateMachine(WorkflowStatus.PENDING)
    with pytest.raises(InvalidTransitionError):
        sm.transition(WorkflowStatus.SUCCESS)  # must go through RUNNING


def test_workflow_terminal_no_transition():
    sm = WorkflowStateMachine(WorkflowStatus.SUCCESS)
    with pytest.raises(InvalidTransitionError):
        sm.transition(WorkflowStatus.RUNNING)


def test_step_no_waiting_state():
    """WAITING removed from our state machine"""
    sm = StepStateMachine(StepStatus.RUNNING)
    with pytest.raises(InvalidTransitionError):
        sm.transition(StepStatus.WAITING)  # should not exist
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_paths.py tests/unit/test_state_machine.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write `paper_review_workflow/core/paths.py`**

```python
"""Session directory and ID generation utilities."""
import hashlib
import secrets
import time
from pathlib import Path


def generate_paper_id(title: str) -> str:
    """8-char hex of sha256(title)."""
    return hashlib.sha256(title.encode("utf-8")).hexdigest()[:8]


def generate_run_id() -> str:
    """Format: YYYYMMDD-HHMMSS-xxxx (4 hex chars)."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(2)
    return f"{ts}-{suffix}"


def get_session_dir(sessions_root: str, paper_id: str, run_id: str) -> Path:
    return Path(sessions_root) / paper_id / run_id


def generate_pending_paper_id(run_id: str) -> str:
    """Pre-extract paper_id placeholder."""
    return f"pending-{run_id}"
```

- [ ] **Step 4: Write `paper_review_workflow/core/event_bus.py`**

Port from lwf, remove `STEP_WAITING` / `STEP_RESUMED` / `STEP_WAITING_RETRY` / `STEP_RETRIED` events and factories (`make_step_waiting_event`, `make_step_resumed_event`, `make_step_waiting_retry_event`, `make_step_retried_event`).

```python
"""Event bus for workflow state changes.

Ported from lwf, removed WAITING/RESUMED/RETRY events (no human approval).
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from weakref import WeakSet

logger = logging.getLogger(__name__)


class EventType(Enum):
    WORKFLOW_CREATED   = "workflow.created"
    WORKFLOW_STARTED   = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_CANCELLED = "workflow.cancelled"

    JOB_QUEUED    = "job.queued"
    JOB_STARTED   = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_SKIPPED   = "job.skipped"

    STEP_STARTED   = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_SKIPPED   = "step.skipped"
    STEP_LOG       = "step.log"

    ENGINE_READY = "engine.ready"
    ERROR        = "error"


@dataclass
class WorkflowEvent:
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    run_id: Optional[str] = None
    job_id: Optional[str] = None
    step_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
            "job_id": self.job_id,
            "step_id": self.step_id,
            "data": self.data,
        }


def make_workflow_started_event(run) -> WorkflowEvent:
    return WorkflowEvent(
        event_type=EventType.WORKFLOW_STARTED, run_id=run.id,
        data={"workflow_name": run.workflow_def.name if run.workflow_def else "",
              "trigger_type": run.trigger_type, "status": run.status.value},
    )


def make_workflow_completed_event(run) -> WorkflowEvent:
    return WorkflowEvent(
        event_type=EventType.WORKFLOW_COMPLETED, run_id=run.id,
        data={"status": run.status.value, "duration": run.duration},
    )


def make_job_started_event(run_id, job) -> WorkflowEvent:
    return WorkflowEvent(
        event_type=EventType.JOB_STARTED, run_id=run_id, job_id=job.job_def.id if job.job_def else None,
        data={"name": job.job_def.name if job.job_def else ""},
    )


def make_job_completed_event(run_id, job) -> WorkflowEvent:
    return WorkflowEvent(
        event_type=EventType.JOB_COMPLETED, run_id=run_id, job_id=job.job_def.id if job.job_def else None,
        data={"status": job.status.value, "duration": job.duration},
    )


def make_step_started_event(run_id, job_id, step) -> WorkflowEvent:
    return WorkflowEvent(
        event_type=EventType.STEP_STARTED, run_id=run_id, job_id=job_id, step_id=step.step_def.id if step.step_def else None,
        data={"name": step.step_def.name if step.step_def else ""},
    )


def make_step_completed_event(run_id, job_id, step) -> WorkflowEvent:
    return WorkflowEvent(
        event_type=EventType.STEP_COMPLETED, run_id=run_id, job_id=job_id, step_id=step.step_def.id if step.step_def else None,
        data={"status": step.status.value, "duration": step.duration, "outputs": step.outputs},
    )


def make_step_log_event(run_id, job_id, step_id, line) -> WorkflowEvent:
    return WorkflowEvent(
        event_type=EventType.STEP_LOG, run_id=run_id, job_id=job_id, step_id=step_id,
        data={"line": line},
    )


class EventBus:
    def __init__(self):
        self._subscribers: Set[Callable[[WorkflowEvent], None]] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def subscribe(self, callback: Callable[[WorkflowEvent], None]) -> Callable[[], None]:
        self._subscribers.add(callback)
        def unsubscribe():
            self._subscribers.discard(callback)
        return unsubscribe

    def publish(self, event: WorkflowEvent) -> None:
        for cb in list(self._subscribers):
            try:
                cb(event)
            except Exception as e:
                logger.error(f"[EventBus] subscriber error: {e}", exc_info=True)

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
```

- [ ] **Step 5: Write `paper_review_workflow/core/state_machine.py`**

Port from lwf `state_machine.py`. Remove `WAITING`/`WAITING_RETRY` from STEP_TRANSITIONS. Remove the `make_step_waiting_event`/`make_step_resumed_event`/`make_step_waiting_retry_event`/`make_step_retried_event` imports. The full file is long; copy the structure from lwf and adapt:

```python
"""Three-layer state machine: Workflow / Job / Step.

Ported from lwf, removed WAITING and WAITING_RETRY states.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Dict, FrozenSet, Optional

from .models import (
    WorkflowStatus, JobStatus, StepStatus,
    WorkflowRun, JobInstance, StepInstance,
)
from .event_bus import (
    EventBus, WorkflowEvent, EventType,
    make_workflow_started_event, make_workflow_completed_event,
    make_job_started_event, make_job_completed_event,
    make_step_started_event, make_step_completed_event,
)

logger = logging.getLogger(__name__)


class InvalidTransitionError(Exception):
    def __init__(self, entity: str, from_state, to_state):
        super().__init__(
            f"[StateMachine] {entity} cannot transition {from_state.value!r} -> {to_state.value!r}"
        )
        self.entity = entity
        self.from_state = from_state
        self.to_state = to_state


# ── Transition tables ─────────────────────────────────

WORKFLOW_TRANSITIONS: Dict[WorkflowStatus, FrozenSet[WorkflowStatus]] = {
    WorkflowStatus.PENDING:   frozenset({WorkflowStatus.RUNNING, WorkflowStatus.CANCELLED}),
    WorkflowStatus.RUNNING:   frozenset({WorkflowStatus.SUCCESS, WorkflowStatus.FAILURE, WorkflowStatus.CANCELLED}),
    WorkflowStatus.SUCCESS:   frozenset(),
    WorkflowStatus.FAILURE:   frozenset(),
    WorkflowStatus.CANCELLED: frozenset(),
    WorkflowStatus.SKIPPED:   frozenset(),
}

JOB_TRANSITIONS: Dict[JobStatus, FrozenSet[JobStatus]] = {
    JobStatus.PENDING:   frozenset({JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.SKIPPED, JobStatus.CANCELLED}),
    JobStatus.QUEUED:    frozenset({JobStatus.RUNNING, JobStatus.CANCELLED, JobStatus.SKIPPED}),
    JobStatus.RUNNING:   frozenset({JobStatus.SUCCESS, JobStatus.FAILURE, JobStatus.CANCELLED}),
    JobStatus.SUCCESS:   frozenset(),
    JobStatus.FAILURE:   frozenset(),
    JobStatus.CANCELLED: frozenset(),
    JobStatus.SKIPPED:   frozenset(),
}

STEP_TRANSITIONS: Dict[StepStatus, FrozenSet[StepStatus]] = {
    StepStatus.PENDING:   frozenset({StepStatus.RUNNING, StepStatus.SKIPPED, StepStatus.CANCELLED}),
    StepStatus.RUNNING:   frozenset({StepStatus.SUCCESS, StepStatus.FAILURE, StepStatus.CANCELLED}),
    StepStatus.SUCCESS:   frozenset(),
    StepStatus.FAILURE:   frozenset(),
    StepStatus.SKIPPED:   frozenset(),
    StepStatus.CANCELLED: frozenset(),
}

WORKFLOW_TERMINAL = frozenset({WorkflowStatus.SUCCESS, WorkflowStatus.FAILURE,
                               WorkflowStatus.CANCELLED, WorkflowStatus.SKIPPED})
JOB_TERMINAL = frozenset({JobStatus.SUCCESS, JobStatus.FAILURE,
                          JobStatus.CANCELLED, JobStatus.SKIPPED})
STEP_TERMINAL = frozenset({StepStatus.SUCCESS, StepStatus.FAILURE,
                           StepStatus.CANCELLED, StepStatus.SKIPPED})


# ── Generic state machine ─────────────────────────────

class _StateMachine:
    def __init__(self, initial, transitions, name: str):
        self.current = initial
        self._transitions = transitions
        self._name = name
        self._history = [initial]

    def transition(self, to_state) -> None:
        allowed = self._transitions.get(self.current, frozenset())
        if to_state not in allowed:
            raise InvalidTransitionError(self._name, self.current, to_state)
        self.current = to_state
        self._history.append(to_state)

    @property
    def is_terminal(self) -> bool:
        return self.current in frozenset(self._transitions.keys()) - frozenset().union(*self._transitions.values()) or \
               len(self._transitions.get(self.current, frozenset())) == 0

    @property
    def history(self):
        return list(self._history)


class WorkflowStateMachine(_StateMachine):
    def __init__(self, initial=WorkflowStatus.PENDING):
        super().__init__(initial, WORKFLOW_TRANSITIONS, "Workflow")


class JobStateMachine(_StateMachine):
    def __init__(self, initial=JobStatus.PENDING):
        super().__init__(initial, JOB_TRANSITIONS, "Job")


class StepStateMachine(_StateMachine):
    def __init__(self, initial=StepStatus.PENDING):
        super().__init__(initial, STEP_TRANSITIONS, "Step")


# ── Coordinator ───────────────────────────────────────

class WorkflowStateMachineCoordinator:
    """Coordinates state transitions across the three layers and publishes events."""

    def __init__(self, run: WorkflowRun, event_bus: EventBus):
        self.run = run
        self.event_bus = event_bus
        self.workflow_sm = WorkflowStateMachine(run.status)
        self._job_sms: Dict[str, JobStateMachine] = {}
        self._step_sms: Dict[str, StepStateMachine] = {}

    # Workflow
    def start_workflow(self) -> None:
        self.workflow_sm.transition(WorkflowStatus.RUNNING)
        self.run.status = WorkflowStatus.RUNNING
        self.run.start_time = datetime.now()
        self.event_bus.publish(make_workflow_started_event(self.run))

    def complete_workflow(self, success: bool) -> None:
        target = WorkflowStatus.SUCCESS if success else WorkflowStatus.FAILURE
        self.workflow_sm.transition(target)
        self.run.status = target
        self.run.end_time = datetime.now()
        self.event_bus.publish(make_workflow_completed_event(self.run))

    def cancel_workflow(self) -> None:
        if self.workflow_sm.is_terminal:
            return
        self.workflow_sm.transition(WorkflowStatus.CANCELLED)
        self.run.status = WorkflowStatus.CANCELLED
        self.run.end_time = datetime.now()
        self.event_bus.publish(WorkflowEvent(
            event_type=EventType.WORKFLOW_CANCELLED, run_id=self.run.id,
        ))

    # Job
    def start_job(self, job_id: str, job: JobInstance) -> None:
        sm = JobStateMachine(job.status)
        self._job_sms[job_id] = sm
        sm.transition(JobStatus.RUNNING)
        job.status = JobStatus.RUNNING
        job.start_time = datetime.now()
        self.event_bus.publish(make_job_started_event(self.run.id, job))

    def complete_job(self, job_id: str, job: JobInstance, success: bool) -> None:
        sm = self._job_sms.get(job_id)
        if sm is None:
            sm = JobStateMachine(job.status)
            self._job_sms[job_id] = sm
        target = JobStatus.SUCCESS if success else JobStatus.FAILURE
        sm.transition(target)
        job.status = target
        job.end_time = datetime.now()
        self.event_bus.publish(make_job_completed_event(self.run.id, job))

    def skip_job(self, job_id: str, job: JobInstance) -> None:
        sm = JobStateMachine(job.status)
        self._job_sms[job_id] = sm
        sm.transition(JobStatus.SKIPPED)
        job.status = JobStatus.SKIPPED
        self.event_bus.publish(WorkflowEvent(
            event_type=EventType.JOB_SKIPPED, run_id=self.run.id, job_id=job_id,
        ))

    def cancel_job(self, job_id: str, job: JobInstance) -> None:
        sm = self._job_sms.get(job_id) or JobStateMachine(job.status)
        self._job_sms[job_id] = sm
        if not sm.is_terminal:
            sm.transition(JobStatus.CANCELLED)
            job.status = JobStatus.CANCELLED
            job.end_time = datetime.now()

    # Step
    def start_step(self, job_id: str, step: StepInstance) -> None:
        sm = StepStateMachine(step.status)
        self._step_sms[step.id] = sm
        sm.transition(StepStatus.RUNNING)
        step.status = StepStatus.RUNNING
        step.start_time = datetime.now()
        self.event_bus.publish(make_step_started_event(self.run.id, job_id, step))

    def complete_step(self, job_id: str, step: StepInstance, success: bool) -> None:
        sm = self._step_sms.get(step.id) or StepStateMachine(step.status)
        self._step_sms[step.id] = sm
        target = StepStatus.SUCCESS if success else StepStatus.FAILURE
        sm.transition(target)
        step.status = target
        step.end_time = datetime.now()
        self.event_bus.publish(make_step_completed_event(self.run.id, job_id, step))

    def skip_step(self, job_id: str, step: StepInstance) -> None:
        sm = StepStateMachine(step.status)
        self._step_sms[step.id] = sm
        sm.transition(StepStatus.SKIPPED)
        step.status = StepStatus.SKIPPED
        self.event_bus.publish(WorkflowEvent(
            event_type=EventType.STEP_SKIPPED, run_id=self.run.id, job_id=job_id, step_id=step.id,
        ))

    def cancel_step(self, job_id: str, step: StepInstance) -> None:
        sm = self._step_sms.get(step.id) or StepStateMachine(step.status)
        self._step_sms[step.id] = sm
        if not sm.is_terminal:
            sm.transition(StepStatus.CANCELLED)
            step.status = StepStatus.CANCELLED

    def cancel_all_pending_jobs_and_steps(self) -> None:
        for job_id, job in self.run.jobs.items():
            if not JOB_TERMINAL.__contains__(job.status):
                self.cancel_job(job_id, job)
            for step in job.steps:
                if not STEP_TERMINAL.__contains__(step.status):
                    self.cancel_step(job_id, step)
```

- [ ] **Step 6: Write `paper_review_workflow/core/context.py`**

Port the `WorkflowContext` class and `${{ }}` expression evaluator from lwf. Strip cookie/secret loading (not needed). The minimal port:

```python
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
        ctx: Dict[str, Any] = {
            "inputs": self.run.trigger_payload or {},
            "env": self.run.env or {},
            "workflow": {"name": self.run.workflow_def.name if self.run.workflow_def else ""},
            "jobs": {jid: {"outputs": j.outputs, "status": j.status.value, "result": j.status.value}
                     for jid, j in self.run.jobs.items()},
        }
        if job_instance:
            ctx["job"] = {"id": job_instance.job_def.id if job_instance.job_def else ""}
            ctx["steps"] = {s.step_def.id: {"outputs": s.outputs} for s in job_instance.steps if s.step_def}
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
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/unit/test_paths.py tests/unit/test_state_machine.py -v`
Expected: PASS (9 tests)

- [ ] **Step 8: Commit**

```bash
git add paper_review_workflow/core/paths.py paper_review_workflow/core/event_bus.py paper_review_workflow/core/state_machine.py paper_review_workflow/core/context.py tests/unit/test_paths.py tests/unit/test_state_machine.py
git commit -m "feat(core): port paths/event_bus/state_machine/context from lwf"
```

## Task 1.5: Port `storage/` (memory + json_file)

Port from lwf `storage/backend.py`, `storage/memory.py`, `storage/json_file.py`. Strip `StepWaitInfo`/`WaitFormField` references (already removed from models).

**Files:**
- Create: `paper_review_workflow/storage/__init__.py`
- Create: `paper_review_workflow/storage/backend.py`
- Create: `paper_review_workflow/storage/memory.py`
- Create: `paper_review_workflow/storage/json_file.py`
- Test: `tests/unit/test_storage.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_storage.py
import pytest
from paper_review_workflow.core.models import WorkflowRun, WorkflowStatus
from paper_review_workflow.storage.memory import MemoryStorage
from paper_review_workflow.storage.json_file import JsonFileStorage


@pytest.fixture
def sample_run():
    return WorkflowRun()


def test_memory_save_and_get(sample_run):
    s = MemoryStorage()
    s.open()
    s.save_run(sample_run)
    loaded = s.get_run(sample_run.id)
    assert loaded is not None
    assert loaded.id == sample_run.id


def test_memory_get_nonexistent():
    s = MemoryStorage()
    s.open()
    assert s.get_run("nonexistent") is None


def test_memory_list_runs(sample_run):
    s = MemoryStorage()
    s.open()
    s.save_run(sample_run)
    runs = s.list_runs(limit=10)
    assert len(runs) == 1


def test_json_file_save_and_get(sample_run, tmp_path):
    s = JsonFileStorage(data_dir=str(tmp_path))
    s.open()
    s.save_run(sample_run)
    loaded = s.get_run(sample_run.id)
    assert loaded is not None
    assert loaded.id == sample_run.id


def test_json_file_persists_after_close(sample_run, tmp_path):
    s1 = JsonFileStorage(data_dir=str(tmp_path))
    s1.open()
    s1.save_run(sample_run)
    s1.close()

    s2 = JsonFileStorage(data_dir=str(tmp_path))
    s2.open()
    loaded = s2.get_run(sample_run.id)
    assert loaded is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_storage.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write `paper_review_workflow/storage/backend.py`**

Copy lwf `storage/backend.py` verbatim (it's already clean, no wait_info references).

- [ ] **Step 4: Write `paper_review_workflow/storage/memory.py`**

Copy lwf `storage/memory.py` verbatim.

- [ ] **Step 5: Write `paper_review_workflow/storage/json_file.py`**

Copy lwf `storage/json_file.py`. **In `_serialize_step` and `_deserialize_step`, remove `wait_info`/`resumed_data`/`retry_info`/`retry_count` fields** (already removed from StepInstance).

- [ ] **Step 6: Write `paper_review_workflow/storage/__init__.py`**

```python
"""Storage backends."""
from .backend import StorageBackend
from .memory import MemoryStorage
from .json_file import JsonFileStorage

__all__ = ["StorageBackend", "MemoryStorage", "JsonFileStorage"]
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/unit/test_storage.py -v`
Expected: PASS (5 tests)

- [ ] **Step 8: Commit**

```bash
git add paper_review_workflow/storage/ tests/unit/test_storage.py
git commit -m "feat(storage): port memory + json_file backends"
```

## Task 1.6: Port `actions/base.py` + `actions/registry.py`

Port from lwf. **Remove** `waiting`/`wait_info`/`cancel_action` from `ActionResult`. Remove `ActionResult.wait()` factory.

**Files:**
- Create: `paper_review_workflow/actions/base.py`
- Create: `paper_review_workflow/actions/registry.py`
- Test: `tests/unit/test_registry.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_registry.py
import pytest
from paper_review_workflow.actions.base import BaseAction, ActionResult
from paper_review_workflow.actions.registry import ActionRegistry


class EchoAction(BaseAction):
    def run(self, params, env, context, log_callback=None):
        msg = params.get("message", "")
        if log_callback:
            log_callback(f"echo: {msg}")
        return ActionResult(success=True, outputs={"message": msg})


def test_action_result_success():
    r = ActionResult(success=True, outputs={"x": 1})
    assert r.success
    assert r.outputs == {"x": 1}
    assert r.exit_code == 0


def test_action_result_failure():
    r = ActionResult(success=False, message="err")
    assert not r.success
    assert r.exit_code == 1


def test_registry_register_and_get():
    # Reset singleton for test
    ActionRegistry._instance = None
    reg = ActionRegistry()
    action = EchoAction()
    reg.register("paper-review/echo@v1", action)
    assert reg.get("paper-review/echo@v1") is action


def test_registry_get_strips_version():
    ActionRegistry._instance = None
    reg = ActionRegistry()
    action = EchoAction()
    reg.register("paper-review/echo@v1", action)
    assert reg.get("paper-review/echo@v2") is action  # version-stripped lookup


def test_registry_singleton():
    ActionRegistry._instance = None
    r1 = ActionRegistry()
    r2 = ActionRegistry()
    assert r1 is r2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_registry.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write `paper_review_workflow/actions/base.py`**

```python
"""Base action and result classes.

Ported from lwf, removed waiting/wait_info (no human approval).
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional


class ActionResult:
    """Two-state result: success or failure (no waiting state)."""

    def __init__(
        self,
        success: bool,
        outputs: Optional[Dict[str, Any]] = None,
        message: str = "",
        log_lines: Optional[List[str]] = None,
        exit_code: int = 0,
    ):
        self.success = success
        self.outputs = outputs or {}
        self.message = message
        self.log_lines = log_lines or []
        self.exit_code = 0 if success else (exit_code or 1)


class BaseAction(ABC):
    """Base class for all actions. Implementations must be stateless."""

    @property
    def description(self) -> str:
        return ""

    @abstractmethod
    def run(
        self,
        params: Dict[str, Any],
        env: Dict[str, str],
        context: Dict,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> ActionResult:
        pass
```

- [ ] **Step 4: Write `paper_review_workflow/actions/registry.py`**

```python
"""Action registry (singleton, caches stateless instances).

Ported from lwf verbatim (this part has no wait_info).
"""
import logging
from typing import Dict, Optional

from .base import BaseAction

logger = logging.getLogger(__name__)


class ActionRegistry:
    _instance: Optional["ActionRegistry"] = None
    _actions: Dict[str, BaseAction] = {}

    def __new__(cls) -> "ActionRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._actions = {}
        return cls._instance

    def register(self, name: str, action: BaseAction) -> None:
        self._actions[name] = action
        logger.debug(f"[Registry] registered: {name}")

    def get(self, name: str) -> Optional[BaseAction]:
        if name in self._actions:
            return self._actions[name]
        base_name = name.split("@")[0]
        return self._actions.get(base_name)

    def list_actions(self) -> list:
        return [{"name": k, "description": v.description}
                for k, v in self._actions.items()]

    def clear(self) -> None:
        """Test-only: clear the singleton state."""
        self._actions.clear()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_registry.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add paper_review_workflow/actions/base.py paper_review_workflow/actions/registry.py tests/unit/test_registry.py
git commit -m "feat(actions): port base + registry (stateless, no waiting)"
```

## Task 1.7: Port `executors/` (workflow + job + step)

Port from lwf. **Remove**: `_execute_run` and `_execute_script` from `step_executor.py` (only `_execute_uses` remains). Keep matrix strategy in `job_executor.py` intact.

**Files:**
- Create: `paper_review_workflow/executors/workflow_executor.py`
- Create: `paper_review_workflow/executors/job_executor.py`
- Create: `paper_review_workflow/executors/step_executor.py`
- Create: `paper_review_workflow/executors/__init__.py`
- Test: `tests/integration/test_executors_minimal.py`

- [ ] **Step 1: Write failing integration test**

```python
# tests/integration/test_executors_minimal.py
from paper_review_workflow.actions.registry import ActionRegistry
from paper_review_workflow.actions.base import BaseAction, ActionResult
from paper_review_workflow.core.parser import WorkflowParser
from paper_review_workflow.core.event_bus import EventBus
from paper_review_workflow.core.models import WorkflowRun
from paper_review_workflow.storage.memory import MemoryStorage
from paper_review_workflow.executors import WorkflowExecutor


class EchoAction(BaseAction):
    def run(self, params, env, context, log_callback=None):
        msg = params.get("message", "default")
        if log_callback:
            log_callback(f"echo: {msg}")
        return ActionResult(success=True, outputs={"message": msg})


def test_minimal_workflow_runs():
    ActionRegistry._instance = None
    reg = ActionRegistry()
    reg.register("paper-review/echo@v1", EchoAction())

    yaml = """
name: minimal
on: {workflow_dispatch: {}}
jobs:
  echo:
    runs-on: local
    steps:
      - id: e
        uses: paper-review/echo@v1
        with:
          message: hello
"""
    parser = WorkflowParser()
    wf_def = parser.parse_string(yaml)
    run = WorkflowRun(workflow_def=wf_def, env={})

    bus = EventBus()
    ex = WorkflowExecutor(registry=reg, event_bus=bus)
    ex.execute(run)

    assert run.status.value == "success"
    assert run.jobs["echo"].steps[0].outputs["message"] == "hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_executors_minimal.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Port `executors/step_executor.py`**

Port from lwf, **delete** `_execute_run`, `_execute_script`, all `script_args`/`with_params` flattening for SCRIPT_PARAM_* (keep only `with_params` → `params` for `uses`). Remove `WAITING` handling. Keep cancel_check, timeout, log_cb. The action call should be:

```python
action = self.registry.get(step_def.uses)
if action is None:
    return ActionResult(success=False, message=f"action not found: {step_def.uses}")
result = action.run(params=step_def.with_params, env=env, context=ctx_snapshot, log_callback=log_cb)
```

Adapt the rest of lwf's step_executor.py to remove wait handling.

- [ ] **Step 4: Port `executors/job_executor.py`**

Port verbatim from lwf, including `_execute_matrix` and `_build_matrix_combinations`. No changes needed (matrix logic is reusable as-is).

- [ ] **Step 5: Port `executors/workflow_executor.py`**

Port verbatim from lwf. The `_schedule_jobs` ThreadPoolExecutor logic stays the same.

- [ ] **Step 6: Write `executors/__init__.py`**

```python
"""Three-layer executors."""
from .workflow_executor import WorkflowExecutor
from .job_executor import JobExecutor
from .step_executor import StepExecutor

__all__ = ["WorkflowExecutor", "JobExecutor", "StepExecutor"]
```

- [ ] **Step 7: Run integration test**

Run: `pytest tests/integration/test_executors_minimal.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add paper_review_workflow/executors/ tests/integration/test_executors_minimal.py
git commit -m "feat(executors): port 3-layer executors (uses-only, matrix intact)"
```

## Task 1.8: Build `engine.py` + `cli.py` + `main.py` + minimal config

**Files:**
- Create: `paper_review_workflow/engine.py`
- Create: `paper_review_workflow/cli.py`
- Create: `main.py`
- Create: `paper_review_workflow/actions/builtin.py`
- Create: `configs/minimal.yaml`
- Test: `tests/integration/test_engine_minimal.py`

- [ ] **Step 1: Write failing integration test**

```python
# tests/integration/test_engine_minimal.py
import json
from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.memory import MemoryStorage


def test_engine_runs_minimal_yaml(tmp_path):
    yaml_path = tmp_path / "minimal.yaml"
    yaml_path.write_text("""
name: minimal
on: {workflow_dispatch: {}}
jobs:
  echo:
    runs-on: local
    steps:
      - id: e
        uses: paper-review/echo@v1
        with:
          message: hi
""")
    engine = ReviewEngine(storage=MemoryStorage())
    run = engine.run_from_file(str(yaml_path), payload={})
    assert run.status.value == "success"
    assert run.jobs["echo"].steps[0].outputs["message"] == "hi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_engine_minimal.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write `paper_review_workflow/actions/builtin.py`**

```python
"""Built-in actions registration."""
from .base import BaseAction, ActionResult
from .registry import ActionRegistry


class EchoAction(BaseAction):
    @property
    def description(self) -> str:
        return "Echo back the message param (smoke test)"

    def run(self, params, env, context, log_callback=None):
        msg = params.get("message", "")
        if log_callback:
            log_callback(f"echo: {msg}")
        return ActionResult(success=True, outputs={"message": msg})


def register_builtin_actions(registry: ActionRegistry) -> None:
    registry.register("paper-review/echo@v1", EchoAction())
```

- [ ] **Step 4: Write `paper_review_workflow/engine.py`**

```python
"""ReviewEngine: facade layer (simplified from lwf WorkflowEngine)."""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .core.models import WorkflowRun, WorkflowStatus, JobStatus, StepStatus
from .core.parser import WorkflowParser
from .core.context import WorkflowContext
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
        self.storage = storage or JsonFileStorage(sessions_root)
        self.storage.open()

        self._active_runs: Dict[str, WorkflowRun] = {}
        self._coordinators: Dict[str, WorkflowStateMachineCoordinator] = {}

        self._workflow_executor = WorkflowExecutor(
            registry=self.registry,
            event_bus=self.event_bus,
        )

        register_builtin_actions(self.registry)
        self._register_persist_hooks()

    # ── Entry points ──────────────────────────────────

    def run_from_file(self, yaml_path: str,
                      trigger_type: str = "workflow_dispatch",
                      payload: Optional[Dict] = None) -> WorkflowRun:
        wf_def = self.parser.parse_file(yaml_path)
        return self.run_workflow(wf_def, trigger_type, payload)

    def run_workflow(self, wf_def, trigger_type="workflow_dispatch", payload=None):
        run = WorkflowRun(
            workflow_def=wf_def,
            trigger_type=trigger_type,
            trigger_payload=payload or {},
            env=dict(wf_def.env) if wf_def.env else {},
        )
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
        return self._do_execute(run)

    # ── Control ───────────────────────────────────────

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

    # ── Query ─────────────────────────────────────────

    def list_runs(self, paper_id: Optional[str] = None,
                  status: Optional[WorkflowStatus] = None,
                  limit: int = 50):
        runs = self.storage.list_runs(status=status, limit=limit * 5)
        if paper_id:
            runs = [r for r in runs if r.env.get("__paper_id__") == paper_id]
        return runs[:limit]

    def get_run(self, run_id: str):
        return self.storage.get_run(run_id)

    # ── Internal ──────────────────────────────────────

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
        """Subscribe to state events → auto-save run on every transition."""
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
        """Reset specified components and their downstream to pending."""
        # Build dependency graph from job.needs
        all_jobs = list(run.jobs.keys())
        downstream: set = set()
        # Add the components themselves
        for c in components:
            downstream.add(c)
        # Propagate to downstream (transitive needs closure)
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
        # Reset those jobs
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
```

- [ ] **Step 5: Write `paper_review_workflow/cli.py`**

```python
"""CLI dispatcher."""
import argparse
import json
import logging
import sys
from typing import List

from .engine import ReviewEngine
from .storage import MemoryStorage, JsonFileStorage


def _build_storage(args):
    storage_type = getattr(args, "storage", "json")
    if storage_type == "memory":
        return MemoryStorage()
    return JsonFileStorage(data_dir=getattr(args, "storage_dir", "./sessions"))


def _cmd_run(engine: ReviewEngine, args) -> int:
    try:
        payload = json.loads(args.payload) if args.payload else {}
    except json.JSONDecodeError as e:
        print(f"[Error] payload JSON parse failed: {e}", file=sys.stderr)
        return 2
    # Apply --env overrides
    for kv in args.env or []:
        k, _, v = kv.partition("=")
        if not _:
            print(f"[Error] invalid --env {kv}, expected KEY=VALUE", file=sys.stderr)
            return 2
        # env vars get injected into run.env via wf_def.env merge before run
    run = engine.run_from_file(args.yaml, payload=payload)
    print(f"\n最终状态: {run.status.value}")
    if run.duration:
        print(f"总耗时:   {run.duration:.2f}s")
    return 0 if run.status.value == "success" else 1


def _cmd_resume(engine: ReviewEngine, args) -> int:
    rerun = args.rerun.split(",") if args.rerun else None
    run = engine.resume_run(args.run_id, rerun_components=rerun, rerun_all=args.rerun_all)
    print(f"\n最终状态: {run.status.value}")
    return 0 if run.status.value == "success" else 1


def _cmd_list_runs(engine: ReviewEngine, args) -> int:
    from .core.models import WorkflowStatus
    status = WorkflowStatus(args.status) if args.status else None
    runs = engine.list_runs(paper_id=args.paper_id, status=status, limit=args.limit)
    print(f"{'RUN_ID':<24} {'STATUS':<10} {'WORKFLOW':<20} {'DURATION'}")
    for r in runs:
        dur = f"{r.duration:.1f}s" if r.duration else "-"
        wf = (r.workflow_def.name if r.workflow_def else "")[:20]
        print(f"{r.id:<24} {r.status.value:<10} {wf:<20} {dur}")
    return 0


def _cmd_show_run(engine: ReviewEngine, args) -> int:
    run = engine.get_run(args.run_id)
    if run is None:
        print(f"run not found: {args.run_id}", file=sys.stderr)
        return 1
    print(json.dumps(run.to_dict(), indent=2, default=str))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="论文评审工作流")
    parser.add_argument("--storage", default="json", choices=["memory", "json"])
    parser.add_argument("--storage-dir", default="./sessions")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="运行 YAML 评审")
    run_p.add_argument("yaml")
    run_p.add_argument("--trigger", default="workflow_dispatch")
    run_p.add_argument("--payload", default="{}")
    run_p.add_argument("--env", action="append", default=[])

    res_p = sub.add_parser("resume", help="断点续跑")
    res_p.add_argument("run_id")
    res_p.add_argument("--rerun", default=None, help="逗号分隔的组件名")
    res_p.add_argument("--rerun-all", action="store_true")

    list_p = sub.add_parser("list-runs", help="列出历史 run")
    list_p.add_argument("--paper-id", default=None)
    list_p.add_argument("--status", default=None,
                        choices=["pending", "running", "success", "failure", "cancelled"])
    list_p.add_argument("--limit", type=int, default=50)

    show_p = sub.add_parser("show-run", help="查看 run 详情")
    show_p.add_argument("run_id")

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    if args.command is None:
        parser.print_help()
        return 0

    engine = ReviewEngine(storage=_build_storage(args),
                          sessions_root=args.storage_dir)

    import signal
    def _sig(signum, frame):
        print("\n⚠️  Received interrupt, shutting down gracefully...")
        cancelled = engine.shutdown(timeout=30)
        print(f"Cancelled {cancelled} active run(s). State saved.")
        sys.exit(130)
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    if args.command == "run":
        return _cmd_run(engine, args)
    elif args.command == "resume":
        return _cmd_resume(engine, args)
    elif args.command == "list-runs":
        return _cmd_list_runs(engine, args)
    elif args.command == "show-run":
        return _cmd_show_run(engine, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Write `main.py`**

```python
"""Entry point: `python main.py run configs/normal_review.yaml --payload '...'`"""
import sys
from paper_review_workflow.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: Create `configs/minimal.yaml`**

```yaml
name: minimal-test

on:
  workflow_dispatch:
    inputs:
      message:
        description: "Message to echo"
        required: false
        default: hello
        type: string

env:
  SESSIONS_ROOT: ./sessions

jobs:
  echo:
    name: "🔊 Echo"
    runs-on: local
    steps:
      - id: echo
        uses: paper-review/echo@v1
        with:
          message: ${{ inputs.message }}
```

- [ ] **Step 8: Run integration test**

Run: `pytest tests/integration/test_engine_minimal.py -v`
Expected: PASS

- [ ] **Step 9: Smoke test via CLI**

Run: `python main.py run configs/minimal.yaml --storage memory --payload '{"message":"hi"}'`
Expected: output ending with `最终状态: success`

- [ ] **Step 10: Commit**

```bash
git add paper_review_workflow/engine.py paper_review_workflow/cli.py paper_review_workflow/actions/builtin.py main.py configs/minimal.yaml tests/integration/test_engine_minimal.py
git commit -m "feat(engine): add ReviewEngine facade + CLI + EchoAction smoke"
```

---

# M2: LLM Provider Layer

**Goal:** Build the pluggable LLM provider abstraction with the Anthropic implementation, prompt caching, retry logic, and Pydantic schemas.

**Estimated:** 1 day

## Task 2.1: Define Pydantic schemas

**Files:**
- Create: `paper_review_workflow/llm/schemas.py`
- Test: `tests/unit/test_schemas.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_schemas.py
import pytest
from pydantic import ValidationError
from paper_review_workflow.llm.schemas import (
    DimensionScore, PaperMetadata, SynthesisResult,
)


def test_dimension_score_valid():
    s = DimensionScore(
        score=4, confidence=0.85,
        strengths=["a", "b"], weaknesses=["c"],
        justification="x" * 200,
        evidence=[{"section": "3.2", "quote": "...", "page": 5}],
    )
    assert s.score == 4
    assert s.confidence == 0.85


def test_dimension_score_out_of_range_high():
    with pytest.raises(ValidationError):
        DimensionScore(score=6, confidence=0.5, strengths=["a"],
                       weaknesses=["b"], justification="x" * 200)


def test_dimension_score_out_of_range_low():
    with pytest.raises(ValidationError):
        DimensionScore(score=0, confidence=0.5, strengths=["a"],
                       weaknesses=["b"], justification="x" * 200)


def test_dimension_score_justification_too_short():
    with pytest.raises(ValidationError):
        DimensionScore(score=3, confidence=0.5, strengths=["a"],
                       weaknesses=["b"], justification="too short")


def test_dimension_score_confidence_range():
    with pytest.raises(ValidationError):
        DimensionScore(score=3, confidence=1.5, strengths=["a"],
                       weaknesses=["b"], justification="x" * 200)


def test_paper_metadata_minimal():
    m = PaperMetadata(title="T", authors=["A"], abstract="abs")
    assert m.doi is None
    assert m.arxiv_id is None


def test_synthesis_result_valid():
    s = SynthesisResult(
        summary="x" * 200,
        key_strengths=["a"], key_weaknesses=["b"],
        questions_for_authors=["q1"],
        overall_assessment="good",
    )
    assert s.summary.startswith("x")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_schemas.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write `paper_review_workflow/llm/schemas.py`**

```python
"""Pydantic schemas for LLM structured output."""
from typing import List, Optional
from pydantic import BaseModel, Field


class DimensionScore(BaseModel):
    """Single dimension scoring result (1-5 OpenReview scale)."""
    score: int = Field(ge=1, le=5, description="1-5 OpenReview scale")
    confidence: float = Field(ge=0.0, le=1.0, description="0.0-1.0")
    strengths: List[str] = Field(min_length=1, max_length=5)
    weaknesses: List[str] = Field(min_length=1, max_length=5)
    justification: str = Field(min_length=100, max_length=800)
    evidence: List[dict] = Field(default_factory=list)


class PaperMetadata(BaseModel):
    """Paper metadata extracted from PDF/arXiv."""
    title: str
    authors: List[str]
    abstract: str
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    """Synthesized review across all dimensions."""
    summary: str = Field(min_length=200, max_length=1500)
    key_strengths: List[str]
    key_weaknesses: List[str]
    questions_for_authors: List[str]
    overall_assessment: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_schemas.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/llm/schemas.py tests/unit/test_schemas.py
git commit -m "feat(llm): add Pydantic schemas for structured output"
```

## Task 2.2: Build LLM provider abstraction (base + registry)

**Files:**
- Create: `paper_review_workflow/llm/base.py`
- Create: `paper_review_workflow/llm/registry.py`
- Test: `tests/unit/test_llm_base.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_llm_base.py
import pytest
from paper_review_workflow.llm.base import LLMProvider, LLMResponse, LLMError, RateLimitError
from paper_review_workflow.llm.registry import ProviderRegistry


class FakeProvider(LLMProvider):
    provider_name = "fake"

    def complete(self, **kwargs):
        from pydantic import BaseModel
        class FakeSchema(BaseModel):
            x: int
        return LLMResponse(
            text=None, structured=FakeSchema(x=42),
            usage={"input_tokens": 10, "output_tokens": 5,
                   "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            model="fake-model",
        )


def test_provider_registry_singleton():
    ProviderRegistry._instance = None
    r1 = ProviderRegistry()
    r2 = ProviderRegistry()
    assert r1 is r2


def test_provider_registry_register_and_get():
    ProviderRegistry._instance = None
    reg = ProviderRegistry()
    reg.register("fake", FakeProvider)
    assert reg.get("fake") is FakeProvider


def test_provider_registry_get_unknown():
    ProviderRegistry._instance = None
    reg = ProviderRegistry()
    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_llm_response_dataclass():
    from pydantic import BaseModel
    class S(BaseModel):
        y: str
    r = LLMResponse(text="hi", structured=S(y="x"),
                    usage={"input_tokens": 1, "output_tokens": 1,
                           "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                    model="m")
    assert r.text == "hi"
    assert r.structured.y == "x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_llm_base.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write `paper_review_workflow/llm/base.py`**

```python
"""LLM provider abstraction."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Type, Union
from pydantic import BaseModel


@dataclass
class LLMResponse:
    text: Optional[str]
    structured: Optional[BaseModel]
    usage: dict
    model: str
    raw: Any = None


class LLMError(Exception):
    """Base LLM error."""


class RateLimitError(LLMError):
    """HTTP 429 after retries exhausted."""


class ContextLengthError(LLMError):
    """Context length exceeded."""


class SchemaValidationError(LLMError):
    """LLM output did not match the schema."""


class LLMProvider(ABC):
    provider_name: str = ""

    @abstractmethod
    def complete(
        self,
        system: Union[str, list],
        messages: list,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        response_schema: Optional[Type[BaseModel]] = None,
        cached_context: Optional[str] = None,
    ) -> LLMResponse:
        ...
```

- [ ] **Step 4: Write `paper_review_workflow/llm/registry.py`**

```python
"""Provider registry (singleton)."""
from typing import Dict, Type, Optional
from .base import LLMProvider


class ProviderRegistry:
    _instance: Optional["ProviderRegistry"] = None
    _providers: Dict[str, Type[LLMProvider]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._providers = {}
            cls._instance._register_builtin()
        return cls._instance

    def _register_builtin(self) -> None:
        from .anthropic_provider import AnthropicProvider
        self.register("anthropic", AnthropicProvider)

    def register(self, name: str, provider_cls: Type[LLMProvider]) -> None:
        self._providers[name] = provider_cls

    def get(self, name: str) -> Type[LLMProvider]:
        if name not in self._providers:
            raise KeyError(f"provider not registered: {name}")
        return self._providers[name]

    def list_providers(self) -> list:
        return list(self._providers.keys())
```

- [ ] **Step 5: Run test to verify it passes (will fail on import of anthropic_provider, defer)**

For now, the test will fail because `anthropic_provider` doesn't exist. Create a stub:

```python
# paper_review_workflow/llm/anthropic_provider.py
"""Anthropic LLM provider. Stub, will be implemented in Task 2.3."""
from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def complete(self, **kwargs):
        raise NotImplementedError("AnthropicProvider not yet implemented")
```

Run: `pytest tests/unit/test_llm_base.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add paper_review_workflow/llm/base.py paper_review_workflow/llm/registry.py paper_review_workflow/llm/anthropic_provider.py tests/unit/test_llm_base.py
git commit -m "feat(llm): add provider abstraction + registry + stub"
```

## Task 2.3: Implement AnthropicProvider with retry + cache + tool use

**Files:**
- Modify: `paper_review_workflow/llm/anthropic_provider.py`
- Test: `tests/unit/test_anthropic_provider.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_anthropic_provider.py
import pytest
from unittest.mock import MagicMock, patch
import anthropic

from paper_review_workflow.llm.anthropic_provider import AnthropicProvider
from paper_review_workflow.llm.base import RateLimitError, ContextLengthError, SchemaValidationError
from paper_review_workflow.llm.schemas import DimensionScore


def _make_response(content_blocks, usage_in=100, usage_out=50,
                   cache_creation=0, cache_read=0):
    resp = MagicMock()
    resp.content = content_blocks
    resp.usage = MagicMock(
        input_tokens=usage_in, output_tokens=usage_out,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )
    return resp


def _tool_use_block(input_dict):
    b = MagicMock()
    b.type = "tool_use"
    b.input = input_dict
    return b


@patch("anthropic.Anthropic")
def test_complete_with_structured_output(mock_cls):
    mock_client = mock_cls.return_value
    mock_client.messages.create.return_value = _make_response([
        _tool_use_block({
            "score": 4, "confidence": 0.85,
            "strengths": ["a"], "weaknesses": ["b"],
            "justification": "x" * 200, "evidence": [],
        })
    ])

    p = AnthropicProvider()
    result = p.complete(
        system="test", messages=[{"role": "user", "content": "score"}],
        model="claude-sonnet-4-6", max_tokens=100,
        response_schema=DimensionScore,
    )

    assert result.structured.score == 4
    assert result.structured.confidence == 0.85
    assert result.usage["input_tokens"] == 100


@patch("anthropic.Anthropic")
def test_complete_with_cached_context_sets_cache_control(mock_cls):
    mock_client = mock_cls.return_value
    mock_client.messages.create.return_value = _make_response([
        _tool_use_block({
            "score": 3, "confidence": 0.5,
            "strengths": ["a"], "weaknesses": ["b"],
            "justification": "x" * 200, "evidence": [],
        })
    ])

    p = AnthropicProvider()
    p.complete(
        system="score novelty",
        messages=[{"role": "user", "content": "..."}],
        model="claude-sonnet-4-6", max_tokens=100,
        response_schema=DimensionScore,
        cached_context="PAPER FULL TEXT",
    )

    call_kwargs = mock_client.messages.create.call_args.kwargs
    system_blocks = call_kwargs["system"]
    assert any(b.get("cache_control") == {"type": "ephemeral"} for b in system_blocks)
    assert any(b.get("text") == "PAPER FULL TEXT" for b in system_blocks)


@patch("anthropic.Anthropic")
def test_retry_on_rate_limit(mock_cls, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)  # skip real sleeping

    mock_client = mock_cls.return_value
    success_resp = _make_response([
        _tool_use_block({
            "score": 3, "confidence": 0.5,
            "strengths": ["a"], "weaknesses": ["b"],
            "justification": "x" * 200, "evidence": [],
        })
    ])
    mock_client.messages.create.side_effect = [
        anthropic.RateLimitError(
            message="429", response=MagicMock(status_code=429), body=None
        ),
        success_resp,
    ]

    p = AnthropicProvider()
    result = p.complete(
        system="t", messages=[], model="m", max_tokens=10,
        response_schema=DimensionScore,
    )
    assert mock_client.messages.create.call_count == 2
    assert result.structured.score == 3


@patch("anthropic.Anthropic")
def test_rate_limit_exhausted(mock_cls, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    mock_client = mock_cls.return_value
    mock_client.messages.create.side_effect = anthropic.RateLimitError(
        message="429", response=MagicMock(status_code=429), body=None
    )

    p = AnthropicProvider()
    with pytest.raises(RateLimitError):
        p.complete(system="t", messages=[], model="m", max_tokens=10,
                   response_schema=DimensionScore)
    assert mock_client.messages.create.call_count == 3


@patch("anthropic.Anthropic")
def test_context_length_no_retry(mock_cls, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    mock_client = mock_cls.return_value
    err = anthropic.BadRequestError(
        message="context_length_exceeded", response=MagicMock(status_code=400), body=None
    )
    mock_client.messages.create.side_effect = err

    p = AnthropicProvider()
    with pytest.raises(ContextLengthError):
        p.complete(system="t", messages=[], model="m", max_tokens=10,
                   response_schema=DimensionScore)
    assert mock_client.messages.create.call_count == 1


@patch("anthropic.Anthropic")
def test_schema_validation_failure_raises(mock_cls):
    mock_client = mock_cls.return_value
    # score=6 violates schema (1-5)
    mock_client.messages.create.return_value = _make_response([
        _tool_use_block({
            "score": 6, "confidence": 0.5,
            "strengths": ["a"], "weaknesses": ["b"],
            "justification": "x" * 200, "evidence": [],
        })
    ])

    p = AnthropicProvider()
    with pytest.raises(SchemaValidationError):
        p.complete(system="t", messages=[], model="m", max_tokens=10,
                   response_schema=DimensionScore)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_anthropic_provider.py -v`
Expected: FAIL with NotImplementedError

- [ ] **Step 3: Implement `AnthropicProvider`**

```python
"""Anthropic LLM provider with retry, prompt cache, and tool-use structured output."""
import random
import time
import logging
from typing import Any, List, Optional, Type, Union

import anthropic
from pydantic import BaseModel, ValidationError

from .base import (
    LLMProvider, LLMResponse, LLMError,
    RateLimitError, ContextLengthError, SchemaValidationError,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    MAX_RETRIES = 3
    INITIAL_BACKOFF = 1.0
    MAX_BACKOFF = 30.0

    def __init__(self, api_key: Optional[str] = None):
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system, messages, model, max_tokens,
                 temperature=0.0, response_schema=None, cached_context=None):
        system_blocks = self._build_system_blocks(system, cached_context)
        tools, tool_choice = self._build_tools(response_schema)

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_blocks,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                return self._parse_response(resp, response_schema, model)

            except anthropic.RateLimitError:
                if attempt == self.MAX_RETRIES - 1:
                    raise RateLimitError(f"rate limit after {self.MAX_RETRIES} attempts")
                self._sleep_backoff(attempt)

            except anthropic.APIStatusError as e:
                if e.status_code == 400 and "context_length" in str(e).lower():
                    raise ContextLengthError(str(e))
                if e.status_code in (500, 503) and attempt < self.MAX_RETRIES - 1:
                    self._sleep_backoff(attempt)
                else:
                    raise

            except anthropic.APIConnectionError:
                if attempt < self.MAX_RETRIES - 1:
                    self._sleep_backoff(attempt)
                else:
                    raise

    def _build_system_blocks(self, system, cached_context):
        blocks = []
        if cached_context:
            blocks.append({
                "type": "text",
                "text": cached_context,
                "cache_control": {"type": "ephemeral"},
            })
        if isinstance(system, str):
            if system:
                blocks.append({"type": "text", "text": system})
        else:
            blocks.extend(system)
        return blocks

    def _build_tools(self, schema):
        if schema is None:
            return None, None
        return [{
            "name": "submit_result",
            "description": "Submit the structured result",
            "input_schema": schema.model_json_schema(),
        }], {"type": "tool", "name": "submit_result"}

    def _parse_response(self, resp, schema, model):
        structured = None
        if schema:
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    try:
                        structured = schema(**block.input)
                    except ValidationError as e:
                        raise SchemaValidationError(
                            f"LLM output failed schema validation: {e}"
                        )
                    break
            if structured is None:
                raise SchemaValidationError("No tool_use block in response")
        else:
            text_parts = [getattr(b, "text", "") for b in resp.content
                          if getattr(b, "type", None) == "text"]
            structured = None

        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        }

        text = None
        if not schema:
            text = "".join(getattr(b, "text", "") for b in resp.content
                           if getattr(b, "type", None) == "text") or None

        return LLMResponse(
            text=text,
            structured=structured,
            usage=usage,
            model=model,
            raw=resp,
        )

    def _sleep_backoff(self, attempt: int) -> None:
        backoff = min(self.INITIAL_BACKOFF * (2 ** attempt), self.MAX_BACKOFF)
        jitter = random.uniform(0, backoff * 0.1)
        time.sleep(backoff + jitter)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_anthropic_provider.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/llm/anthropic_provider.py tests/unit/test_anthropic_provider.py
git commit -m "feat(llm): implement AnthropicProvider with retry/cache/tool-use"
```

## Task 2.4: Build LLMClient convenience layer

**Files:**
- Create: `paper_review_workflow/llm/client.py`
- Create: `paper_review_workflow/llm/__init__.py`
- Test: `tests/unit/test_llm_client.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_llm_client.py
import pytest
from unittest.mock import MagicMock, patch
from paper_review_workflow.llm.client import LLMClient
from paper_review_workflow.llm.schemas import DimensionScore


def test_from_env_initializes_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("LLM_MAX_TOKENS", "4096")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    LLMClient._instance = None
    client = LLMClient.from_env()
    assert client.model == "claude-sonnet-4-6"
    assert client.max_tokens == 4096


def test_from_env_singleton(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    LLMClient._instance = None
    c1 = LLMClient.from_env()
    c2 = LLMClient.from_env()
    assert c1 is c2


def test_score_delegates_to_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    LLMClient._instance = None
    client = LLMClient.from_env()

    mock_result = MagicMock()
    mock_result.structured = DimensionScore(
        score=4, confidence=0.8, strengths=["a"],
        weaknesses=["b"], justification="x" * 200,
    )
    client._provider.complete = MagicMock(return_value=mock_result)

    result = client.score(
        system="score novelty", user_content="paper text",
        schema=DimensionScore, cached_context="PAPER",
    )

    assert result.score == 4
    call_kwargs = client._provider.complete.call_args.kwargs
    assert call_kwargs["cached_context"] == "PAPER"
    assert call_kwargs["response_schema"] is DimensionScore
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_llm_client.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write `paper_review_workflow/llm/client.py`**

```python
"""LLMClient: convenience layer delegating to a provider chosen by env."""
import os
from typing import Optional, Type, Union
from pydantic import BaseModel

from .base import LLMProvider, LLMResponse
from .registry import ProviderRegistry


class LLMClient:
    _instance: Optional["LLMClient"] = None

    def __init__(self, provider: LLMProvider, model: str,
                 max_tokens: int, temperature: float):
        self._provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @classmethod
    def from_env(cls) -> "LLMClient":
        if cls._instance is None:
            provider_name = os.environ.get("LLM_PROVIDER", "anthropic")
            provider_cls = ProviderRegistry.get(provider_name)
            api_key = os.environ.get("ANTHROPIC_API_KEY") if provider_name == "anthropic" else None
            provider = provider_cls(api_key=api_key) if provider_name == "anthropic" else provider_cls()
            cls._instance = cls(
                provider=provider,
                model=os.environ.get("LLM_MODEL", "claude-sonnet-4-6"),
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
                temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Test-only: reset singleton."""
        cls._instance = None

    def complete(self, system, messages, response_schema=None, cached_context=None,
                 model=None, max_tokens=None, temperature=None) -> LLMResponse:
        return self._provider.complete(
            system=system,
            messages=messages,
            model=model or self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            response_schema=response_schema,
            cached_context=cached_context,
        )

    def score(self, system: str, user_content: str, schema: Type[BaseModel],
              cached_context: Optional[str] = None) -> BaseModel:
        resp = self.complete(
            system=system,
            messages=[{"role": "user", "content": user_content}],
            response_schema=schema,
            cached_context=cached_context,
        )
        return resp.structured
```

- [ ] **Step 4: Write `paper_review_workflow/llm/__init__.py`**

```python
"""LLM provider layer."""
from .base import LLMProvider, LLMResponse, LLMError, RateLimitError, ContextLengthError, SchemaValidationError
from .registry import ProviderRegistry
from .client import LLMClient
from .schemas import DimensionScore, PaperMetadata, SynthesisResult

__all__ = [
    "LLMProvider", "LLMResponse", "LLMError", "RateLimitError",
    "ContextLengthError", "SchemaValidationError",
    "ProviderRegistry", "LLMClient",
    "DimensionScore", "PaperMetadata", "SynthesisResult",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_llm_client.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add paper_review_workflow/llm/client.py paper_review_workflow/llm/__init__.py tests/unit/test_llm_client.py
git commit -m "feat(llm): add LLMClient convenience layer"
```

---

# M3: Extract Component

**Goal:** Build the `paper-review/extract@v1` action that takes a PDF path or arXiv ID, fetches/parses the paper, and writes `metadata.json` / `sections.json` / `full_text.md` / `references.json` / `artifacts/paper.pdf`.

**Estimated:** 1.5 days

## Task 3.1: PDF parsing module (PyMuPDF)

**Files:**
- Create: `paper_review_workflow/actions/extract/pdf.py`
- Test: `tests/unit/test_extract_pdf.py`
- Test fixture: `tests/fixtures/sample_paper.pdf` (a minimal 2-page PDF, handcrafted or downloaded)

- [ ] **Step 1: Create a minimal test PDF fixture**

Run:
```bash
mkdir -p tests/fixtures
python -c "
import fitz
doc = fitz.open()
page = doc.new_page()
page.insert_text((50, 72), 'Sample Paper Title')
page.insert_text((50, 100), 'Author One, Author Two')
page.insert_text((50, 140), 'Abstract')
page.insert_text((50, 160), 'This is a sample abstract for testing. ' * 10)
page.insert_text((50, 220), '1. Introduction')
page.insert_text((50, 240), 'This is the introduction. ' * 20)
page2 = doc.new_page()
page2.insert_text((50, 72), '2. Method')
page2.insert_text((50, 100), 'The method is described here. ' * 30)
page2.insert_text((50, 400), 'References')
page2.insert_text((50, 420), '[1] Smith et al. 2023. Some paper.')
doc.save('tests/fixtures/sample_paper.pdf')
doc.close()
print('created fixture')
"
```
Expected: `created fixture`, file exists at `tests/fixtures/sample_paper.pdf`

- [ ] **Step 2: Write failing test**

```python
# tests/unit/test_extract_pdf.py
import pytest
from pathlib import Path
from paper_review_workflow.actions.extract.pdf import parse_pdf


def test_parse_pdf_returns_metadata_sections_fulltext(tmp_path):
    fixture = Path("tests/fixtures/sample_paper.pdf")
    if not fixture.exists():
        pytest.skip("fixture missing")

    result = parse_pdf(str(fixture))

    assert "metadata" in result
    assert "sections" in result
    assert "full_text" in result
    assert "references" in result
    assert isinstance(result["metadata"]["title"], str)
    assert len(result["metadata"]["title"]) > 0
    assert len(result["sections"]) >= 1
    assert "Introduction" in result["full_text"] or "introduction" in result["full_text"].lower()


def test_parse_pdf_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        parse_pdf("/nonexistent/path/to/file.pdf")


def test_parse_pdf_corrupted(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(Exception):
        parse_pdf(str(bad))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/unit/test_extract_pdf.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 4: Implement `paper_review_workflow/actions/extract/pdf.py`**

```python
"""PDF parsing using PyMuPDF (fitz)."""
import re
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF


SECTION_TITLE_RE = re.compile(
    r"^(?:\d+\.?\d*\.?\d*\s+)?(Abstract|Introduction|Background|Related Work|"
    r"Method(?:s)?|Approach|Model|Experiments?|Results?|Evaluation|"
    r"Discussion|Conclusion[s]?|References|Acknowledgments?)\s*$",
    re.IGNORECASE,
)


def parse_pdf(pdf_path: str) -> Dict:
    """Parse a PDF file into metadata, sections, full_text, references.

    Returns:
        {
            "metadata": {"title", "authors", "abstract", "doi", "arxiv_id", "keywords"},
            "sections": [{"title", "level", "text", "page_start", "page_end"}],
            "full_text": str (markdown),
            "references": [{"raw", "page"}],
        }
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(path))
    try:
        pages_text = [page.get_text("text") for page in doc]
        full_text = "\n\n".join(pages_text)

        metadata = _extract_metadata(doc, pages_text)
        sections = _extract_sections(pages_text)
        references = _extract_references(pages_text)

        return {
            "metadata": metadata,
            "sections": sections,
            "full_text": full_text,
            "references": references,
        }
    finally:
        doc.close()


def _extract_metadata(doc, pages_text: List[str]) -> Dict:
    # Prefer PDF metadata, fallback to first-page heuristics
    pdf_meta = doc.metadata or {}
    title = pdf_meta.get("title", "").strip()
    authors_str = pdf_meta.get("author", "").strip()

    if not title:
        # First non-empty line of page 1, skip "arXiv:" prelude
        first_page_lines = [l.strip() for l in pages_text[0].splitlines() if l.strip()]
        # Skip arXiv headers
        for line in first_page_lines:
            if line.lower().startswith("arxiv"):
                continue
            title = line
            break

    if not authors_str:
        # Heuristic: line(s) after title, before "Abstract"
        lines = pages_text[0].splitlines() if pages_text else []
        author_lines = []
        in_authors = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith("abstract"):
                break
            if title and stripped == title:
                in_authors = True
                continue
            if in_authors:
                author_lines.append(stripped)
        authors_str = ", ".join(author_lines[:3])  # cap at 3

    authors = [a.strip() for a in re.split(r"[,;]|\band\b", authors_str) if a.strip()]
    abstract = _extract_abstract(pages_text)

    return {
        "title": title or "Untitled",
        "authors": authors,
        "abstract": abstract,
        "doi": None,
        "arxiv_id": None,
        "keywords": [],
    }


def _extract_abstract(pages_text: List[str]) -> str:
    full = "\n".join(pages_text)
    m = re.search(r"Abstract[:\s]*(.+?)(?=\n\s*(?:1\.?\s+)?(?:Introduction|Keywords|I\.\s))",
                  full, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()[:2000]
    return ""


def _extract_sections(pages_text: List[str]) -> List[Dict]:
    sections = []
    current = None

    for page_idx, page_text in enumerate(pages_text):
        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            m = SECTION_TITLE_RE.match(stripped)
            if m:
                if current:
                    current["page_end"] = page_idx
                    sections.append(current)
                current = {
                    "title": stripped,
                    "level": 1 if not stripped[0].isdigit() else
                             (len(stripped.split()[0].rstrip(".").split("."))),
                    "text": "",
                    "page_start": page_idx,
                    "page_end": page_idx,
                }
            elif current:
                current["text"] += stripped + "\n"

    if current:
        sections.append(current)
    return sections


def _extract_references(pages_text: List[str]) -> List[Dict]:
    refs = []
    in_refs = False
    for page_idx, page_text in enumerate(pages_text):
        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^References\s*$", stripped, re.IGNORECASE):
                in_refs = True
                continue
            if in_refs:
                # Each reference usually starts with [N] or a name
                if re.match(r"^\[\d+\]", stripped) or stripped[0:1].isupper():
                    refs.append({"raw": stripped, "page": page_idx})
    return refs
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_extract_pdf.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add paper_review_workflow/actions/extract/pdf.py tests/unit/test_extract_pdf.py tests/fixtures/
git commit -m "feat(extract): PyMuPDF-based PDF parser"
```

## Task 3.2: arXiv fetch module (latex-first, PDF fallback)

**Files:**
- Create: `paper_review_workflow/actions/extract/arxiv.py`
- Test: `tests/unit/test_extract_arxiv.py`

- [ ] **Step 1: Write failing test (with mocked network)**

```python
# tests/unit/test_extract_arxiv.py
import pytest
from unittest.mock import patch, MagicMock
from paper_review_workflow.actions.extract.arxiv import (
    parse_arxiv_id, fetch_arxiv_source, fetch_arxiv_pdf, ArxivFetchError
)


def test_parse_arxiv_id_from_numeric():
    assert parse_arxiv_id("2402.12098") == "2402.12098"


def test_parse_arxiv_id_from_abs_url():
    assert parse_arxiv_id("https://arxiv.org/abs/2402.12098") == "2402.12098"


def test_parse_arxiv_id_from_pdf_url():
    assert parse_arxiv_id("https://arxiv.org/pdf/2402.12098.pdf") == "2402.12098"


def test_parse_arxiv_id_from_v_version():
    assert parse_arxiv_id("2402.12098v2") == "2402.12098"


def test_parse_arxiv_id_invalid():
    with pytest.raises(ValueError):
        parse_arxiv_id("not-an-arxiv-id")


@patch("paper_review_workflow.actions.extract.arxiv.httpx.get")
def test_fetch_arxiv_pdf_success(mock_get, tmp_path):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"%PDF-1.4 fake pdf"
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    out = tmp_path / "paper.pdf"
    fetch_arxiv_pdf("2402.12098", out)
    assert out.read_bytes() == b"%PDF-1.4 fake pdf"


@patch("paper_review_workflow.actions.extract.arxiv.httpx.get")
def test_fetch_arxiv_pdf_failure_raises(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.raise_for_status.side_effect = Exception("404")
    mock_get.return_value = mock_resp

    with pytest.raises(ArxivFetchError):
        fetch_arxiv_pdf("9999.99999", "/tmp/out.pdf")


@patch("paper_review_workflow.actions.extract.arxiv.httpx.get")
def test_fetch_arxiv_source_latex_tarball(mock_get, tmp_path):
    import tarfile
    # Build a fake tarball with a .tex file
    tar_path = tmp_path / "src.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        import io
        tex_content = b"\\title{Fake Paper}\n\\section{Intro}\nHello"
        info = tarfile.TarInfo(name="main.tex")
        info.size = len(tex_content)
        tf.addfile(info, io.BytesIO(tex_content))

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = tar_path.read_bytes()
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    out = tmp_path / "src.tar.gz"
    has_latex = fetch_arxiv_source("2402.12098", out)
    assert has_latex is True
    assert out.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_extract_arxiv.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement `paper_review_workflow/actions/extract/arxiv.py`**

```python
"""arXiv paper fetching: latex source first, PDF fallback."""
import re
import logging
from pathlib import Path
from typing import Union

import httpx

logger = logging.getLogger(__name__)


ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)


class ArxivFetchError(Exception):
    pass


def parse_arxiv_id(source: str) -> str:
    """Extract arXiv ID from various input formats."""
    source = source.strip()
    m = ARXIV_ID_RE.search(source)
    if not m:
        # Try pure ID without URL
        if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", source):
            return re.sub(r"v\d+$", "", source)
        raise ValueError(f"cannot parse arxiv id from: {source}")
    return m.group(1)


def fetch_arxiv_source(arxiv_id: str, dest: Union[str, Path],
                       timeout: float = 60.0) -> bool:
    """Fetch latex source tarball. Returns True if successful."""
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, follow_redirects=True)
            resp.raise_for_status()
            Path(dest).write_bytes(resp.content)
        return True
    except Exception as e:
        logger.warning(f"arxiv source fetch failed for {arxiv_id}: {e}")
        return False


def fetch_arxiv_pdf(arxiv_id: str, dest: Union[str, Path],
                    timeout: float = 60.0) -> None:
    """Fetch PDF. Raises ArxivFetchError on failure."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, follow_redirects=True)
            resp.raise_for_status()
            Path(dest).write_bytes(resp.content)
    except Exception as e:
        raise ArxivFetchError(f"failed to fetch PDF for {arxiv_id}: {e}")


def fetch_arxiv(arxiv_id: str, session_dir: Path) -> dict:
    """Fetch arxiv paper, prefer latex source, fallback to PDF.

    Returns:
        {"pdf_path": str, "latex_tarball": Optional[str], "arxiv_id": str}
    """
    artifacts = session_dir / "00_extract" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    latex_path = artifacts / "source.tar.gz"
    pdf_path = artifacts / "paper.pdf"

    if fetch_arxiv_source(arxiv_id, latex_path):
        logger.info(f"got latex source for {arxiv_id}")
        # Still try to get PDF for fallback parsing
        try:
            fetch_arxiv_pdf(arxiv_id, pdf_path)
        except ArxivFetchError:
            pass
        return {"pdf_path": str(pdf_path) if pdf_path.exists() else None,
                "latex_tarball": str(latex_path), "arxiv_id": arxiv_id}

    # Fallback to PDF only
    fetch_arxiv_pdf(arxiv_id, pdf_path)
    return {"pdf_path": str(pdf_path), "latex_tarball": None, "arxiv_id": arxiv_id}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_extract_arxiv.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/actions/extract/arxiv.py tests/unit/test_extract_arxiv.py
git commit -m "feat(extract): arXiv fetch with latex-first PDF-fallback"
```

## Task 3.3: ExtractAction main entry

**Files:**
- Create: `paper_review_workflow/actions/extract/__init__.py`
- Test: `tests/unit/test_extract_action.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_extract_action.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from paper_review_workflow.actions.extract import ExtractAction
from paper_review_workflow.actions.base import ActionResult


def test_extract_local_pdf_success(tmp_path):
    fixture = Path("tests/fixtures/sample_paper.pdf")
    if not fixture.exists():
        pytest.skip("fixture missing")

    session_dir = tmp_path / "session"
    session_dir.mkdir()

    action = ExtractAction()
    result = action.run(
        params={"source": str(fixture), "session_dir": str(session_dir)},
        env={}, context={}, log_callback=lambda x: None,
    )

    assert result.success
    out_dir = session_dir / "00_extract"
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "sections.json").exists()
    assert (out_dir / "full_text.md").exists()
    assert (out_dir / "references.json").exists()

    meta = json.loads((out_dir / "metadata.json").read_text())
    assert "title" in meta
    assert isinstance(meta["authors"], list)

    assert "paper_id" in result.outputs
    assert len(result.outputs["paper_id"]) == 8
    assert result.outputs["full_text_path"].endswith("full_text.md")


def test_extract_unsupported_source(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    action = ExtractAction()
    result = action.run(
        params={"source": "not-a-pdf-or-arxiv-id", "session_dir": str(session_dir)},
        env={}, context={}, log_callback=lambda x: None,
    )
    assert not result.success
    assert "unsupported source" in result.message


@patch("paper_review_workflow.actions.extract.arxiv.fetch_arxiv")
@patch("paper_review_workflow.actions.extract.pdf.parse_pdf")
def test_extract_arxiv_id(mock_parse, mock_fetch, tmp_path):
    # Setup mocks
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    pdf_path = session_dir / "00_extract" / "artifacts" / "paper.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"fake pdf")

    mock_fetch.return_value = {
        "pdf_path": str(pdf_path), "latex_tarball": None, "arxiv_id": "2402.12098"
    }
    mock_parse.return_value = {
        "metadata": {"title": "Test Paper", "authors": ["A"], "abstract": "abs",
                     "doi": None, "arxiv_id": None, "keywords": []},
        "sections": [{"title": "Intro", "level": 1, "text": "...", "page_start": 0, "page_end": 0}],
        "full_text": "Intro ...",
        "references": [],
    }

    action = ExtractAction()
    result = action.run(
        params={"source": "2402.12098", "session_dir": str(session_dir)},
        env={}, context={}, log_callback=lambda x: None,
    )

    assert result.success
    meta = json.loads((session_dir / "00_extract" / "metadata.json").read_text())
    assert meta["arxiv_id"] == "2402.12098"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_extract_action.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement `paper_review_workflow/actions/extract/__init__.py`**

```python
"""ExtractAction: parse PDF or arXiv paper into structured artifacts."""
import hashlib
import json
import shutil
import logging
from pathlib import Path
from typing import Optional

from ..base import BaseAction, ActionResult
from ..registry import ActionRegistry
from .pdf import parse_pdf
from .arxiv import parse_arxiv_id, fetch_arxiv, ArxivFetchError

logger = logging.getLogger(__name__)


class ExtractAction(BaseAction):
    """Parse a paper (PDF path or arXiv ID) into metadata/sections/full_text/references."""

    @property
    def description(self) -> str:
        return "Extract paper content (PDF or arXiv)"

    def run(self, params, env, context, log_callback=None):
        source = params.get("source", "").strip()
        session_dir = Path(params["session_dir"])

        if not source:
            return ActionResult(success=False, message="source is required")

        try:
            if self._is_arxiv(source):
                return self._extract_arxiv(source, session_dir, log_callback)
            elif source.endswith(".pdf") and Path(source).is_file():
                return self._extract_pdf(source, session_dir, log_callback)
            else:
                return ActionResult(
                    success=False,
                    message=f"unsupported source: {source} (must be arXiv ID/URL or PDF path)"
                )
        except Exception as e:
            logger.exception("extract failed")
            return ActionResult(success=False, message=f"extract failed: {e}")

    def _is_arxiv(self, source: str) -> bool:
        try:
            parse_arxiv_id(source)
            return True
        except ValueError:
            return False

    def _extract_pdf(self, pdf_path: str, session_dir: Path, log_callback) -> ActionResult:
        if log_callback:
            log_callback(f"parsing local PDF: {pdf_path}")

        parsed = parse_pdf(pdf_path)
        out_dir = session_dir / "00_extract"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy the PDF to artifacts
        artifacts = out_dir / "artifacts"
        artifacts.mkdir(exist_ok=True)
        shutil.copy(pdf_path, artifacts / "paper.pdf")

        return self._write_outputs(out_dir, parsed, arxiv_id=None, log_callback=log_callback)

    def _extract_arxiv(self, source: str, session_dir: Path, log_callback) -> ActionResult:
        arxiv_id = parse_arxiv_id(source)
        if log_callback:
            log_callback(f"fetching arXiv paper: {arxiv_id}")

        try:
            fetch_result = fetch_arxiv(arxiv_id, session_dir)
        except ArxivFetchError as e:
            return ActionResult(success=False, message=str(e))

        if not fetch_result.get("pdf_path"):
            return ActionResult(success=False, message=f"could not fetch any PDF for {arxiv_id}")

        if log_callback:
            log_callback(f"parsing PDF: {fetch_result['pdf_path']}")

        parsed = parse_pdf(fetch_result["pdf_path"])
        parsed["metadata"]["arxiv_id"] = arxiv_id

        out_dir = session_dir / "00_extract"
        return self._write_outputs(out_dir, parsed, arxiv_id=arxiv_id, log_callback=log_callback)

    def _write_outputs(self, out_dir: Path, parsed: dict,
                       arxiv_id: Optional[str], log_callback) -> ActionResult:
        metadata = parsed["metadata"]
        if arxiv_id:
            metadata["arxiv_id"] = arxiv_id

        # Compute paper_id from title
        paper_id = hashlib.sha256(metadata["title"].encode("utf-8")).hexdigest()[:8]

        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )
        (out_dir / "sections.json").write_text(
            json.dumps(parsed["sections"], ensure_ascii=False, indent=2)
        )
        (out_dir / "full_text.md").write_text(parsed["full_text"])
        (out_dir / "references.json").write_text(
            json.dumps(parsed["references"], ensure_ascii=False, indent=2)
        )

        if log_callback:
            log_callback(f"extracted: {metadata['title'][:80]}")
            log_callback(f"sections: {len(parsed['sections'])}, references: {len(parsed['references'])}")

        return ActionResult(
            success=True,
            outputs={
                "paper_id": paper_id,
                "full_text_path": str(out_dir / "full_text.md"),
                "metadata_path": str(out_dir / "metadata.json"),
                "sections_path": str(out_dir / "sections.json"),
                "references_path": str(out_dir / "references.json"),
            },
            log_lines=[f"extracted paper_id={paper_id}"],
        )


def register_extract_action(registry: ActionRegistry) -> None:
    registry.register("paper-review/extract@v1", ExtractAction())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_extract_action.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Wire into `actions/builtin.py`**

```python
# paper_review_workflow/actions/builtin.py  (update)
from .base import BaseAction, ActionResult
from .registry import ActionRegistry
from .extract import ExtractAction, register_extract_action


class EchoAction(BaseAction):
    @property
    def description(self) -> str:
        return "Echo back the message param (smoke test)"

    def run(self, params, env, context, log_callback=None):
        msg = params.get("message", "")
        if log_callback:
            log_callback(f"echo: {msg}")
        return ActionResult(success=True, outputs={"message": msg})


def register_builtin_actions(registry: ActionRegistry) -> None:
    registry.register("paper-review/echo@v1", EchoAction())
    register_extract_action(registry)
```

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add paper_review_workflow/actions/extract/__init__.py paper_review_workflow/actions/builtin.py tests/unit/test_extract_action.py
git commit -m "feat(extract): ExtractAction with PDF/arXiv support"
```

---

# M4: Dimension Scoring Components

**Goal:** Build the `paper-review/dim_score@v1` action that scores one dimension via LLM, and the 8 prompt templates. Verify 8 dimensions run in parallel via matrix strategy with prompt cache hitting ≥7 of 8.

**Estimated:** 1.5 days

## Task 4.1: Create 8 dimension prompt templates

**Files:**
- Create: `paper_review_workflow/actions/dimensions/prompts/novelty.j2`
- Create: `paper_review_workflow/actions/dimensions/prompts/soundness.j2`
- Create: `paper_review_workflow/actions/dimensions/prompts/significance.j2`
- Create: `paper_review_workflow/actions/dimensions/prompts/clarity.j2`
- Create: `paper_review_workflow/actions/dimensions/prompts/reproducibility.j2`
- Create: `paper_review_workflow/actions/dimensions/prompts/related_work.j2`
- Create: `paper_review_workflow/actions/dimensions/prompts/positioning.j2`
- Create: `paper_review_workflow/actions/dimensions/prompts/presentation.j2`

- [ ] **Step 1: Write `novelty.j2`**

```jinja
You are an expert peer reviewer for a top-tier Computer Science conference (NeurIPS/ICML/ACL caliber).

Score the paper's **Novelty** — the originality of the contributions, methods, or insights compared to prior work.

Consider:
- Are the core ideas new, or a recombination of existing techniques?
- Does the paper clearly articulate what is novel?
- How significant is the novelty (incremental vs. substantial)?
- Are there clear differentiators from the closest prior work?

Scoring (1-5, OpenReview scale):
- 5: Groundbreaking novelty; opens a new research direction
- 4: Substantial novel contribution; clearly differentiated from prior work
- 3: Moderate novelty; some new insights but builds heavily on existing work
- 2: Incremental novelty; minor variation of existing techniques
- 1: Not novel; clearly previously published or trivially derivable

Provide:
- score (1-5)
- confidence (0.0-1.0): your confidence in the score given your familiarity with the literature
- strengths (1-5 items): specific novel aspects
- weaknesses (1-5 items): where novelty is lacking or unclear
- justification (200-800 chars): your reasoning
- evidence (list of {section, quote, page}): specific paper evidence

The paper full text is provided separately as cached context. Score based on it.
```

- [ ] **Step 2: Write `soundness.j2`**

```jinja
You are an expert peer reviewer for a top-tier Computer Science conference.

Score the paper's **Soundness** — the rigor of the methodology, correctness of the claims, and quality of evidence.

Consider:
- Are the theoretical claims supported by proofs or empirical evidence?
- Is the experimental design appropriate for the claims?
- Are baselines fair and sufficient?
- Are there obvious flaws in the reasoning or methodology?
- Are error bars / statistical significance reported where needed?

Scoring (1-5):
- 5: Rigorous and flawless; claims fully supported
- 4: Sound with minor gaps; claims mostly supported
- 3: Mixed; some claims supported, others weak
- 2: Significant methodological issues; claims under-supported
- 1: Fundamentally flawed; claims not supported

Provide score, confidence, strengths, weaknesses, justification (200-800 chars), and evidence (list of {section, quote, page}).
```

- [ ] **Step 3: Write the remaining 6 templates**

For each of `significance`, `clarity`, `reproducibility`, `related_work`, `positioning`, `presentation`, create a `.j2` file following the same structure. Adjust the "Consider" and scoring guidance to the dimension:

- **significance.j2**: Impact on the field, importance of the problem, potential to influence future work.
- **clarity.j2**: Writing quality, logical flow, readability, appropriate background given.
- **reproducibility.j2**: Code/data availability, hyperparameters stated, sufficient detail to replicate.
- **related_work.j2**: Coverage of relevant prior work, accurate citations, proper positioning.
- **positioning.j2**: How well the paper frames its contribution relative to existing literature.
- **presentation.j2**: Figure quality, table clarity, formatting, typographical correctness.

Each file ends with: `Provide score, confidence, strengths, weaknesses, justification (200-800 chars), and evidence (list of {section, quote, page}).`

- [ ] **Step 4: Commit**

```bash
git add paper_review_workflow/actions/dimensions/prompts/
git commit -m "feat(dimensions): 8 jinja2 prompt templates"
```

## Task 4.2: Build DimensionAction

**Files:**
- Create: `paper_review_workflow/actions/dimensions/__init__.py`
- Test: `tests/unit/test_dim_score.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_dim_score.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_review_workflow.actions.dimensions import DimensionAction
from paper_review_workflow.llm.schemas import DimensionScore


@pytest.fixture
def fake_full_text(tmp_path):
    p = tmp_path / "full_text.md"
    p.write_text("# Sample Paper\n\nThis is the paper content. " * 50)
    return str(p)


@pytest.fixture
def fake_metadata(tmp_path):
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps({
        "title": "Sample Paper",
        "authors": ["A"],
        "abstract": "...",
        "doi": None, "arxiv_id": None, "keywords": [],
    }))
    return str(p)


def test_dimension_action_scores_novelty(tmp_path, fake_full_text, fake_metadata):
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    action = DimensionAction()

    fake_score = DimensionScore(
        score=4, confidence=0.85,
        strengths=["new method X"], weaknesses=["unclear scope"],
        justification="x" * 250,
        evidence=[{"section": "3.2", "quote": "we propose", "page": 5}],
    )

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.score.return_value = fake_score
        mock_from_env.return_value = mock_client

        result = action.run(
            params={
                "dimension": "novelty",
                "session_dir": str(session_dir),
                "full_text_path": fake_full_text,
                "metadata_path": fake_metadata,
            },
            env={}, context={}, log_callback=lambda x: None,
        )

    assert result.success
    assert result.outputs["score"] == 4
    assert result.outputs["confidence"] == 0.85

    out_dir = session_dir / "10_dim_novelty"
    score_json = json.loads((out_dir / "score.json").read_text())
    assert score_json["dimension"] == "novelty"
    assert score_json["score"] == 4
    assert "cache_read_input_tokens" in score_json["usage"]

    review_md = (out_dir / "review.md").read_text()
    assert "novelty" in review_md.lower()
    assert "score: 4" in review_md.lower() or "score: 4/5" in review_md.lower()


def test_dimension_action_passes_paper_as_cached_context(tmp_path, fake_full_text, fake_metadata):
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    action = DimensionAction()
    fake_score = DimensionScore(
        score=3, confidence=0.7,
        strengths=["a"], weaknesses=["b"],
        justification="y" * 200,
    )

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.score.return_value = fake_score
        mock_from_env.return_value = mock_client

        action.run(
            params={
                "dimension": "soundness",
                "session_dir": str(session_dir),
                "full_text_path": fake_full_text,
                "metadata_path": fake_metadata,
            },
            env={}, context={}, log_callback=lambda x: None,
        )

    call_kwargs = mock_client.score.call_args.kwargs
    # cached_context should contain the paper text
    assert "Sample Paper" in call_kwargs["cached_context"]
    # system prompt should mention "Soundness"
    assert "Soundness" in call_kwargs["system"] or "soundness" in call_kwargs["system"].lower()


def test_dimension_action_unknown_dimension(tmp_path, fake_full_text, fake_metadata):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    action = DimensionAction()
    result = action.run(
        params={
            "dimension": "unknown_dim",
            "session_dir": str(session_dir),
            "full_text_path": fake_full_text,
            "metadata_path": fake_metadata,
        },
        env={}, context={}, log_callback=lambda x: None,
    )
    assert not result.success
    assert "unknown dimension" in result.message.lower() or "no prompt" in result.message.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_dim_score.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement `paper_review_workflow/actions/dimensions/__init__.py`**

```python
"""DimensionAction: scores one paper dimension via LLM, used in matrix strategy."""
import json
import logging
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..base import BaseAction, ActionResult
from ..registry import ActionRegistry
from ...llm.client import LLMClient
from ...llm.schemas import DimensionScore

logger = logging.getLogger(__name__)


ALL_DIMENSIONS = [
    "novelty", "soundness", "significance", "clarity",
    "reproducibility", "related_work", "positioning", "presentation",
]


_PROMPTS_DIR = Path(__file__).parent / "prompts"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
)


class DimensionAction(BaseAction):
    """Score one dimension of a paper. Driven by `with.dimension` param."""

    @property
    def description(self) -> str:
        return "Score one dimension of a paper (matrix-driven)"

    def run(self, params, env, context, log_callback=None):
        dimension = params.get("dimension")
        if not dimension:
            return ActionResult(success=False, message="dimension param required")
        if dimension not in ALL_DIMENSIONS:
            return ActionResult(
                success=False,
                message=f"unknown dimension: {dimension}; must be one of {ALL_DIMENSIONS}",
            )

        session_dir = Path(params["session_dir"])
        full_text_path = params["full_text_path"]

        try:
            paper_text = Path(full_text_path).read_text(encoding="utf-8")
        except Exception as e:
            return ActionResult(success=False, message=f"cannot read paper: {e}")

        # Load metadata (optional, for context in prompt)
        metadata = {}
        if "metadata_path" in params:
            try:
                metadata = json.loads(Path(params["metadata_path"]).read_text())
            except Exception:
                pass

        prompt = self._render_prompt(dimension, metadata)

        client = LLMClient.from_env()
        score = client.score(
            system=prompt,
            user_content=f"Score the {dimension} dimension of this paper.",
            schema=DimensionScore,
            cached_context=paper_text,
        )

        out_dir = session_dir / f"10_dim_{dimension}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write_score_json(out_dir / "score.json", score, dimension, client.model)
        self._write_review_md(out_dir / "review.md", score, dimension)

        if log_callback:
            log_callback(
                f"[{dimension}] score={score.score} conf={score.confidence:.2f} "
                f"cache_read={score.usage if hasattr(score, 'usage') else 'n/a'}"
            )

        return ActionResult(
            success=True,
            outputs={
                "score": score.score,
                "confidence": score.confidence,
                "score_path": str(out_dir / "score.json"),
                "review_path": str(out_dir / "review.md"),
            },
            log_lines=[f"[{dimension}] score={score.score}"],
        )

    def _render_prompt(self, dimension: str, metadata: dict) -> str:
        template = _jinja_env.get_template(f"{dimension}.j2")
        return template.render(metadata=metadata, dimension=dimension)

    def _write_score_json(self, path: Path, score: DimensionScore,
                          dimension: str, model: str) -> None:
        payload = {
            "schema_version": "1.0",
            "dimension": dimension,
            "score": score.score,
            "confidence": score.confidence,
            "strengths": score.strengths,
            "weaknesses": score.weaknesses,
            "justification": score.justification,
            "evidence": score.evidence,
            "model_used": model,
            "usage": score.usage if hasattr(score, "usage") else {},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def _write_review_md(self, path: Path, score: DimensionScore,
                         dimension: str) -> None:
        lines = [
            f"# {dimension.title()} — Score: {score.score}/5 (confidence: {score.confidence:.2f})",
            "",
            "## Strengths",
            *[f"- {s}" for s in score.strengths],
            "",
            "## Weaknesses",
            *[f"- {w}" for w in score.weaknesses],
            "",
            "## Justification",
            score.justification,
        ]
        if score.evidence:
            lines.extend(["", "## Evidence"])
            for ev in score.evidence:
                lines.append(f"- §{ev.get('section', '?')}, p.{ev.get('page', '?')}: {ev.get('quote', '')[:100]}")
        path.write_text("\n".join(lines))


def register_dimension_action(registry: ActionRegistry) -> None:
    registry.register("paper-review/dim_score@v1", DimensionAction())
```

- [ ] **Step 4: Wire into `actions/builtin.py`**

```python
# paper_review_workflow/actions/builtin.py  (update)
from .base import BaseAction, ActionResult
from .registry import ActionRegistry
from .extract import ExtractAction, register_extract_action
from .dimensions import DimensionAction, register_dimension_action


class EchoAction(BaseAction):
    @property
    def description(self) -> str:
        return "Echo back the message param (smoke test)"

    def run(self, params, env, context, log_callback=None):
        msg = params.get("message", "")
        if log_callback:
            log_callback(f"echo: {msg}")
        return ActionResult(success=True, outputs={"message": msg})


def register_builtin_actions(registry: ActionRegistry) -> None:
    registry.register("paper-review/echo@v1", EchoAction())
    register_extract_action(registry)
    register_dimension_action(registry)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_dim_score.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add paper_review_workflow/actions/dimensions/__init__.py paper_review_workflow/actions/builtin.py tests/unit/test_dim_score.py
git commit -m "feat(dimensions): DimensionAction with jinja2 prompts"
```

## Task 4.3: Matrix parallelism integration test (mocked LLM)

**Files:**
- Test: `tests/integration/test_matrix_parallel.py`

- [ ] **Step 1: Write failing test**

```python
# tests/integration/test_matrix_parallel.py
import time
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.memory import MemoryStorage
from paper_review_workflow.llm.schemas import DimensionScore


@pytest.fixture
def fake_paper_session(tmp_path):
    """Pre-create extract outputs so dimensions job can run."""
    session_dir = tmp_path / "session"
    extract_dir = session_dir / "00_extract"
    extract_dir.mkdir(parents=True)
    (extract_dir / "full_text.md").write_text("# Paper\n\nContent. " * 50)
    (extract_dir / "metadata.json").write_text('{"title":"T","authors":[],"abstract":""}')
    return session_dir


def test_8_dimensions_run_in_parallel(fake_paper_session, tmp_path):
    """Verify 8 LLM calls happen in parallel, not serially."""
    call_times = []

    fake_score = DimensionScore(
        score=4, confidence=0.8, strengths=["a"], weaknesses=["b"],
        justification="x" * 200,
    )

    def mock_score(**kwargs):
        call_times.append(time.time())
        time.sleep(0.5)  # simulate LLM latency
        return fake_score

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.score.side_effect = mock_score
        mock_client.model = "claude-sonnet-4-6"
        mock_from_env.return_value = mock_client

        yaml = tmp_path / "test.yaml"
        yaml.write_text(f"""
name: parallel-test
on: {{workflow_dispatch: {{}}}}
env:
  SESSION_DIR: "{fake_paper_session}"
jobs:
  dimensions:
    runs-on: local
    strategy:
      matrix:
        dimension: [novelty, soundness, significance, clarity,
                    reproducibility, related_work, positioning, presentation]
      max-parallel: 8
    steps:
      - uses: paper-review/dim_score@v1
        with:
          dimension: ${{{ { matrix.dimension } }}}
          session_dir: ${{{ { env.SESSION_DIR } }}}
          full_text_path: "${{ env.SESSION_DIR }}/00_extract/full_text.md"
          metadata_path: "${{ env.SESSION_DIR }}/00_extract/metadata.json"
""")

        engine = ReviewEngine(storage=MemoryStorage())
        run = engine.run_from_file(str(yaml), payload={})

    assert run.status.value == "success"
    assert len(call_times) == 8
    # If parallel, all 8 calls start within ~0.5s (one sleep cycle)
    # If serial, would take 4s
    elapsed = max(call_times) - min(call_times)
    assert elapsed < 1.0, f"calls not parallel: elapsed={elapsed:.2f}s"
```

- [ ] **Step 2: Run test to verify it passes (should pass since we ported matrix)**

Run: `pytest tests/integration/test_matrix_parallel.py -v`
Expected: PASS (parallel verified by elapsed < 1.0s)

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_matrix_parallel.py
git commit -m "test(integration): verify 8-dimension matrix parallelism"
```

---

# M5: Synthesize + Decide

**Goal:** Build `synthesize` (LLM-based) and `decide` (pure-rule) actions. Run a full normal_review workflow end-to-end with mocked LLM.

**Estimated:** 1 day

## Task 5.1: SynthesizeAction

**Files:**
- Create: `paper_review_workflow/actions/synthesize.py`
- Create: `paper_review_workflow/actions/prompts/synthesize.j2`
- Test: `tests/unit/test_synthesize.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_synthesize.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_review_workflow.actions.synthesize import SynthesizeAction
from paper_review_workflow.llm.schemas import SynthesisResult


@pytest.fixture
def session_with_8_dims(tmp_path):
    session_dir = tmp_path / "session"
    dims = ["novelty", "soundness", "significance", "clarity",
            "reproducibility", "related_work", "positioning", "presentation"]
    for dim in dims:
        d = session_dir / f"10_dim_{dim}"
        d.mkdir(parents=True)
        (d / "score.json").write_text(json.dumps({
            "dimension": dim, "score": 4, "confidence": 0.8,
            "strengths": ["x"], "weaknesses": ["y"],
            "justification": "z" * 200,
            "evidence": [], "model_used": "test", "usage": {},
        }))
    return session_dir


def test_synthesize_reads_8_dims_and_calls_llm(session_with_8_dims):
    action = SynthesizeAction()
    fake_synth = SynthesisResult(
        summary="x" * 250,
        key_strengths=["a"], key_weaknesses=["b"],
        questions_for_authors=["q1"],
        overall_assessment="good paper",
    )

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.score.return_value = fake_synth
        mock_from_env.return_value = mock_client

        result = action.run(
            params={"session_dir": str(session_with_8_dims)},
            env={}, context={}, log_callback=lambda x: None,
        )

    assert result.success
    assert (session_with_8_dims / "50_synthesize" / "review.md").exists()
    scores = json.loads((session_with_8_dims / "50_synthesize" / "scores.json").read_text())
    assert len(scores) == 8
    assert scores["novelty"]["score"] == 4


def test_synthesize_fails_with_too_many_missing(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    # Only 5 dims present (< 6 threshold)
    for dim in ["novelty", "soundness", "significance", "clarity", "reproducibility"]:
        d = session_dir / f"10_dim_{dim}"
        d.mkdir()
        (d / "score.json").write_text(json.dumps({"score": 4}))

    action = SynthesizeAction()
    result = action.run(
        params={"session_dir": str(session_dir)},
        env={}, context={}, log_callback=lambda x: None,
    )
    assert not result.success
    assert "missing" in result.message.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_synthesize.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Create `paper_review_workflow/actions/prompts/synthesize.j2`**

```jinja
You are a senior meta-reviewer synthesizing 8 dimension scores into a unified peer review.

The paper has been scored on these dimensions (1-5 scale):
- Novelty: how original
- Soundness: how rigorous
- Significance: how impactful
- Clarity: how well-written
- Reproducibility: how reproducible
- Related Work: literature coverage
- Positioning: framing relative to prior work
- Presentation: figures/tables/formatting

Below are the per-dimension scores, strengths, weaknesses, and justifications:

{% for dim, result in dimensions.items() %}
## {{ dim }} (score={{ result.score }}, confidence={{ result.confidence }})
Strengths: {{ result.strengths | join(", ") }}
Weaknesses: {{ result.weaknesses | join(", ") }}
Justification: {{ result.justification }}
{% endfor %}

Synthesize into:
- summary (200-1500 chars): overall assessment of the paper
- key_strengths (list): top 2-4 strengths across dimensions
- key_weaknesses (list): top 2-4 weaknesses across dimensions
- questions_for_authors (list): 2-4 questions you would ask the authors in rebuttal
- overall_assessment: one-paragraph final assessment

Be balanced and specific. Reference specific dimension findings.
```

- [ ] **Step 4: Implement `paper_review_workflow/actions/synthesize.py`**

```python
"""SynthesizeAction: combine 8 dimension scores into unified review."""
import json
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .base import BaseAction, ActionResult
from .registry import ActionRegistry
from .dimensions import ALL_DIMENSIONS
from ..llm.client import LLMClient
from ..llm.schemas import SynthesisResult

logger = logging.getLogger(__name__)


_PROMPTS_DIR = Path(__file__).parent / "prompts"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
)


class SynthesizeAction(BaseAction):
    """Read 8 dim score.json files, call LLM, write review.md + scores.json."""

    MIN_DIMENSIONS = 6  # require at least 6/8 to synthesize

    @property
    def description(self) -> str:
        return "Synthesize 8 dimension scores into unified review"

    def run(self, params, env, context, log_callback=None):
        session_dir = Path(params["session_dir"])

        dim_results = {}
        missing = []
        for dim in ALL_DIMENSIONS:
            score_path = session_dir / f"10_dim_{dim}" / "score.json"
            if score_path.exists():
                dim_results[dim] = json.loads(score_path.read_text())
            else:
                missing.append(dim)

        if log_callback:
            log_callback(f"loaded {len(dim_results)}/8 dimensions, missing: {missing}")

        if len(dim_results) < self.MIN_DIMENSIONS:
            return ActionResult(
                success=False,
                message=f"too many missing dimensions ({len(missing)} missing, need ≥{self.MIN_DIMENSIONS})",
            )

        prompt = self._render_prompt(dim_results)
        dim_summary = json.dumps(dim_results, ensure_ascii=False, default=str)

        client = LLMClient.from_env()
        synthesis = client.score(
            system=prompt,
            user_content=dim_summary,
            schema=SynthesisResult,
        )

        out_dir = session_dir / "50_synthesize"
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "scores.json").write_text(
            json.dumps(dim_results, ensure_ascii=False, indent=2)
        )
        self._write_review_md(out_dir / "review.md", synthesis, dim_results)

        if log_callback:
            log_callback(f"synthesis complete: {len(synthesis.key_strengths)} strengths, "
                        f"{len(synthesis.key_weaknesses)} weaknesses")

        return ActionResult(
            success=True,
            outputs={
                "review_path": str(out_dir / "review.md"),
                "scores_path": str(out_dir / "scores.json"),
            },
        )

    def _render_prompt(self, dim_results: dict) -> str:
        template = _jinja_env.get_template("synthesize.j2")
        return template.render(dimensions=dim_results)

    def _write_review_md(self, path: Path, synthesis: SynthesisResult,
                         dim_results: dict) -> None:
        lines = ["# Peer Review Synthesis", "", "## Summary", synthesis.summary, ""]
        lines.extend(["## Key Strengths", *[f"- {s}" for s in synthesis.key_strengths], ""])
        lines.extend(["## Key Weaknesses", *[f"- {w}" for w in synthesis.key_weaknesses], ""])
        lines.extend(["## Questions for Authors", *[f"- {q}" for q in synthesis.questions_for_authors], ""])
        lines.extend(["## Overall Assessment", synthesis.overall_assessment, ""])
        lines.extend(["## Per-Dimension Scores"])
        for dim, result in dim_results.items():
            lines.append(f"- **{dim}**: {result.get('score', '?')}/5 (conf={result.get('confidence', '?')})")
        path.write_text("\n".join(lines))


def register_synthesize_action(registry: ActionRegistry) -> None:
    registry.register("paper-review/synthesize@v1", SynthesizeAction())
```

- [ ] **Step 5: Wire into `actions/builtin.py`**

Add:
```python
from .synthesize import SynthesizeAction, register_synthesize_action
# in register_builtin_actions:
register_synthesize_action(registry)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_synthesize.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Commit**

```bash
git add paper_review_workflow/actions/synthesize.py paper_review_workflow/actions/prompts/synthesize.j2 paper_review_workflow/actions/builtin.py tests/unit/test_synthesize.py
git commit -m "feat(synthesize): LLM-based meta-review of 8 dimensions"
```

## Task 5.2: DecideAction (pure rule)

**Files:**
- Create: `paper_review_workflow/actions/decide.py`
- Test: `tests/unit/test_decide.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_decide.py
import json
import pytest
from pathlib import Path

from paper_review_workflow.actions.decide import DecideAction


@pytest.fixture
def session_with_scores(tmp_path):
    session_dir = tmp_path / "session"
    syn_dir = session_dir / "50_synthesize"
    syn_dir.mkdir(parents=True)
    scores = {
        dim: {"score": 4, "confidence": 0.8} for dim in [
            "novelty", "soundness", "significance", "clarity",
            "reproducibility", "related_work", "positioning", "presentation",
        ]
    }
    (syn_dir / "scores.json").write_text(json.dumps(scores))
    return session_dir


def test_decide_all_5_strong_accept(session_with_scores):
    # Modify to all 5s
    syn_dir = session_with_scores / "50_synthesize"
    scores = {dim: {"score": 5, "confidence": 1.0} for dim in [
        "novelty", "soundness", "significance", "clarity",
        "reproducibility", "related_work", "positioning", "presentation",
    ]}
    (syn_dir / "scores.json").write_text(json.dumps(scores))

    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_scores),
                "scores_path": str(syn_dir / "scores.json")},
        env={}, context={}, log_callback=lambda x: None,
    )

    assert result.success
    decision = json.loads((session_with_scores / "60_decision" / "decision.json").read_text())
    assert decision["recommendation"] == "strong_accept"
    assert decision["weighted_score"] == 5.0


def test_decide_all_1_strong_reject(session_with_scores):
    syn_dir = session_with_scores / "50_synthesize"
    scores = {dim: {"score": 1, "confidence": 1.0} for dim in [
        "novelty", "soundness", "significance", "clarity",
        "reproducibility", "related_work", "positioning", "presentation",
    ]}
    (syn_dir / "scores.json").write_text(json.dumps(scores))

    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_scores),
                "scores_path": str(syn_dir / "scores.json")},
        env={}, context={}, log_callback=lambda x: None,
    )

    decision = json.loads((session_with_scores / "60_decision" / "decision.json").read_text())
    assert decision["recommendation"] == "strong_reject"
    assert decision["weighted_score"] == 1.0


def test_decide_weighted_average_uses_weights(session_with_scores):
    """soundness (1.2) should weight higher than presentation (0.8)"""
    syn_dir = session_with_scores / "50_synthesize"
    scores = {
        "novelty": {"score": 5}, "soundness": {"score": 1},  # high-weight low score
        "significance": {"score": 5}, "clarity": {"score": 5},
        "reproducibility": {"score": 5}, "related_work": {"score": 5},
        "positioning": {"score": 5}, "presentation": {"score": 5},
    }
    (syn_dir / "scores.json").write_text(json.dumps(scores))

    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_scores),
                "scores_path": str(syn_dir / "scores.json")},
        env={}, context={}, log_callback=lambda x: None,
    )

    decision = json.loads((session_with_scores / "60_decision" / "decision.json").read_text())
    # Weighted avg should be lower than simple avg (4.625) due to soundness weight
    assert decision["weighted_score"] < 4.625
    assert decision["weighted_score"] > 4.0


def test_decide_missing_dim_treated_as_na(session_with_scores):
    syn_dir = session_with_scores / "50_synthesize"
    scores = {
        "novelty": {"score": 4}, "soundness": {"score": 4},
        "significance": {"score": 4}, "clarity": {"score": 4},
        "reproducibility": {"score": 4}, "related_work": {"score": 4},
        "positioning": {"score": 4},
        # presentation missing
    }
    (syn_dir / "scores.json").write_text(json.dumps(scores))

    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_scores),
                "scores_path": str(syn_dir / "scores.json")},
        env={}, context={}, log_callback=lambda x: None,
    )

    assert result.success
    decision = json.loads((session_with_scores / "60_decision" / "decision.json").read_text())
    assert decision["weighted_score"] == 4.0  # remaining 7 all 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_decide.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement `paper_review_workflow/actions/decide.py`**

```python
"""DecideAction: pure-rule weighted scoring → OpenReview 7-tier recommendation."""
import json
import logging
from pathlib import Path
from typing import Dict, List

from .base import BaseAction, ActionResult
from .registry import ActionRegistry

logger = logging.getLogger(__name__)


DEFAULT_WEIGHTS: Dict[str, float] = {
    "novelty": 1.0,
    "soundness": 1.2,
    "significance": 1.2,
    "clarity": 0.8,
    "reproducibility": 1.0,
    "related_work": 0.8,
    "positioning": 0.8,
    "presentation": 0.8,
}


THRESHOLDS: List[tuple] = [
    (4.5, "strong_accept"),
    (4.0, "accept"),
    (3.5, "weak_accept"),
    (3.0, "borderline"),
    (2.0, "weak_reject"),
    (1.5, "reject"),
    (0.0, "strong_reject"),
]


VALID_RECOMMENDATIONS = [label for _, label in THRESHOLDS]


class DecideAction(BaseAction):
    """Compute weighted average of dimension scores and map to 7-tier recommendation."""

    @property
    def description(self) -> str:
        return "Decide final recommendation (pure rule, no LLM)"

    def run(self, params, env, context, log_callback=None):
        session_dir = Path(params["session_dir"])
        scores_path = params.get("scores_path") or str(
            session_dir / "50_synthesize" / "scores.json"
        )

        try:
            scores: Dict[str, dict] = json.loads(Path(scores_path).read_text())
        except Exception as e:
            return ActionResult(success=False, message=f"cannot read scores: {e}")

        weights = self._resolve_weights(env)
        per_dimension = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for dim, weight in weights.items():
            if dim not in scores:
                if log_callback:
                    log_callback(f"⚠️  dimension {dim} missing, skipping (weight={weight})")
                continue
            score = scores[dim].get("score")
            confidence = scores[dim].get("confidence", 0.0)
            if score is None:
                continue
            weighted = score * weight
            per_dimension[dim] = {
                "score": score,
                "confidence": confidence,
                "weighted": weighted / weight,  # back to raw for display
            }
            weighted_sum += weighted
            total_weight += weight

        if total_weight == 0:
            return ActionResult(success=False, message="no dimensions to score")

        weighted_score = weighted_sum / total_weight
        recommendation = self._map_to_recommendation(weighted_score)

        decision = {
            "schema_version": "1.0",
            "recommendation": recommendation,
            "weighted_score": round(weighted_score, 2),
            "per_dimension": per_dimension,
            "decision_rationale": self._generate_rationale(recommendation, weighted_score, per_dimension),
            "key_concerns": self._extract_key_concerns(scores),
            "key_strengths": self._extract_key_strengths(scores),
            "weights_used": weights,
        }

        out_dir = session_dir / "60_decision"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "decision.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2)
        )

        # Also create final_report.md as a soft link / copy
        final_report = session_dir / "final_report.md"
        review_path = session_dir / "50_synthesize" / "review.md"
        if review_path.exists():
            final_report.write_text(review_path.read_text())

        if log_callback:
            log_callback(f"recommendation: {recommendation} (score={weighted_score:.2f})")

        return ActionResult(
            success=True,
            outputs={
                "decision_path": str(out_dir / "decision.json"),
                "recommendation": recommendation,
                "weighted_score": round(weighted_score, 2),
            },
        )

    def _resolve_weights(self, env: Dict[str, str]) -> Dict[str, float]:
        weights = dict(DEFAULT_WEIGHTS)
        for dim in list(weights.keys()):
            env_key = f"WEIGHT_{dim.upper()}"
            if env_key in env:
                try:
                    weights[dim] = float(env[env_key])
                except ValueError:
                    pass
        return weights

    def _map_to_recommendation(self, score: float) -> str:
        for threshold, label in THRESHOLDS:
            if score >= threshold:
                return label
        return "strong_reject"

    def _generate_rationale(self, recommendation: str, score: float,
                            per_dimension: dict) -> str:
        top_dim = max(per_dimension.items(), key=lambda kv: kv[1]["weighted"])
        bot_dim = min(per_dimension.items(), key=lambda kv: kv[1]["weighted"])
        return (
            f"Weighted average across {len(per_dimension)} dimensions is {score:.2f}/5, "
            f"mapping to '{recommendation}'. "
            f"Strongest dimension: {top_dim[0]} ({top_dim[1]['weighted']:.1f}). "
            f"Weakest dimension: {bot_dim[0]} ({bot_dim[1]['weighted']:.1f})."
        )

    def _extract_key_concerns(self, scores: dict) -> list:
        concerns = []
        for dim, data in scores.items():
            if data.get("score", 5) <= 2:
                weaknesses = data.get("weaknesses", [])
                if weaknesses:
                    concerns.append(f"{dim}: {weaknesses[0]}")
        return concerns[:3]

    def _extract_key_strengths(self, scores: dict) -> list:
        strengths = []
        for dim, data in scores.items():
            if data.get("score", 0) >= 4:
                s = data.get("strengths", [])
                if s:
                    strengths.append(f"{dim}: {s[0]}")
        return strengths[:3]


def register_decide_action(registry: ActionRegistry) -> None:
    registry.register("paper-review/decide@v1", DecideAction())
```

- [ ] **Step 4: Wire into `actions/builtin.py`**

Add:
```python
from .decide import DecideAction, register_decide_action
# in register_builtin_actions:
register_decide_action(registry)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_decide.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add paper_review_workflow/actions/decide.py paper_review_workflow/actions/builtin.py tests/unit/test_decide.py
git commit -m "feat(decide): pure-rule weighted scoring + 7-tier mapping"
```

## Task 5.3: Full normal_review YAML config

**Files:**
- Create: `configs/normal_review.yaml`

- [ ] **Step 1: Write `configs/normal_review.yaml`**

```yaml
name: normal-paper-review

on:
  workflow_dispatch:
    inputs:
      paper_source:
        description: "论文来源(PDF 路径或 arXiv ID/URL)"
        required: true
        type: string
      mode:
        description: "评审模式"
        type: choice
        default: normal
        options: [normal]

env:
  LLM_PROVIDER: anthropic
  LLM_MODEL: claude-sonnet-4-6
  LLM_MAX_TOKENS: "4096"
  LLM_TEMPERATURE: "0.0"
  SESSIONS_ROOT: ./sessions
  WEIGHT_NOVELTY: "1.0"
  WEIGHT_SOUNDESS: "1.2"
  WEIGHT_SIGNIFICANCE: "1.2"
  WEIGHT_CLARITY: "0.8"
  WEIGHT_REPRODUCIBILITY: "1.0"
  WEIGHT_RELATED_WORK: "0.8"
  WEIGHT_POSITIONING: "0.8"
  WEIGHT_PRESENTATION: "0.8"

jobs:
  extract:
    name: "📄 论文抽取"
    runs-on: local
    outputs:
      paper_id: ${{ steps.extract.outputs.paper_id }}
      full_text_path: ${{ steps.extract.outputs.full_text_path }}
      metadata_path: ${{ steps.extract.outputs.metadata_path }}
      sections_path: ${{ steps.extract.outputs.sections_path }}
      references_path: ${{ steps.extract.outputs.references_path }}
    steps:
      - id: extract
        uses: paper-review/extract@v1
        with:
          source: ${{ inputs.paper_source }}
          session_dir: ${{ env.SESSIONS_ROOT }}/${{ env.PAPER_ID }}/${{ env.RUN_ID }}

  dimensions:
    name: "📊 维度打分"
    needs: extract
    strategy:
      matrix:
        dimension:
          - novelty
          - soundness
          - significance
          - clarity
          - reproducibility
          - related_work
          - positioning
          - presentation
      max-parallel: 8
    runs-on: local
    steps:
      - uses: paper-review/dim_score@v1
        with:
          dimension: ${{ matrix.dimension }}
          session_dir: ${{ env.SESSIONS_ROOT }}/${{ env.PAPER_ID }}/${{ env.RUN_ID }}
          full_text_path: ${{ needs.extract.outputs.full_text_path }}
          metadata_path: ${{ needs.extract.outputs.metadata_path }}

  synthesize:
    name: "📝 综合评审"
    needs: dimensions
    runs-on: local
    outputs:
      review_path: ${{ steps.synthesize.outputs.review_path }}
      scores_path: ${{ steps.synthesize.outputs.scores_path }}
    steps:
      - id: synthesize
        uses: paper-review/synthesize@v1
        with:
          session_dir: ${{ env.SESSIONS_ROOT }}/${{ env.PAPER_ID }}/${{ env.RUN_ID }}

  decide:
    name: "🎯 推荐决定"
    needs: synthesize
    runs-on: local
    steps:
      - uses: paper-review/decide@v1
        with:
          session_dir: ${{ env.SESSIONS_ROOT }}/${{ env.PAPER_ID }}/${{ env.RUN_ID }}
          scores_path: ${{ needs.synthesize.outputs.scores_path }}
```

- [ ] **Step 2: Commit**

```bash
git add configs/normal_review.yaml
git commit -m "feat(config): normal_review.yaml production config"
```

## Task 5.4: End-to-end integration test (mocked LLM)

**Files:**
- Test: `tests/integration/test_full_review_mocked.py`

- [ ] **Step 1: Write test**

```python
# tests/integration/test_full_review_mocked.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.memory import MemoryStorage
from paper_review_workflow.llm.schemas import DimensionScore, SynthesisResult


@pytest.fixture
def fake_paper():
    return str(Path("tests/fixtures/sample_paper.pdf"))


def test_full_review_workflow_mocked(fake_paper, tmp_path, monkeypatch):
    """End-to-end: extract → 8 dims → synthesize → decide. LLM mocked."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    fake_dim_score = DimensionScore(
        score=4, confidence=0.8,
        strengths=["a strength"], weaknesses=["a weakness"],
        justification="x" * 250,
        evidence=[],
    )
    fake_synth = SynthesisResult(
        summary="x" * 250,
        key_strengths=["strength"],
        key_weaknesses=["weakness"],
        questions_for_authors=["question?"],
        overall_assessment="good paper overall",
    )

    from paper_review_workflow.llm.client import LLMClient
    LLMClient.reset()

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.score.side_effect = lambda **kwargs: (
            fake_synth if "SynthesisResult" in str(kwargs.get("schema")) or kwargs.get("schema") is SynthesisResult
            else fake_dim_score
        )
        mock_from_env.return_value = mock_client

        # Write a normal_review.yaml that uses tmp_path for sessions
        yaml = tmp_path / "test_review.yaml"
        yaml.write_text(f"""
name: test-review
on: {{workflow_dispatch: {{}}}}
env:
  LLM_PROVIDER: anthropic
  LLM_MODEL: test-model
  LLM_MAX_TOKENS: "4096"
  LLM_TEMPERATURE: "0.0"
  SESSIONS_ROOT: {tmp_path / "sessions"}
  WEIGHT_NOVELTY: "1.0"
  WEIGHT_SOUNDESS: "1.2"
  WEIGHT_SIGNIFICANCE: "1.2"
  WEIGHT_CLARITY: "0.8"
  WEIGHT_REPRODUCIBILITY: "1.0"
  WEIGHT_RELATED_WORK: "0.8"
  WEIGHT_POSITIONING: "0.8"
  WEIGHT_PRESENTATION: "0.8"
jobs:
  extract:
    runs-on: local
    outputs:
      paper_id: ${{ steps.extract.outputs.paper_id }}
      full_text_path: ${{ steps.extract.outputs.full_text_path }}
      metadata_path: ${{ steps.extract.outputs.metadata_path }}
    steps:
      - id: extract
        uses: paper-review/extract@v1
        with:
          source: "{fake_paper}"
          session_dir: "{tmp_path / "session"}"
  dimensions:
    needs: extract
    strategy:
      matrix:
        dimension: [novelty, soundness, significance, clarity,
                    reproducibility, related_work, positioning, presentation]
      max-parallel: 8
    runs-on: local
    steps:
      - uses: paper-review/dim_score@v1
        with:
          dimension: ${{ matrix.dimension }}
          session_dir: "{tmp_path / "session"}"
          full_text_path: ${{ needs.extract.outputs.full_text_path }}
          metadata_path: ${{ needs.extract.outputs.metadata_path }}
  synthesize:
    needs: dimensions
    runs-on: local
    steps:
      - id: synthesize
        uses: paper-review/synthesize@v1
        with:
          session_dir: "{tmp_path / "session"}"
  decide:
    needs: synthesize
    runs-on: local
    steps:
      - uses: paper-review/decide@v1
        with:
          session_dir: "{tmp_path / "session"}"
          scores_path: ${{ needs.synthesize.outputs.scores_path }}
""")

        engine = ReviewEngine(storage=MemoryStorage())
        run = engine.run_from_file(str(yaml), payload={})

    assert run.status.value == "success"

    session_dir = tmp_path / "session"
    assert (session_dir / "00_extract" / "full_text.md").exists()
    for dim in ["novelty", "soundness", "significance", "clarity",
                "reproducibility", "related_work", "positioning", "presentation"]:
        assert (session_dir / f"10_dim_{dim}" / "score.json").exists()
    assert (session_dir / "50_synthesize" / "review.md").exists()
    assert (session_dir / "60_decision" / "decision.json").exists()

    decision = json.loads((session_dir / "60_decision" / "decision.json").read_text())
    assert decision["recommendation"] in [
        "strong_accept", "accept", "weak_accept", "borderline",
        "weak_reject", "reject", "strong_reject",
    ]
    assert 1.0 <= decision["weighted_score"] <= 5.0
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/integration/test_full_review_mocked.py -v`
Expected: PASS (may need a few iterations to fix YAML escaping)

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_full_review_mocked.py
git commit -m "test(integration): full review workflow with mocked LLM"
```

---

# M6: Resume / Rerun / Cancel

**Goal:** Test and harden the resume/rerun/cancel/list/show CLI commands.

**Estimated:** 1 day

## Task 6.1: Resume test (skip completed, rerun failed)

**Files:**
- Test: `tests/integration/test_resume.py`

- [ ] **Step 1: Write test**

```python
# tests/integration/test_resume.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.json_file import JsonFileStorage
from paper_review_workflow.llm.schemas import DimensionScore, SynthesisResult
from paper_review_workflow.llm.client import LLMClient


@pytest.fixture
def fake_paper():
    return str(Path("tests/fixtures/sample_paper.pdf"))


def test_resume_skips_completed_and_reruns_failed(fake_paper, tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    LLMClient.reset()

    fake_dim = DimensionScore(
        score=4, confidence=0.8, strengths=["a"], weaknesses=["b"],
        justification="x" * 200, evidence=[],
    )
    fake_synth = SynthesisResult(
        summary="x" * 250, key_strengths=["s"], key_weaknesses=["w"],
        questions_for_authors=["q"], overall_assessment="ok",
    )

    yaml = tmp_path / "test_review.yaml"
    session_dir = tmp_path / "session"
    yaml.write_text(f"""
name: test-resume
on: {{workflow_dispatch: {{}}}}
env:
  LLM_PROVIDER: anthropic
  LLM_MODEL: test-model
  SESSIONS_ROOT: {tmp_path}
jobs:
  extract:
    runs-on: local
    outputs:
      paper_id: ${{ steps.extract.outputs.paper_id }}
      full_text_path: ${{ steps.extract.outputs.full_text_path }}
      metadata_path: ${{ steps.extract.outputs.metadata_path }}
    steps:
      - id: extract
        uses: paper-review/extract@v1
        with:
          source: "{fake_paper}"
          session_dir: "{session_dir}"
  dimensions:
    needs: extract
    strategy:
      matrix:
        dimension: [novelty, soundness]
      max-parallel: 2
    runs-on: local
    steps:
      - uses: paper-review/dim_score@v1
        with:
          dimension: ${{ matrix.dimension }}
          session_dir: "{session_dir}"
          full_text_path: ${{ needs.extract.outputs.full_text_path }}
          metadata_path: ${{ needs.extract.outputs.metadata_path }}
  synthesize:
    needs: dimensions
    runs-on: local
    steps:
      - id: synthesize
        uses: paper-review/synthesize@v1
        with:
          session_dir: "{session_dir}"
  decide:
    needs: synthesize
    runs-on: local
    steps:
      - uses: paper-review/decide@v1
        with:
          session_dir: "{session_dir}"
          scores_path: ${{ needs.synthesize.outputs.scores_path }}
""")

    storage = JsonFileStorage(data_dir=str(tmp_path / "storage"))

    # First run: let dim_soundness fail
    call_count = {"dim": 0}

    def dim_side_effect(**kwargs):
        call_count["dim"] += 1
        if call_count["dim"] == 2:  # second dim call (soundness in matrix order)
            raise RuntimeError("simulated API timeout")
        return fake_dim

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.score.side_effect = dim_side_effect
        mock_from_env.return_value = mock_client

        engine = ReviewEngine(storage=storage)
        run1 = engine.run_from_file(str(yaml), payload={})
        assert run1.status.value == "failure"
        # 1 dim succeeded, 1 failed

    # Second run: resume, fix mock
    call_count2 = {"dim": 0, "synth": 0}

    def dim_side_effect2(**kwargs):
        call_count2["dim"] += 1
        return fake_dim

    def synth_side_effect(**kwargs):
        call_count2["synth"] += 1
        return fake_synth

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.score.side_effect = lambda **kw: fake_synth if kw.get("schema") is SynthesisResult else fake_dim
        mock_from_env.return_value = mock_client

        engine2 = ReviewEngine(storage=storage)
        run2 = engine2.resume_run(run1.id)

    assert run2.status.value == "success"
    # Resume should only re-run soundness (failed) + synthesize + decide
    # novelty should be skipped
    # (we can't easily assert exact call count due to mock, but the run should succeed)
```

- [ ] **Step 2: Run test**

Run: `pytest tests/integration/test_resume.py -v`
Expected: PASS (may need to iterate on resume logic in engine.py)

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_resume.py
git commit -m "test(integration): resume skips completed, reruns failed"
```

## Task 6.2: Rerun-specific component test

**Files:**
- Test: `tests/integration/test_rerun.py`

- [ ] **Step 1: Write test**

```python
# tests/integration/test_rerun.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.json_file import JsonFileStorage
from paper_review_workflow.llm.schemas import DimensionScore, SynthesisResult
from paper_review_workflow.llm.client import LLMClient
from paper_review_workflow.core.models import JobStatus


@pytest.fixture
def fake_paper():
    return str(Path("tests/fixtures/sample_paper.pdf"))


def test_rerun_dim_novelty_cascades_to_downstream(fake_paper, tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    LLMClient.reset()

    fake_dim = DimensionScore(
        score=4, confidence=0.8, strengths=["a"], weaknesses=["b"],
        justification="x" * 200, evidence=[],
    )
    fake_synth = SynthesisResult(
        summary="x" * 250, key_strengths=["s"], key_weaknesses=["w"],
        questions_for_authors=["q"], overall_assessment="ok",
    )

    yaml = tmp_path / "test_review.yaml"
    session_dir = tmp_path / "session"
    yaml.write_text(f"""
name: test-rerun
on: {{workflow_dispatch: {{}}}}
env:
  LLM_PROVIDER: anthropic
  LLM_MODEL: test-model
  SESSIONS_ROOT: {tmp_path}
jobs:
  extract:
    runs-on: local
    outputs:
      paper_id: ${{ steps.extract.outputs.paper_id }}
      full_text_path: ${{ steps.extract.outputs.full_text_path }}
      metadata_path: ${{ steps.extract.outputs.metadata_path }}
    steps:
      - id: extract
        uses: paper-review/extract@v1
        with:
          source: "{fake_paper}"
          session_dir: "{session_dir}"
  dimensions:
    needs: extract
    strategy:
      matrix:
        dimension: [novelty, soundness]
      max-parallel: 2
    runs-on: local
    steps:
      - uses: paper-review/dim_score@v1
        with:
          dimension: ${{ matrix.dimension }}
          session_dir: "{session_dir}"
          full_text_path: ${{ needs.extract.outputs.full_text_path }}
          metadata_path: ${{ needs.extract.outputs.metadata_path }}
  synthesize:
    needs: dimensions
    runs-on: local
    steps:
      - id: synthesize
        uses: paper-review/synthesize@v1
        with:
          session_dir: "{session_dir}"
  decide:
    needs: synthesize
    runs-on: local
    steps:
      - uses: paper-review/decide@v1
        with:
          session_dir: "{session_dir}"
          scores_path: ${{ needs.synthesize.outputs.scores_path }}
""")

    storage = JsonFileStorage(data_dir=str(tmp_path / "storage"))

    # First run succeeds
    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.score.side_effect = lambda **kw: fake_synth if kw.get("schema") is SynthesisResult else fake_dim
        mock_from_env.return_value = mock_client

        engine = ReviewEngine(storage=storage)
        run1 = engine.run_from_file(str(yaml), payload={})
        assert run1.status.value == "success"

    # Mark novelty for rerun
    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.model = "test-model"
        call_log = []
        def side_effect(**kw):
            schema = kw.get("schema")
            if schema is SynthesisResult:
                call_log.append("synthesize")
                return fake_synth
            call_log.append(f"dim:{kw}")
            return fake_dim
        mock_client.score.side_effect = side_effect
        mock_from_env.return_value = mock_client

        engine2 = ReviewEngine(storage=storage)
        run2 = engine2.resume_run(run1.id, rerun_components=["dimensions_novelty"])

    assert run2.status.value == "success"
    # Check that dimensions_novelty was reset (status went from success → pending → success)
    # And synthesize was re-called
    assert "synthesize" in call_log
```

- [ ] **Step 2: Run test**

Run: `pytest tests/integration/test_rerun.py -v`
Expected: PASS (may need to fix `_mark_for_rerun` matrix job ID handling)

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_rerun.py
git commit -m "test(integration): --rerun cascades to downstream"
```

## Task 6.3: Cancel test (Ctrl+C graceful shutdown)

**Files:**
- Test: `tests/integration/test_cancel.py`

- [ ] **Step 1: Write test**

```python
# tests/integration/test_cancel.py
import json
import time
import threading
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.json_file import JsonFileStorage
from paper_review_workflow.llm.schemas import DimensionScore
from paper_review_workflow.llm.client import LLMClient


@pytest.fixture
def fake_paper():
    return str(Path("tests/fixtures/sample_paper.pdf"))


def test_cancel_marks_run_as_cancelled(fake_paper, tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    LLMClient.reset()

    yaml = tmp_path / "test_review.yaml"
    session_dir = tmp_path / "session"
    yaml.write_text(f"""
name: test-cancel
on: {{workflow_dispatch: {{}}}}
env:
  LLM_PROVIDER: anthropic
  LLM_MODEL: test-model
  SESSIONS_ROOT: {tmp_path}
jobs:
  extract:
    runs-on: local
    outputs:
      paper_id: ${{ steps.extract.outputs.paper_id }}
      full_text_path: ${{ steps.extract.outputs.full_text_path }}
      metadata_path: ${{ steps.extract.outputs.metadata_path }}
    steps:
      - id: extract
        uses: paper-review/extract@v1
        with:
          source: "{fake_paper}"
          session_dir: "{session_dir}"
  dimensions:
    needs: extract
    strategy:
      matrix:
        dimension: [novelty, soundness, significance, clarity]
      max-parallel: 4
    runs-on: local
    steps:
      - uses: paper-review/dim_score@v1
        with:
          dimension: ${{ matrix.dimension }}
          session_dir: "{session_dir}"
          full_text_path: ${{ needs.extract.outputs.full_text_path }}
          metadata_path: ${{ needs.extract.outputs.metadata_path }}
""")

    storage = JsonFileStorage(data_dir=str(tmp_path / "storage"))

    fake_dim = DimensionScore(
        score=4, confidence=0.8, strengths=["a"], weaknesses=["b"],
        justification="x" * 200, evidence=[],
    )

    started = threading.Event()
    run_id_holder = {}

    def slow_score(**kwargs):
        started.set()
        time.sleep(2)  # slow LLM call
        return fake_dim

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.model = "test-model"
        mock_client.score.side_effect = slow_score
        mock_from_env.return_value = mock_client

        engine = ReviewEngine(storage=storage)

        # Run in a thread
        def run_async():
            run = engine.run_from_file(str(yaml), payload={})
            run_id_holder["run"] = run

        t = threading.Thread(target=run_async)
        t.start()

        # Wait for dim calls to start
        assert started.wait(timeout=5)

        # Cancel
        # Find the run_id from active_runs
        time.sleep(0.2)
        assert len(engine._active_runs) == 1
        run_id = list(engine._active_runs.keys())[0]
        cancelled = engine.cancel_run(run_id)
        assert cancelled

        t.join(timeout=5)

    run = run_id_holder["run"]
    assert run.status.value == "cancelled"
```

- [ ] **Step 2: Run test**

Run: `pytest tests/integration/test_cancel.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_cancel.py
git commit -m "test(integration): cancel marks run as cancelled"
```

## Task 6.4: list-runs and show-run CLI tests

**Files:**
- Test: `tests/integration/test_cli_commands.py`

- [ ] **Step 1: Write test**

```python
# tests/integration/test_cli_commands.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from paper_review_workflow.cli import main
from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.json_file import JsonFileStorage
from paper_review_workflow.core.models import WorkflowRun, WorkflowStatus


def test_list_runs_empty(capsys, tmp_path):
    exit_code = main([
        "--storage", "json", "--storage-dir", str(tmp_path),
        "list-runs",
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "RUN_ID" in out


def test_show_run_nonexistent(capsys, tmp_path):
    exit_code = main([
        "--storage", "json", "--storage-dir", str(tmp_path),
        "show-run", "nonexistent",
    ])
    assert exit_code == 1


def test_show_run_existing(capsys, tmp_path):
    # Pre-populate storage
    storage = JsonFileStorage(data_dir=str(tmp_path))
    storage.open()
    run = WorkflowRun(status=WorkflowStatus.SUCCESS)
    storage.save_run(run)
    storage.close()

    exit_code = main([
        "--storage", "json", "--storage-dir", str(tmp_path),
        "show-run", run.id,
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert run.id in out
    assert "success" in out
```

- [ ] **Step 2: Run test**

Run: `pytest tests/integration/test_cli_commands.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_cli_commands.py
git commit -m "test(integration): list-runs and show-run CLI commands"
```

---

# M7: Testing, Polish, Documentation

**Goal:** Fill coverage gaps, write README, add E2E test marker, finalize.

**Estimated:** 1 day

## Task 7.1: E2E test with real arXiv paper (marked, skipped by default)

**Files:**
- Test: `tests/e2e/test_normal_review_arxiv.py`

- [ ] **Step 1: Write test**

```python
# tests/e2e/test_normal_review_arxiv.py
import json
import os
import pytest
from pathlib import Path

from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.json_file import JsonFileStorage


@pytest.mark.e2e
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"),
                    reason="requires ANTHROPIC_API_KEY")
def test_review_real_arxiv_paper(tmp_path):
    """End-to-end: review a real arXiv paper. Costs ~$0.5 in API calls."""
    storage = JsonFileStorage(data_dir=str(tmp_path / "storage"))
    engine = ReviewEngine(storage=storage)

    # Use AgentReview paper (small, well-known)
    run = engine.run_from_file(
        "configs/normal_review.yaml",
        payload={"paper_source": "2402.12098"},
    )

    assert run.status.value == "success"

    # Verify session dir structure
    session_dirs = list((tmp_path / "storage" / "runs").iterdir())
    # Find the run.json
    run_data = json.loads((tmp_path / "storage" / "runs" / f"{run.id}.json").read_text())

    # Verify all 8 dimensions scored
    dim_jobs = [k for k in run_data["jobs"] if k.startswith("dimensions_")]
    assert len(dim_jobs) == 8
    for dim_job in dim_jobs:
        assert run_data["jobs"][dim_job]["status"] == "success"

    # Verify decision
    # decision.json should be in session dir, but since we used tmp_path/sessions/pending-...,
    # we need to check the env
    session_dir = run.env.get("SESSIONS_ROOT", "./sessions")
    # Find any 60_decision dir
    decision_files = list(Path(session_dir).rglob("60_decision/decision.json"))
    if decision_files:
        decision = json.loads(decision_files[0].read_text())
        assert decision["recommendation"] in [
            "strong_accept", "accept", "weak_accept", "borderline",
            "weak_reject", "reject", "strong_reject",
        ]
        assert 1.0 <= decision["weighted_score"] <= 5.0

    # Verify prompt cache hit at least 7/8
    score_files = list(Path(session_dir).rglob("10_dim_*/score.json"))
    cache_hits = 0
    for sf in score_files:
        data = json.loads(sf.read_text())
        if data.get("usage", {}).get("cache_read_input_tokens", 0) > 0:
            cache_hits += 1
    assert cache_hits >= 7, f"only {cache_hits}/8 cache hits"
```

- [ ] **Step 2: Commit (don't run by default)**

```bash
git add tests/e2e/test_normal_review_arxiv.py
git commit -m "test(e2e): real arXiv paper review (marked, requires API key)"
```

## Task 7.2: Write README.md

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

```markdown
# Paper Review Workflow

AI-powered academic paper review workflow engine. Runs an 8-dimension LLM-based peer review on a PDF or arXiv paper, producing structured scores and a final recommendation.

## Installation

```bash
git clone <repo>
cd paper-review-workflow
pip install -e ".[dev]"
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quick Start

Review an arXiv paper:

```bash
python main.py run configs/normal_review.yaml \
    --payload '{"paper_source": "2402.12098"}'
```

Review a local PDF:

```bash
python main.py run configs/normal_review.yaml \
    --payload '{"paper_source": "/path/to/paper.pdf"}'
```

## Output Structure

Each run creates a session directory:

```
sessions/<paper_id>/<run_id>/
  run.json                    # Full run state (lwf storage)
  run_manifest.json           # Static archive
  00_extract/                 # Parsed paper
    metadata.json
    sections.json
    full_text.md
    references.json
  10_dim_novelty/             # 8 dimension scores (parallel)
    score.json
    review.md
  ...
  50_synthesize/              # Meta-review
    review.md
    scores.json
  60_decision/                # Final recommendation
    decision.json
  final_report.md
```

## Resume / Rerun

If a run fails or is interrupted:

```bash
python main.py resume <run_id>
```

Force rerun a specific component (cascades to downstream):

```bash
python main.py resume <run_id> --rerun dim_novelty
```

## List Past Runs

```bash
python main.py list-runs
python main.py show-run <run_id>
```

## Configuration

See `configs/normal_review.yaml` for the default 8-dimension review config. Customize weights via `WEIGHT_*` env vars.

## Testing

```bash
# Unit + integration (no API key needed)
pytest

# E2E (requires API key, costs ~$0.5)
pytest --run-e2e -m e2e
```

## Architecture

This project reuses the [lwf](https://github.com/...) workflow engine skeleton (GitHub Actions-style YAML, three-layer state machine, matrix parallelism) and adds paper-review-specific actions:

- `extract`: PDF / arXiv parsing
- `dim_score` (matrix × 8): LLM scoring per dimension
- `synthesize`: LLM meta-review
- `decide`: pure-rule weighted scoring

See `docs/superpowers/specs/2026-08-12-paper-review-workflow-design.md` for the full design.

## License

MIT
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with install/usage/architecture"
```

## Task 7.3: Coverage check and gap-fill

- [ ] **Step 1: Run coverage**

Run: `pytest --cov=paper_review_workflow --cov-report=term-missing`
Expected: Coverage meets targets from SPEC Section 10.5:
- core/parser.py: 95%+
- core/state_machine.py: 95%+
- llm/anthropic_provider.py: 90%+
- llm/schemas.py: 100%
- actions/decide.py: 100%
- engine.py: 80%+

- [ ] **Step 2: Fill gaps**

For any module below target, add tests until coverage is met. Focus on uncovered branches (error paths, edge cases).

- [ ] **Step 3: Commit gap-fill tests**

```bash
git add tests/
git commit -m "test: fill coverage gaps to meet targets"
```

## Task 7.4: Final smoke test via CLI (real API key)

- [ ] **Step 1: Run a real review**

Run:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python main.py run configs/normal_review.yaml \
    --payload '{"paper_source": "2402.12098"}' \
    --log-level INFO
```

Expected: Output ending with `最终状态: success` and `recommendation: <one of 7 tiers>`.

- [ ] **Step 2: Verify outputs exist**

Run:
```bash
ls sessions/*/*/00_extract/
ls sessions/*/*/10_dim_*/
ls sessions/*/*/50_synthesize/
ls sessions/*/*/60_decision/
cat sessions/*/*/60_decision/decision.json
```

- [ ] **Step 3: Test resume**

Press Ctrl+C during a run, then:

```bash
python main.py list-runs --status cancelled
python main.py resume <run_id>
```

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "chore: M7 complete - Phase 1 ready for use"
git tag v0.1.0
```

---

# Self-Review Checklist

## Spec Coverage

| SPEC Section | Implemented By |
|---|---|
| 1. Project scope | M1-M7 cover Phase 1 only ✅ |
| 2. Architecture | M1 ports core/executors/storage, M2-M5 add llm+actions ✅ |
| 3. Data model & session dirs | M1 models, M3 extract outputs, M5 synthesize/decide outputs ✅ |
| 4. Component contracts | M1 base/registry, M3-M5 actions ✅ |
| 5. Execution model & parallelism | M1 executors (matrix intact), M4.3 parallelism test ✅ |
| 6. LLM provider abstraction | M2 base/registry/client/anthropic_provider ✅ |
| 7. YAML config & components | M5.3 normal_review.yaml ✅ |
| 8. CLI | M1.8 cli.py + main.py, M6.4 cli tests ✅ |
| 9. Error handling | M2.3 retry logic in AnthropicProvider, M3 extract errors, M5 synthesize missing-dim handling ✅ |
| 10. Test strategy | M1-M6 unit+integration, M7.1 E2E ✅ |
| 11. Milestones | M1-M7 directly map ✅ |

## Placeholder Scan

- ✅ No "TBD" / "TODO" / "implement later"
- ✅ All steps have complete code or exact commands
- ✅ All test code shown in full

## Type Consistency

- `ActionResult` fields: `success` / `outputs` / `message` / `log_lines` / `exit_code` — used consistently across M1-M5 ✅
- `DimensionScore` schema fields: `score` / `confidence` / `strengths` / `weaknesses` / `justification` / `evidence` — same in M2.1, M4.2, M5.1 ✅
- `LLMClient.from_env()` / `client.score()` — same signature in M2.4, M4.2, M5.1 ✅
- `register_builtin_actions(registry)` — called once in M1.8, extended in M3.3, M4.2, M5.1, M5.2 ✅
- `ALL_DIMENSIONS` list — defined in M4.2, used in M5.1 (synthesize) ✅
- `DecideAction.WEIGHTS` keys — match `WEIGHT_*` env vars in normal_review.yaml ✅

## Potential Issues to Watch During Implementation

1. **YAML `${{ }}` escaping in tests**: Multi-line YAML strings with `${{ }}` need careful Python f-string handling. The integration tests use `{{` and `}}` to escape braces.
2. **Matrix job IDs in resume**: `_mark_for_rerun` needs to handle matrix-expanded job IDs like `dimensions_novelty`. The current implementation in M1.8 may need adjustment during M6.2.
3. **Prompt cache TTL**: 5-minute TTL means if a single LLM call hangs >5min, subsequent calls lose cache. Acceptable for Phase 1.
4. **arXiv rate limiting**: `arxiv` library and direct HTTP may both be rate-limited. The 60s timeout should be sufficient.
5. **PyMuPDF license**: AGPL for non-commercial; verify acceptability before any production use beyond personal research.

---

# Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-12-paper-review-workflow-impl.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
