import json
import threading
import time
from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.memory import MemoryStorage
from paper_review_workflow.actions.base import BaseAction, ActionResult
from paper_review_workflow.actions.registry import ActionRegistry
from paper_review_workflow.core.models import WorkflowStatus


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


class SlowAction(BaseAction):
    """Action that sleeps so we can test cancellation mid-execution"""
    def run(self, params, env, context, log_callback=None):
        time.sleep(2)
        return ActionResult(success=True, outputs={"done": True})


def test_cancel_run_works_mid_execution(tmp_path):
    """C1 fix: cancel_run must find the coordinator during execution"""
    ActionRegistry._instance = None

    engine = ReviewEngine(storage=MemoryStorage())

    # Register a slow action
    engine.registry.register("paper-review/slow@v1", SlowAction())

    yaml_path = tmp_path / "slow.yaml"
    yaml_path.write_text("""
name: slow-test
on: {workflow_dispatch: {}}
jobs:
  slow:
    runs-on: local
    steps:
      - id: s
        uses: paper-review/slow@v1
""")

    run_id_holder = {}

    def run_in_thread():
        run = engine.run_from_file(str(yaml_path), payload={})
        run_id_holder["run"] = run

    t = threading.Thread(target=run_in_thread)
    t.start()

    # Wait for execution to start
    time.sleep(0.3)
    assert len(engine._active_runs) == 1, "run should be active"
    run_id = list(engine._active_runs.keys())[0]

    # Cancel -- this should find the coordinator and cancel
    cancelled = engine.cancel_run(run_id)
    assert cancelled, "cancel_run should return True for active run"

    t.join(timeout=5)
    run = run_id_holder["run"]
    assert run.status.value == "cancelled", f"expected cancelled, got {run.status.value}"


def test_extra_env_overrides(tmp_path):
    """I1 fix: --env KEY=VALUE should override run env"""
    ActionRegistry._instance = None

    engine = ReviewEngine(storage=MemoryStorage())

    yaml_path = tmp_path / "env_test.yaml"
    yaml_path.write_text("""
name: env-test
on: {workflow_dispatch: {}}
env:
  DEFAULT_VAR: default
  OVERRIDE_VAR: original
jobs:
  echo:
    runs-on: local
    steps:
      - id: e
        uses: paper-review/echo@v1
""")

    run = engine.run_from_file(
        str(yaml_path),
        payload={},
        extra_env={"OVERRIDE_VAR": "overridden", "NEW_VAR": "added"},
    )

    assert run.env["DEFAULT_VAR"] == "default"
    assert run.env["OVERRIDE_VAR"] == "overridden"
    assert run.env["NEW_VAR"] == "added"

