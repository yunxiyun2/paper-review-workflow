from paper_review_workflow.actions.registry import ActionRegistry
from paper_review_workflow.actions.base import BaseAction, ActionResult
from paper_review_workflow.core.parser import WorkflowParser
from paper_review_workflow.core.event_bus import EventBus
from paper_review_workflow.core.models import WorkflowRun
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
