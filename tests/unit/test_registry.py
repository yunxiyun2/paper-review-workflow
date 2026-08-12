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
