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
