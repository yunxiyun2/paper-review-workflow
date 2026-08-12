"""Built-in actions registration."""
from .base import BaseAction, ActionResult
from .registry import ActionRegistry
from .extract import ExtractAction, register_extract_action
from .dimensions import DimensionAction, register_dimension_action
from .synthesize import SynthesizeAction, register_synthesize_action


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
    register_synthesize_action(registry)
