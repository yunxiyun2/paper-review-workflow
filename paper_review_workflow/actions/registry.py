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
        if base_name in self._actions:
            return self._actions[base_name]
        # Fall back: match any registered key with the same version-stripped base
        for key, action in self._actions.items():
            if key.split("@")[0] == base_name:
                return action
        return None

    def list_actions(self) -> list:
        return [{"name": k, "description": v.description}
                for k, v in self._actions.items()]

    def clear(self) -> None:
        """Test-only: clear the singleton state."""
        self._actions.clear()
