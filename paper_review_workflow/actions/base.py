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
