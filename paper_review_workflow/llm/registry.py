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
