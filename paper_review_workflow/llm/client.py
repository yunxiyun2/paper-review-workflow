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
            registry = ProviderRegistry()
            provider_cls = registry.get(provider_name)
            provider = provider_cls.from_env()

            # Anthropic has default model; others require LLM_MODEL
            model = os.environ.get("LLM_MODEL")
            if not model:
                if provider_name == "anthropic":
                    model = "claude-sonnet-4-6"
                else:
                    from .base import LLMError
                    raise LLMError(f"LLM_MODEL not set (required for provider '{provider_name}')")

            cls._instance = cls(
                provider=provider,
                model=model,
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
