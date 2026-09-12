"""LLMClient: convenience layer delegating to a provider chosen by env."""
import os
from typing import Dict, Optional, Type, Union
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
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "LLMClient":
        """Build a client from an env mapping (defaults to os.environ).

        The os.environ path keeps the process-wide singleton; a per-run `env`
        mapping (e.g. workflow env with user-supplied API keys) always builds a
        fresh instance so concurrent runs never leak credentials into each other.
        """
        if env is None and cls._instance is not None:
            return cls._instance
        src = env if env is not None else os.environ
        provider_name = src.get("LLM_PROVIDER", "anthropic")
        registry = ProviderRegistry()
        provider_cls = registry.get(provider_name)
        provider = provider_cls.from_env(src)

        # Anthropic/zhipu have default models; others require LLM_MODEL
        model = src.get("LLM_MODEL") or getattr(provider_cls, "DEFAULT_MODEL", "")
        if not model:
            if provider_name == "anthropic":
                model = "claude-sonnet-4-6"
            else:
                from .base import LLMError
                raise LLMError(f"LLM_MODEL not set (required for provider '{provider_name}')")

        client = cls(
            provider=provider,
            model=model,
            max_tokens=int(src.get("LLM_MAX_TOKENS", "4096")),
            temperature=float(src.get("LLM_TEMPERATURE", "0.0")),
        )
        if env is None:
            cls._instance = client
        return client

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
