"""Anthropic LLM provider. Stub, will be implemented in Task M2.3."""
from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def complete(self, **kwargs):
        raise NotImplementedError("AnthropicProvider not yet implemented")
