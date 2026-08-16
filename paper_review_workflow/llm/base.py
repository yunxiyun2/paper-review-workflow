"""LLM provider abstraction."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Type, Union
from pydantic import BaseModel


@dataclass
class LLMResponse:
    text: Optional[str]
    structured: Optional[BaseModel]
    usage: dict
    model: str
    raw: Any = None


class LLMError(Exception):
    """Base LLM error."""


class RateLimitError(LLMError):
    """HTTP 429 after retries exhausted."""


class ContextLengthError(LLMError):
    """Context length exceeded."""


class SchemaValidationError(LLMError):
    """LLM output did not match the schema."""


class LLMProvider(ABC):
    provider_name: str = ""

    @classmethod
    @abstractmethod
    def from_env(cls) -> "LLMProvider":
        """Read provider-specific env vars and construct the provider.
        Each subclass implements this to read its own API key + config."""
        ...

    @abstractmethod
    def complete(
        self,
        system: Union[str, list],
        messages: list,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        response_schema: Optional[Type[BaseModel]] = None,
        cached_context: Optional[str] = None,
    ) -> LLMResponse:
        ...
