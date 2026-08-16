"""DeepSeek LLM provider — reuses openai SDK with different base_url.
Uses json_object + schema-in-prompt (DeepSeek doesn't support json_schema)."""
import json
import random
import time
import logging
import os
from typing import Optional, Type, Union

from pydantic import BaseModel, ValidationError

from .base import (
    LLMProvider, LLMResponse, LLMError,
    RateLimitError, ContextLengthError, SchemaValidationError,
)

logger = logging.getLogger(__name__)


class DeepSeekProvider(LLMProvider):
    provider_name = "deepseek"
    BASE_URL = "https://api.deepseek.com"
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 1.0
    MAX_BACKOFF = 30.0

    def __init__(self, api_key: Optional[str] = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)

    @classmethod
    def from_env(cls) -> "DeepSeekProvider":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise LLMError("DEEPSEEK_API_KEY not set")
        return cls(api_key=api_key)

    def complete(self, system, messages, model, max_tokens,
                 temperature=0.0, response_schema=None, cached_context=None):
        system_text = self._build_system_text(system, cached_context, response_schema)

        response_format = None
        if response_schema:
            response_format = {"type": "json_object"}

        openai_messages = [{"role": "system", "content": system_text}] + messages

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=openai_messages,
                    response_format=response_format,
                )
                return self._parse_response(resp, response_schema, model)
            except Exception as e:
                if self._is_rate_limit(e):
                    if attempt == self.MAX_RETRIES - 1:
                        raise RateLimitError(f"rate limit after {self.MAX_RETRIES} attempts")
                    self._sleep_backoff(attempt)
                elif self._is_context_length(e):
                    raise ContextLengthError(str(e))
                elif self._is_retryable(e):
                    if attempt < self.MAX_RETRIES - 1:
                        self._sleep_backoff(attempt)
                    else:
                        raise
                else:
                    raise

    def _build_system_text(self, system, cached_context, response_schema=None):
        parts = []
        if cached_context:
            parts.append(cached_context)
        if isinstance(system, str):
            if system:
                parts.append(system)
        else:
            parts.extend(system)
        # Append schema description for DeepSeek (doesn't support json_schema natively)
        if response_schema:
            parts.append(
                "You must respond with JSON matching this schema:\n" +
                json.dumps(response_schema.model_json_schema(), indent=2)
            )
        return "\n\n".join(parts)

    def _parse_response(self, resp, schema, model):
        text_content = resp.choices[0].message.content
        structured = None
        if schema:
            try:
                data = json.loads(text_content)
                structured = schema(**data)
            except (json.JSONDecodeError, ValidationError) as e:
                raise SchemaValidationError(f"LLM output failed schema validation: {e}")
        usage = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        return LLMResponse(
            text=text_content if not schema else None,
            structured=structured,
            usage=usage,
            model=model,
            raw=resp,
        )

    def _is_rate_limit(self, e):
        try:
            from openai import RateLimitError as OpenAIRLE
            return isinstance(e, OpenAIRLE)
        except ImportError:
            return False

    def _is_context_length(self, e):
        try:
            from openai import BadRequestError
            if isinstance(e, BadRequestError) and "context_length" in str(e).lower():
                return True
        except ImportError:
            pass
        return False

    def _is_retryable(self, e):
        try:
            from openai import APIStatusError, APIConnectionError
            if isinstance(e, APIConnectionError):
                return True
            if isinstance(e, APIStatusError) and e.status_code in (500, 503):
                return True
        except ImportError:
            pass
        return False

    def _sleep_backoff(self, attempt):
        backoff = min(self.INITIAL_BACKOFF * (2 ** attempt), self.MAX_BACKOFF)
        jitter = random.uniform(0, backoff * 0.1)
        time.sleep(backoff + jitter)
