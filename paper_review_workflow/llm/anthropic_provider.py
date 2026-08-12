"""Anthropic LLM provider with retry, prompt cache, and tool-use structured output."""
import random
import time
import logging
from typing import Optional

import anthropic
from pydantic import ValidationError

from .base import (
    LLMProvider, LLMResponse,
    RateLimitError, ContextLengthError, SchemaValidationError,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    MAX_RETRIES = 3
    INITIAL_BACKOFF = 1.0
    MAX_BACKOFF = 30.0

    def __init__(self, api_key: Optional[str] = None):
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system, messages, model, max_tokens,
                 temperature=0.0, response_schema=None, cached_context=None):
        system_blocks = self._build_system_blocks(system, cached_context)
        tools, tool_choice = self._build_tools(response_schema)

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_blocks,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                return self._parse_response(resp, response_schema, model)

            except anthropic.RateLimitError:
                if attempt == self.MAX_RETRIES - 1:
                    raise RateLimitError(f"rate limit after {self.MAX_RETRIES} attempts")
                self._sleep_backoff(attempt)

            except anthropic.APIStatusError as e:
                if e.status_code == 400 and "context_length" in str(e).lower():
                    raise ContextLengthError(str(e))
                if e.status_code in (500, 503) and attempt < self.MAX_RETRIES - 1:
                    self._sleep_backoff(attempt)
                else:
                    raise

            except anthropic.APIConnectionError:
                if attempt < self.MAX_RETRIES - 1:
                    self._sleep_backoff(attempt)
                else:
                    raise

    def _build_system_blocks(self, system, cached_context):
        blocks = []
        if cached_context:
            blocks.append({
                "type": "text",
                "text": cached_context,
                "cache_control": {"type": "ephemeral"},
            })
        if isinstance(system, str):
            if system:
                blocks.append({"type": "text", "text": system})
        else:
            blocks.extend(system)
        return blocks

    def _build_tools(self, schema):
        if schema is None:
            return None, None
        return [{
            "name": "submit_result",
            "description": "Submit the structured result",
            "input_schema": schema.model_json_schema(),
        }], {"type": "tool", "name": "submit_result"}

    def _parse_response(self, resp, schema, model):
        structured = None
        if schema:
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    try:
                        structured = schema(**block.input)
                    except ValidationError as e:
                        raise SchemaValidationError(
                            f"LLM output failed schema validation: {e}"
                        )
                    break
            if structured is None:
                raise SchemaValidationError("No tool_use block in response")

        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        }

        text = None
        if not schema:
            text = "".join(getattr(b, "text", "") for b in resp.content
                           if getattr(b, "type", None) == "text") or None

        return LLMResponse(
            text=text,
            structured=structured,
            usage=usage,
            model=model,
            raw=resp,
        )

    def _sleep_backoff(self, attempt: int) -> None:
        backoff = min(self.INITIAL_BACKOFF * (2 ** attempt), self.MAX_BACKOFF)
        jitter = random.uniform(0, backoff * 0.1)
        time.sleep(backoff + jitter)
