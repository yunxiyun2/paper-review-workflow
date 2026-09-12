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
    MAX_RETRIES = 5
    INITIAL_BACKOFF = 3.0
    MAX_BACKOFF = 60.0

    def __init__(self, api_key: Optional[str] = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)

    @classmethod
    def from_env(cls, env=None) -> "DeepSeekProvider":
        import os
        src = env if env is not None else os.environ
        api_key = src.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise LLMError("DEEPSEEK_API_KEY not set")
        return cls(api_key=api_key)

    def complete(self, system, messages, model, max_tokens,
                 temperature=0.0, response_schema=None, cached_context=None):
        system_text = self._build_system_text(system, cached_context, response_schema)

        response_format = None
        if response_schema:
            response_format = {"type": "json_object"}

        repair_note = ""
        for attempt in range(self.MAX_RETRIES):
            try:
                openai_messages = [{"role": "system", "content": system_text + repair_note}] + messages
                resp = self._client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=openai_messages,
                    response_format=response_format,
                )
                return self._parse_response(resp, response_schema, model)
            except SchemaValidationError as e:
                # Self-repair loop: feed the validation error back and let the
                # model correct its own output (covers over-length fields,
                # schema echo, truncated/fenced JSON...).
                if attempt >= self.MAX_RETRIES - 1:
                    raise
                repair_note = (
                    "\n\nYour previous response FAILED validation with this error:\n"
                    + str(e)
                    + "\nRespond again with a single corrected JSON object that strictly "
                    "matches the schema. Fix the reported problem (e.g. keep string fields "
                    "within their length limits). Output the JSON object only."
                )
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
        # Append schema description for DeepSeek/Zhipu (no native json_schema support).
        # Strip "title" keys first — they are pure noise and models sometimes echo
        # schema metadata back instead of producing content.
        if response_schema:
            schema = self._strip_keys(response_schema.model_json_schema(), {"title"})
            parts.append(
                "OUTPUT CONTRACT: Respond with a single JSON object that conforms to this "
                "JSON schema. Output the JSON object itself — never repeat or echo the schema.\n"
                "JSON schema:\n" + json.dumps(schema, indent=2)
            )
        return "\n\n".join(parts)

    @staticmethod
    def _strip_keys(node, keys_to_remove):
        if isinstance(node, dict):
            return {k: DeepSeekProvider._strip_keys(v, keys_to_remove)
                    for k, v in node.items() if k not in keys_to_remove}
        if isinstance(node, list):
            return [DeepSeekProvider._strip_keys(v, keys_to_remove) for v in node]
        return node

    def _parse_response(self, resp, schema, model):
        message = resp.choices[0].message
        text_content = (message.content or "").strip()
        structured = None
        if schema:
            if not text_content:
                raise SchemaValidationError(
                    "model returned empty content"
                    + ("" if not getattr(message, "reasoning_content", None)
                       else " (reasoning present but no final content — "
                            "try a larger max_tokens so thinking does not exhaust the budget)")
                )
            data = json.loads(self._extract_json(text_content))
            try:
                structured = schema(**data)
            except ValidationError as e:
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

    @staticmethod
    def _extract_json(text: str) -> str:
        """Pull the JSON object out of a model response: strips markdown code
        fences and, failing that, slices from the first '{' to the last '}'."""
        import re
        t = text.strip()
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
        try:
            json.loads(t)
            return t
        except json.JSONDecodeError:
            start, end = t.find("{"), t.rfind("}")
            if start != -1 and end > start:
                candidate = t[start:end + 1]
                json.loads(candidate)  # raise JSONDecodeError if still broken
                return candidate
            raise

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
