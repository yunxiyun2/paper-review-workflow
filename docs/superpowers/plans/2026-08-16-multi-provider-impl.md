# Multi LLM Provider (Phase 2 #2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenAI and DeepSeek LLM providers so users can switch between Anthropic/OpenAI/DeepSeek via environment variables — no code changes needed.

**Architecture:** Each provider implements `from_env()` classmethod to read its own API key. `LLMClient.from_env()` delegates to `provider_cls.from_env()`. OpenAI uses `json_schema` response format (strict). DeepSeek uses `json_object` + schema-in-prompt (weaker but compatible). Both reuse the `openai` SDK; DeepSeek just sets a different `base_url`. Retry logic is shared across all 3 providers (3x exponential backoff). Anthropic keeps default model `claude-sonnet-4-6`; OpenAI/DeepSeek require `LLM_MODEL` env var.

**Tech Stack:** Python 3.10+, `openai>=1.0` SDK (used by both OpenAI + DeepSeek), Pydantic v2, pytest with mocking.

**Reference SPEC:** `docs/superpowers/specs/2026-08-16-multi-provider-design.md`

---

## File Structure Overview

```
paper_review_workflow/llm/
├── base.py                     # Modify: add from_env() abstractmethod
├── registry.py                 # Modify: register openai + deepseek
├── client.py                   # Modify: from_env() delegates to provider_cls.from_env()
├── anthropic_provider.py       # Modify: add from_env() classmethod
├── openai_provider.py          # ★ Create: OpenAI provider (json_schema)
├── deepseek_provider.py        # ★ Create: DeepSeek provider (json_object + base_url)
├── schemas.py                  # No change
└── prompts/                    # No change

tests/unit/
├── test_anthropic_provider.py  # Modify: add from_env tests
├── test_openai_provider.py      # ★ Create
├── test_deepseek_provider.py    # ★ Create
└── test_llm_client.py           # Modify: multi-provider from_env tests

tests/integration/
├── test_venue_review_openai.py     # ★ Create
└── test_venue_review_deepseek.py   # ★ Create

tests/e2e/
├── test_e2e_openai_arxiv.py       # ★ Create (marked)
└── test_e2e_deepseek_arxiv.py     # ★ Create (marked)

pyproject.toml                # Modify: add openai>=1.0
```

---

# M1: base.py + AnthropicProvider from_env + LLMClient Refactor

**Goal:** Add `from_env()` abstractmethod to `LLMProvider`, implement it on `AnthropicProvider`, refactor `LLMClient.from_env()` to delegate to `provider_cls.from_env()` instead of hardcoding Anthropic logic.

**Estimated:** 0.5 day

## Task 1.1: Add `from_env()` to `LLMProvider` ABC + `AnthropicProvider`

**Files:**
- Modify: `paper_review_workflow/llm/base.py`
- Modify: `paper_review_workflow/llm/anthropic_provider.py`
- Modify: `tests/unit/test_anthropic_provider.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_anthropic_provider.py`:

```python
def test_anthropic_from_env_reads_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    from paper_review_workflow.llm.anthropic_provider import AnthropicProvider
    provider = AnthropicProvider.from_env()
    assert provider is not None
    assert provider.provider_name == "anthropic"


def test_anthropic_from_env_missing_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from paper_review_workflow.llm.base import LLMError
    from paper_review_workflow.llm.anthropic_provider import AnthropicProvider
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY not set"):
        AnthropicProvider.from_env()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_anthropic_provider.py::test_anthropic_from_env_reads_api_key -v`
Expected: FAIL with `AttributeError: type object 'AnthropicProvider' has no attribute 'from_env'`

- [ ] **Step 3: Add `from_env()` abstractmethod to `LLMProvider` in `base.py`**

Read `paper_review_workflow/llm/base.py`. In the `LLMProvider` class, after the `provider_name` class attribute and before the `complete()` abstractmethod, add:

```python
    @classmethod
    @abstractmethod
    def from_env(cls) -> "LLMProvider":
        """Read provider-specific env vars and construct the provider.
        Each subclass implements this to read its own API key + config."""
        ...
```

Make sure `abstractmethod` is imported from `abc` (should already be).

- [ ] **Step 4: Add `from_env()` classmethod to `AnthropicProvider`**

Read `paper_review_workflow/llm/anthropic_provider.py`. After the `__init__` method, add:

```python
    @classmethod
    def from_env(cls) -> "AnthropicProvider":
        """Read ANTHROPIC_API_KEY from environment and construct provider."""
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY not set")
        return cls(api_key=api_key)
```

Make sure `LLMError` is imported from `.base` (should already be in the existing imports).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_anthropic_provider.py -v`
Expected: All tests pass (existing tests + 2 new from_env tests)

- [ ] **Step 6: Commit**

```bash
git add paper_review_workflow/llm/base.py paper_review_workflow/llm/anthropic_provider.py tests/unit/test_anthropic_provider.py
git commit -m "feat(llm): add from_env() abstractmethod + AnthropicProvider.from_env()"
```

## Task 1.2: Refactor `LLMClient.from_env()` to use `provider_cls.from_env()`

**Files:**
- Modify: `paper_review_workflow/llm/client.py`
- Modify: `tests/unit/test_llm_client.py`

- [ ] **Step 1: Read current `LLMClient.from_env()`**

Run: `cat paper_review_workflow/llm/client.py`

The current implementation hardcodes Anthropic logic:
```python
api_key = os.environ.get("ANTHROPIC_API_KEY") if provider_name == "anthropic" else None
provider = provider_cls(api_key=api_key) if provider_name == "anthropic" else provider_cls()
```

- [ ] **Step 2: Write failing tests**

Append to `tests/unit/test_llm_client.py`:

```python
def test_from_env_openai_provider(monkeypatch):
    """LLMClient.from_env() should use OpenAIProvider.from_env()"""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    LLMClient.reset()
    client = LLMClient.from_env()
    assert client.model == "gpt-4o"


def test_from_env_deepseek_provider(monkeypatch):
    """LLMClient.from_env() should use DeepSeekProvider.from_env()"""
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    LLMClient.reset()
    client = LLMClient.from_env()
    assert client.model == "deepseek-chat"


def test_from_env_openai_missing_llm_model_raises(monkeypatch):
    """LLM_MODEL required for non-anthropic providers"""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    LLMClient.reset()
    from paper_review_workflow.llm.base import LLMError
    with pytest.raises(LLMError, match="LLM_MODEL not set"):
        LLMClient.from_env()


def test_from_env_anthropic_keeps_default_model(monkeypatch):
    """Anthropic still has default model claude-sonnet-4-6"""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    LLMClient.reset()
    client = LLMClient.from_env()
    assert client.model == "claude-sonnet-4-6"
```

Note: These tests will FAIL until OpenAIProvider and DeepSeekProvider are created (M2/M3). For now, only `test_from_env_anthropic_keeps_default_model` and the existing `test_from_env_openai_provider` (which mocks `from_env`) will work. The `test_from_env_openai_missing_llm_model_raises` test needs the provider registered.

Actually, the `test_from_env_openai_provider` and `test_from_env_deepseek_provider` tests above call `provider_cls.from_env()` which doesn't exist yet for OpenAI/DeepSeek. These tests will be addressed in M4 after both providers exist.

For now, skip the openai/deepseek tests — they'll be re-enabled in M4. Just write `test_from_env_anthropic_keeps_default_model` and `test_from_env_openai_missing_llm_model_raises` (which will also fail until openai is registered).

Simplify: only write tests that can pass now:

```python
def test_from_env_anthropic_keeps_default_model(monkeypatch):
    """Anthropic still has default model claude-sonnet-4-6 even without LLM_MODEL"""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    LLMClient.reset()
    client = LLMClient.from_env()
    assert client.model == "claude-sonnet-4-6"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/unit/test_llm_client.py::test_from_env_anthropic_keeps_default_model -v`
Expected: FAIL (current `from_env` doesn't have default model logic — it uses `os.environ.get("LLM_MODEL", "claude-sonnet-4-6")` which is already a default, but the test checks that even without LLM_MODEL set, it works. This should actually PASS with the current code since the default is already `claude-sonnet-4-6`. But the refactored code should NOT hardcode the default — it should conditionally apply it only for anthropic.)

Actually the current code already has `os.environ.get("LLM_MODEL", "claude-sonnet-4-6")` which means it always defaults to claude-sonnet-4-6 regardless of provider. This is wrong for openai/deepseek. The refactored code should:
- Anthropic: default to `claude-sonnet-4-6` if LLM_MODEL not set
- OpenAI/DeepSeek: raise error if LLM_MODEL not set

So the test `test_from_env_anthropic_keeps_default_model` should PASS even with current code. Let me write a test that FAILS with current code:

```python
def test_from_env_openai_missing_llm_model_raises(monkeypatch):
    """LLM_MODEL required for non-anthropic providers (not hardcoded default)"""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    LLMClient.reset()
    from paper_review_workflow.llm.base import LLMError
    with pytest.raises(LLMError, match="LLM_MODEL not set"):
        LLMClient.from_env()
```

This will FAIL because:
1. OpenAIProvider doesn't exist yet (provider_cls.from_env() will fail)
2. The current code doesn't raise LLMError for missing LLM_MODEL

Both issues will be fixed when we refactor `from_env()`. But OpenAIProvider needs to be registered first. So let's register a stub OpenAIProvider that raises NotImplementedError for now, or just write the test and let it fail.

Actually, the simplest approach: write the test, refactor `LLMClient.from_env()`, and the test will fail because `ProviderRegistry.get("openai")` raises `KeyError` (provider not registered). We'll register it in M4.

For now, let's just refactor `LLMClient.from_env()` and verify the anthropic tests still pass:

- [ ] **Step 4: Refactor `LLMClient.from_env()`**

Read `paper_review_workflow/llm/client.py`. Replace the `from_env()` classmethod with:

```python
    @classmethod
    def from_env(cls) -> "LLMClient":
        if cls._instance is None:
            provider_name = os.environ.get("LLM_PROVIDER", "anthropic")
            registry = ProviderRegistry()
            provider_cls = registry.get(provider_name)
            provider = provider_cls.from_env()  # Delegate to provider

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
```

- [ ] **Step 5: Run existing tests to verify no regressions**

Run: `pytest tests/unit/test_llm_client.py -v`
Expected: All existing tests pass. The new `test_from_env_anthropic_keeps_default_model` should pass.

- [ ] **Step 6: Run full suite**

Run: `pytest tests/ --ignore=tests/unit/test_extract_arxiv.py -k "not test_extract_arxiv_id" 2>&1 | tail -5`
Expected: All tests pass (no regressions)

- [ ] **Step 7: Commit**

```bash
git add paper_review_workflow/llm/client.py tests/unit/test_llm_client.py
git commit -m "refactor(llm): LLMClient.from_env() delegates to provider_cls.from_env()"
```

---

# M2: OpenAIProvider

**Goal:** Implement `OpenAIProvider` with `from_env()`, `complete()`, `json_schema` response format, retry logic, and prompt cache (automatic, no cache_control).

**Estimated:** 1 day

## Task 2.1: Add `openai` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read current pyproject.toml**

Run: `cat pyproject.toml`

- [ ] **Step 2: Add openai dependency**

Edit `pyproject.toml` — in the `dependencies` array, add `"openai>=1.0"` after the existing entries:

```toml
dependencies = [
    "anthropic>=0.40.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "pymupdf>=1.24.0",
    "arxiv>=2.1.0",
    "httpx>=0.27.0",
    "jinja2>=3.1.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "websockets>=13.0",
    "openai>=1.0",
]
```

- [ ] **Step 3: Install**

Run: `pip install -e ".[dev]"`
Expected: Successfully installs openai

- [ ] **Step 4: Verify import**

Run: `python -c "from openai import OpenAI; print('openai SDK OK')"`
Expected: `openai SDK OK`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore(deps): add openai>=1.0 for OpenAI/DeepSeek providers"
```

## Task 2.2: Implement `OpenAIProvider`

**Files:**
- Create: `paper_review_workflow/llm/openai_provider.py`
- Test: `tests/unit/test_openai_provider.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_openai_provider.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
import json

from paper_review_workflow.llm.openai_provider import OpenAIProvider
from paper_review_workflow.llm.base import LLMError, RateLimitError, ContextLengthError, SchemaValidationError
from paper_review_workflow.llm.schemas import DimensionScore


def _make_openai_response(content_text, prompt_tokens=100, completion_tokens=50):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content_text
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


def _valid_score_json():
    return json.dumps({
        "score": 7,
        "confidence": 0.85,
        "strengths": ["rigorous proof"],
        "weaknesses": ["limited baselines"],
        "justification": "x" * 250,
        "evidence": [],
    })


def test_openai_from_env_reads_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAIProvider.from_env()
    assert provider is not None
    assert provider.provider_name == "openai"


def test_openai_from_env_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY not set"):
        OpenAIProvider.from_env()


@patch("openai.OpenAI")
def test_openai_complete_with_structured_output(mock_cls):
    mock_client = mock_cls.return_value
    mock_client.chat.completions.create.return_value = _make_openai_response(_valid_score_json())

    provider = OpenAIProvider(api_key="sk-test")
    result = provider.complete(
        system="score soundness",
        messages=[{"role": "user", "content": "paper text"}],
        model="gpt-4o",
        max_tokens=4096,
        response_schema=DimensionScore,
    )

    assert result.structured.score == 7
    assert result.structured.confidence == 0.85
    assert result.usage["input_tokens"] == 100
    assert result.usage["output_tokens"] == 50
    assert result.model == "gpt-4o"

    # Verify response_format was json_schema
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"]["type"] == "json_schema"
    assert call_kwargs["response_format"]["json_schema"]["strict"] is True


@patch("openai.OpenAI")
def test_openai_complete_cached_context_in_system(mock_cls):
    mock_client = mock_cls.return_value
    mock_client.chat.completions.create.return_value = _make_openai_response(_valid_score_json())

    provider = OpenAIProvider(api_key="sk-test")
    provider.complete(
        system="score novelty",
        messages=[{"role": "user", "content": "..."}],
        model="gpt-4o",
        max_tokens=4096,
        response_schema=DimensionScore,
        cached_context="PAPER FULL TEXT",
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    system_msg = call_kwargs["messages"][0]
    assert "PAPER FULL TEXT" in system_msg["content"]
    assert system_msg["role"] == "system"


@patch("openai.OpenAI")
def test_openai_retry_on_rate_limit(mock_cls, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    mock_client = mock_cls.return_value
    from openai import RateLimitError as OpenAIRateLimitError

    success_resp = _make_openai_response(_valid_score_json())
    mock_client.chat.completions.create.side_effect = [
        OpenAIRateLimitError(
            message="429",
            response=MagicMock(status_code=429),
            body=None,
        ),
        success_resp,
    ]

    provider = OpenAIProvider(api_key="sk-test")
    result = provider.complete(
        system="t", messages=[], model="gpt-4o", max_tokens=4096,
        response_schema=DimensionScore,
    )
    assert mock_client.chat.completions.create.call_count == 2
    assert result.structured.score == 7


@patch("openai.OpenAI")
def test_openai_rate_limit_exhausted(mock_cls, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    mock_client = mock_cls.return_value
    from openai import RateLimitError as OpenAIRateLimitError
    mock_client.chat.completions.create.side_effect = OpenAIRateLimitError(
        message="429", response=MagicMock(status_code=429), body=None
    )

    provider = OpenAIProvider(api_key="sk-test")
    with pytest.raises(RateLimitError):
        provider.complete(system="t", messages=[], model="gpt-4o", max_tokens=4096,
                         response_schema=DimensionScore)
    assert mock_client.chat.completions.create.call_count == 3


@patch("openai.OpenAI")
def test_openai_schema_validation_failure(mock_cls):
    mock_client = mock_cls.return_value
    # score=99 violates schema (1-10 for NeurIPS, but DimensionScore is 1-5)
    mock_client.chat.completions.create.return_value = _make_openai_response(
        json.dumps({"score": 99, "confidence": 0.5, "strengths": ["a"],
                    "weaknesses": ["b"], "justification": "x" * 200})
    )

    provider = OpenAIProvider(api_key="sk-test")
    with pytest.raises(SchemaValidationError):
        provider.complete(system="t", messages=[], model="gpt-4o", max_tokens=4096,
                         response_schema=DimensionScore)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_openai_provider.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `paper_review_workflow/llm/openai_provider.py`**

```python
"""OpenAI LLM provider with json_schema structured output + automatic prompt cache."""
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


class OpenAIProvider(LLMProvider):
    provider_name = "openai"
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 1.0
    MAX_BACKOFF = 30.0

    def __init__(self, api_key: Optional[str] = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)

    @classmethod
    def from_env(cls) -> "OpenAIProvider":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY not set")
        return cls(api_key=api_key)

    def complete(self, system, messages, model, max_tokens,
                 temperature=0.0, response_schema=None, cached_context=None):
        system_text = self._build_system_text(system, cached_context)

        response_format = None
        if response_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "submit_result",
                    "schema": response_schema.model_json_schema(),
                    "strict": True,
                }
            }

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

    def _build_system_text(self, system, cached_context):
        parts = []
        if cached_context:
            parts.append(cached_context)
        if isinstance(system, str):
            if system:
                parts.append(system)
        else:
            parts.extend(system)
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_openai_provider.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/llm/openai_provider.py tests/unit/test_openai_provider.py
git commit -m "feat(llm): OpenAIProvider with json_schema + retry + auto cache"
```

---

# M3: DeepSeekProvider

**Goal:** Implement `DeepSeekProvider` — reuses `openai` SDK with different `base_url`, uses `json_object` response format + schema-in-prompt (since DeepSeek doesn't support `json_schema`).

**Estimated:** 0.5 day

## Task 3.1: Implement `DeepSeekProvider`

**Files:**
- Create: `paper_review_workflow/llm/deepseek_provider.py`
- Test: `tests/unit/test_deepseek_provider.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_deepseek_provider.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
import json

from paper_review_workflow.llm.deepseek_provider import DeepSeekProvider
from paper_review_workflow.llm.base import LLMError, RateLimitError, SchemaValidationError
from paper_review_workflow.llm.schemas import DimensionScore


def _make_openai_response(content_text, prompt_tokens=100, completion_tokens=50):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content_text
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


def _valid_score_json():
    return json.dumps({
        "score": 7,
        "confidence": 0.85,
        "strengths": ["rigorous proof"],
        "weaknesses": ["limited baselines"],
        "justification": "x" * 250,
        "evidence": [],
    })


def test_deepseek_from_env_reads_api_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    provider = DeepSeekProvider.from_env()
    assert provider is not None
    assert provider.provider_name == "deepseek"


def test_deepseek_from_env_missing_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(LLMError, match="DEEPSEEK_API_KEY not set"):
        DeepSeekProvider.from_env()


@patch("openai.OpenAI")
def test_deepseek_complete_with_json_object_format(mock_cls):
    """DeepSeek uses json_object + schema in prompt (not json_schema)"""
    mock_client = mock_cls.return_value
    mock_client.chat.completions.create.return_value = _make_openai_response(_valid_score_json())

    provider = DeepSeekProvider(api_key="sk-test")
    result = provider.complete(
        system="score soundness",
        messages=[{"role": "user", "content": "paper text"}],
        model="deepseek-chat",
        max_tokens=4096,
        response_schema=DimensionScore,
    )

    assert result.structured.score == 7
    assert result.structured.confidence == 0.85

    # Verify response_format was json_object (NOT json_schema)
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"]["type"] == "json_object"


@patch("openai.OpenAI")
def test_deepseek_schema_embedded_in_system_prompt(mock_cls):
    """Schema description should be appended to system prompt for DeepSeek"""
    mock_client = mock_cls.return_value
    mock_client.chat.completions.create.return_value = _make_openai_response(_valid_score_json())

    provider = DeepSeekProvider(api_key="sk-test")
    provider.complete(
        system="score soundness",
        messages=[{"role": "user", "content": "..."}],
        model="deepseek-chat",
        max_tokens=4096,
        response_schema=DimensionScore,
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    system_content = call_kwargs["messages"][0]["content"]
    # System prompt should contain "JSON matching this schema"
    assert "JSON matching this schema" in system_content
    # And should contain the schema properties (e.g., "score", "confidence")
    assert "score" in system_content
    assert "confidence" in system_content


@patch("openai.OpenAI")
def test_deepseek_cached_context_in_system(mock_cls):
    mock_client = mock_cls.return_value
    mock_client.chat.completions.create.return_value = _make_openai_response(_valid_score_json())

    provider = DeepSeekProvider(api_key="sk-test")
    provider.complete(
        system="score novelty",
        messages=[{"role": "user", "content": "..."}],
        model="deepseek-chat",
        max_tokens=4096,
        response_schema=DimensionScore,
        cached_context="PAPER FULL TEXT",
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    system_content = call_kwargs["messages"][0]["content"]
    assert "PAPER FULL TEXT" in system_content


@patch("openai.OpenAI")
def test_deepseek_schema_validation_failure(mock_cls):
    mock_client = mock_cls.return_value
    mock_client.chat.completions.create.return_value = _make_openai_response(
        json.dumps({"score": 99, "confidence": 0.5, "strengths": ["a"],
                    "weaknesses": ["b"], "justification": "x" * 200})
    )

    provider = DeepSeekProvider(api_key="sk-test")
    with pytest.raises(SchemaValidationError):
        provider.complete(system="t", messages=[], model="deepseek-chat", max_tokens=4096,
                         response_schema=DimensionScore)


@patch("openai.OpenAI")
def test_deepseek_uses_correct_base_url(mock_cls):
    """DeepSeekProvider should set base_url to https://api.deepseek.com"""
    DeepSeekProvider(api_key="sk-test")
    # The OpenAI client should have been called with base_url
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["base_url"] == "https://api.deepseek.com"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_deepseek_provider.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `paper_review_workflow/llm/deepseek_provider.py`**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_deepseek_provider.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/llm/deepseek_provider.py tests/unit/test_deepseek_provider.py
git commit -m "feat(llm): DeepSeekProvider with json_object + schema-in-prompt"
```

---

# M4: Registry + LLMClient Integration

**Goal:** Register OpenAI/DeepSeek in `ProviderRegistry`, add LLMClient multi-provider tests, verify switching works.

**Estimated:** 0.5 day

## Task 4.1: Register providers in ProviderRegistry

**Files:**
- Modify: `paper_review_workflow/llm/registry.py`
- Test: `tests/unit/test_llm_base.py` (existing test file for registry)

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_llm_base.py`:

```python
def test_registry_has_3_providers():
    """ProviderRegistry should auto-register anthropic, openai, deepseek"""
    ProviderRegistry._instance = None
    reg = ProviderRegistry()
    providers = reg.list_providers()
    assert "anthropic" in providers
    assert "openai" in providers
    assert "deepseek" in providers
    assert len(providers) == 3


def test_registry_get_openai_provider():
    ProviderRegistry._instance = None
    reg = ProviderRegistry()
    provider_cls = reg.get("openai")
    assert provider_cls.provider_name == "openai"


def test_registry_get_deepseek_provider():
    ProviderRegistry._instance = None
    reg = ProviderRegistry()
    provider_cls = reg.get("deepseek")
    assert provider_cls.provider_name == "deepseek"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_llm_base.py::test_registry_has_3_providers -v`
Expected: FAIL (only anthropic registered currently)

- [ ] **Step 3: Modify `paper_review_workflow/llm/registry.py`**

Read `paper_review_workflow/llm/registry.py`. In the `_register_builtin()` method, add openai + deepseek:

```python
    def _register_builtin(self) -> None:
        from .anthropic_provider import AnthropicProvider
        self.register("anthropic", AnthropicProvider)
        from .openai_provider import OpenAIProvider
        self.register("openai", OpenAIProvider)
        from .deepseek_provider import DeepSeekProvider
        self.register("deepseek", DeepSeekProvider)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_llm_base.py -v`
Expected: PASS (all tests including 3 new)

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/llm/registry.py tests/unit/test_llm_base.py
git commit -m "feat(llm): register OpenAI + DeepSeek in ProviderRegistry"
```

## Task 4.2: LLMClient multi-provider integration tests

**Files:**
- Modify: `tests/unit/test_llm_client.py`

- [ ] **Step 1: Write multi-provider tests**

Append to `tests/unit/test_llm_client.py`:

```python
def test_from_env_openai_provider(monkeypatch):
    """LLMClient.from_env() should use OpenAIProvider.from_env()"""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    LLMClient.reset()
    client = LLMClient.from_env()
    assert client.model == "gpt-4o"
    assert client._provider.provider_name == "openai"


def test_from_env_deepseek_provider(monkeypatch):
    """LLMClient.from_env() should use DeepSeekProvider.from_env()"""
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    LLMClient.reset()
    client = LLMClient.from_env()
    assert client.model == "deepseek-chat"
    assert client._provider.provider_name == "deepseek"


def test_from_env_openai_missing_api_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    LLMClient.reset()
    from paper_review_workflow.llm.base import LLMError
    with pytest.raises(LLMError, match="OPENAI_API_KEY not set"):
        LLMClient.from_env()


def test_from_env_deepseek_missing_api_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    LLMClient.reset()
    from paper_review_workflow.llm.base import LLMError
    with pytest.raises(LLMError, match="DEEPSEEK_API_KEY not set"):
        LLMClient.from_env()


def test_from_env_openai_missing_llm_model_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    LLMClient.reset()
    from paper_review_workflow.llm.base import LLMError
    with pytest.raises(LLMError, match="LLM_MODEL not set"):
        LLMClient.from_env()


def test_from_env_anthropic_keeps_default_model(monkeypatch):
    """Anthropic still has default model claude-sonnet-4-6 even without LLM_MODEL"""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    LLMClient.reset()
    client = LLMClient.from_env()
    assert client.model == "claude-sonnet-4-6"
    assert client._provider.provider_name == "anthropic"
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/unit/test_llm_client.py -v`
Expected: PASS (all tests including 6 new multi-provider tests)

- [ ] **Step 3: Run full suite**

Run: `pytest tests/ --ignore=tests/unit/test_extract_arxiv.py -k "not test_extract_arxiv_id" 2>&1 | tail -5`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_llm_client.py
git commit -m "test(llm): multi-provider LLMClient.from_env() integration tests"
```

---

# M5: Integration Tests + E2E + README + Tag

**Goal:** Integration tests (mock LLM, OpenAI + DeepSeek venue review), E2E tests (marked, real API), README update, tag v0.5.0.

**Estimated:** 1 day

## Task 5.1: Venue review integration test (OpenAI mock)

**Files:**
- Create: `tests/integration/test_venue_review_openai.py`

- [ ] **Step 1: Write test**

Create `tests/integration/test_venue_review_openai.py`:

```python
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.memory import MemoryStorage
from paper_review_workflow.llm.client import LLMClient
from paper_review_workflow.llm.base import LLMResponse


@pytest.fixture
def mock_openai_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    LLMClient.reset()

    fake_score = MagicMock()
    fake_score.score = 7
    fake_score.confidence = 0.85
    fake_score.strengths = ["rigorous"]
    fake_score.weaknesses = ["limited"]
    fake_score.justification = "x" * 250
    fake_score.evidence = []

    fake_synth = MagicMock()
    fake_synth.summary = "x" * 250
    fake_synth.key_strengths = ["strong"]
    fake_synth.key_weaknesses = ["weak"]
    fake_synth.questions_for_authors = ["why?"]
    fake_synth.overall_assessment = "good"

    fake_dim_resp = MagicMock(spec=LLMResponse)
    fake_dim_resp.structured = fake_score
    fake_dim_resp.usage = {"input_tokens": 100, "output_tokens": 50,
                           "cache_creation_input_tokens": 0,
                           "cache_read_input_tokens": 0}
    fake_dim_resp.model = "gpt-4o"

    fake_synth_resp = MagicMock(spec=LLMResponse)
    fake_synth_resp.structured = fake_synth
    fake_synth_resp.usage = {"input_tokens": 500, "output_tokens": 200,
                             "cache_creation_input_tokens": 0,
                             "cache_read_input_tokens": 0}
    fake_synth_resp.model = "gpt-4o"

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.model = "gpt-4o"
        def side_effect(**kw):
            schema = kw.get("response_schema")
            if hasattr(schema, "__name__") and "Synthesis" in schema.__name__:
                return fake_synth_resp
            return fake_dim_resp
        mock_client.complete.side_effect = side_effect
        mock_from_env.return_value = mock_client
        yield mock_client


def test_openai_provider_full_review(mock_openai_llm, tmp_path):
    """End-to-end: OpenAI provider (mocked) + NeurIPS venue produces scores + decision."""
    engine = ReviewEngine(storage=MemoryStorage())

    yaml_path = tmp_path / "openai_test.yaml"
    yaml_content = """
name: openai-neurips-test
on: {workflow_dispatch: {}}
env:
  LLM_PROVIDER: openai
  LLM_MODEL: gpt-4o
  VENUE: neurips
  SESSIONS_ROOT: __SESSIONS_ROOT__
jobs:
  extract:
    runs-on: local
    outputs:
      paper_id: __PAPER_ID__
      full_text_path: __FULL_TEXT_PATH__
      metadata_path: __METADATA_PATH__
    steps:
      - id: extract
        uses: paper-review/extract@v1
        with:
          source: "__FAKE_PAPER__"
          session_dir: "__SESSION_DIR__"
  dimensions:
    needs: extract
    strategy:
      matrix:
        dimension: [soundness, presentation, contribution]
      max-parallel: 3
    runs-on: local
    steps:
      - uses: paper-review/dim_score@v1
        with:
          dimension: __DIM_MATRIX__
          session_dir: "__SESSION_DIR__"
          full_text_path: __FULL_TEXT_PATH__
          metadata_path: __METADATA_PATH__
  synthesize:
    needs: dimensions
    runs-on: local
    steps:
      - id: synthesize
        uses: paper-review/synthesize@v1
        with:
          session_dir: "__SESSION_DIR__"
  decide:
    needs: synthesize
    runs-on: local
    steps:
      - uses: paper-review/decide@v1
        with:
          session_dir: "__SESSION_DIR__"
          scores_path: __SCORES_PATH__
"""
    yaml_content = yaml_content.replace("__SESSIONS_ROOT__", str(tmp_path))
    yaml_content = yaml_content.replace("__FAKE_PAPER__", str(Path("tests/fixtures/sample_paper.pdf")))
    yaml_content = yaml_content.replace("__SESSION_DIR__", str(tmp_path / "session"))
    yaml_content = yaml_content.replace("__PAPER_ID__", "${{ steps.extract.outputs.paper_id }}")
    yaml_content = yaml_content.replace("__FULL_TEXT_PATH__", "${{ needs.extract.outputs.full_text_path }}")
    yaml_content = yaml_content.replace("__METADATA_PATH__", "${{ needs.extract.outputs.metadata_path }}")
    yaml_content = yaml_content.replace("__SCORES_PATH__", "${{ needs.synthesize.outputs.scores_path }}")
    yaml_content = yaml_content.replace("__DIM_MATRIX__", "${{ matrix.dimension }}")
    yaml_path.write_text(yaml_content)

    from paper_review_workflow.actions.synthesize import SynthesizeAction
    with patch.object(SynthesizeAction, "MIN_DIMENSIONS", 2):
        run = engine.run_from_file(str(yaml_path), payload={})

    assert run.status.value == "success", f"run failed: {run.status.value}"

    session_dir = tmp_path / "session"
    for dim in ["soundness", "presentation", "contribution"]:
        score_json = json.loads((session_dir / f"10_dim_{dim}" / "score.json").read_text())
        assert score_json["venue"] == "neurips"
        assert 1 <= score_json["score"] <= 10

    decision = json.loads((session_dir / "60_decision" / "decision.json").read_text())
    assert decision["venue"] == "neurips"
    assert decision["score_range"] == [1, 10]
    assert decision["recommendation"] in [
        "strong_accept", "accept", "weak_accept", "borderline",
        "weak_reject", "reject", "strong_reject",
    ]
```

- [ ] **Step 2: Run test**

Run: `pytest tests/integration/test_venue_review_openai.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_venue_review_openai.py
git commit -m "test(integration): OpenAI provider venue review (mocked)"
```

## Task 5.2: Venue review integration test (DeepSeek mock)

**Files:**
- Create: `tests/integration/test_venue_review_deepseek.py`

- [ ] **Step 1: Write test**

Create `tests/integration/test_venue_review_deepseek.py` — same structure as OpenAI test but with DeepSeek env vars:

```python
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.memory import MemoryStorage
from paper_review_workflow.llm.client import LLMClient
from paper_review_workflow.llm.base import LLMResponse


@pytest.fixture
def mock_deepseek_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    LLMClient.reset()

    fake_score = MagicMock()
    fake_score.score = 7
    fake_score.confidence = 0.85
    fake_score.strengths = ["rigorous"]
    fake_score.weaknesses = ["limited"]
    fake_score.justification = "x" * 250
    fake_score.evidence = []

    fake_synth = MagicMock()
    fake_synth.summary = "x" * 250
    fake_synth.key_strengths = ["strong"]
    fake_synth.key_weaknesses = ["weak"]
    fake_synth.questions_for_authors = ["why?"]
    fake_synth.overall_assessment = "good"

    fake_dim_resp = MagicMock(spec=LLMResponse)
    fake_dim_resp.structured = fake_score
    fake_dim_resp.usage = {"input_tokens": 100, "output_tokens": 50,
                           "cache_creation_input_tokens": 0,
                           "cache_read_input_tokens": 0}
    fake_dim_resp.model = "deepseek-chat"

    fake_synth_resp = MagicMock(spec=LLMResponse)
    fake_synth_resp.structured = fake_synth
    fake_synth_resp.usage = {"input_tokens": 500, "output_tokens": 200,
                             "cache_creation_input_tokens": 0,
                             "cache_read_input_tokens": 0}
    fake_synth_resp.model = "deepseek-chat"

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.model = "deepseek-chat"
        def side_effect(**kw):
            schema = kw.get("response_schema")
            if hasattr(schema, "__name__") and "Synthesis" in schema.__name__:
                return fake_synth_resp
            return fake_dim_resp
        mock_client.complete.side_effect = side_effect
        mock_from_env.return_value = mock_client
        yield mock_client


def test_deepseek_provider_full_review(mock_deepseek_llm, tmp_path):
    """End-to-end: DeepSeek provider (mocked) + NeurIPS venue produces scores + decision."""
    engine = ReviewEngine(storage=MemoryStorage())

    yaml_path = tmp_path / "deepseek_test.yaml"
    yaml_content = """
name: deepseek-neurips-test
on: {workflow_dispatch: {}}
env:
  LLM_PROVIDER: deepseek
  LLM_MODEL: deepseek-chat
  VENUE: neurips
  SESSIONS_ROOT: __SESSIONS_ROOT__
jobs:
  extract:
    runs-on: local
    outputs:
      paper_id: __PAPER_ID__
      full_text_path: __FULL_TEXT_PATH__
      metadata_path: __METADATA_PATH__
    steps:
      - id: extract
        uses: paper-review/extract@v1
        with:
          source: "__FAKE_PAPER__"
          session_dir: "__SESSION_DIR__"
  dimensions:
    needs: extract
    strategy:
      matrix:
        dimension: [soundness, presentation, contribution]
      max-parallel: 3
    runs-on: local
    steps:
      - uses: paper-review/dim_score@v1
        with:
          dimension: __DIM_MATRIX__
          session_dir: "__SESSION_DIR__"
          full_text_path: __FULL_TEXT_PATH__
          metadata_path: __METADATA_PATH__
  synthesize:
    needs: dimensions
    runs-on: local
    steps:
      - id: synthesize
        uses: paper-review/synthesize@v1
        with:
          session_dir: "__SESSION_DIR__"
  decide:
    needs: synthesize
    runs-on: local
    steps:
      - uses: paper-review/decide@v1
        with:
          session_dir: "__SESSION_DIR__"
          scores_path: __SCORES_PATH__
"""
    yaml_content = yaml_content.replace("__SESSIONS_ROOT__", str(tmp_path))
    yaml_content = yaml_content.replace("__FAKE_PAPER__", str(Path("tests/fixtures/sample_paper.pdf")))
    yaml_content = yaml_content.replace("__SESSION_DIR__", str(tmp_path / "session"))
    yaml_content = yaml_content.replace("__PAPER_ID__", "${{ steps.extract.outputs.paper_id }}")
    yaml_content = yaml_content.replace("__FULL_TEXT_PATH__", "${{ needs.extract.outputs.full_text_path }}")
    yaml_content = yaml_content.replace("__METADATA_PATH__", "${{ needs.extract.outputs.metadata_path }}")
    yaml_content = yaml_content.replace("__SCORES_PATH__", "${{ needs.synthesize.outputs.scores_path }}")
    yaml_content = yaml_content.replace("__DIM_MATRIX__", "${{ matrix.dimension }}")
    yaml_path.write_text(yaml_content)

    from paper_review_workflow.actions.synthesize import SynthesizeAction
    with patch.object(SynthesizeAction, "MIN_DIMENSIONS", 2):
        run = engine.run_from_file(str(yaml_path), payload={})

    assert run.status.value == "success", f"run failed: {run.status.value}"

    session_dir = tmp_path / "session"
    for dim in ["soundness", "presentation", "contribution"]:
        score_json = json.loads((session_dir / f"10_dim_{dim}" / "score.json").read_text())
        assert score_json["venue"] == "neurips"
        assert 1 <= score_json["score"] <= 10

    decision = json.loads((session_dir / "60_decision" / "decision.json").read_text())
    assert decision["venue"] == "neurips"
    assert decision["score_range"] == [1, 10]
    assert decision["recommendation"] in [
        "strong_accept", "accept", "weak_accept", "borderline",
        "weak_reject", "reject", "strong_reject",
    ]
```

- [ ] **Step 2: Run test**

Run: `pytest tests/integration/test_venue_review_deepseek.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_venue_review_deepseek.py
git commit -m "test(integration): DeepSeek provider venue review (mocked)"
```

## Task 5.3: E2E tests (marked)

**Files:**
- Create: `tests/e2e/test_e2e_openai_arxiv.py`
- Create: `tests/e2e/test_e2e_deepseek_arxiv.py`

- [ ] **Step 1: Write E2E tests**

Create `tests/e2e/test_e2e_openai_arxiv.py`:

```python
import os
import time
import json
import pytest
from fastapi.testclient import TestClient
from paper_review_workflow.api import create_app
from paper_review_workflow.storage.memory import MemoryStorage


@pytest.mark.e2e
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"),
                    reason="requires OPENAI_API_KEY")
def test_e2e_real_openai_arxiv_review():
    """E2E: Real OpenAI API + real arXiv paper + NeurIPS venue."""
    app = create_app(storage=MemoryStorage(), configs_dir="configs")
    with TestClient(app) as client:
        r = client.post("/api/runs", json={
            "workflow_name": "neurips-paper-review",
            "inputs": {"paper_source": "2402.12098", "mode": "neurips"},
        })
        assert r.status_code == 202
        run_id = r.json()["run_id"]

        for _ in range(300):
            r = client.get(f"/api/runs/{run_id}")
            data = r.json()
            if data["status"] in ("success", "failure", "cancelled"):
                break
            time.sleep(1)
        else:
            assert False, "run did not complete within 5 minutes"

        assert data["status"] == "success", f"expected success, got {data['status']}"
        dim_jobs = [k for k in data["jobs"] if k.startswith("dimensions_")]
        assert len(dim_jobs) == 3  # NeurIPS has 3 dims
        assert data["jobs"]["decide"]["status"] == "success"
```

Create `tests/e2e/test_e2e_deepseek_arxiv.py`:

```python
import os
import time
import json
import pytest
from fastapi.testclient import TestClient
from paper_review_workflow.api import create_app
from paper_review_workflow.storage.memory import MemoryStorage


@pytest.mark.e2e
@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"),
                    reason="requires DEEPSEEK_API_KEY")
def test_e2e_real_deepseek_arxiv_review():
    """E2E: Real DeepSeek API + real arXiv paper + NeurIPS venue."""
    os.environ["LLM_PROVIDER"] = "deepseek"
    os.environ["LLM_MODEL"] = os.environ.get("LLM_MODEL", "deepseek-chat")
    app = create_app(storage=MemoryStorage(), configs_dir="configs")
    with TestClient(app) as client:
        r = client.post("/api/runs", json={
            "workflow_name": "neurips-paper-review",
            "inputs": {"paper_source": "2402.12098", "mode": "neurips"},
        })
        assert r.status_code == 202
        run_id = r.json()["run_id"]

        for _ in range(300):
            r = client.get(f"/api/runs/{run_id}")
            data = r.json()
            if data["status"] in ("success", "failure", "cancelled"):
                break
            time.sleep(1)
        else:
            assert False, "run did not complete within 5 minutes"

        assert data["status"] == "success", f"expected success, got {data['status']}"
        dim_jobs = [k for k in data["jobs"] if k.startswith("dimensions_")]
        assert len(dim_jobs) == 3
        assert data["jobs"]["decide"]["status"] == "success"
```

- [ ] **Step 2: Verify test discovery**

Run: `pytest tests/e2e/test_e2e_openai_arxiv.py tests/e2e/test_e2e_deepseek_arxiv.py --collect-only`
Expected: 2 tests collected (skipped without API keys)

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_e2e_openai_arxiv.py tests/e2e/test_e2e_deepseek_arxiv.py
git commit -m "test(e2e): real OpenAI + DeepSeek arXiv review (marked)"
```

## Task 5.4: Update README + tag v0.5.0

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read current README**

Run: `cat README.md`

- [ ] **Step 2: Add multi-provider section**

After the "Configuration" section (or wherever appropriate), add:

```markdown
## Multi-Provider LLM Support

The engine supports 3 LLM providers:

| Provider | Env Var | Default Model | Structured Output |
|----------|---------|---------------|-------------------|
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` | tool use (strict) |
| OpenAI | `OPENAI_API_KEY` | (user must set `LLM_MODEL`) | json_schema (strict) |
| DeepSeek | `DEEPSEEK_API_KEY` | (user must set `LLM_MODEL`) | json_object + prompt schema |

Switch providers via environment variables:

```bash
# OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-4o  # user specifies current model name

# DeepSeek
export LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=sk-...
export LLM_MODEL=deepseek-chat

# Anthropic (default)
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
# LLM_MODEL defaults to claude-sonnet-4-6
```

All providers share the same retry strategy (3x exponential backoff) and schema validation (`SchemaValidationError` on invalid LLM output).
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ --ignore=tests/unit/test_extract_arxiv.py -k "not test_extract_arxiv_id" 2>&1 | tail -5`
Expected: All tests pass

- [ ] **Step 4: Commit + tag**

```bash
git add README.md
git commit -m "docs: add multi-provider LLM documentation to README"
git tag v0.5.0
```

- [ ] **Step 5: Push**

```bash
git push origin main
git push --tags
```

---

# Self-Review Checklist

## Spec Coverage

| SPEC Section | Implemented By |
|---|---|
| 1. Scope | M1-M5 cover Phase 2 #2 only ✅ |
| 2. Architecture | M1 base.py refactored, M2 OpenAI, M3 DeepSeek, M4 registry ✅ |
| 3.1 LLMProvider from_env | M1 Task 1.1 ✅ |
| 3.2 AnthropicProvider.from_env | M1 Task 1.1 ✅ |
| 3.3 OpenAIProvider | M2 Task 2.2 ✅ |
| 3.4 DeepSeekProvider | M3 Task 3.1 ✅ |
| 3.5 LLMClient.from_env refactor | M1 Task 1.2 ✅ |
| 3.6 ProviderRegistry | M4 Task 4.1 ✅ |
| 4. Structured output comparison | M2 (json_schema) + M3 (json_object + prompt) ✅ |
| 5. Prompt cache | M2 (auto) + M3 (no cache) ✅ |
| 6. Retry strategy | M2 + M3 (shared pattern) ✅ |
| 7. Testing strategy | M1-M4 unit, M5 integration + E2E ✅ |
| 8. Milestones | M1-M5 directly map ✅ |
| 9. 验收标准 | All covered ✅ |

## Placeholder Scan

- ✅ No "TBD"/"TODO"
- ✅ All code shown in full
- ✅ All test code shown
- ✅ No "implement later"

## Type Consistency

- `from_env()` → returns `LLMProvider` subclass — consistent across M1 (Anthropic) + M2 (OpenAI) + M3 (DeepSeek) ✅
- `complete()` signature — same as Phase 1 `LLMProvider.complete()` ✅
- `LLMClient.from_env()` → delegates to `provider_cls.from_env()` — consistent M1 + M4 ✅
- `ProviderRegistry.register(name, cls)` / `.get(name)` — consistent M4 + existing Phase 1 ✅
- `_build_system_text()` — M2 OpenAI + M3 DeepSeek have same signature (M3 adds response_schema param) ✅
- `_parse_response()` — same signature M2 + M3 ✅
- `_sleep_backoff(attempt)` — same pattern M2 + M3 (matching AnthropicProvider) ✅

---

# Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-16-multi-provider-impl.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
