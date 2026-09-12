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
        "score": 4,
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

    assert result.structured.score == 4
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
    # System prompt should contain the output contract (anti-echo instruction)
    assert "never repeat or echo the schema" in system_content
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
