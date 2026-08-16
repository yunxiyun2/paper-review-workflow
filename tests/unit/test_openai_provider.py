import pytest
from unittest.mock import MagicMock, patch
import json

from paper_review_workflow.llm.openai_provider import OpenAIProvider
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

    assert result.structured.score == 4
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
    assert result.structured.score == 4


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
    # score=99 violates schema (1-5 for DimensionScore)
    mock_client.chat.completions.create.return_value = _make_openai_response(
        json.dumps({"score": 99, "confidence": 0.5, "strengths": ["a"],
                    "weaknesses": ["b"], "justification": "x" * 200})
    )

    provider = OpenAIProvider(api_key="sk-test")
    with pytest.raises(SchemaValidationError):
        provider.complete(system="t", messages=[], model="gpt-4o", max_tokens=4096,
                         response_schema=DimensionScore)
