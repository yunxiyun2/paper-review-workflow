import pytest
from unittest.mock import MagicMock, patch
import anthropic

from paper_review_workflow.llm.anthropic_provider import AnthropicProvider
from paper_review_workflow.llm.base import RateLimitError, ContextLengthError, SchemaValidationError
from paper_review_workflow.llm.schemas import DimensionScore


def _make_response(content_blocks, usage_in=100, usage_out=50,
                   cache_creation=0, cache_read=0):
    resp = MagicMock()
    resp.content = content_blocks
    resp.usage = MagicMock(
        input_tokens=usage_in, output_tokens=usage_out,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )
    return resp


def _tool_use_block(input_dict):
    b = MagicMock()
    b.type = "tool_use"
    b.input = input_dict
    return b


@patch("anthropic.Anthropic")
def test_complete_with_structured_output(mock_cls):
    mock_client = mock_cls.return_value
    mock_client.messages.create.return_value = _make_response([
        _tool_use_block({
            "score": 4, "confidence": 0.85,
            "strengths": ["a"], "weaknesses": ["b"],
            "justification": "x" * 200, "evidence": [],
        })
    ])

    p = AnthropicProvider()
    result = p.complete(
        system="test", messages=[{"role": "user", "content": "score"}],
        model="claude-sonnet-4-6", max_tokens=100,
        response_schema=DimensionScore,
    )

    assert result.structured.score == 4
    assert result.structured.confidence == 0.85
    assert result.usage["input_tokens"] == 100


@patch("anthropic.Anthropic")
def test_complete_with_cached_context_sets_cache_control(mock_cls):
    mock_client = mock_cls.return_value
    mock_client.messages.create.return_value = _make_response([
        _tool_use_block({
            "score": 3, "confidence": 0.5,
            "strengths": ["a"], "weaknesses": ["b"],
            "justification": "x" * 200, "evidence": [],
        })
    ])

    p = AnthropicProvider()
    p.complete(
        system="score novelty",
        messages=[{"role": "user", "content": "..."}],
        model="claude-sonnet-4-6", max_tokens=100,
        response_schema=DimensionScore,
        cached_context="PAPER FULL TEXT",
    )

    call_kwargs = mock_client.messages.create.call_args.kwargs
    system_blocks = call_kwargs["system"]
    assert any(b.get("cache_control") == {"type": "ephemeral"} for b in system_blocks)
    assert any(b.get("text") == "PAPER FULL TEXT" for b in system_blocks)


@patch("anthropic.Anthropic")
def test_retry_on_rate_limit(mock_cls, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)  # skip real sleeping

    mock_client = mock_cls.return_value
    success_resp = _make_response([
        _tool_use_block({
            "score": 3, "confidence": 0.5,
            "strengths": ["a"], "weaknesses": ["b"],
            "justification": "x" * 200, "evidence": [],
        })
    ])
    mock_client.messages.create.side_effect = [
        anthropic.RateLimitError(
            message="429", response=MagicMock(status_code=429), body=None
        ),
        success_resp,
    ]

    p = AnthropicProvider()
    result = p.complete(
        system="t", messages=[], model="m", max_tokens=10,
        response_schema=DimensionScore,
    )
    assert mock_client.messages.create.call_count == 2
    assert result.structured.score == 3


@patch("anthropic.Anthropic")
def test_rate_limit_exhausted(mock_cls, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    mock_client = mock_cls.return_value
    mock_client.messages.create.side_effect = anthropic.RateLimitError(
        message="429", response=MagicMock(status_code=429), body=None
    )

    p = AnthropicProvider()
    with pytest.raises(RateLimitError):
        p.complete(system="t", messages=[], model="m", max_tokens=10,
                   response_schema=DimensionScore)
    assert mock_client.messages.create.call_count == 3


@patch("anthropic.Anthropic")
def test_context_length_no_retry(mock_cls, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    mock_client = mock_cls.return_value
    err = anthropic.BadRequestError(
        message="context_length_exceeded", response=MagicMock(status_code=400), body=None
    )
    mock_client.messages.create.side_effect = err

    p = AnthropicProvider()
    with pytest.raises(ContextLengthError):
        p.complete(system="t", messages=[], model="m", max_tokens=10,
                   response_schema=DimensionScore)
    assert mock_client.messages.create.call_count == 1


@patch("anthropic.Anthropic")
def test_schema_validation_failure_raises(mock_cls):
    mock_client = mock_cls.return_value
    # score=6 violates schema (1-5)
    mock_client.messages.create.return_value = _make_response([
        _tool_use_block({
            "score": 6, "confidence": 0.5,
            "strengths": ["a"], "weaknesses": ["b"],
            "justification": "x" * 200, "evidence": [],
        })
    ])

    p = AnthropicProvider()
    with pytest.raises(SchemaValidationError):
        p.complete(system="t", messages=[], model="m", max_tokens=10,
                   response_schema=DimensionScore)
