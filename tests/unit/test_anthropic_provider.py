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


@patch("anthropic.Anthropic")
def test_retry_on_500_then_success(mock_cls, monkeypatch):
    """APIStatusError 500 should retry then succeed."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    mock_client = mock_cls.return_value
    success_resp = _make_response([
        _tool_use_block({
            "score": 4, "confidence": 0.7,
            "strengths": ["a"], "weaknesses": ["b"],
            "justification": "x" * 200, "evidence": [],
        })
    ])
    mock_client.messages.create.side_effect = [
        anthropic.APIStatusError(
            message="500", response=MagicMock(status_code=500), body=None
        ),
        success_resp,
    ]
    p = AnthropicProvider()
    result = p.complete(
        system="t", messages=[], model="m", max_tokens=10,
        response_schema=DimensionScore,
    )
    assert mock_client.messages.create.call_count == 2
    assert result.structured.score == 4


@patch("anthropic.Anthropic")
def test_500_exhausted_retries_raises(mock_cls, monkeypatch):
    """APIStatusError 500 after all retries should re-raise."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    mock_client = mock_cls.return_value
    mock_client.messages.create.side_effect = anthropic.APIStatusError(
        message="500", response=MagicMock(status_code=500), body=None
    )
    p = AnthropicProvider()
    with pytest.raises(anthropic.APIStatusError):
        p.complete(
            system="t", messages=[], model="m", max_tokens=10,
            response_schema=DimensionScore,
        )
    assert mock_client.messages.create.call_count == 3


@patch("anthropic.Anthropic")
def test_400_non_context_length_raises_immediately(mock_cls, monkeypatch):
    """400 that is NOT context_length should raise immediately (no retry)."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    mock_client = mock_cls.return_value
    mock_client.messages.create.side_effect = anthropic.APIStatusError(
        message="bad request", response=MagicMock(status_code=400), body=None
    )
    p = AnthropicProvider()
    with pytest.raises(anthropic.APIStatusError):
        p.complete(
            system="t", messages=[], model="m", max_tokens=10,
            response_schema=DimensionScore,
        )
    assert mock_client.messages.create.call_count == 1


@patch("anthropic.Anthropic")
def test_retry_on_api_connection_error_then_success(mock_cls, monkeypatch):
    """APIConnectionError should retry then succeed."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    mock_client = mock_cls.return_value
    success_resp = _make_response([
        _tool_use_block({
            "score": 3, "confidence": 0.5,
            "strengths": ["a"], "weaknesses": ["b"],
            "justification": "x" * 200, "evidence": [],
        })
    ])
    mock_client.messages.create.side_effect = [
        anthropic.APIConnectionError(
            message="conn err", request=MagicMock()
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
def test_api_connection_error_exhausted_retries_raises(mock_cls, monkeypatch):
    """APIConnectionError after all retries should re-raise."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    mock_client = mock_cls.return_value
    mock_client.messages.create.side_effect = anthropic.APIConnectionError(
        message="conn err", request=MagicMock()
    )
    p = AnthropicProvider()
    with pytest.raises(anthropic.APIConnectionError):
        p.complete(
            system="t", messages=[], model="m", max_tokens=10,
            response_schema=DimensionScore,
        )
    assert mock_client.messages.create.call_count == 3


@patch("anthropic.Anthropic")
def test_build_system_blocks_with_list_system(mock_cls):
    """When system is a list of blocks, it should be extended as-is."""
    mock_client = mock_cls.return_value
    mock_client.messages.create.return_value = _make_response([
        _tool_use_block({
            "score": 4, "confidence": 0.85,
            "strengths": ["a"], "weaknesses": ["b"],
            "justification": "x" * 200, "evidence": [],
        })
    ])
    p = AnthropicProvider()
    system_blocks_input = [{"type": "text", "text": "block1"}]
    p.complete(
        system=system_blocks_input,
        messages=[{"role": "user", "content": "score"}],
        model="claude-sonnet-4-6", max_tokens=100,
        response_schema=DimensionScore,
    )
    call_kwargs = mock_client.messages.create.call_args.kwargs
    system_blocks = call_kwargs["system"]
    assert any(b.get("text") == "block1" for b in system_blocks)


@patch("anthropic.Anthropic")
def test_build_tools_returns_none_when_no_schema(mock_cls):
    """When response_schema is None, tools/tool_choice should be None."""
    mock_client = mock_cls.return_value
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "hello world"
    mock_client.messages.create.return_value = _make_response([text_block])
    p = AnthropicProvider()
    result = p.complete(
        system="t", messages=[], model="m", max_tokens=10,
        response_schema=None,
    )
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["tools"] is None
    assert call_kwargs["tool_choice"] is None
    assert result.text == "hello world"


@patch("anthropic.Anthropic")
def test_parse_response_no_tool_use_raises_schema_error(mock_cls):
    """When schema is provided but no tool_use block in response, raise."""
    mock_client = mock_cls.return_value
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "I cannot comply"
    mock_client.messages.create.return_value = _make_response([text_block])
    p = AnthropicProvider()
    with pytest.raises(SchemaValidationError):
        p.complete(
            system="t", messages=[], model="m", max_tokens=10,
            response_schema=DimensionScore,
        )


@patch("anthropic.Anthropic")
def test_parse_response_text_when_no_schema(mock_cls):
    """When response_schema is None, text content is joined and returned."""
    mock_client = mock_cls.return_value
    b1 = MagicMock()
    b1.type = "text"
    b1.text = "hello "
    b2 = MagicMock()
    b2.type = "text"
    b2.text = "world"
    mock_client.messages.create.return_value = _make_response([b1, b2])
    p = AnthropicProvider()
    result = p.complete(
        system="t", messages=[], model="m", max_tokens=10,
        response_schema=None,
    )
    assert result.text == "hello world"
    assert result.structured is None


@patch("anthropic.Anthropic")
def test_parse_response_empty_text_returns_none(mock_cls):
    """When response_schema is None and no text blocks, text should be None."""
    mock_client = mock_cls.return_value
    b1 = MagicMock()
    b1.type = "tool_use"
    b1.text = ""
    mock_client.messages.create.return_value = _make_response([b1])
    p = AnthropicProvider()
    result = p.complete(
        system="t", messages=[], model="m", max_tokens=10,
        response_schema=None,
    )
    assert result.text is None
