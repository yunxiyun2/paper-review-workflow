import pytest
from unittest.mock import MagicMock, patch
import json

from paper_review_workflow.llm.zhipu_provider import ZhipuProvider
from paper_review_workflow.llm.registry import ProviderRegistry
from paper_review_workflow.llm.base import LLMError
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
        "confidence": 0.9,
        "strengths": ["clear contribution"],
        "weaknesses": ["weak related work"],
        "justification": "x" * 250,
        "evidence": [],
    })


def test_zhipu_registered_in_registry():
    registry = ProviderRegistry()
    assert "zhipu" in registry.list_providers()
    assert registry.get("zhipu") is ZhipuProvider


def test_zhipu_from_env_reads_api_key():
    provider = ZhipuProvider.from_env({"ZHIPU_API_KEY": "zhu-test"})
    assert provider is not None
    assert provider.provider_name == "zhipu"


def test_zhipu_from_env_missing_key_raises():
    with pytest.raises(LLMError, match="ZHIPU_API_KEY not set"):
        ZhipuProvider.from_env({})


def test_zhipu_falls_back_to_os_environ(monkeypatch):
    monkeypatch.setenv("ZHIPU_API_KEY", "zhu-env")
    provider = ZhipuProvider.from_env()
    assert provider is not None


def test_zhipu_base_url():
    provider = ZhipuProvider(api_key="zhu-test")
    assert provider.BASE_URL == "https://open.bigmodel.cn/api/paas/v4/"
    assert provider.DEFAULT_MODEL == "glm-4.6"


@patch("openai.OpenAI")
def test_zhipu_complete_uses_json_object_and_bigmodel_base_url(mock_cls):
    mock_client = mock_cls.return_value
    mock_client.chat.completions.create.return_value = _make_openai_response(_valid_score_json())

    provider = ZhipuProvider(api_key="zhu-test")
    result = provider.complete(
        system="score soundness",
        messages=[{"role": "user", "content": "paper text"}],
        model="glm-4.6",
        max_tokens=4096,
        response_schema=DimensionScore,
    )

    assert result.structured.score == 4
    assert result.structured.confidence == 0.9
    # json_object response_format + schema appended to prompt (zhipu has no json_schema)
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "schema" in kwargs["messages"][0]["content"]
    # client built against the zhipu base_url
    assert mock_cls.call_args.kwargs["base_url"] == ZhipuProvider.BASE_URL


@patch("openai.OpenAI")
def test_zhipu_complete_no_schema_no_response_format(mock_cls):
    mock_client = mock_cls.return_value
    mock_client.chat.completions.create.return_value = _make_openai_response("plain text")

    provider = ZhipuProvider(api_key="zhu-test")
    result = provider.complete(
        system="summarize",
        messages=[{"role": "user", "content": "hi"}],
        model="glm-4.6",
        max_tokens=1024,
    )
    assert result.text == "plain text"
    assert result.structured is None
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs.get("response_format") is None


@patch("openai.OpenAI")
def test_zhipu_schema_validation_error(mock_cls):
    mock_client = mock_cls.return_value
    mock_client.chat.completions.create.return_value = _make_openai_response('{"score": "not-a-number"}')

    provider = ZhipuProvider(api_key="zhu-test")
    from paper_review_workflow.llm.base import SchemaValidationError
    with pytest.raises(SchemaValidationError):
        provider.complete(
            system="score",
            messages=[{"role": "user", "content": "paper"}],
            model="glm-4.6",
            max_tokens=4096,
            response_schema=DimensionScore,
        )


@patch("openai.OpenAI")
def test_zhipu_extracts_json_from_markdown_fence(mock_cls):
    """glm-5.3 (thinking model) sometimes wraps JSON in markdown fences"""
    mock_client = mock_cls.return_value
    fenced = "```json\n" + _valid_score_json() + "\n```"
    mock_client.chat.completions.create.return_value = _make_openai_response(fenced)

    provider = ZhipuProvider(api_key="zhu-test")
    result = provider.complete(
        system="score", messages=[{"role": "user", "content": "paper"}],
        model="glm-5.3", max_tokens=8192, response_schema=DimensionScore,
    )
    assert result.structured.score == 4


@patch("openai.OpenAI")
def test_zhipu_repairs_schema_validation_failure(mock_cls):
    """On SchemaValidationError the provider retries with the error fed back"""
    mock_client = mock_cls.return_value
    too_long = json.dumps({
        "score": 4, "confidence": 0.9,
        "strengths": ["s"], "weaknesses": ["w"],
        "justification": "x" * 900,  # exceeds 800-char limit
        "evidence": [],
    })
    mock_client.chat.completions.create.side_effect = [
        _make_openai_response(too_long),
        _make_openai_response(_valid_score_json()),
    ]

    provider = ZhipuProvider(api_key="zhu-test")
    result = provider.complete(
        system="score", messages=[{"role": "user", "content": "paper"}],
        model="glm-5.3", max_tokens=8192, response_schema=DimensionScore,
    )
    assert result.structured.score == 4
    assert mock_client.chat.completions.create.call_count == 2
    second_call = mock_client.chat.completions.create.call_args_list[1]
    system_content = second_call.kwargs["messages"][0]["content"]
    assert "FAILED validation" in system_content
    assert "800" in system_content  # the original error is fed back


@patch("openai.OpenAI")
def test_zhipu_empty_content_raises_for_repair(mock_cls):
    mock_client = mock_cls.return_value
    mock_client.chat.completions.create.return_value = _make_openai_response("")

    provider = ZhipuProvider(api_key="zhu-test")
    from paper_review_workflow.llm.base import SchemaValidationError
    with pytest.raises(SchemaValidationError):
        provider.complete(
            system="score", messages=[{"role": "user", "content": "paper"}],
            model="glm-5.3", max_tokens=8192, response_schema=DimensionScore,
        )
