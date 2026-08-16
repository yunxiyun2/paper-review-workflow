import pytest
from unittest.mock import MagicMock, patch
from paper_review_workflow.llm.client import LLMClient
from paper_review_workflow.llm.schemas import DimensionScore


def test_from_env_initializes_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("LLM_MAX_TOKENS", "4096")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    LLMClient._instance = None
    client = LLMClient.from_env()
    assert client.model == "claude-sonnet-4-6"
    assert client.max_tokens == 4096


def test_from_env_singleton(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    LLMClient._instance = None
    c1 = LLMClient.from_env()
    c2 = LLMClient.from_env()
    assert c1 is c2


def test_score_delegates_to_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    LLMClient._instance = None
    client = LLMClient.from_env()

    mock_result = MagicMock()
    mock_result.structured = DimensionScore(
        score=4, confidence=0.8, strengths=["a"],
        weaknesses=["b"], justification="x" * 200,
    )
    client._provider.complete = MagicMock(return_value=mock_result)

    result = client.score(
        system="score novelty", user_content="paper text",
        schema=DimensionScore, cached_context="PAPER",
    )

    assert result.score == 4
    call_kwargs = client._provider.complete.call_args.kwargs
    assert call_kwargs["cached_context"] == "PAPER"
    assert call_kwargs["response_schema"] is DimensionScore


def test_reset_for_tests(monkeypatch):
    """LLMClient.reset() clears the singleton"""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    LLMClient._instance = None
    c1 = LLMClient.from_env()
    assert LLMClient._instance is c1
    LLMClient.reset()
    assert LLMClient._instance is None


def test_complete_passes_defaults(monkeypatch):
    """complete() uses client's model/max_tokens/temperature as defaults"""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_MAX_TOKENS", "2048")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
    LLMClient._instance = None
    client = LLMClient.from_env()

    mock_resp = MagicMock()
    client._provider.complete = MagicMock(return_value=mock_resp)

    client.complete(system="s", messages=[{"role": "user", "content": "c"}])

    call_kwargs = client._provider.complete.call_args.kwargs
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["max_tokens"] == 2048
    assert call_kwargs["temperature"] == 0.5


def test_from_env_anthropic_keeps_default_model(monkeypatch):
    """Anthropic still has default model claude-sonnet-4-6 even without LLM_MODEL"""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    LLMClient.reset()
    client = LLMClient.from_env()
    assert client.model == "claude-sonnet-4-6"
    assert client._provider.provider_name == "anthropic"


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
