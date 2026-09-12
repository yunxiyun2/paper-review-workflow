import pytest
from pydantic import BaseModel
from paper_review_workflow.llm.base import LLMProvider, LLMResponse, LLMError, RateLimitError
from paper_review_workflow.llm.registry import ProviderRegistry


class FakeProvider(LLMProvider):
    provider_name = "fake"

    @classmethod
    def from_env(cls) -> "FakeProvider":
        return cls()

    def complete(self, **kwargs):
        class FakeSchema(BaseModel):
            x: int
        return LLMResponse(
            text=None, structured=FakeSchema(x=42),
            usage={"input_tokens": 10, "output_tokens": 5,
                   "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            model="fake-model",
        )


def test_provider_registry_singleton():
    ProviderRegistry._instance = None
    r1 = ProviderRegistry()
    r2 = ProviderRegistry()
    assert r1 is r2


def test_provider_registry_register_and_get():
    ProviderRegistry._instance = None
    reg = ProviderRegistry()
    reg.register("fake", FakeProvider)
    assert reg.get("fake") is FakeProvider


def test_provider_registry_get_unknown():
    ProviderRegistry._instance = None
    reg = ProviderRegistry()
    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_llm_response_dataclass():
    class S(BaseModel):
        y: str
    r = LLMResponse(
        text="hi", structured=S(y="x"),
        usage={"input_tokens": 1, "output_tokens": 1,
               "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        model="m",
    )
    assert r.text == "hi"
    assert r.structured.y == "x"


def test_llm_provider_is_abstract():
    """LLMProvider cannot be instantiated directly"""
    with pytest.raises(TypeError):
        LLMProvider()


def test_anthropic_provider_registered_by_default():
    """ProviderRegistry auto-registers AnthropicProvider in __init__"""
    ProviderRegistry._instance = None
    reg = ProviderRegistry()
    assert "anthropic" in reg.list_providers()


def test_registry_has_4_providers():
    """ProviderRegistry should auto-register anthropic, openai, deepseek, zhipu"""
    ProviderRegistry._instance = None
    reg = ProviderRegistry()
    providers = reg.list_providers()
    assert "anthropic" in providers
    assert "openai" in providers
    assert "deepseek" in providers
    assert "zhipu" in providers
    assert len(providers) == 4


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
