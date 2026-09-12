"""Zhipu (智谱) LLM provider — OpenAI-compatible endpoint at open.bigmodel.cn.
Uses json_object + schema-in-prompt (same strategy as DeepSeek)."""
from typing import Optional

from .base import LLMError
from .deepseek_provider import DeepSeekProvider


class ZhipuProvider(DeepSeekProvider):
    provider_name = "zhipu"
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
    DEFAULT_MODEL = "glm-4.6"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key)

    @classmethod
    def from_env(cls, env=None) -> "ZhipuProvider":
        import os
        src = env if env is not None else os.environ
        api_key = src.get("ZHIPU_API_KEY")
        if not api_key:
            raise LLMError("ZHIPU_API_KEY not set")
        return cls(api_key=api_key)
