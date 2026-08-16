# Multi LLM Provider (Phase 2 #2) — SPEC-PRD

**版本**: 1.0
**日期**: 2026-08-16
**状态**: 设计确认中
**Phase**: 2 (subsystem #2 of 7)

---

## 1. 项目定位与范围

### 1.1 一句话定位

为 paper-review-workflow 增加 OpenAI 和 DeepSeek LLM provider,让评审工作流不锁定 Anthropic,用户可通过环境变量切换 LLM 后端。

### 1.2 范围(本 SPEC 覆盖)

- 新增 `paper_review_workflow/llm/openai_provider.py` — OpenAI provider(GPT-4o 系列 / o3 / GPT-5 等,模型名由用户指定)
- 新增 `paper_review_workflow/llm/deepseek_provider.py` — DeepSeek provider(复用 openai SDK + base_url)
- 改造 `paper_review_workflow/llm/base.py` — `LLMProvider` ABC 加 `from_env()` abstractmethod
- 改造 `paper_review_workflow/llm/anthropic_provider.py` — 加 `from_env()` classmethod
- 改造 `paper_review_workflow/llm/client.py` — `LLMClient.from_env()` 不硬编码,调 `provider_cls.from_env()`;OpenAI/DeepSeek 必须设 `LLM_MODEL`
- 改造 `paper_review_workflow/llm/registry.py` — 注册 openai/deepseek
- `pyproject.toml` 加 `openai>=1.0` 依赖

### 1.3 不做

- Gemini provider(用户未选)
- 其他国产 provider(如通义千问、文心一言等)
- provider 自动选择逻辑(用户手动设 `LLM_PROVIDER`)
- 前端 provider 切换 UI(Phase 2 #3 前端已建,不做 provider 选择器)
- CI/CD + PyPI 发布(Phase 2 #7)

---

## 2. 整体架构

### 2.1 目录结构

```
paper_review_workflow/llm/
├── base.py                     # 改造:LLMProvider 加 from_env() abstractmethod
├── registry.py                 # 改造:_register_builtin 加 openai/deepseek
├── client.py                   # 改造:from_env() 调 provider_cls.from_env()
├── anthropic_provider.py       # 改造:加 from_env() classmethod
├── openai_provider.py          # ★ 新增:OpenAI provider
├── deepseek_provider.py        # ★ 新增:DeepSeek provider(用 openai SDK + base_url)
├── schemas.py                  # 不变
└── prompts/                    # 不变
```

### 2.2 架构层次

```
┌──────────────────────────────────────────────┐
│  DimensionAction / SynthesizeAction           │
│   (调 LLMClient.from_env().complete())        │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  LLMClient (singleton)                       │
│   from_env() → provider_cls.from_env()        │
│   _default_model() → Anthropic 有默认,      │
│                      OpenAI/DeepSeek 必填     │
└──────────────┬───────────────────────────────┘
               │ 委托
               ▼
┌──────────────────────────────────────────────┐
│  LLMProvider (ABC)                           │
│   from_env() ← abstractmethod                │
│   complete() ← abstractmethod                │
└──┬───────────┬───────────┬───────────────────┘
   │           │           │
   ▼           ▼           ▼
┌──────┐ ┌──────────┐ ┌──────────┐
│Anthro│ │ OpenAI   │ │ DeepSeek │
│pic   │ │ Provider │ │ Provider │
│      │ │          │ │          │
│from_ │ │ from_env │ │ from_env │
│env() │ │ (OPENAI_ │ │ (DEEPSEEK│
│(ANTH │ │ API_KEY) │ │ _API_KEY)│
│ROPIC │ │          │ │          │
│_API_ │ │ json_    │ │ json_    │
│KEY)  │ │ schema   │ │ object + │
│      │ │          │ │ prompt   │
│cache │ │ auto     │ │ no cache │
│_ctrl │ │ cache    │ │          │
└──────┘ └──────────┘ └──────────┘
```

### 2.3 关键设计原则

1. **`from_env()` 是 classmethod + abstractmethod**:每个 provider 负责读自己的 env var + 构造自己
2. **`LLMClient.from_env()` 不硬编码**:`provider_cls.from_env()` 统一入口
3. **OpenAI 用 `json_schema`**:`response_format={"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": True}}`
4. **DeepSeek 用 `json_object` + prompt schema**:DeepSeek 不支持 `json_schema`,用 `json_object` + 在 system prompt 里描述 schema
5. **DeepSeek 复用 `openai` SDK**:只改 `base_url`,不需额外依赖
6. **默认 model**:Anthropic 默认 `claude-sonnet-4-6`(已知有效);OpenAI/DeepSeek 不设默认,`LLM_MODEL` 必填

---

## 3. Provider 接口设计

### 3.1 `LLMProvider` ABC 改造

```python
# paper_review_workflow/llm/base.py

class LLMProvider(ABC):
    provider_name: str = ""

    @classmethod
    @abstractmethod
    def from_env(cls) -> "LLMProvider":
        """Read provider-specific env vars and construct the provider.
        Each subclass implements this to read its own API key + config."""
        ...

    @abstractmethod
    def complete(
        self,
        system: Union[str, list],
        messages: list,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        response_schema: Optional[Type[BaseModel]] = None,
        cached_context: Optional[str] = None,
    ) -> LLMResponse:
        ...
```

### 3.2 `AnthropicProvider.from_env()`

```python
class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"
    # __init__(api_key) + complete() 不变(Phase 1 M2.3 实现)

    @classmethod
    def from_env(cls) -> "AnthropicProvider":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY not set")
        return cls(api_key=api_key)
```

### 3.3 `OpenAIProvider`

```python
# paper_review_workflow/llm/openai_provider.py

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
        # 1. 拼 system text(OpenAI 自动缓存,不加 cache_control)
        system_text = self._build_system_text(system, cached_context)

        # 2. 构造 response_format(OpenAI json_schema)
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

        # 3. 指数退避重试
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "system", "content": system_text}] + messages,
                    response_format=response_format,
                )
                return self._parse_response(resp, response_schema, model)
            except Exception as e:
                # 重试逻辑(429/500/503/连接错误)
                # 400 context_length 不重试
                ...

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
        structured = None
        text_content = resp.choices[0].message.content
        if schema:
            try:
                data = json.loads(text_content)
                structured = schema(**data)
            except (json.JSONDecodeError, ValidationError) as e:
                raise SchemaValidationError(f"LLM output failed schema validation: {e}")
        usage = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "cache_creation_input_tokens": 0,  # OpenAI 不报告 cache 分项
            "cache_read_input_tokens": 0,
        }
        return LLMResponse(
            text=text_content if not schema else None,
            structured=structured,
            usage=usage,
            model=model,
            raw=resp,
        )
```

### 3.4 `DeepSeekProvider`

```python
# paper_review_workflow/llm/deepseek_provider.py

class DeepSeekProvider(LLMProvider):
    provider_name = "deepseek"
    BASE_URL = "https://api.deepseek.com"
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 1.0
    MAX_BACKOFF = 30.0

    def __init__(self, api_key: Optional[str] = None):
        from openai import OpenAI
        # DeepSeek 用 OpenAI SDK + 不同 base_url
        self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)

    @classmethod
    def from_env(cls) -> "DeepSeekProvider":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise LLMError("DEEPSEEK_API_KEY not set")
        return cls(api_key=api_key)

    def complete(self, system, messages, model, max_tokens,
                 temperature=0.0, response_schema=None, cached_context=None):
        # DeepSeek 用 json_object(不支持 json_schema)
        system_text = self._build_system_text(system, cached_context)

        response_format = None
        if response_schema:
            # 在 system prompt 里嵌入 schema 说明
            system_text += "\n\nYou must respond with JSON matching this schema:\n" + \
                           json.dumps(response_schema.model_json_schema(), indent=2)
            response_format = {"type": "json_object"}

        # 重试 + 调用(同 OpenAI 模式)
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "system", "content": system_text}] + messages,
                    response_format=response_format,
                )
                return self._parse_response(resp, response_schema, model)
            except Exception as e:
                # 重试逻辑
                ...

    def _build_system_text(self, system, cached_context):
        # 同 OpenAIProvider
        ...

    def _parse_response(self, resp, schema, model):
        # 同 OpenAIProvider
        ...
```

### 3.5 `LLMClient.from_env()` 重构

```python
# paper_review_workflow/llm/client.py

class LLMClient:
    @classmethod
    def from_env(cls) -> "LLMClient":
        if cls._instance is None:
            provider_name = os.environ.get("LLM_PROVIDER", "anthropic")
            registry = ProviderRegistry()
            provider_cls = registry.get(provider_name)
            provider = provider_cls.from_env()  # ★ 不再硬编码 Anthropic

            # Anthropic 有默认 model,其他 provider 必须设 LLM_MODEL
            model = os.environ.get("LLM_MODEL")
            if not model:
                if provider_name == "anthropic":
                    model = "claude-sonnet-4-6"
                else:
                    raise LLMError(f"LLM_MODEL not set (required for provider '{provider_name}')")

            cls._instance = cls(
                provider=provider,
                model=model,
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
                temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
            )
        return cls._instance
```

### 3.6 `ProviderRegistry._register_builtin()`

```python
def _register_builtin(self) -> None:
    from .anthropic_provider import AnthropicProvider
    self.register("anthropic", AnthropicProvider)
    from .openai_provider import OpenAIProvider
    self.register("openai", OpenAIProvider)
    from .deepseek_provider import DeepSeekProvider
    self.register("deepseek", DeepSeekProvider)
```

---

## 4. 结构化输出机制对比

| Provider | 机制 | Schema 传递方式 | 严格性 |
|---|---|---|---|
| Anthropic | tool use + `tool_choice` | `input_schema` 在 tool 定义里 | 强制(JSON 必须匹配 schema) |
| OpenAI | `response_format={"type":"json_schema"}` | `json_schema.schema` 在 response_format 里 | 强制(strict=True) |
| DeepSeek | `response_format={"type":"json_object"}` + prompt | schema 描述在 system prompt 里 | 弱(LLM 自由返回 JSON,可能不匹配) |

**DeepSeek 的 schema 校验**:由于 `json_object` 不强制 schema,`_parse_response` 仍做 Pydantic 校验,失败则 `SchemaValidationError`(同 Anthropic/OpenAI)。LLM 偶尔返回不匹配 schema 的 JSON 会被捕获。

---

## 5. Prompt Cache 处理

| Provider | Cache 机制 | `cached_context` 参数处理 |
|---|---|---|
| Anthropic | `cache_control: {"type": "ephemeral"}` | 拼 system block + 加 `cache_control` 标记 |
| OpenAI | 自动 prompt prefix cache | 拼 system text,**不加标记**(OpenAI 自动缓存) |
| DeepSeek | 无 cache | 拼 system text,**无缓存优势** |

---

## 6. 重试策略(3 provider 统一)

所有 provider 复用 AnthropicProvider 的重试模式:
- **3 次重试**,指数退避(1s → 2s → 4s)+ 10% 抖动,最大 30s
- **429/500/503/连接错误**:重试
- **400 context_length**:不重试,抛 `ContextLengthError`
- **schema 校验失败**:不重试,抛 `SchemaValidationError`
- **3 次后仍失败**:抛 `RateLimitError`(429)或原始异常

---

## 7. 测试策略

### 7.1 测试金字塔

```
        ┌─────────────┐
        │  E2E (1%)    │  真实 OpenAI/DeepSeek API(marked,跳过默认)
        └─────────────┘
       ┌───────────────┐
       │ Integration    │  LLMClient.from_env() 切换 provider + venue 评审
       │    (20%)       │
       └───────────────┘
     ┌───────────────────┐
     │     Unit (79%)     │  Provider mock 测试
     └───────────────────┘
```

### 7.2 关键测试场景

**单元**(mock SDK 调用):
- `test_openai_provider.py`:6+ tests — 结构化输出(json_schema)、cached_context 拼入 system、重试(429/500/503)、context_length 不重试、schema 校验失败、API key 缺失
- `test_deepseek_provider.py`:6+ tests — json_object + prompt schema、cached_context、重试、context_length、schema 校验、API key 缺失
- `test_llm_client_multi_provider.py`:`LLMClient.from_env()` 切 openai/deepseek/anthropic + 默认 model + API key 缺失报错 + `LLM_MODEL` 缺失报错(openai/deepseek)
- `test_provider_registry.py`:`list_providers()` 返回 3 个,`get("openai")`/`get("deepseek")` 正确

**集成**:
- `test_venue_review_openai.py`:用 OpenAI provider(mock)跑通 NeurIPS 评审
- `test_venue_review_deepseek.py`:用 DeepSeek provider(mock)跑通 NeurIPS 评审

**E2E**(marked):
- `test_e2e_openai_arxiv.py`:真实 OpenAI API + arXiv 论文
- `test_e2e_deepseek_arxiv.py`:真实 DeepSeek API + arXiv 论文

---

## 8. 实施阶段

| 里程碑 | 范围 | 预估工时 |
|---|---|---|
| **M1: base.py + AnthropicProvider from_env + LLMClient 重构** | `LLMProvider` 加 `from_env()` abstractmethod + `AnthropicProvider.from_env()` + `LLMClient.from_env()` 重构 + 单元测试 | 0.5 天 |
| **M2: OpenAIProvider** | `openai_provider.py` + `openai` 依赖 + 单元测试(6 tests) | 1 天 |
| **M3: DeepSeekProvider** | `deepseek_provider.py`(复用 openai SDK) + 单元测试(6 tests) | 0.5 天 |
| **M4: Registry + LLMClient 集成** | `ProviderRegistry` 注册 openai/deepseek + 集成测试 | 0.5 天 |
| **M5: 集成测试 + E2E + README + tag** | 2 个 venue_review 集成测试 + 2 个 E2E marked + README + tag v0.5.0 | 1 天 |

**Phase 2 #2 总预估: ~3.5 天**

---

## 9. 验收标准

- [ ] `LLMProvider` ABC 有 `from_env()` abstractmethod
- [ ] `AnthropicProvider.from_env()` 读 `ANTHROPIC_API_KEY`
- [ ] `OpenAIProvider` 用 `openai` SDK,`response_format={"type":"json_schema",...}`
- [ ] `DeepSeekProvider` 用 `openai` SDK + `base_url="https://api.deepseek.com"`,`response_format={"type":"json_object"}` + prompt schema
- [ ] `LLMClient.from_env()` 不硬编码 Anthropic,调 `provider_cls.from_env()`
- [ ] Anthropic 有默认 model `claude-sonnet-4-6`;OpenAI/DeepSeek 必须设 `LLM_MODEL`
- [ ] `ProviderRegistry.list_providers()` 返回 `["anthropic", "openai", "deepseek"]`
- [ ] 3 个 provider 都有重试逻辑(3 次指数退避)
- [ ] 3 个 provider 都有 schema 校验失败 → `SchemaValidationError`
- [ ] `pyproject.toml` 加 `openai>=1.0`
- [ ] 单元测试覆盖重试/cache/schema/missing key
- [ ] 集成测试用 mock OpenAI/DeepSeek 跑通 NeurIPS 评审
- [ ] E2E 测试 marked(跳过默认)
- [ ] README 更新(多 provider 使用说明)
- [ ] tag v0.5.0

---

## 10. 配置示例

```bash
# OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-4o  # 用户指定当前可用模型名

# DeepSeek
export LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=sk-...
export LLM_MODEL=deepseek-chat

# Anthropic(不变)
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_MODEL=claude-sonnet-4-6  # 可省略,有默认值
```

---

## 附录 A: 依赖更新

`pyproject.toml` 增加依赖:

```toml
dependencies = [
    # ... 现有 ...
    "openai>=1.0",  # OpenAI + DeepSeek 都用此 SDK
]
```

DeepSeek 不需要单独的 SDK,复用 `openai` 包 + `base_url` 参数。

---

**END OF SPEC**
