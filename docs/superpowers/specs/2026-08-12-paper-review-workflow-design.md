# Paper Review Workflow (PRW) — SPEC-PRD

**版本**: 1.0
**日期**: 2026-08-12
**状态**: 设计确认中

---

## 1. 项目定位与范围

### 1.1 一句话定位

一个 Python 单体本地工作流引擎,通过可组合的 AI 组件对学术论文进行多维度评审打分,生成结构化评审报告。

### 1.2 双模式

- **`normal` 模式(Phase 1 本次范围)**: 通用 8 维度评审规则,适用于 CS 论文通用评审。
- **`venue-specific` 模式(Phase 2 后续)**: 模拟特定会议/期刊(NeurIPS/ICML/ACL 等)的评审规则。

### 1.3 Phase 1 范围(本 SPEC 覆盖)

- 复用 lwf(`for-staging/lwf`)骨架,改造为论文评审工作流
- 实现 normal 模式 YAML 配置 + 8 维度评审组件
- 输入支持:本地 PDF + arXiv ID/URL
- LLM Provider 抽象层,仅实现 Anthropic
- CLI 双模式入口(`run` / `resume` + 查询命令)
- filesystem 即记忆 + lwf storage 持久化 + 断点续跑

### 1.4 Phase 1 不做

- venue-specific 模式
- 其他 LLM Provider(OpenAI/Gemini/DeepSeek)
- 简易前端(组件装配 YAML)
- OpenReview XML 导出
- 人工审批/交互式步骤(lwf 已有,但本场景不需要)
- FastAPI server + WebSocket
- 多论文批量评审
- CI/CD pipeline 与 PyPI 发布

---

## 2. 整体架构

### 2.1 目录结构

```
paper-review-workflow/
├── main.py                          # 双模式入口
├── pyproject.toml                   # uv/pip 包管理
├── requirements.txt                 # 兼容 lwf 习惯
├── configs/                         # 评审 YAML 配置
│   ├── normal_review.yaml
│   ├── minimal.yaml                 # M1 验证用
│   └── examples/
├── examples/                        # 示例输入
├── sessions/                        # 评审产物根目录(运行时生成)
└── paper_review_workflow/           # 核心包(从 lwf workflow_engine/ 改造)
    ├── __init__.py
    ├── engine.py                    # ReviewEngine 门面层
    ├── cli.py                       # argparse 子命令分发
    ├── core/
    │   ├── models.py                # WorkflowRun / JobInstance / StepInstance
    │   ├── parser.py                # YAML 解析(jobs/steps/needs/uses)
    │   ├── context.py               # WorkflowContext + 表达式求值
    │   ├── state_machine.py         # 三层状态机
    │   ├── event_bus.py             # 事件总线
    │   └── paths.py                 # session 路径生成
    ├── executors/
    │   ├── workflow_executor.py     # Job 调度(needs 拓扑 + 并行)
    │   ├── job_executor.py          # matrix 策略执行
    │   └── step_executor.py         # Action 分发 + 重试 + 超时
    ├── actions/                     # ★ 评审专用组件
    │   ├── base.py                  # BaseAction + ActionResult
    │   ├── registry.py              # ActionRegistry(单例)
    │   ├── builtin.py               # register_builtin_actions()
    │   ├── extract/                 # 论文抽取
    │   │   ├── __init__.py          # ExtractAction
    │   │   ├── pdf.py               # PyMuPDF 解析
    │   │   └── arxiv.py             # arXiv latex + PDF 回退
    │   ├── dimensions/
    │   │   ├── base_dim.py          # DimensionAction 通用基类
    │   │   └── prompts/             # 8 个维度 prompt 模板
    │   ├── synthesize.py            # 汇总 8 维 → review.md
    │   └── decide.py                # 加权评分 → decision.json
    ├── llm/                         # ★ LLM Provider 抽象层
    │   ├── base.py                  # LLMProvider ABC + LLMResponse
    │   ├── anthropic_provider.py    # Phase 1 唯一实现
    │   ├── registry.py              # ProviderRegistry
    │   ├── client.py                # LLMClient 委托层
    │   ├── schemas.py               # Pydantic schema
    │   └── prompts/                 # Jinja2 模板
    ├── storage/                     # 复用 lwf
    │   ├── backend.py
    │   ├── memory.py
    │   └── json_file.py
    └── api/                         # (Phase 2) FastAPI+WS
        └── server.py
```

### 2.2 架构层次

```
┌──────────────────────────────────────────────────────┐
│  CLI  (main.py, cli.py)                              │
├──────────────────────────────────────────────────────┤
│  Engine  (engine.py)                                 │
│   - ReviewEngine 门面层                              │
│   - run / resume / cancel / shutdown / list / show   │
├──────────────────────────────────────────────────────┤
│  Executors  (executors/)                             │
│   - WorkflowExecutor: Job 拓扑+并行                  │
│   - JobExecutor: matrix 策略                         │
│   - StepExecutor: Action 分发+重试+超时              │
├──────────────────────────────────────────────────────┤
│  Actions  (actions/)                                 │
│   - extract / dimensions(matrix) / synthesize / decide│
│   - 通过 registry 查找,实例缓存(无状态)             │
├──────────────────────────────────────────────────────┤
│  LLM  (llm/)                                         │
│   - LLMProvider 抽象 + AnthropicProvider             │
│   - 同步 Anthropic SDK + tool use + prompt cache     │
├──────────────────────────────────────────────────────┤
│  Core  (core/)                                       │
│   - models / parser / context / state_machine        │
│   - event_bus / paths                                │
├──────────────────────────────────────────────────────┤
│  Storage  (storage/)                                 │
│   - memory / json_file                               │
└──────────────────────────────────────────────────────┘
```

### 2.3 关键设计原则

1. **依赖方向单向向下**: CLI → Engine → Actions → LLM → Core → Storage
2. **Actions 严格无状态**: 实例缓存跨 run 共享,所有参数从 `params`/`env`/`context` 注入
3. **LLM Provider 可插拔**: 通过环境变量 `LLM_PROVIDER` 选择,Phase 1 仅 anthropic
4. **filesystem 即记忆**: 组件通过 `run_manifest.json` 的路径表 + 约定目录互相定位产物,不直接互相调用
5. **事件驱动持久化**: 状态变更 → EventBus → 自动 `storage.save_run`,断点续跑基础

---

## 3. 数据模型与 Session 目录结构

### 3.1 Session 目录结构

```
sessions/
└── <paper_id>/                          # 论文级
    └── <run_id>/                        # 单次评审,格式 20260812-143022-a1b2
        ├── run.json                     # ★ lwf storage:WorkflowRun 序列化(含三层状态)
        ├── run_manifest.json            # ★ 静态档案(paper_id/mode/config/llm/weights)
        ├── run.log                      # 全局日志
        ├── inputs/
        │   └── source.yaml              # paper_source + 配置快照
        ├── 00_extract/
        │   ├── metadata.json
        │   ├── sections.json
        │   ├── full_text.md
        │   ├── references.json
        │   ├── extract.log
        │   └── artifacts/
        │       └── paper.pdf
        ├── 10_dim_novelty/
        │   ├── score.json
        │   ├── review.md
        │   └── dim_novelty.log
        ├── 10_dim_soundness/
        ├── 10_dim_significance/
        ├── 10_dim_clarity/
        ├── 10_dim_reproducibility/
        ├── 10_dim_related_work/
        ├── 10_dim_positioning/
        ├── 10_dim_presentation/
        ├── 50_synthesize/
        │   ├── review.md
        │   ├── scores.json
        │   └── synthesize.log
        ├── 60_decision/
        │   └── decision.json
        └── final_report.md              # 软链接 → 50_synthesize/review.md
```

### 3.2 run.json(lwf storage 持久化的 WorkflowRun)

```json
{
  "id": "20260812-143022-a1b2",
  "workflow_name": "normal-paper-review",
  "status": "running",
  "trigger_type": "workflow_dispatch",
  "trigger_payload": {"paper_source": "2402.12098"},
  "start_time": "2026-08-12T14:30:22Z",
  "end_time": null,
  "duration": null,
  "run_number": 1,
  "jobs": {
    "extract": {
      "id": "...", "name": "📄 论文抽取",
      "status": "completed",
      "start_time": "...", "end_time": "...",
      "outputs": {
        "paper_id": "a1b2c3d4",
        "full_text_path": "00_extract/full_text.md",
        "metadata_path": "00_extract/metadata.json",
        "sections_path": "00_extract/sections.json",
        "references_path": "00_extract/references.json"
      },
      "steps": [
        {"id": "extract", "name": "extract", "status": "success",
         "outputs": {...}, "log": [...], "duration": 12.4}
      ]
    },
    "dimensions_novelty":      {"status": "completed", "outputs": {"score": 4, "confidence": 0.85}},
    "dimensions_soundness":    {"status": "failed", "error_msg": "API timeout"},
    "dimensions_significance": {"status": "completed", ...},
    "dimensions_clarity":      {"status": "completed", ...},
    "dimensions_reproducibility":{"status": "completed", ...},
    "dimensions_related_work": {"status": "completed", ...},
    "dimensions_positioning":  {"status": "completed", ...},
    "dimensions_presentation": {"status": "completed", ...},
    "synthesize":              {"status": "pending"},
    "decide":                  {"status": "pending"}
  }
}
```

**关键**: matrix strategy 展开后,每个维度是独立 `JobInstance`,各自有独立 status。断点续跑时只重跑失败的单个维度。

### 3.3 run_manifest.json(静态档案)

```json
{
  "schema_version": "1.0",
  "run_id": "20260812-143022-a1b2",
  "paper_id": "a1b2c3d4",
  "created_at": "2026-08-12T14:30:22Z",
  "status": "running",

  "paper_source": {
    "type": "arxiv",
    "value": "2402.12098",
    "hash": "sha256:..."
  },

  "mode": "normal",

  "config_snapshot": {
    "config_path": "configs/normal_review.yaml",
    "config_hash": "sha256:...",
    "config_content": {}
  },

  "llm_config": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "max_tokens": 4096,
    "temperature": 0.0
  },

  "paper_metadata": {
    "title": "...",
    "authors": [],
    "abstract": "...",
    "doi": null,
    "arxiv_id": "2402.12098",
    "keywords": []
  },

  "weights": {
    "novelty": 1.0, "soundness": 1.2, "significance": 1.2,
    "clarity": 0.8, "reproducibility": 1.0,
    "related_work": 0.8, "positioning": 0.8, "presentation": 0.8
  }
}
```

### 3.4 score.json schema(每个维度)

```json
{
  "schema_version": "1.0",
  "dimension": "novelty",
  "score": 4,
  "confidence": 0.85,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "justification": "...(200-800 字)",
  "evidence": [
    {"section": "3.2", "quote": "...", "page": 5}
  ],
  "model_used": "claude-sonnet-4-6",
  "usage": {
    "input_tokens": 12345,
    "output_tokens": 678,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 30000
  }
}
```

### 3.5 decision.json schema

```json
{
  "schema_version": "1.0",
  "recommendation": "weak_accept",
  "weighted_score": 3.85,
  "per_dimension": {
    "novelty":           {"score": 4, "confidence": 0.85, "weighted": 4.0},
    "soundness":         {"score": 4, "confidence": 0.90, "weighted": 4.0}
  },
  "decision_rationale": "...(200-400 字)",
  "key_concerns": ["..."],
  "key_strengths": ["..."],
  "weights_used": {}
}
```

### 3.6 断点续跑规则

1. Engine 启动时,从 storage 加载 `WorkflowRun` 对象
2. 跳过 `status == completed` 的 Job
3. 对 `status == failed` 或 `running`(中断的)的 Job,重置为 `pending` 重新执行
4. `--rerun <components>` 时,重置指定组件及其下游为 `pending`
5. `--rerun-all` 时,所有 Job 重置为 `pending`,清空产物目录。`run_manifest.json` 保留顶层元数据(run_id/paper_id/created_at/paper_source/mode/config_snapshot/llm_config/weights),但 `paper_metadata` 字段重置为 null,由 extract 重新填充

### 3.7 SESSION_DIR 与 paper_id 的时序

`paper_id` 由 extract 组件完成后才知道(依赖 title)。处理流程:

1. **run 启动时**:Engine 生成 `run_id`(时间戳+随机),先用临时 paper_id `pending-<run_id>` 创建 session 目录 `sessions/pending-<run_id>/<run_id>/`
2. **extract 完成**:从论文 title 计算 `paper_id`,把 session 目录 rename 到 `sessions/<paper_id>/<run_id>/`
3. **后续组件**:通过 `env.SESSION_DIR` 读取真实路径(extract 完成后由 Engine 更新 env)
4. **manifest 更新**:extract 完成后,Engine 写入 `paper_id` 和 `paper_metadata` 到 `run_manifest.json`

`env.SESSION_DIR` 在 YAML 中作为占位符,实际值由 Engine 在 extract 完成后注入到下游 Job 的 env 中。extract 自身从 `params["session_dir"]` 拿到的是初始临时路径。

---

## 4. 组件接口契约

### 4.1 BaseAction 抽象基类(照搬 lwf 同步接口)

```python
# paper_review_workflow/actions/base.py
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

class ActionResult:
    """二态:success/fail(去掉 lwf 的 waiting 态,论文评审无人工审批)"""
    def __init__(
        self,
        success: bool,
        outputs: Optional[Dict[str, Any]] = None,
        message: str = "",
        log_lines: Optional[List[str]] = None,
        exit_code: int = 0,
    ):
        self.success = success
        self.outputs = outputs or {}
        self.message = message
        self.log_lines = log_lines or []
        self.exit_code = 0 if success else (exit_code or 1)

class BaseAction(ABC):
    """与 lwf 完全一致的接口"""

    @property
    def description(self) -> str:
        return ""

    @abstractmethod
    def run(
        self,
        params: Dict[str, Any],         # 来自 YAML with 字段
        env: Dict[str, str],            # 合并后的环境变量
        context: Dict,                  # 完整 workflow context 快照
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> ActionResult:
        pass
```

### 4.2 Registry(照搬 lwf,缓存实例)

```python
# paper_review_workflow/actions/registry.py
class ActionRegistry:
    """单例,缓存实例(与 lwf 一致)。Action 必须严格无状态"""
    _instance: Optional["ActionRegistry"] = None
    _actions: Dict[str, BaseAction] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._actions = {}
        return cls._instance

    def register(self, name: str, action: BaseAction) -> None:
        self._actions[name] = action

    def get(self, name: str) -> Optional[BaseAction]:
        if name in self._actions:
            return self._actions[name]
        base_name = name.split("@")[0]
        return self._actions.get(base_name)
```

### 4.3 维度组件共享基类

```python
# paper_review_workflow/actions/dimensions/base_dim.py
class DimensionAction(BaseAction):
    """通用维度打分 action,通过 with.dimension 区分 8 个维度"""

    @property
    def description(self) -> str:
        return "Score one dimension of a paper"

    def run(self, params, env, context, log_callback=None) -> ActionResult:
        dimension = params["dimension"]
        session_dir = params["session_dir"]
        full_text_path = params["full_text_path"]

        # 1. 读论文全文(从 extract 产物)
        with open(full_text_path) as f:
            paper_text = f.read()

        # 2. 加载 prompt 模板
        prompt = self._render_prompt(dimension)

        # 3. 调 LLM(同步,tool use 强制结构化输出)
        client = LLMClient.from_env()
        score = client.score(
            system=prompt,
            user_content=f"Score the {dimension} dimension.",
            schema=DimensionScore,
            cached_context=paper_text,  # ★ 8 维共享 prompt cache
        )

        # 4. 落盘 JSON + MD
        out_dir = Path(session_dir) / f"10_dim_{dimension}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write_score_json(out_dir / "score.json", score, dimension)
        self._write_review_md(out_dir / "review.md", score, dimension)

        # 5. 返回 outputs(写入 context,供下游引用)
        return ActionResult(
            success=True,
            outputs={
                "score": score.score,
                "confidence": score.confidence,
                "score_path": str(out_dir / "score.json"),
                "review_path": str(out_dir / "review.md"),
            },
            log_lines=[f"[{dimension}] score={score.score} conf={score.confidence}"],
        )
```

### 4.4 组件清单

| Action 名 | 文件 | 职责 | 调 LLM |
|---|---|---|---|
| `paper-review/extract@v1` | `actions/extract/__init__.py` | 解析 PDF/arXiv → metadata/sections/full_text/references | 是(元数据补全) |
| `paper-review/dim_score@v1` | `actions/dimensions/base_dim.py` | 单维度打分(通用,matrix + with.dimension 区分) | 是 |
| `paper-review/synthesize@v1` | `actions/synthesize.py` | 读 8 维 score.json → 调 LLM 综合评审报告 | 是 |
| `paper-review/decide@v1` | `actions/decide.py` | 加权评分 + 映射 7 档决定(纯规则) | 否 |

---

## 5. 执行模型与并发

### 5.1 执行流拓扑

```
extract (单 Job)
    │ needs: extract
    ▼
┌──dimensions matrix Job ────────────────────────────┐
│  matrix_pool (ThreadPoolExecutor, max_parallel=8)  │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│  │novel │sound │signif│clari │repro │relat │posit │present│
│  │_ty   │_ness │_icanc│_ty   │_ducib│_work │_ionin│_ation │
│  └──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──┬───┴──┬───┘
│     └──────┴──────┴──────┴──────┴──────┴──────┴──────┘
│                wait(FIRST_COMPLETED) 流式收尾
└────────────────────────┬───────────────────────────┘
                         │ needs: dimensions
                         ▼
                synthesize (单 Job,读 FS 8 维结果)
                         │ needs: synthesize
                         ▼
                    decide (单 Job,纯规则计算)
```

### 5.2 三层并发模型

| 层级 | 机制 | 来源 | 并发度 |
|---|---|---|---|
| Job 级 | `workflow_executor._schedule_jobs` 的 `ThreadPoolExecutor` | lwf 复用 | max_workers = len(jobs) |
| Matrix 级 | `job_executor._execute_matrix` 的 `matrix_pool` | lwf 复用 | max_parallel = len(combos) = 8 |
| LLM 调用级 | Anthropic SDK 同步阻塞,线程内串行 | 自封装 | 单线程内 1 次,跨 matrix 线程并行 |

### 5.3 filesystem 即记忆

每个 matrix sub-job 把产物写到约定路径 `10_dim_<dimension>/score.json`。`synthesize` Job 不依赖 lwf 的 matrix outputs 聚合(因为 lwf `_execute_matrix` 不聚合子 Job outputs 到父 Job),而是直接 glob 文件系统:

```python
class SynthesizeAction(BaseAction):
    def run(self, params, env, context, log_callback):
        session_dir = params["session_dir"]
        dim_results = {}
        missing_dims = []
        for dim in ALL_8_DIMENSIONS:
            score_path = Path(session_dir) / f"10_dim_{dim}" / "score.json"
            if score_path.exists():
                dim_results[dim] = json.loads(score_path.read_text())
            else:
                missing_dims.append(dim)
        if len(missing_dims) > 2:
            return ActionResult(success=False,
                message=f"too many missing dimensions: {missing_dims}")
        # 调 LLM 综合(缺失维度标 N/A,权重归零)
        ...
```

### 5.4 维度独立性(评审学理)

8 个维度在评审学理上**完全独立**,无先后依赖。每个维度只读 extract 产物,不需要看其他维度的结果。事实上,维度独立打分**可以避免锚定偏差**(anchoring bias),NeurIPS 等会议官方指南也强调维度独立评分。

`synthesize` 扮演 meta-reviewer 角色,读 8 维结果综合成统一报告。

### 5.5 Prompt Cache 共享策略

8 个维度共享同一份论文全文(10K-50K token),通过 Anthropic prompt cache 节省成本。

**Prompt cache 是 Anthropic 服务端的能力**:
- 我们在 API 请求里声明 `cache_control: {"type": "ephemeral"}`
- Anthropic 服务端自动存储 prefix,5 分钟 TTL
- 后续请求前缀字节级一致时自动命中,按 1/10 价格计费
- 首次写入按 1.25 倍价格

**实现**:`LLMClient.score()` 的 `cached_context` 参数自动转成带 `cache_control` 的 system block。

**预期收益**:8 维总 input 成本从 $0.72 降到 $0.18,节省 76%。

**约束**:
- 前缀必须字节级一致(8 维读同一份 `full_text.md` 保证)
- 最小 1024 tokens(论文全文远超)
- TTL 5 分钟(8 维并行通常在 5 分钟内完成)

---

## 6. LLM Provider 抽象层

### 6.1 抽象接口

```python
# llm/base.py
class LLMProvider(ABC):
    provider_name: str = ""

    @abstractmethod
    def complete(
        self,
        system: str | list[dict],
        messages: list[dict],
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        response_schema: Optional[Type[BaseModel]] = None,
        cached_context: Optional[str] = None,
    ) -> LLMResponse:
        """同步阻塞调用。子类负责:
        1. 把 cached_context 包装成 provider 特定的 cache_control 格式
        2. 把 response_schema 转成 provider 特定的结构化输出机制
        3. 实现重试(指数退避,429/500/503)
        4. 返回统一 LLMResponse
        """
```

### 6.2 AnthropicProvider 实现

- 同步 `Anthropic` client,内部连接池线程安全
- `cached_context` → `cache_control: {"type": "ephemeral"}` system block
- `response_schema` → tool use + `tool_choice` 强制结构化输出
- 指数退避重试:429/500/503/连接错误,3 次,初始 1s,最大 30s,带 10% 抖动
- 上下文超限(400)不重试,直接抛 `ContextLengthError`

### 6.3 LLMClient 委托层

```python
class LLMClient:
    """全局共享,从环境变量加载 provider,类级单例"""
    _instance: Optional["LLMClient"] = None

    @classmethod
    def from_env(cls) -> "LLMClient":
        if cls._instance is None:
            provider_name = os.environ.get("LLM_PROVIDER", "anthropic")
            provider = ProviderRegistry.get(provider_name)()
            cls._instance = cls(
                provider=provider,
                model=os.environ.get("LLM_MODEL", "claude-sonnet-4-6"),
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
                temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
            )
        return cls._instance

    def score(self, system, user_content, schema, cached_context=None) -> BaseModel:
        """便捷方法:调一次 LLM,返回结构化对象"""
        resp = self.complete(
            system=system,
            messages=[{"role": "user", "content": user_content}],
            response_schema=schema,
            cached_context=cached_context,
        )
        return resp.structured
```

### 6.4 Pydantic Schemas

```python
class DimensionScore(BaseModel):
    score: int = Field(ge=1, le=5, description="1-5 OpenReview 制")
    confidence: float = Field(ge=0.0, le=1.0)
    strengths: list[str] = Field(min_length=1, max_length=5)
    weaknesses: list[str] = Field(min_length=1, max_length=5)
    justification: str = Field(min_length=100, max_length=800)
    evidence: list[dict] = Field(default_factory=list)

class PaperMetadata(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    doi: str | None = None
    arxiv_id: str | None = None
    keywords: list[str] = Field(default_factory=list)

class SynthesisResult(BaseModel):
    summary: str = Field(min_length=200, max_length=1500)
    key_strengths: list[str]
    key_weaknesses: list[str]
    questions_for_authors: list[str]
    overall_assessment: str
```

### 6.5 环境变量配置

```yaml
# configs/normal_review.yaml
env:
  LLM_PROVIDER: anthropic
  LLM_MODEL: claude-sonnet-4-6
  LLM_MAX_TOKENS: "4096"
  LLM_TEMPERATURE: "0.0"
  ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
```

### 6.6 线程安全分析

| 对象 | 线程安全 | 说明 |
|---|---|---|
| `LLMClient._instance` | ✅ | Python GIL 保证首次赋值原子,后续只读 |
| `AnthropicProvider._client` | ✅ | Anthropic SDK 内部用 urllib3 连接池,线程安全 |
| 8 个 matrix 线程并发调 `client.messages.create` | ✅ | 各自阻塞等待,SDK 内部连接池处理并发 |

### 6.7 Phase 2 扩展点

新增 OpenAI/Gemini/DeepSeek provider:
1. 新建 `<provider>_provider.py`,实现 `LLMProvider.complete()`
2. 把 `response_schema` 转成 provider 特定格式(OpenAI `response_format` / Gemini `response_schema`)
3. 把 `cached_context` 转成 provider 特定 cache 机制
4. `ProviderRegistry._register_builtin()` 注册
5. **组件代码不变**,只改 YAML 的 `LLM_PROVIDER`

---

## 7. YAML 配置与组件清单

### 7.1 默认 normal 模式 YAML

```yaml
# configs/normal_review.yaml
name: normal-paper-review

on:
  workflow_dispatch:
    inputs:
      paper_source:
        description: "论文来源(PDF 路径或 arXiv ID/URL)"
        required: true
        type: string
      mode:
        description: "评审模式"
        type: choice
        default: normal
        options: [normal]

env:
  LLM_PROVIDER: anthropic
  LLM_MODEL: claude-sonnet-4-6
  LLM_MAX_TOKENS: "4096"
  LLM_TEMPERATURE: "0.0"
  SESSIONS_ROOT: ./sessions
  SESSION_DIR: "${{ env.SESSIONS_ROOT }}/${{ env.PAPER_ID }}/${{ env.RUN_ID }}"
  WEIGHT_NOVELTY: "1.0"
  WEIGHT_SOUNDESS: "1.2"
  WEIGHT_SIGNIFICANCE: "1.2"
  WEIGHT_CLARITY: "0.8"
  WEIGHT_REPRODUCIBILITY: "1.0"
  WEIGHT_RELATED_WORK: "0.8"
  WEIGHT_POSITIONING: "0.8"
  WEIGHT_PRESENTATION: "0.8"

jobs:
  extract:
    name: "📄 论文抽取"
    runs-on: local
    outputs:
      paper_id: ${{ steps.extract.outputs.paper_id }}
      full_text_path: ${{ steps.extract.outputs.full_text_path }}
      metadata_path: ${{ steps.extract.outputs.metadata_path }}
      sections_path: ${{ steps.extract.outputs.sections_path }}
      references_path: ${{ steps.extract.outputs.references_path }}
    steps:
      - id: extract
        uses: paper-review/extract@v1
        with:
          source: ${{ inputs.paper_source }}
          session_dir: ${{ env.SESSION_DIR }}

  dimensions:
    name: "📊 维度打分"
    needs: extract
    strategy:
      matrix:
        dimension:
          - novelty
          - soundness
          - significance
          - clarity
          - reproducibility
          - related_work
          - positioning
          - presentation
      max-parallel: 8
    runs-on: local
    steps:
      - uses: paper-review/dim_score@v1
        with:
          dimension: ${{ matrix.dimension }}
          session_dir: ${{ env.SESSION_DIR }}
          full_text_path: ${{ needs.extract.outputs.full_text_path }}
          metadata_path: ${{ needs.extract.outputs.metadata_path }}

  synthesize:
    name: "📝 综合评审"
    needs: dimensions
    runs-on: local
    outputs:
      review_path: ${{ steps.synthesize.outputs.review_path }}
      scores_path: ${{ steps.synthesize.outputs.scores_path }}
    steps:
      - id: synthesize
        uses: paper-review/synthesize@v1
        with:
          session_dir: ${{ env.SESSION_DIR }}

  decide:
    name: "🎯 推荐决定"
    needs: synthesize
    runs-on: local
    steps:
      - uses: paper-review/decide@v1
        with:
          session_dir: ${{ env.SESSION_DIR }}
          scores_path: ${{ needs.synthesize.outputs.scores_path }}
```

### 7.2 组件清单

| Action 名 | 文件 | 职责 | 输入(with) | 输出(outputs) | 调 LLM |
|---|---|---|---|---|---|
| `paper-review/extract@v1` | `actions/extract/__init__.py` | 解析 PDF/arXiv | source, session_dir | paper_id, full_text_path, metadata_path, sections_path, references_path | 是 |
| `paper-review/dim_score@v1` | `actions/dimensions/base_dim.py` | 单维度打分 | dimension, session_dir, full_text_path, metadata_path | score, confidence, score_path, review_path | 是 |
| `paper-review/synthesize@v1` | `actions/synthesize.py` | 综合 8 维 | session_dir | review_path, scores_path | 是 |
| `paper-review/decide@v1` | `actions/decide.py` | 加权决定 | session_dir, scores_path | decision_path, recommendation, weighted_score | 否 |

### 7.3 Extract 组件设计

- 输入判断:arXiv ID(纯数字或 arxiv.org URL)vs 本地 PDF 路径
- arXiv:优先 latex 源码 tarball,失败回退 PDF
- PDF:PyMuPDF 解析,提取文本/章节/参考文献
- LLM 补全 metadata(若 PyMuPDF 抽不到 abstract/keywords)
- 落盘:metadata.json / sections.json / full_text.md / references.json / artifacts/paper.pdf
- 计算 paper_id(title 的 sha256 前 8 位)

### 7.4 Decide 组件(纯规则,不调 LLM)

```python
class DecideAction(BaseAction):
    WEIGHTS = {
        "novelty": 1.0, "soundness": 1.2, "significance": 1.2,
        "clarity": 0.8, "reproducibility": 1.0,
        "related_work": 0.8, "positioning": 0.8, "presentation": 0.8,
    }

    THRESHOLDS = [
        (4.5, "strong_accept"),
        (4.0, "accept"),
        (3.5, "weak_accept"),
        (3.0, "borderline"),
        (2.0, "weak_reject"),
        (1.5, "reject"),
        (0.0, "strong_reject"),
    ]
```

---

## 8. CLI 接口与运行模式

### 8.1 命令结构

```bash
# 双模式入口
python main.py run <yaml> [options]     # CLI 直接运行(Phase 1 主用)
python main.py server [options]         # API 服务器(Phase 2)

# 断点续跑与查询
python main.py resume <run_id> [--rerun <components>] [--rerun-all]
python main.py list-runs [--paper-id <id>] [--status <status>] [--limit 50]
python main.py show-run <run_id>
python main.py clean <run_id> [--force]
```

### 8.2 `run` 子命令

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `yaml` | 是 | - | YAML 配置文件路径(`-` 从 stdin) |
| `--trigger` | 否 | `workflow_dispatch` | 触发类型 |
| `--payload` | 否 | `{}` | JSON 字符串,含 inputs |
| `--env` | 否 | - | 覆盖环境变量,可多次 |
| `--log-level` | 否 | `INFO` | DEBUG/INFO/WARNING/ERROR |
| `--storage` | 否 | `json` | memory/json |
| `--storage-dir` | 否 | `./sessions` | JSON 存储目录 |

### 8.3 `resume` 子命令

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `run_id` | 是 | - | 续跑的 run ID |
| `--rerun` | 否 | - | 逗号分隔的组件名,重置这些及下游为 pending |
| `--rerun-all` | 否 | - | 重置所有组件,清空产物目录 |

### 8.4 ReviewEngine 门面层

```python
class ReviewEngine:
    """论文评审工作流引擎(门面层,简化自 lwf WorkflowEngine)"""

    def __init__(self, storage=None, sessions_root="./sessions"):
        self.parser = WorkflowParser()
        self.registry = ActionRegistry()
        self.event_bus = EventBus()
        self.storage = storage or JsonFileStorage(sessions_root)
        self.storage.open()
        self._active_runs = {}
        self._coordinators = {}
        self._workflow_executor = WorkflowExecutor(
            registry=self.registry, event_bus=self.event_bus,
            on_coordinator_created=self._on_coordinator_created,
            on_workflow_started=self._on_workflow_started,
        )
        register_builtin_actions(self.registry)
        self._register_persist_hooks()  # 事件驱动持久化

    def run_from_file(self, yaml_path, trigger_type="workflow_dispatch", payload=None): ...
    def run_workflow(self, wf_def, trigger_type="workflow_dispatch", payload=None): ...
    def resume_run(self, run_id, rerun_components=None, rerun_all=False): ...
    def cancel_run(self, run_id) -> bool: ...
    def shutdown(self, timeout=30) -> int: ...
    def list_runs(self, paper_id=None, status=None, limit=50): ...
    def get_run(self, run_id): ...
```

### 8.5 lwf 删除/保留清单

**删掉**:
- 5 种触发器(只留 `workflow_dispatch`)
- 流程定义版本管理(`def_store`)
- Webhook 接收
- `workflow-wait` 协议(人工审批)
- `script` / `run` 字段(执行子进程)
- `claude_code_sdk` 集成
- Supabase / Lotus 远程存储
- FastAPI server + WebSocket(Phase 2)

**保留**:
- 三层状态机(Workflow/Job/Step)
- 三层 Executor(workflow/job/step)
- EventBus + 事件驱动持久化
- WorkflowParser + 表达式求值
- matrix strategy
- JsonFileStorage
- 优雅关闭(简化版)

### 8.6 退出码

| 退出码 | 含义 |
|---|---|
| 0 | run 成功完成 |
| 1 | run 失败 |
| 2 | 配置错误 |
| 3 | 环境错误 |
| 130 | 用户 Ctrl+C |

---

## 9. 错误处理与重试策略

### 9.1 错误分类矩阵

| 错误类型 | 处理策略 | 重试 |
|---|---|---|
| 配置错误 | 立即终止,退出码 2 | 否 |
| 环境错误 | 立即终止,退出码 3 | 否 |
| LLM 限流(429) | 指数退避 3 次 | 是 |
| LLM 服务端错误(500/503) | 指数退避 3 次 | 是 |
| LLM 上下文超限 | 不重试,组件失败 | 否 |
| LLM schema 校验失败 | 重试 1 次(加严 prompt) | 是 1 次 |
| LLM 超时(>120s) | 不重试,组件失败 | 否 |
| PDF 解析失败 | 不重试,extract 失败 | 否 |
| arXiv 抓取失败 | 回退到 PDF | 1 次 |
| 磁盘写入失败 | 不重试,组件失败 | 否 |
| 用户取消 | 优雅关闭,保存状态 | - |
| 组件异常 | 不重试,组件失败 | 否 |

### 9.2 LLM 重试细节

- 429/500/503/连接错误:指数退避 3 次,初始 1s,最大 30s,带 10% 抖动
- 上下文超限(400 `context_length`):不重试,抛 `ContextLengthError`
- schema 校验失败:重试 1 次,在 prompt 里强调约束

### 9.3 维度级失败隔离

8 维度并行,某维度失败不阻塞其他维度。`synthesize` 加条件:`至少 6/8 维度成功`。少于 6 维成功则不合成,直接 fail。

`synthesize` 内部对缺失维度降级处理:标 N/A,加权计算时权重归零。

### 9.4 优雅关闭

```python
def _signal_handler(signum, frame):
    cancelled = engine.shutdown(timeout=30)
    print(f"Cancelled {cancelled} active run(s). State saved.")
    sys.exit(130)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
```

### 9.5 日志与可观测性

- 日志层级:`run.log` 全局 + 各组件 `<component>.log`
- 关键事件通过 EventBus 广播:workflow/job/step 的 started/completed/failed/log
- 终端输出:INFO 打印进度,DEBUG 打印 LLM 详情,WARNING 打印降级
- token 用量审计:每次 LLM 调用的 usage 写入 `score.json`,run 结束汇总成本

---

## 10. 测试策略

### 10.1 测试金字塔

```
        ┌─────────────┐
        │  E2E (5%)   │  端到端跑真实论文
        └─────────────┘
       ┌───────────────┐
       │ Integration   │  多组件协作 / 并行 / 续跑
       │    (25%)      │
       └───────────────┘
     ┌───────────────────┐
     │     Unit (70%)     │  单组件 / 函数 / schema
     └───────────────────┘
```

### 10.2 测试目录

```
tests/
├── unit/
│   ├── test_parser.py
│   ├── test_state_machine.py
│   ├── test_context.py
│   ├── test_paths.py
│   ├── test_llm_provider.py
│   ├── test_schemas.py
│   ├── test_extract_pdf.py
│   ├── test_extract_arxiv.py
│   ├── test_dim_score.py
│   ├── test_synthesize.py
│   └── test_decide.py
├── integration/
│   ├── test_matrix_parallel.py
│   ├── test_resume.py
│   ├── test_rerun.py
│   ├── test_cancel.py
│   └── test_storage_persistence.py
└── e2e/
    ├── test_normal_review_pdf.py
    └── test_normal_review_arxiv.py
```

### 10.3 关键测试场景

- **LLM 重试**:mock 429 三次后成功 / 三次后失败 / 上下文超限不重试
- **Schema 校验**:`score=6` 报错 / `justification` 过短报错
- **Decide 纯规则**:全 5 分→strong_accept / 全 1 分→strong_reject / 加权计算正确
- **Matrix 并行**:8 次调用在 0.5s 内启动(并行,非串行)
- **断点续跑**:7 维成功 + 1 维失败 → 续跑只重跑 1 维 + synthesize + decide
- **选择性重跑**:`--rerun dim_novelty` 级联重跑 synthesize + decide
- **E2E**:真实 arXiv 论文跑通,cache 命中 ≥7

### 10.4 Mock 策略

- 单元测试:全局 mock `LLMClient.from_env()`,返回定制结构化结果
- 集成测试:同上,验证多组件协作
- E2E:不 mock,用真实 API key,成本 ~$0.5/test

### 10.5 覆盖率目标

| 模块 | 覆盖率 |
|---|---|
| core/parser.py | 95%+ |
| core/state_machine.py | 95%+ |
| llm/anthropic_provider.py | 90%+ |
| llm/schemas.py | 100% |
| actions/extract/ | 85%+ |
| actions/dimensions/ | 85%+ |
| actions/synthesize.py | 85%+ |
| actions/decide.py | 100% |
| engine.py | 80%+ |

---

## 11. 实施阶段划分

### 11.1 Phase 1 内部里程碑

| 里程碑 | 范围 | 验收标准 | 预估工时 |
|---|---|---|---|
| **M1: 骨架搭建** | 从 lwf 复制 + 改造 core/executors/storage/engine,删除不需要的 | `python main.py run configs/minimal.yaml` 跑通空 action | 1 天 |
| **M2: LLM Provider 层** | base/registry/client/anthropic_provider + schemas + 3 个 prompt 模板 | 单元测试覆盖重试/cache/schema | 1 天 |
| **M3: Extract 组件** | PyMuPDF + arXiv 抓取 + LLM 补全 metadata | 真实 arXiv ID 跑通 extract | 1.5 天 |
| **M4: 维度组件** | 通用 DimensionAction + 8 个 prompt + matrix 验证 | 8 维并行,cache 命中 ≥7 | 1.5 天 |
| **M5: Synthesize + Decide** | synthesize 调 LLM + decide 纯规则 | 完整 run 跑通 | 1 天 |
| **M6: 断点续跑 + CLI** | resume/list-runs/show-run/--rerun/优雅关闭 | 续跑测试通过 | 1 天 |
| **M7: 测试补全 + 文档** | unit + integration + README | 覆盖率达标 | 1 天 |

**Phase 1 总预估: ~8 天**

### 11.2 M1 骨架搭建任务

1. 创建项目结构
2. 从 lwf 复制核心模块并改造:
   - `core/parser.py`:删 `script`/`run`/`script_args` 字段
   - `core/models.py`:删 `ActionType.RUN`/`SCRIPT`、`StepWaitInfo`、`WorkflowDefRecord`
   - `core/state_machine.py`:删 `WAITING`/`WAITING_RETRY` 状态
   - `executors/step_executor.py`:删 `_execute_run`/`_execute_script`,只留 `_execute_uses`
   - `engine.py`:删触发器/版本管理/`_release_waiting_steps`
3. `pyproject.toml` + `requirements.txt`
4. `main.py`(简化)+ `cli.py`
5. 最小验证:`EchoAction` + `configs/minimal.yaml`

### 11.3 Phase 1 验收标准

- [ ] `python main.py run configs/normal_review.yaml --payload '{"paper_source":"2402.12098"}'` 端到端跑通
- [ ] 8 维度并行执行,7+ 维度 cache 命中
- [ ] 产物完整:`run.json` / `run_manifest.json` / `00_extract/*` / `10_dim_*/{score.json,review.md}` / `50_synthesize/{review.md,scores.json}` / `60_decision/decision.json` / `final_report.md`
- [ ] decision 字段合法:`recommendation` ∈ 7 档,`weighted_score` ∈ [1,5]
- [ ] Ctrl+C 中断后,`python main.py resume <run_id>` 能续跑
- [ ] `--rerun dim_novelty` 能级联重跑 synthesize + decide
- [ ] 单元测试覆盖率达标,`pytest` 全过
- [ ] E2E 测试通过
- [ ] README 含安装/使用/配置说明

### 11.4 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| Anthropic API 限流 | 中 | 8 维并行可能触发 429 | 指数退避重试 3 次 |
| PyMuPDF 解析质量差 | 中 | 全文混乱影响打分 | M3 用真实论文验证,必要时引入 marker fallback |
| arXiv latex 源码不可用 | 中 | 回退到 PDF | 已有回退策略 |
| 论文过长(>200K token) | 低 | 上下文超限 | extract 检测长度,超限时截断主要章节 + 警告 |
| matrix 并发触发连接池耗尽 | 低 | 部分维度失败 | Anthropic SDK 默认连接池够 8 并发 |
| lwf 代码改造工作量超预期 | 中 | M1 延期 | M1 拆细,先跑通最小路径再补全 |

---

## 附录 A: lwf 参考模块清单

**lwf 项目路径**: `/Users/dengyunxi/workspace/for-staging/lwf`

**可直接复用的模块**:
- `main.py` — 双模式入口(改造)
- `workflow_engine/core/parser.py` — YAML 解析(删 script/run)
- `workflow_engine/core/state_machine.py` — 三层状态机(删 WAITING)
- `workflow_engine/core/event_bus.py` — 事件总线(直接复用)
- `workflow_engine/core/models.py` — 数据模型(精简)
- `workflow_engine/core/context.py` — 表达式求值(直接复用)
- `workflow_engine/actions/registry.py` — Action 注册表(直接复用)
- `workflow_engine/executors/workflow_executor.py` — Job 调度(直接复用)
- `workflow_engine/executors/job_executor.py` — matrix 策略(直接复用)
- `workflow_engine/executors/step_executor.py` — Action 分发(精简)
- `workflow_engine/storage/backend.py` + `memory.py` + `json_file.py`(直接复用)
- `examples/comprehensive_demo.yml` — 综合演示(参考)

**不复用**:
- `actions/builtin.py`(lwf 的 RunCommandAction/ScriptFileAction/ClaudeCodeAction 等)
- `actions/claude_code_client.py`(用 anthropic SDK 直调)
- `storage/supabase_storage.py` / `lotus_storage.py`(只用 memory/json)
- `api/server.py`(Phase 2)
- `triggers/`(只留 workflow_dispatch)
- `core/cookie_manager.py`(无外部认证需求)

---

## 附录 B: 8 维度评审标准(normal 模式)

| 维度 | 评分范围 | 关注点 |
|---|---|---|
| **Novelty** | 1-5 | 论文主张的新颖性,与现有文献对比 |
| **Soundness** | 1-5 | 方法/论证严谨性,实验设计是否合理 |
| **Significance** | 1-5 | 问题重要性,潜在影响 |
| **Clarity** | 1-5 | 写作清晰度,逻辑连贯性 |
| **Reproducibility** | 1-5 | 代码/数据可得性,实验可复现 |
| **Related Work** | 1-5 | 文献覆盖度,定位准确性 |
| **Positioning** | 1-5 | 论文如何定位自己的贡献 |
| **Presentation** | 1-5 | 图表/排版/格式质量 |

**OpenReview 7 档推荐**:
`strong_accept` / `accept` / `weak_accept` / `borderline` / `weak_reject` / `reject` / `strong_reject`

**映射阈值**(基于加权平均分,1-5 制):
- ≥4.5 → strong_accept
- ≥4.0 → accept
- ≥3.5 → weak_accept
- ≥3.0 → borderline
- ≥2.0 → weak_reject
- ≥1.5 → reject
- <1.5 → strong_reject

---

**END OF SPEC**
