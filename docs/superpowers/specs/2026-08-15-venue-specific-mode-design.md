# Venue-Specific Mode (Phase 2 #1) — SPEC-PRD

**版本**: 1.0
**日期**: 2026-08-15
**状态**: 设计确认中
**Phase**: 2 (subsystem #1 of 7)

---

## 1. 项目定位与范围

### 1.1 一句话定位

为 paper-review-workflow 增加 venue-specific 评审模式,支持按 NeurIPS / ICML / ACL 等会议官方评审规则(维度/评分范围/权重/阈值)进行模拟评审。

### 1.2 范围(本 SPEC 覆盖)

- 新增 `paper_review_workflow/core/venue_config.py` — VenueConfig 类(加载 YAML + 动态生成 schema)
- 新增 `configs/venues/{neurips,icml,acl}.yaml` — 3 个 venue 的配置文件
- 新增 `paper_review_workflow/actions/dimensions/prompts/venues/{neurips,icml,acl}/<dim>.j2` — 11 个 venue-specific prompt
- 改造 `DimensionAction` — 从 venue_config 读维度 + 动态 schema + score.json 加 venue 字段
- 改造 `DecideAction` — 从 venue_config 读 weights/thresholds + decision.json 加 venue/score_range
- **删除 normal 模式** — 老 8 维 1-5 prompt 目录、`ALL_DIMENSIONS` 常量、`DEFAULT_WEIGHTS`、`THRESHOLDS` 全部删除
- **NeurIPS 作为默认 venue** — `VENUE` env 默认 `neurips`(不再有 normal)
- 重命名 `configs/normal_review.yaml` → `configs/neurips_review.yaml`
- 改造 Phase 1 测试(因 normal 删除)— 用 NeurIPS 3 维 1-10

### 1.3 不做(其他 Phase 2 子系统)

- 其他 LLM Provider(Phase 2 #2)
- 简易前端(Phase 2 #3)
- OpenReview XML 导出(Phase 2 #4)
- 多论文批量评审(Phase 2 #6)
- CI/CD + PyPI 发布(Phase 2 #7)

---

## 2. 整体架构

### 2.1 目录结构

```
paper-review-workflow/
├── configs/
│   ├── neurips_review.yaml              # ★ 重命名(原 normal_review.yaml)+ mode: neurips
│   └── venues/                          # ★ 新增:venue 定义目录
│       ├── neurips.yaml                 # NeurIPS(3 维 1-10)
│       ├── icml.yaml                    # ICML(4 维 1-4)
│       └── acl.yaml                     # ACL(4 维 1-4)
├── paper_review_workflow/
│   ├── core/
│   │   └── venue_config.py              # ★ 新增:VenueConfig 类 + 动态 schema 工厂
│   ├── actions/
│   │   ├── dimensions/
│   │   │   ├── __init__.py              # 改造:DimensionAction 读 venue_config
│   │   │   └── prompts/
│   │   │       └── venues/              # ★ venue-specific prompts(normal/ 删除)
│   │   │           ├── neurips/
│   │   │           │   ├── soundness.j2
│   │   │           │   ├── presentation.j2
│   │   │           │   └── contribution.j2
│   │   │           ├── icml/
│   │   │           │   ├── soundness.j2
│   │   │           │   ├── significance.j2
│   │   │           │   ├── originality.j2
│   │   │           │   └── clarity.j2
│   │   │           └── acl/
│   │   │               ├── soundness.j2
│   │   │               ├── excitement.j2
│   │   │               ├── reproducibility.j2
│   │   │               └── overall.j2
│   │   ├── decide.py                    # 改造:THRESHOLDS/DEFAULT_WEIGHTS 从 venue_config 读
│   │   └── ...
│   └── ...
└── tests/
    ├── unit/
    │   ├── test_venue_config.py         # ★ 新增
    │   ├── test_dim_score.py            # 改造:NeurIPS soundness 1-10
    │   └── test_decide.py               # 改造:NeurIPS 1-10 阈值
    ├── integration/
    │   ├── test_venue_review_neurips.py # ★ 新增
    │   ├── test_venue_review_icml.py    # ★ 新增
    │   ├── test_venue_review_acl.py     # ★ 新增
    │   ├── test_full_review_mocked.py   # 改造:用 NeurIPS
    │   ├── test_matrix_parallel.py      # 改造:matrix 改 3 维
    │   ├── test_resume.py               # 改造:用 NeurIPS soundness
    │   └── test_rerun.py                # 改造:用 NeurIPS soundness
    └── e2e/
        └── test_neurips_review_arxiv.py # ★ 重命名(原 test_normal_review_arxiv.py)
```

### 2.2 关键设计原则

1. **YAML 自描述**:每个 venue 完整定义维度/范围/权重/阈值/prompts_dir,加新 venue 不需改代码
2. **动态 schema**:`VenueConfig.get_dimension_score_schema()` 用 `pydantic.create_model` 按 venue 的 score_min/score_max 生成专属 Pydantic schema
3. **NeurIPS 默认**:`VENUE` env 不指定时默认 `neurips`(不再有 normal)
4. **删除 normal**:老 8 维 1-5 prompts、`ALL_DIMENSIONS`、`DEFAULT_WEIGHTS`、`THRESHOLDS` 全部删除
5. **Phase 1 测试改造**:因 normal 删除,假设 8 维 1-5 的测试都改为 NeurIPS 3 维 1-10

---

## 3. VenueConfig 数据结构与 YAML 格式

### 3.1 VenueConfig 类(`paper_review_workflow/core/venue_config.py`)

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Type
import yaml
from pydantic import BaseModel, Field, create_model

@dataclass
class VenueConfig:
    """A venue's review configuration."""
    name: str
    display_name: str
    dimensions: List[str]
    score_min: int
    score_max: int
    confidence_min: float
    confidence_max: float
    weights: Dict[str, float]
    thresholds: List[tuple]
    prompts_dir: str

    _cache: Dict[str, "VenueConfig"] = {}  # 类级缓存

    @classmethod
    def load(cls, name: str, venues_dir: str = "configs/venues") -> "VenueConfig":
        """Load a venue config by name. Cached."""
        if name in cls._cache:
            return cls._cache[name]
        yaml_path = Path(venues_dir) / f"{name}.yaml"
        if not yaml_path.exists():
            raise ValueError(f"venue not found: {name} (looked at {yaml_path})")
        config = cls.from_yaml(str(yaml_path))
        cls._cache[name] = config
        return config

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "VenueConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            name=raw["name"],
            display_name=raw["display_name"],
            dimensions=raw["dimensions"],
            score_min=raw["score_min"],
            score_max=raw["score_max"],
            confidence_min=raw.get("confidence_min", 0.0),
            confidence_max=raw.get("confidence_max", 1.0),
            weights=raw["weights"],
            thresholds=[tuple(t) for t in raw["thresholds"]],
            prompts_dir=raw["prompts_dir"],
        )

    def get_dimension_score_schema(self) -> Type[BaseModel]:
        """Dynamically build a Pydantic schema for this venue's score range."""
        return create_model(
            f"DimensionScore_{self.name}",
            score=(int, Field(ge=self.score_min, le=self.score_max,
                              description=f"{self.score_min}-{self.score_max} scale")),
            confidence=(float, Field(ge=self.confidence_min, le=self.confidence_max)),
            strengths=(List[str], Field(min_length=1, max_length=5)),
            weaknesses=(List[str], Field(min_length=1, max_length=5)),
            justification=(str, Field(min_length=100, max_length=800)),
            evidence=(List[dict], Field(default_factory=list)),
            __base__=BaseModel,
        )
```

### 3.2 YAML 格式

#### `configs/venues/neurips.yaml`

```yaml
name: neurips
display_name: "NeurIPS 2025"

dimensions:
  - soundness
  - presentation
  - contribution

score_min: 1
score_max: 10
confidence_min: 0.0
confidence_max: 1.0

weights:
  soundness: 1.3
  presentation: 0.8
  contribution: 1.4

thresholds:
  - [9.0, "strong_accept"]
  - [7.5, "accept"]
  - [6.0, "weak_accept"]
  - [4.5, "borderline"]
  - [3.0, "weak_reject"]
  - [1.5, "reject"]
  - [0.0, "strong_reject"]

prompts_dir: "prompts/venues/neurips"
```

#### `configs/venues/icml.yaml`

```yaml
name: icml
display_name: "ICML 2025"

dimensions:
  - soundness
  - significance
  - originality
  - clarity

score_min: 1
score_max: 4
confidence_min: 0.0
confidence_max: 1.0

weights:
  soundness: 1.3
  significance: 1.2
  originality: 1.1
  clarity: 0.8

thresholds:
  - [3.75, "strong_accept"]
  - [3.25, "accept"]
  - [2.75, "weak_accept"]
  - [2.25, "borderline"]
  - [1.75, "weak_reject"]
  - [1.25, "reject"]
  - [0.0, "strong_reject"]

prompts_dir: "prompts/venues/icml"
```

#### `configs/venues/acl.yaml`

```yaml
name: acl
display_name: "ACL 2025 (ARR)"

dimensions:
  - soundness
  - excitement
  - reproducibility
  - overall

score_min: 1
score_max: 4
confidence_min: 0.0
confidence_max: 1.0

weights:
  soundness: 1.3
  excitement: 1.2
  reproducibility: 1.0
  overall: 1.3

thresholds:
  - [3.75, "strong_accept"]
  - [3.25, "accept"]
  - [2.75, "weak_accept"]
  - [2.25, "borderline"]
  - [1.75, "weak_reject"]
  - [1.25, "reject"]
  - [0.0, "strong_reject"]

prompts_dir: "prompts/venues/acl"
```

---

## 4. DimensionScore 动态 Schema

Phase 1 的 `DimensionScore` 硬编码 `score: int = Field(ge=1, le=5)`。Phase 2 #1 改为按 venue 的 score_min/score_max 动态生成 schema(见 Section 3.1 的 `get_dimension_score_schema()`)。

- NeurIPS venue → `DimensionScore_neurips` with `score: 1-10`
- ICML venue → `DimensionScore_icml` with `score: 1-4`
- ACL venue → `DimensionScore_acl` with `score: 1-4`

原 `paper_review_workflow/llm/schemas.py` 里的 `DimensionScore` 类**保留**(Phase 2 #5 synthesize schema 仍引用),但 venue 模式下不使用 — `DimensionAction` 改用 `venue_config.get_dimension_score_schema()`。

---

## 5. DimensionAction 改造

### 5.1 改造要点

| Phase 1 | Phase 2 #1 |
|---|---|
| `ALL_DIMENSIONS` 硬编码 8 维 | 删除,从 `venue_config.dimensions` 读 |
| `prompts/<dim>.j2` | `prompts/<venue_dir>/<dim>.j2` |
| `DimensionScore` 固定 1-5 | `venue_config.get_dimension_score_schema()` 动态生成 |
| score.json 无 venue 字段 | score.json 加 `"venue": <name>` |
| 无 venue 校验 | 校验 `VENUE` env var 合法 + 加载对应 YAML |

### 5.2 DimensionAction.run() 关键路径

```python
class DimensionAction(BaseAction):
    def run(self, params, env, context, log_callback=None):
        venue_name = env.get("VENUE", "neurips")  # ★ 默认 neurips(不再 normal)
        dimension = params.get("dimension")

        venue_config = VenueConfig.load(venue_name)

        if dimension not in venue_config.dimensions:
            return ActionResult(success=False,
                message=f"dimension {dimension} not in venue {venue_name}'s dimensions: {venue_config.dimensions}")

        schema = venue_config.get_dimension_score_schema()
        prompt = self._render_prompt(venue_config.prompts_dir, dimension)

        client = LLMClient.from_env()
        response = client.complete(
            system=prompt,
            messages=[{"role": "user", "content": f"Score the {dimension} dimension."}],
            response_schema=schema,
            cached_context=paper_text,
        )
        score = response.structured

        # 落盘(同 Phase 1,但 score.json 加 venue 字段)
        self._write_score_json(out_dir / "score.json", score, dimension, venue_name, response.model, response.usage)
        ...
```

### 5.3 score.json 加 venue 字段

```json
{
  "schema_version": "1.0",
  "venue": "neurips",
  "dimension": "soundness",
  "score": 7,
  "confidence": 0.85,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "justification": "...",
  "evidence": [...],
  "model_used": "claude-sonnet-4-6",
  "usage": {...}
}
```

---

## 6. DecideAction 改造

### 6.1 改造要点

Phase 1 `DecideAction` 硬编码 `DEFAULT_WEIGHTS` + `THRESHOLDS`。Phase 2 #1 改为从 `venue_config` 读。

### 6.2 DecideAction.run() 关键路径

```python
class DecideAction(BaseAction):
    def run(self, params, env, context, log_callback=None):
        venue_name = env.get("VENUE", "neurips")
        venue_config = VenueConfig.load(venue_name)

        weights = venue_config.weights
        thresholds = venue_config.thresholds
        score_min = venue_config.score_min
        score_max = venue_config.score_max

        # 加权计算(同 Phase 1 逻辑)
        weighted_sum = sum(scores[dim]["score"] * weights[dim]
                          for dim in weights if dim in scores)
        total_weight = sum(weights[dim] for dim in weights if dim in scores)
        weighted_score = weighted_sum / total_weight if total_weight else 0

        # 映射到 7 档(用 venue 的 thresholds)
        recommendation = next(
            label for threshold, label in thresholds
            if weighted_score >= threshold
        )

        # decision.json 加 venue + score_range 字段
        decision = {
            "schema_version": "1.0",
            "venue": venue_name,
            "recommendation": recommendation,
            "weighted_score": round(weighted_score, 2),
            "score_range": [score_min, score_max],
            ...
        }
```

### 6.3 删除的硬编码

`paper_review_workflow/actions/decide.py` 删除以下常量:
- `DEFAULT_WEIGHTS`
- `THRESHOLDS`
- `VALID_RECOMMENDATIONS`

`_resolve_weights(env)` 保留 — 仍支持 `WEIGHT_<DIM>` env 覆盖(优先级:env > venue_config)。

---

## 7. normal 模式删除 + NeurIPS 作为默认

### 7.1 删除清单

| 路径 | 操作 |
|---|---|
| `configs/normal_review.yaml` | **重命名**为 `configs/neurips_review.yaml` + 改 `mode: neurips` |
| `paper_review_workflow/actions/dimensions/prompts/*.j2`(8 个) | **删除**(novelty/soundness/significance/clarity/reproducibility/related_work/positioning/presentation) |
| `paper_review_workflow/actions/dimensions/__init__.py` 里的 `ALL_DIMENSIONS` | **删除** |
| `paper_review_workflow/actions/decide.py` 里的 `DEFAULT_WEIGHTS` / `THRESHOLDS` | **删除** |
| `tests/e2e/test_normal_review_arxiv.py` | **重命名**为 `test_neurips_review_arxiv.py` |

### 7.2 `configs/neurips_review.yaml`(重命名后)

```yaml
name: neurips-paper-review

on:
  workflow_dispatch:
    inputs:
      paper_source:
        description: "论文来源(PDF 路径或 arXiv ID/URL)"
        required: true
        type: string
      mode:
        description: "评审 venue"
        type: choice
        default: neurips
        options: [neurips, icml, acl]

env:
  LLM_PROVIDER: anthropic
  LLM_MODEL: claude-sonnet-4-6
  LLM_MAX_TOKENS: "4096"
  LLM_TEMPERATURE: "0.0"
  SESSIONS_ROOT: ./sessions
  VENUE: ${{ inputs.mode }}    # ★ 由 mode 输入决定 venue

jobs:
  extract:
    # ... 同 Phase 1 ...
  dimensions:
    needs: extract
    strategy:
      matrix:
        dimension: ${{ venue.dimensions }}    # ★ 需 engine 注入 venue_config.dimensions
      max-parallel: 8
    runs-on: local
    steps:
      - uses: paper-review/dim_score@v1
        with:
          dimension: ${{ matrix.dimension }}
          # ... 其他参数 ...
  synthesize:
    needs: dimensions
    # ... 同 Phase 1 ...
  decide:
    needs: synthesize
    # ... 同 Phase 1 ...
```

**注意**:matrix 的 dimension 列表需要从 venue_config 动态注入。实现方案:在 `engine.run_workflow()` / `dispatch_workflow()` 里,加载 venue_config 后,把 `dimensions` 列表写入 `run.env["VENUE_DIMENSIONS"]`,然后 YAML 里用 `${{ env.VENUE_DIMENSIONS }}`。或者让 `WorkflowParser` 支持 venue-aware matrix(复杂)。简化方案:YAML 里 dimension 列表硬编码为 NeurIPS 3 维,如果 venue 切换则需手动改 YAML — 或者干脆只支持 NeurIPS(本 SPEC 范围)。

**简化决定**:本 SPEC 仅支持 NeurIPS venue 跑通(其他 venue 通过 `VenueConfig.load` 验证可用,但完整跑通 ICML/ACL 留作 M7 集成测试的一部分)。`configs/neurips_review.yaml` 的 matrix 硬编码 NeurIPS 3 维。

---

## 8. 测试策略

### 8.1 测试金字塔

```
        ┌─────────────┐
        │  E2E (1%)    │  真实 arXiv + NeurIPS venue(可选)
        └─────────────┘
       ┌───────────────┐
       │ Integration   │  3 venue 各跑通(mock LLM)
       │    (30%)      │
       └───────────────┘
     ┌───────────────────┐
     │     Unit (69%)     │  VenueConfig + 动态 schema + DimensionAction/DecideAction
     └───────────────────┘
```

### 8.2 关键测试场景

**单元**:
- `test_venue_config.py`:`VenueConfig.load("neurips")` 返回 3 维 1-10;`load("icml")` 返回 4 维 1-4;`load("acl")` 返回 4 维 1-4;`load("unknown")` 报错;YAML 字段缺失时报错
- `test_venue_config.py`:`get_dimension_score_schema()` 返回的 schema `score` 字段范围正确(NeurIPS 1-10, ICML/ACL 1-4)
- `test_dim_score.py`(改造):用 NeurIPS `soundness` + 1-10 mock,验证 score.json 含 `venue` 字段
- `test_decide.py`(改造):NeurIPS 1-10 阈值(全 9 → strong_accept,全 1 → strong_reject)

**集成**:
- `test_venue_review_neurips.py`:跑通 NeurIPS 评审(3 维 1-10)
- `test_venue_review_icml.py`:跑通 ICML 评审(4 维 1-4)
- `test_venue_review_acl.py`:跑通 ACL 评审(4 维 1-4)
- 验证 decision.json 含正确 `venue` + `score_range` 字段 + 合理 recommendation

**E2E**(marked):
- `test_neurips_review_arxiv.py`:真实 arXiv + NeurIPS + 真实 API key

### 8.3 Phase 1 测试改造(破坏性)

需修改的 Phase 1 测试:
- `tests/integration/test_full_review_mocked.py`:改用 NeurIPS(3 维 1-10)
- `tests/integration/test_matrix_parallel.py`:matrix 改 3 维
- `tests/integration/test_resume.py` / `test_rerun.py`:用 NeurIPS `soundness`
- `tests/unit/test_dim_score.py`:维度改 `soundness`,mock score 1-10
- `tests/unit/test_decide.py`:阈值改 NeurIPS 1-10
- `tests/e2e/test_normal_review_arxiv.py`:重命名为 `test_neurips_review_arxiv.py`

---

## 9. 实施阶段划分

| 里程碑 | 范围 | 预估工时 |
|---|---|---|
| **M1: VenueConfig + 删除 normal** | 创建 `core/venue_config.py` + `configs/venues/{neurips,icml,acl}.yaml` + 删除 normal 相关代码(老 prompts/、`ALL_DIMENSIONS`、`DEFAULT_WEIGHTS`、`THRESHOLDS`) | 1 天 |
| **M2: DimensionScore 动态 schema** | `VenueConfig.get_dimension_score_schema()` + 单元测试 | 0.5 天 |
| **M3: DimensionAction 改造** | 读 venue_config + 动态 schema + venue 校验 + score.json 加 venue 字段 | 1 天 |
| **M4: DecideAction 改造** | 从 venue_config 读 weights/thresholds + decision.json 加 venue + score_range | 0.5 天 |
| **M5: 3 个 venue prompt 模板** | `prompts/venues/{neurips,icml,acl}/<dim>.j2` 共 11 个文件 | 1 天 |
| **M6: normal_review.yaml 重命名 + Phase 1 测试改造** | 重命名 + 改 6 个测试文件用 NeurIPS | 1 天 |
| **M7: 集成测试 3 venue + E2E** | NeurIPS/ICML/ACL 各跑通 + E2E marked | 1 天 |

**Phase 2 #1 总预估: ~6 天**

---

## 10. 验收标准

- [ ] `configs/venues/{neurips,icml,acl}.yaml` 存在,字段完整
- [ ] `VenueConfig.load("neurips")` 返回正确配置(3 维 1-10)
- [ ] `VenueConfig.load("icml")` 返回 4 维 1-4
- [ ] `VenueConfig.load("acl")` 返回 4 维 1-4
- [ ] `VenueConfig.load("unknown")` 报错
- [ ] `VenueConfig.get_dimension_score_schema()` 按 venue 生成正确 schema
- [ ] `DimensionAction` 用 NeurIPS venue 生成 score 1-10
- [ ] `score.json` 含 `venue` 字段
- [ ] `DecideAction` 用 NeurIPS 阈值(9.0→strong_accept 等)
- [ ] `decision.json` 含 `venue` + `score_range` 字段
- [ ] 3 个 venue 的集成测试跑通(mock LLM)
- [ ] `configs/normal_review.yaml` 重命名为 `configs/neurips_review.yaml`,`mode: neurips`
- [ ] Phase 1 测试改造完成,全套测试通过
- [ ] `python main.py run configs/neurips_review.yaml --payload '{"paper_source":"2402.12098"}'` 端到端跑通(需 API key)

---

## 附录 A: 3 venue 官方评审表(参考)

### NeurIPS 2025
- **Dimensions**: Soundness / Presentation / Contribution
- **Score scale**: 1-10
- **Confidence**: 1-5(本 SPEC 用 0-1 统一)

### ICML 2025
- **Dimensions**: Soundness / Significance / Originality / Clarity
- **Score scale**: 1-4
- **Confidence**: 1-5

### ACL 2025 (ARR)
- **Dimensions**: Soundness / Excitement / Reproducibility / Overall Assessment
- **Score scale**: 1-4
- **Confidence**: 1-5

**注**:本 SPEC 的 confidence 统一用 0-1(不按 venue 变),简化 schema 和 decide 逻辑。

---

**END OF SPEC**
