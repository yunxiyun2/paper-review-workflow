# Simple Frontend (Phase 2 #3) — SPEC-PRD

**版本**: 1.0
**日期**: 2026-08-16
**状态**: 设计确认中
**Phase**: 2 (subsystem #3 of 7)

---

## 1. 项目定位与范围

### 1.1 一句话定位

为 paper-review-workflow 增加一个简易 Web 前端(纯 HTML + vanilla JS + CSS),让用户通过浏览器装配 YAML 配置、触发评审、实时监控进度、查看决策,无需 CLI。

### 1.2 范围(本 SPEC 覆盖)

- 创建 `paper_review_workflow/api/static/index.html` — 单文件 SPA(HTML + CSS + JS 内联)
- 改造 `paper_review_workflow/api/server.py` — 加 StaticFiles 挂载 + `GET /` 返回 index.html + `GET /api/venues` 新 endpoint
- 4 个 tab:装配(YAML 装配器)/ 触发(dispatch)/ 监控(run 列表 + WS 实时)/ 决策(decision + review.md)
- 集成测试(StaticFiles + venues endpoint + index.html 可访问)

### 1.3 不做(Phase 2 其他子系统 + YAGNI)

- 其他 LLM Provider(Phase 2 #2)
- OpenReview XML 导出(Phase 2 #4)
- 多论文批量评审(Phase 2 #6)
- CI/CD + PyPI 发布(Phase 2 #7)
- 前端单元测试(无 npm 无 jest)
- E2E 浏览器测试(需 Playwright/Selenium,不引入)
- 响应式设计(本地桌面使用,不针对移动端)
- 国际化(中文 UI)
- 暗/亮主题切换(只做暗色)

---

## 2. 整体架构

### 2.1 目录结构

```
paper-review-workflow/
├── paper_review_workflow/
│   └── api/
│       ├── server.py                    # 改造:加 StaticFiles + GET / + GET /api/venues
│       └── static/                     # ★ 前端代码目录
│           ├── index.html               # ★ 单文件 SPA(HTML + CSS + JS 内联)
│           └── .gitkeep
└── tests/
    └── integration/
        └── test_api_static_frontend.py # ★ 新增:验证 StaticFiles + venues + index.html
```

### 2.2 架构层次

```
┌──────────────────────────────────────────────┐
│  Browser (http://localhost:8000)             │
│   - index.html (HTML + CSS + JS)             │
│   - 4 tabs: 装配/触发/监控/决策               │
└──────────────┬───────────────────────────────┘
               │ REST + WebSocket
               ▼
┌──────────────────────────────────────────────┐
│  FastAPI App  (api/server.py)                │
│   - GET / → index.html                       │
│   - GET /api/venues (★ 新)                   │
│   - GET /api/workflows (现有)                │
│   - POST /api/runs (现有)                    │
│   - GET /api/runs/{id} (现有)                │
│   - POST /api/runs/{id}/cancel/resume (现有) │
│   - WS /ws/runs/{id} (现有)                  │
│   - StaticFiles /static (★ 新)               │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  ReviewEngine + VenueConfig (现有)           │
└──────────────────────────────────────────────┘
```

### 2.3 关键设计原则

1. **单文件 `index.html`**:HTML + CSS + JS 全内联,零构建、零依赖。预计 ~800-1000 行。
2. **复用 Phase 2 #5 API**:8 个 REST + 2 个 WS 全部复用,只加 1 个新 endpoint。
3. **venue 配置驱动**:前端从 `GET /api/venues` 动态获取,不硬编码。加新 venue 不改前端。
4. **WebSocket 复用**:监控 tab 用 Phase 2 #5 的 `/ws/runs/{run_id}`,已有 FilteredWS 过滤。
5. **YAGNI**:不做前端单元测试、E2E 浏览器测试、响应式、国际化、主题切换。

---

## 3. 4 个 Tab 设计

### 3.1 Tab 1: 装配(Assemble)

```
┌─────────────────────────────────────────────────────────┐
│  YAML 配置装配器                                         │
├─────────────────────────────────────────────────────────┤
│  Workflow Name: [____________________]                  │
│  Venue: [NeurIPS ▼]  (从 GET /api/venues 获取)          │
│  ┌─ Dimensions & Weights ─────────────────────────────┐│
│  │  soundness    [=====●===] 1.3   (range 0.5-2.0)    ││
│  │  presentation [===●=====] 0.8                       ││
│  │  contribution [====●====] 1.4                       ││
│  └────────────────────────────────────────────────────┘│
│  paper_source 默认值: [____________________] (可选)    │
│  [Generate YAML]  [Download]  [Register & Dispatch]     │
│  ┌─ 生成的 YAML ──────────────────────────────────────┐│
│  │  name: my-neurips-review                           ││
│  │  ...                                               ││
│  └────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

- **Venue 下拉**: `fetch("/api/venues")` 获取,切换时自动更新维度滑块
- **权重滑块**: 0.5-2.0 范围,步长 0.1,默认值来自 venue 配置
- **Generate YAML**: 前端 JS 生成 YAML 字符串
- **Download**: `Blob` + `URL.createObjectURL` 下载 `.yaml`
- **Register & Dispatch**: `POST /api/workflows/register` + `POST /api/runs` → 跳转监控

### 3.2 Tab 2: 触发(Dispatch)

```
┌─────────────────────────────────────────────────────────┐
│  触发评审                                                │
├─────────────────────────────────────────────────────────┤
│  Workflow: [neurips-paper-review ▼]  (GET /api/workflows)│
│  paper_source: [____________________] (必填)           │
│  mode/venue:  [neurips ▼]  (neurips/icml/acl)           │
│  [🚀 Dispatch]                                          │
│  ┌─ 响应 ─────────────────────────────────────────────┐│
│  │  run_id: 20260816-...                              ││
│  │  status: pending                                   ││
│  │  [→ 前往监控]  (跳转到 Tab 3 并连接 WS)             ││
│  └────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

### 3.3 Tab 3: 监控(Monitor)

```
┌─────────────────────────────────────────────────────────┐
│  Run 列表                          [刷新] [自动更新 ☑]  │
├──────────────────┬──────────────────────────────────────┤
│  Run 列表         │  Run 详情                             │
│  ┌──────────────┐│  run_id: 20260816-...                │
│  │● 20260816... ││  status: [running]                   │
│  │  running      ││  duration: 15.3s                     │
│  ├──────────────┤│  Jobs:                               │
│  │  20260815... ││  ✅ extract       (12.4s)           │
│  │  success     ││  🔄 soundness     (5.2s) ← running  │
│  └──────────────┘│  ⏳ synthesize    (pending)          │
│                   │  [取消] [续跑]                       │
│                   │  ┌─ 实时日志 ──────────────────────┐│
│                   │  │ [14:30:22] workflow.started     ││
│                   │  │ [14:30:34] job.completed: extr ││
│                   │  └─────────────────────────────────┘│
└──────────────────┴──────────────────────────────────────┘
```

- **左侧 run 列表**: `GET /api/runs` 获取,点击切换详情
- **右侧详情**: `GET /api/runs/{id}` 获取 jobs/steps 状态
- **实时日志**: `ws://host/ws/runs/{run_id}` 事件追加到日志框
- **自动更新**: run 列表每 5s 刷新(GET /api/runs),右侧详情通过 WS 实时更新
- **取消/续跑**: `POST /api/runs/{id}/cancel` / `POST /api/runs/{id}/resume`

### 3.4 Tab 4: 决策(Decision)

```
┌─────────────────────────────────────────────────────────┐
│  查看决策                                                │
├─────────────────────────────────────────────────────────┤
│  Select Run: [20260816-... ▼]  (只列 success/cancelled) │
│  ┌─ 推荐决定 ─────────────────────────────────────────┐│
│  │  Recommendation: [weak_accept]                     ││
│  │  Weighted Score: 3.85 / 10                        ││
│  │  Venue: NeurIPS  Score Range: 1-10               ││
│  └────────────────────────────────────────────────────┘│
│  ┌─ 各维度分数 ───────────────────────────────────────┐│
│  │  soundness    ████████░░ 7/10  (conf: 0.85)        ││
│  │  presentation █████░░░░░ 5/10  (conf: 0.70)       ││
│  │  contribution ████████░░ 8/10  (conf: 0.90)        ││
│  └────────────────────────────────────────────────────┘│
│  ┌─ 综合评审 ──────────────────────────────────────────┐│
│  │  ## Summary                                        ││
│  │  This paper proposes...                            ││
│  └────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

- **Run 下拉**: `GET /api/runs?status=success` 获取已完成的 run
- **decision.json**: 从 `GET /api/runs/{id}` 的 `jobs.decide.outputs` 读取
- **review.md**: 从 synthesize job 的 `outputs.review_path` 读取,渲染为 Markdown
- **维度分数条**: CSS 进度条,`score / score_max` 计算宽度

---

## 4. 后端改动

### 4.1 `GET /api/venues` 新 endpoint

```python
@app.get("/api/venues")
async def list_venues():
    """返回所有 venue 配置(供前端装配器用)"""
    from ..core.venue_config import VenueConfig

    venues_dir = os.environ.get("PAPER_REVIEW_CONFIGS_DIR", "configs/venues")
    venues = []
    for yml_file in sorted(Path(venues_dir).glob("*.yaml")):
        try:
            # 清除缓存以支持开发模式热加载
            VenueConfig._cache.pop(yml_file.stem, None) if hasattr(VenueConfig, '_cache') else None
            config = VenueConfig.from_yaml(str(yml_file))
            venues.append({
                "name": config.name,
                "display_name": config.display_name,
                "dimensions": config.dimensions,
                "score_min": config.score_min,
                "score_max": config.score_max,
                "weights": config.weights,
                "thresholds": [{"threshold": t, "label": l} for t, l in config.thresholds],
                "prompts_dir": config.prompts_dir,
            })
        except Exception as e:
            logger.warning(f"[API] failed to load venue {yml_file}: {e}")
    return {"total": len(venues), "venues": venues}
```

响应示例:
```json
{
  "total": 3,
  "venues": [
    {
      "name": "neurips",
      "display_name": "NeurIPS 2025",
      "dimensions": ["soundness", "presentation", "contribution"],
      "score_min": 1,
      "score_max": 10,
      "weights": {"soundness": 1.3, "presentation": 0.8, "contribution": 1.4},
      "thresholds": [{"threshold": 9.0, "label": "strong_accept"}, ...],
      "prompts_dir": "venues/neurips"
    },
    {"name": "icml", ...},
    {"name": "acl", ...}
  ]
}
```

### 4.2 StaticFiles 挂载 + `GET /` 返回 index.html

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# 在 create_app() 里,所有 endpoint 之后:
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """返回前端 index.html"""
    index_path = _static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return HTMLResponse("<h1>Frontend not built</h1>", status_code=404)
```

### 4.3 CORS 不变

已有 CORS middleware(Phase 2 #5),允许 `*` origin。前端从同源访问,不需要额外配置。

---

## 5. 前端实现细节

### 5.1 HTML 结构

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>Paper Review Workflow</title>
  <style>
    /* CSS 变量 + 暗色主题(借鉴 lwf 配色) */
    :root { --bg:#0d1117; --surface:#161b22; --accent:#58a6ff; ... }
    /* 布局:topbar + main-panel */
  </style>
</head>
<body>
  <div class="topbar">
    <span class="logo">📄 Paper Review</span>
    <nav class="tabs">
      <button class="tab-btn active" data-tab="assemble">装配</button>
      <button class="tab-btn" data-tab="dispatch">触发</button>
      <button class="tab-btn" data-tab="monitor">监控</button>
      <button class="tab-btn" data-tab="decision">决策</button>
    </nav>
    <span class="ws-status"><span class="ws-dot"></span> <span id="ws-label">未连接</span></span>
  </div>

  <div class="main-panel">
    <div id="tab-assemble" class="tab-content active"><!-- 装配器 --></div>
    <div id="tab-dispatch" class="tab-content"><!-- 触发器 --></div>
    <div id="tab-monitor" class="tab-content"><!-- 监控器 --></div>
    <div id="tab-decision" class="tab-content"><!-- 决策查看 --></div>
  </div>

  <script>
    // 所有 JS 逻辑内联
  </script>
</body>
</html>
```

### 5.2 JS 模块结构(内联)

```javascript
// ── 全局状态 ──
const state = {
  venues: [],
  workflows: [],
  runs: [],
  activeRunId: null,
  ws: null,
};

// ── Tab 切换 ──
function switchTab(tabName) { ... }

// ── API 调用 ──
async function apiGet(path) { return fetch(path).then(r => r.json()); }
async function apiPost(path, body) {
  return fetch(path, {method:"POST", headers:{"Content-Type":"application/json"},
                    body:JSON.stringify(body)}).then(r => r.json());
}

// ── 装配 tab ──
async function loadVenues() {
  state.venues = await apiGet("/api/venues");
  renderVenueDropdown();
}
function renderVenueDropdown() { ... }
function renderWeightSliders(venue) { ... }
function generateYAML() { ... }
function downloadYAML(yamlStr) { ... }  // Blob 下载
async function registerAndDispatch(yamlStr, paperSource) {
  const reg = await apiPost("/api/workflows/register", {yaml_content: yamlStr});
  const run = await apiPost("/api/runs", {workflow_name: reg.name, inputs: {paper_source: paperSource}});
  switchTab("monitor");
  selectRun(run.run_id);
}

// ── 触发 tab ──
async function loadWorkflows() {
  state.workflows = await apiGet("/api/workflows");
  renderWorkflowDropdown();
}
async function dispatchRun() {
  const run = await apiPost("/api/runs", {
    workflow_name: selectedWorkflow,
    inputs: {paper_source: paperSource, mode: venue}
  });
  showRunId(run.run_id);
}

// ── 监控 tab ──
async function loadRuns() {
  state.runs = await apiGet("/api/runs");
  renderRunList();
}
function selectRun(runId) {
  state.activeRunId = runId;
  connectWebSocket(runId);
  loadRunDetail(runId);
}
function connectWebSocket(runId) {
  if (state.ws) state.ws.close();
  state.ws = new WebSocket(`ws://${location.host}/ws/runs/${runId}`);
  state.ws.onopen = () => updateWsStatus("connected");
  state.ws.onmessage = (e) => handleWsEvent(JSON.parse(e.data));
  state.ws.onclose = () => updateWsStatus("disconnected");
}
function handleWsEvent(event) {
  appendLog(event);
  if (event.event.includes("completed") || event.event.includes("started")) {
    loadRunDetail(state.activeRunId);
  }
}
async function loadRunDetail(runId) {
  const run = await apiGet(`/api/runs/${runId}`);
  renderRunDetail(run);
}
async function cancelRun(runId) { await apiPost(`/api/runs/${runId}/cancel`, {}); }
async function resumeRun(runId) { await apiPost(`/api/runs/${runId}/resume`, {}); }

// ── 决策 tab ──
async function loadCompletedRuns() {
  const data = await apiGet("/api/runs?status=success");
  renderRunDropdown(data.runs);
}
async function loadDecision(runId) {
  const run = await apiGet(`/api/runs/${runId}`);
  const decision = run.jobs?.decide?.outputs;
  renderDecision(decision, run);
}

// ── 初始化 ──
loadVenues();
loadWorkflows();
loadRuns();
setInterval(loadRuns, 5000);  // 5s 自动刷新 run 列表
```

### 5.3 YAML 生成(前端 JS)

```javascript
function generateYAML() {
  const venue = state.venues.find(v => v.name === selectedVenue);
  const weights = collectWeightValues();
  return `name: ${workflowName}
on:
  workflow_dispatch:
    inputs:
      paper_source:
        description: "论文来源"
        required: true
        type: string
      mode:
        type: choice
        default: ${venue.name}
        options: [${state.venues.map(v=>v.name).join(", ")}]
env:
  VENUE: \${{ inputs.mode }}
  ${venue.dimensions.map(d => `WEIGHT_${d.toUpperCase()}: "${weights[d]}"`).join("\n  ")}
jobs:
  extract:
    ...
  dimensions:
    strategy:
      matrix:
        dimension: [${venue.dimensions.join(", ")}]
      max-parallel: ${venue.dimensions.length}
    ...
`;
}
```

### 5.4 Markdown 渲染(简易,不引第三方库)

```javascript
function renderMarkdown(md) {
  return md
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/\n/g, '<br>');
}
```

---

## 6. 测试策略

| 测试 | 方法 |
|---|---|
| `GET /` 返回 index.html | `TestClient.get("/")` 检查 200 + `text/html` |
| `GET /static/index.html` 可访问 | `TestClient.get("/static/index.html")` 检查 200 |
| `GET /api/venues` 返回 3 个 venue | `TestClient.get("/api/venues")` 检查 total=3,含 neurips/icml/acl |
| venue 配置字段完整 | 检查每个 venue 有 dimensions/score_min/score_max/weights |
| Tab 切换 JS 工作 | 手动测试(不引入 Playwright) |

**不做**(YAGNI):前端单元测试、E2E 浏览器测试、响应式测试、国际化测试。

---

## 7. 实施阶段划分

| 里程碑 | 范围 | 预估工时 |
|---|---|---|
| **M1: 后端改动** | `GET /api/venues` + StaticFiles 挂载 + `GET /` 返回 index.html + 集成测试 | 0.5 天 |
| **M2: 前端骨架** | `index.html` 顶栏 + tab 切换 + 暗色主题 CSS + 全局状态 + API 调用工具函数 | 0.5 天 |
| **M3: 装配 tab** | venue 下拉 + 权重滑块 + YAML 生成 + 下载 + 注册+触发 | 1 天 |
| **M4: 触发 tab** | workflow 下拉 + paper_source 输入 + dispatch + 跳转监控 | 0.5 天 |
| **M5: 监控 tab** | run 列表 + 详情 + WS 连接 + 实时日志 + 取消/续跑 | 1.5 天 |
| **M6: 决策 tab** | run 下拉 + decision 展示 + 维度分数条 + review.md 渲染 | 1 天 |
| **M7: 集成测试 + README + tag** | StaticFiles/venues endpoint 测试 + README 更新 + tag v0.4.0 | 0.5 天 |

**Phase 2 #3 总预估: ~5.5 天**

---

## 8. 验收标准

- [ ] `GET /` 返回 `index.html`(200, `text/html`)
- [ ] `GET /static/index.html` 可直接访问
- [ ] `GET /api/venues` 返回 3 个 venue(neurips/icml/acl),字段完整
- [ ] 4 个 tab 可切换(装配/触发/监控/决策)
- [ ] 装配 tab:选 venue → 维度滑块自动更新 → 生成 YAML → 下载
- [ ] 装配 tab:Register & Dispatch → 自动触发 run → 跳转监控
- [ ] 触发 tab:选 workflow + 输入 paper_source → dispatch → 显示 run_id
- [ ] 监控 tab:run 列表自动刷新(5s)+ 点击 run → 详情 + WS 连接
- [ ] 监控 tab:WS 事件实时追加到日志框 + job 状态自动更新
- [ ] 监控 tab:取消/续跑按钮工作
- [ ] 决策 tab:选已完成 run → 显示 recommendation + 分数条 + review.md
- [ ] 暗色主题,布局正确(sidebar + main-panel)
- [ ] `pytest tests/` 全过(含新前端测试)
- [ ] README 更新(含前端访问说明)
- [ ] tag v0.4.0

---

## 附录 A: lwf 前端参考

**lwf 项目路径**: `/Users/dengyunxi/workspace/for-staging/lwf/static/index.html`(2910 行)

**复用模式**:
- 纯 HTML + vanilla JS + CSS 单文件
- 暗色主题(CSS 变量 `--bg`/`--surface`/`--accent` 等)
- topbar + sidebar + main-panel 布局
- WebSocket 状态指示灯
- tab 切换(纯 JS,无框架)

**不复用**:
- lwf 的 Webhook/Schedule/Cookie 管理功能
- lwf 的流程定义版本管理 UI
- lwf 的表单审批交互(我们无人工审批)

---

**END OF SPEC**
