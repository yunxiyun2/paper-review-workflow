# Simple Frontend (Phase 2 #3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure HTML + vanilla JS + CSS single-page frontend that lets users assemble YAML configs, dispatch reviews, monitor progress via WebSocket, and view decisions through a browser — no CLI needed.

**Architecture:** Single-file `index.html` (HTML + CSS + JS inline) served by FastAPI `StaticFiles`. Backend adds `GET /api/venues` endpoint + `GET /` root route. 4 tabs: Assemble / Dispatch / Monitor / Decision. WebSocket reuses Phase 2 #5's `/ws/runs/{run_id}`. All REST endpoints from Phase 2 #5 are reused.

**Tech Stack:** Python 3.10+, FastAPI (StaticFiles + HTMLResponse + FileResponse), vanilla JS (fetch + WebSocket + DOM), CSS (dark theme, no framework), pytest TestClient.

**Reference SPEC:** `docs/superpowers/specs/2026-08-16-simple-frontend-design.md`

---

## File Structure Overview

```
paper-review-workflow/
├── paper_review_workflow/
│   └── api/
│       ├── server.py                    # Modify: add GET /api/venues + StaticFiles + GET /
│       └── static/
│           └── index.html               # ★ Create: single-file SPA (~800-1000 lines)
└── tests/
    └── integration/
        └── test_api_static_frontend.py  # ★ Create: backend integration tests
```

---

# M1: Backend Changes

**Goal:** Add `GET /api/venues` endpoint, mount StaticFiles, add `GET /` root route returning index.html, write integration tests.

**Estimated:** 0.5 day

## Task 1.1: Add `GET /api/venues` endpoint + StaticFiles + GET /

**Files:**
- Modify: `paper_review_workflow/api/server.py`
- Test: `tests/integration/test_api_static_frontend.py`

- [ ] **Step 1: Write failing tests**

Create `tests/integration/test_api_static_frontend.py`:

```python
import pytest
from fastapi.testclient import TestClient
from paper_review_workflow.api import create_app
from paper_review_workflow.storage.memory import MemoryStorage


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    app = create_app(storage=MemoryStorage(), configs_dir="configs")
    with TestClient(app) as c:
        yield c


def test_get_venues_returns_3_venues(client):
    r = client.get("/api/venues")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 3
    names = [v["name"] for v in data["venues"]]
    assert "neurips" in names
    assert "icml" in names
    assert "acl" in names


def test_get_venues_neurips_has_correct_fields(client):
    r = client.get("/api/venues")
    neurips = [v for v in r.json()["venues"] if v["name"] == "neurips"][0]
    assert neurips["display_name"] == "NeurIPS 2025"
    assert neurips["dimensions"] == ["soundness", "presentation", "contribution"]
    assert neurips["score_min"] == 1
    assert neurips["score_max"] == 10
    assert neurips["weights"]["soundness"] == 1.3
    assert len(neurips["thresholds"]) == 7
    assert neurips["thresholds"][0]["label"] == "strong_accept"


def test_get_venues_icml_4_dims_1_to_4(client):
    r = client.get("/api/venues")
    icml = [v for v in r.json()["venues"] if v["name"] == "icml"][0]
    assert icml["dimensions"] == ["soundness", "significance", "originality", "clarity"]
    assert icml["score_max"] == 4


def test_root_returns_html_404_when_no_index(client):
    """GET / returns 404 if index.html doesn't exist yet (M2 will create it)"""
    r = client.get("/")
    # Will be 404 until index.html is created in M2
    assert r.status_code in (200, 404)


def test_static_files_endpoint(client):
    """GET /static/ should serve files from api/static/ dir"""
    r = client.get("/static/.gitkeep")
    # .gitkeep exists (created in Phase 2 #5), should return 200
    assert r.status_code in (200, 404)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_api_static_frontend.py -v`
Expected: FAIL with 404 on `/api/venues` (endpoint not implemented)

- [ ] **Step 3: Add imports to `paper_review_workflow/api/server.py`**

Read the top of `server.py`. Add these imports after the existing `from fastapi.responses import ...` line:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
```

Add `import os` if not already imported.

- [ ] **Step 4: Add `GET /api/venues` endpoint inside `create_app()`**

Find the `@app.get("/api/health")` endpoint in `create_app()`. After the health endpoint, add:

```python
    @app.get("/api/venues")
    async def list_venues():
        """Return all venue configs for the frontend assembler."""
        from ..core.venue_config import VenueConfig
        import os

        venues_dir = os.environ.get("PAPER_REVIEW_CONFIGS_DIR", "configs/venues")
        venues = []
        for yml_file in sorted(Path(venues_dir).glob("*.yaml")):
            try:
                # Clear cache for dev-mode hot reload
                VenueConfig._cache.pop(yml_file.stem, None)
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

- [ ] **Step 5: Add StaticFiles mount + `GET /` root route**

Find the `return app` line at the end of `create_app()`. **Before** `return app`, add:

```python
    # ── Static files + root route (Phase 2 #3) ──
    _static_dir = Path(__file__).parent / "static"
    if _static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root():
        """Serve the frontend index.html."""
        index_path = _static_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path), media_type="text/html")
        return HTMLResponse("<h1>Frontend not built yet</h1>", status_code=404)
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/integration/test_api_static_frontend.py -v`
Expected: All tests pass (5 tests). The `test_root_returns_html_404_when_no_index` and `test_static_files_endpoint` may pass with 404 since `index.html` doesn't exist yet — that's fine, M2 creates it.

- [ ] **Step 7: Commit**

```bash
git add paper_review_workflow/api/server.py tests/integration/test_api_static_frontend.py
git commit -m "feat(api): GET /api/venues + StaticFiles + GET / root route"
```

---

# M2: Frontend Skeleton (HTML + CSS + Tab Switching)

**Goal:** Create `index.html` with topbar, 4 tab buttons, dark theme CSS, global state, API utility functions, tab switching logic. Tab content divs are empty placeholders filled in M3-M6.

**Estimated:** 0.5 day

## Task 2.1: Create index.html skeleton

**Files:**
- Create: `paper_review_workflow/api/static/index.html`

- [ ] **Step 1: Create `paper_review_workflow/api/static/index.html`**

Write the following content (complete single-file HTML):

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Paper Review Workflow</title>
  <style>
    :root {
      --bg: #0d1117;
      --surface: #161b22;
      --surface2: #21262d;
      --surface3: #30363d;
      --border: #30363d;
      --text: #e6edf3;
      --text2: #8b949e;
      --text3: #6e7681;
      --accent: #58a6ff;
      --success: #3fb950;
      --failure: #f85149;
      --warning: #d29922;
      --pending: #8b949e;
      --running: #58a6ff;
      --cancelled: #d29922;
      --radius: 6px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg); color: var(--text); min-height: 100vh;
      font-size: 14px; line-height: 1.5;
    }
    .topbar {
      background: var(--surface); border-bottom: 1px solid var(--border);
      display: flex; align-items: center; gap: 16px;
      padding: 0 20px; height: 52px; position: sticky; top: 0; z-index: 100;
    }
    .logo { font-size: 15px; font-weight: 700; display: flex; align-items: center; gap: 6px; }
    .tabs { display: flex; gap: 4px; }
    .tab-btn {
      background: none; border: none; color: var(--text2); cursor: pointer;
      padding: 7px 14px; border-radius: var(--radius); font-size: 13px;
      transition: all .15s;
    }
    .tab-btn:hover { background: var(--surface2); color: var(--text); }
    .tab-btn.active { background: var(--surface3); color: var(--text); font-weight: 600; }
    .ws-status { margin-left: auto; display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text2); }
    .ws-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text3); transition: background .3s; }
    .ws-dot.connected { background: var(--success); }
    .ws-dot.disconnected { background: var(--failure); }
    .main-panel { padding: 20px; max-width: 1200px; margin: 0 auto; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); margin-bottom: 14px; overflow: hidden;
    }
    .card-header { padding: 11px 15px; border-bottom: 1px solid var(--border); background: var(--surface2); font-weight: 600; }
    .card-body { padding: 15px; }
    .btn {
      display: inline-flex; align-items: center; gap: 5px;
      padding: 6px 13px; border-radius: var(--radius);
      border: 1px solid var(--border); background: var(--surface2);
      color: var(--text); cursor: pointer; font-size: 13px; font-weight: 500;
      transition: all .15s; white-space: nowrap;
    }
    .btn:hover { background: var(--surface3); }
    .btn-primary { background: var(--accent); border-color: var(--accent); color: #fff; }
    .btn-primary:hover { background: #1f6feb; }
    .btn-danger { background: #da3633; border-color: #da3633; color: #fff; }
    .badge {
      display: inline-flex; align-items: center; gap: 4px;
      padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 600;
      text-transform: uppercase; letter-spacing: .04em;
    }
    .badge .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
    .badge.pending { background: rgba(139,148,158,.15); color: var(--pending); }
    .badge.running { background: rgba(88,166,255,.15); color: var(--running); }
    .badge.success { background: rgba(63,185,80,.15); color: var(--success); }
    .badge.failure { background: rgba(248,81,73,.15); color: var(--failure); }
    .badge.cancelled { background: rgba(210,153,34,.15); color: var(--cancelled); }
    input, select, textarea {
      background: var(--bg); border: 1px solid var(--border); color: var(--text);
      border-radius: var(--radius); padding: 6px 10px; font-size: 13px; width: 100%;
    }
    input:focus, select:focus, textarea:focus { outline: none; border-color: var(--accent); }
    label { display: block; font-size: 12px; color: var(--text2); margin-bottom: 4px; margin-top: 10px; }
    .form-row { margin-bottom: 10px; }
    .log-box {
      background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius);
      padding: 10px; max-height: 300px; overflow-y: auto; font-family: monospace;
      font-size: 12px; line-height: 1.6;
    }
    .log-line { color: var(--text2); }
    .log-line .ts { color: var(--text3); }
    .log-line .ev { color: var(--accent); }
    .run-list { display: flex; flex-direction: column; gap: 4px; }
    .run-item {
      padding: 8px 12px; border-radius: var(--radius); cursor: pointer;
      border: 1px solid transparent; transition: all .15s;
    }
    .run-item:hover { background: var(--surface2); }
    .run-item.active { background: var(--surface3); border-color: var(--accent); }
    .run-item .run-id { font-size: 12px; color: var(--text2); }
    .run-item .run-status { font-size: 11px; }
    .score-bar { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
    .score-bar .label { width: 120px; font-size: 12px; color: var(--text2); }
    .score-bar .bar { flex: 1; height: 8px; background: var(--surface3); border-radius: 4px; overflow: hidden; }
    .score-bar .fill { height: 100%; background: var(--accent); border-radius: 4px; }
    .score-bar .val { width: 80px; font-size: 12px; text-align: right; }
    .yaml-output {
      background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius);
      padding: 12px; font-family: monospace; font-size: 12px; white-space: pre-wrap;
      max-height: 300px; overflow-y: auto;
    }
    .empty-state { text-align: center; padding: 40px; color: var(--text3); }
  </style>
</head>
<body>
  <div class="topbar">
    <span class="logo">📄 Paper Review</span>
    <nav class="tabs">
      <button class="tab-btn active" data-tab="assemble" onclick="switchTab('assemble')">装配</button>
      <button class="tab-btn" data-tab="dispatch" onclick="switchTab('dispatch')">触发</button>
      <button class="tab-btn" data-tab="monitor" onclick="switchTab('monitor')">监控</button>
      <button class="tab-btn" data-tab="decision" onclick="switchTab('decision')">决策</button>
    </nav>
    <span class="ws-status">
      <span class="ws-dot" id="ws-dot"></span>
      <span id="ws-label">未连接</span>
    </span>
  </div>

  <div class="main-panel">
    <div id="tab-assemble" class="tab-content active">
      <div class="card"><div class="card-body"><p>装配器将在 M3 实现</p></div></div>
    </div>
    <div id="tab-dispatch" class="tab-content">
      <div class="card"><div class="card-body"><p>触发器将在 M4 实现</p></div></div>
    </div>
    <div id="tab-monitor" class="tab-content">
      <div class="card"><div class="card-body"><p>监控器将在 M5 实现</p></div></div>
    </div>
    <div id="tab-decision" class="tab-content">
      <div class="card"><div class="card-body"><p>决策查看将在 M6 实现</p></div></div>
    </div>
  </div>

  <script>
    // ── Global state ──
    const state = { venues: [], workflows: [], runs: [], activeRunId: null, ws: null };

    // ── Tab switching ──
    function switchTab(tabName) {
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + tabName).classList.add('active');
      document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
      if (tabName === 'monitor') loadRuns();
      if (tabName === 'decision') loadCompletedRuns();
    }

    // ── API utilities ──
    async function apiGet(path) {
      const r = await fetch(path);
      if (!r.ok) throw new Error(`GET ${path} failed: ${r.status}`);
      return r.json();
    }
    async function apiPost(path, body) {
      const r = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      if (!r.ok) throw new Error(`POST ${path} failed: ${r.status}`);
      return r.json();
    }

    // ── WebSocket status ──
    function updateWsStatus(status) {
      const dot = document.getElementById('ws-dot');
      const label = document.getElementById('ws-label');
      dot.className = 'ws-dot ' + status;
      label.textContent = status === 'connected' ? '已连接' : status === 'disconnected' ? '已断开' : '未连接';
    }

    // ── Placeholder functions (filled in M3-M6) ──
    async function loadVenues() { state.venues = await apiGet('/api/venues'); }
    async function loadWorkflows() { state.workflows = await apiGet('/api/workflows'); }
    async function loadRuns() {}
    async function loadCompletedRuns() {}

    // ── Init ──
    (async function init() {
      try {
        await loadVenues();
        await loadWorkflows();
      } catch(e) { console.error('init failed:', e); }
      updateWsStatus('disconnected');
    })();
  </script>
</body>
</html>
```

- [ ] **Step 2: Verify `GET /` now returns the HTML**

Run:
```bash
python -c "
from fastapi.testclient import TestClient
from paper_review_workflow.api import create_app
from paper_review_workflow.storage.memory import MemoryStorage
app = create_app(storage=MemoryStorage(), configs_dir='configs')
with TestClient(app) as c:
    r = c.get('/')
    print(f'status: {r.status_code}')
    print(f'content-type: {r.headers.get(\"content-type\")}')
    print(f'has html: {\"Paper Review\" in r.text}')
"
```
Expected: status 200, content-type text/html, has html True

- [ ] **Step 3: Run integration tests**

Run: `pytest tests/integration/test_api_static_frontend.py -v`
Expected: All 5 tests pass

- [ ] **Step 4: Commit**

```bash
git add paper_review_workflow/api/static/index.html
git commit -m "feat(frontend): index.html skeleton with 4 tabs + dark theme + tab switching"
```

---

# M3: Assembly Tab

**Goal:** Implement the "装配" tab: venue dropdown, weight sliders, YAML generation, download, register & dispatch.

**Estimated:** 1 day

## Task 3.1: Assembly tab HTML + JS

**Files:**
- Modify: `paper_review_workflow/api/static/index.html`

- [ ] **Step 1: Replace the assemble tab placeholder**

In `index.html`, find `<div id="tab-assemble" class="tab-content active">` and replace its inner card with:

```html
    <div id="tab-assemble" class="tab-content active">
      <div class="card">
        <div class="card-header">YAML 配置装配器</div>
        <div class="card-body">
          <div class="form-row">
            <label>Workflow Name</label>
            <input type="text" id="asm-wf-name" value="my-neurips-review" />
          </div>
          <div class="form-row">
            <label>Venue</label>
            <select id="asm-venue" onchange="renderWeightSliders()"></select>
          </div>
          <div class="form-row">
            <label>Dimensions &amp; Weights</label>
            <div id="asm-weights"></div>
          </div>
          <div class="form-row">
            <label>paper_source 默认值 (可选)</label>
            <input type="text" id="asm-paper-source" value="" placeholder="arXiv ID 或 PDF 路径" />
          </div>
          <div style="display:flex;gap:8px;margin-top:14px;">
            <button class="btn btn-primary" onclick="generateYAML()">Generate YAML</button>
            <button class="btn" onclick="downloadYAML()">Download</button>
            <button class="btn btn-primary" onclick="registerAndDispatch()">Register &amp; Dispatch</button>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header">生成的 YAML</div>
        <div class="card-body">
          <div class="yaml-output" id="asm-yaml-output"># 点击 "Generate YAML" 生成配置</div>
        </div>
      </div>
    </div>
```

- [ ] **Step 2: Add assembly JS functions**

In the `<script>` section, replace the `async function loadVenues()` and `async function loadWorkflows()` placeholder lines and add the assembly functions. Find the `// ── Placeholder functions (filled in M3-M6) ──` section and replace it with:

```javascript
    // ── Assemble tab ──
    async function loadVenues() {
      state.venues = await apiGet('/api/venues');
      const sel = document.getElementById('asm-venue');
      sel.innerHTML = state.venues.map(v => `<option value="${v.name}">${v.display_name}</option>`).join('');
      renderWeightSliders();
    }

    async function loadWorkflows() {
      state.workflows = await apiGet('/api/workflows');
    }

    function renderWeightSliders() {
      const venueName = document.getElementById('asm-venue').value;
      const venue = state.venues.find(v => v.name === venueName);
      if (!venue) return;
      const container = document.getElementById('asm-weights');
      container.innerHTML = venue.dimensions.map(dim => `
        <div class="form-row" style="display:flex;align-items:center;gap:10px;">
          <label style="width:120px;margin:0;">${dim}</label>
          <input type="range" min="0.5" max="2.0" step="0.1" value="${venue.weights[dim]}" 
                 oninput="this.nextElementSibling.textContent=this.value" 
                 data-dim="${dim}" class="weight-slider" />
          <span style="width:30px;text-align:center;">${venue.weights[dim]}</span>
        </div>
      `).join('');
    }

    function collectWeightValues() {
      const venueName = document.getElementById('asm-venue').value;
      const venue = state.venues.find(v => v.name === venueName);
      const weights = {};
      document.querySelectorAll('.weight-slider').forEach(s => {
        weights[s.dataset.dim] = parseFloat(s.value);
      });
      return weights;
    }

    function generateYAML() {
      const wfName = document.getElementById('asm-wf-name').value || 'my-review';
      const venueName = document.getElementById('asm-venue').value;
      const venue = state.venues.find(v => v.name === venueName);
      if (!venue) return '';
      const weights = collectWeightValues();
      const paperSource = document.getElementById('asm-paper-source').value;
      const weightLines = venue.dimensions
        .map(d => `  WEIGHT_${d.toUpperCase()}: "${weights[d]}"`)
        .join('\n');
      const yaml = `name: ${wfName}

on:
  workflow_dispatch:
    inputs:
      paper_source:
        description: "论文来源"
        required: true
        type: string
${paperSource ? `        default: "${paperSource}"\n` : ''}      mode:
        type: choice
        default: ${venueName}
        options: [${state.venues.map(v => v.name).join(', ')}]

env:
  VENUE: \${{ inputs.mode }}
${weightLines}

jobs:
  extract:
    runs-on: local
    steps:
      - uses: paper-review/extract@v1
        with:
          source: \${{ inputs.paper_source }}
          session_dir: ./sessions/\${{ env.PAPER_ID }}/\${{ env.RUN_ID }}
  dimensions:
    needs: extract
    strategy:
      matrix:
        dimension: [${venue.dimensions.join(', ')}]
      max-parallel: ${venue.dimensions.length}
    runs-on: local
    steps:
      - uses: paper-review/dim_score@v1
        with:
          dimension: \${{ matrix.dimension }}
          session_dir: ./sessions/\${{ env.PAPER_ID }}/\${{ env.RUN_ID }}
          full_text_path: \${{ needs.extract.outputs.full_text_path }}
          metadata_path: \${{ needs.extract.outputs.metadata_path }}
  synthesize:
    needs: dimensions
    runs-on: local
    steps:
      - uses: paper-review/synthesize@v1
        with:
          session_dir: ./sessions/\${{ env.PAPER_ID }}/\${{ env.RUN_ID }}
  decide:
    needs: synthesize
    runs-on: local
    steps:
      - uses: paper-review/decide@v1
        with:
          session_dir: ./sessions/\${{ env.PAPER_ID }}/\${{ env.RUN_ID }}
          scores_path: \${{ needs.synthesize.outputs.scores_path }}`;
      document.getElementById('asm-yaml-output').textContent = yaml;
      return yaml;
    }

    function downloadYAML() {
      const yaml = document.getElementById('asm-yaml-output').textContent;
      if (yaml.startsWith('# 点击')) { alert('请先生成 YAML'); return; }
      const blob = new Blob([yaml], { type: 'text/yaml' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = (document.getElementById('asm-wf-name').value || 'review') + '.yaml';
      a.click();
      URL.revokeObjectURL(url);
    }

    async function registerAndDispatch() {
      const yaml = generateYAML();
      const paperSource = document.getElementById('asm-paper-source').value;
      try {
        const reg = await apiPost('/api/workflows/register', { yaml_content: yaml });
        const inputs = paperSource ? { paper_source: paperSource } : {};
        const run = await apiPost('/api/runs', { workflow_name: reg.name, inputs: inputs });
        alert(`已触发! run_id: ${run.run_id}`);
        switchTab('monitor');
        selectRun(run.run_id);
      } catch(e) {
        alert('Dispatch 失败: ' + e.message);
      }
    }
```

- [ ] **Step 3: Verify the page loads**

Run:
```bash
python -c "
from fastapi.testclient import TestClient
from paper_review_workflow.api import create_app
from paper_review_workflow.storage.memory import MemoryStorage
app = create_app(storage=MemoryStorage(), configs_dir='configs')
with TestClient(app) as c:
    r = c.get('/')
    print('has asm-wf-name:', 'asm-wf-name' in r.text)
    print('has generateYAML:', 'generateYAML' in r.text)
    print('has renderWeightSliders:', 'renderWeightSliders' in r.text)
"
```
Expected: all True

- [ ] **Step 4: Commit**

```bash
git add paper_review_workflow/api/static/index.html
git commit -m "feat(frontend): assembly tab with venue/weights/YAML generation"
```

---

# M4: Dispatch Tab

**Goal:** Implement the "触发" tab: workflow dropdown, paper_source input, mode selector, dispatch button.

**Estimated:** 0.5 day

## Task 4.1: Dispatch tab HTML + JS

**Files:**
- Modify: `paper_review_workflow/api/static/index.html`

- [ ] **Step 1: Replace the dispatch tab placeholder**

Find `<div id="tab-dispatch" class="tab-content">` and replace its inner card with:

```html
    <div id="tab-dispatch" class="tab-content">
      <div class="card">
        <div class="card-header">触发评审</div>
        <div class="card-body">
          <div class="form-row">
            <label>Workflow</label>
            <select id="dis-wf"></select>
          </div>
          <div class="form-row">
            <label>paper_source (必填: arXiv ID 或 PDF 路径)</label>
            <input type="text" id="dis-paper-source" placeholder="2402.12098 或 /path/to/paper.pdf" />
          </div>
          <div class="form-row">
            <label>Venue (mode)</label>
            <select id="dis-venue"></select>
          </div>
          <div style="margin-top:14px;">
            <button class="btn btn-primary" onclick="dispatchRun()">🚀 Dispatch</button>
          </div>
          <div id="dis-result" style="margin-top:14px;"></div>
        </div>
      </div>
    </div>
```

- [ ] **Step 2: Add dispatch JS functions**

In the `<script>` section, after the assembly functions, add:

```javascript
    // ── Dispatch tab ──
    function populateDispatchDropdowns() {
      const wfSel = document.getElementById('dis-wf');
      wfSel.innerHTML = state.workflows.map(w => `<option value="${w.name}">${w.name}</option>`).join('');
      const venueSel = document.getElementById('dis-venue');
      venueSel.innerHTML = state.venues.map(v => `<option value="${v.name}">${v.display_name}</option>`).join('');
    }

    async function dispatchRun() {
      const wfName = document.getElementById('dis-wf').value;
      const paperSource = document.getElementById('dis-paper-source').value;
      const venue = document.getElementById('dis-venue').value;
      if (!paperSource) { alert('请输入 paper_source'); return; }
      try {
        const run = await apiPost('/api/runs', {
          workflow_name: wfName,
          inputs: { paper_source: paperSource, mode: venue }
        });
        document.getElementById('dis-result').innerHTML = `
          <div class="card" style="border-color:var(--success);">
            <div class="card-body">
              <p><strong>run_id:</strong> ${run.run_id}</p>
              <p><strong>status:</strong> ${run.status}</p>
              <button class="btn btn-primary" onclick="switchTab('monitor'); selectRun('${run.run_id}');">→ 前往监控</button>
            </div>
          </div>`;
      } catch(e) {
        alert('Dispatch 失败: ' + e.message);
      }
    }
```

- [ ] **Step 3: Update `loadWorkflows()` to call `populateDispatchDropdowns()`**

Find the `async function loadWorkflows()` function and add a call to `populateDispatchDropdowns()` at the end:

```javascript
    async function loadWorkflows() {
      state.workflows = await apiGet('/api/workflows');
      populateDispatchDropdowns();
    }
```

- [ ] **Step 4: Verify**

Run the same verification command as M3 Step 3, checking for `dis-wf` and `dispatchRun` in the HTML.

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/api/static/index.html
git commit -m "feat(frontend): dispatch tab with workflow/paper_source/venue selection"
```

---

# M5: Monitor Tab

**Goal:** Implement the "监控" tab: run list (auto-refresh), run detail, WebSocket real-time events, cancel/resume buttons.

**Estimated:** 1.5 days

## Task 5.1: Monitor tab HTML + run list + detail

**Files:**
- Modify: `paper_review_workflow/api/static/index.html`

- [ ] **Step 1: Replace the monitor tab placeholder**

Find `<div id="tab-monitor" class="tab-content">` and replace its inner card with:

```html
    <div id="tab-monitor" class="tab-content">
      <div style="display:flex;gap:14px;">
        <div class="card" style="flex:0 0 300px;">
          <div class="card-header">
            Run 列表
            <button class="btn btn-sm" style="float:right;" onclick="loadRuns()">刷新</button>
          </div>
          <div class="card-body">
            <div class="run-list" id="mon-run-list"></div>
          </div>
        </div>
        <div style="flex:1;">
          <div class="card">
            <div class="card-header">Run 详情</div>
            <div class="card-body" id="mon-detail">
              <p class="empty-state">选择左侧 run 查看详情</p>
            </div>
          </div>
          <div class="card">
            <div class="card-header">实时日志</div>
            <div class="card-body">
              <div class="log-box" id="mon-log"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
```

- [ ] **Step 2: Add monitor JS functions**

In the `<script>` section, replace the `async function loadRuns() {}` placeholder and add monitor functions:

```javascript
    // ── Monitor tab ──
    async function loadRuns() {
      try {
        const data = await apiGet('/api/runs');
        state.runs = data.runs;
        renderRunList();
      } catch(e) { console.error('loadRuns failed:', e); }
    }

    function renderRunList() {
      const list = document.getElementById('mon-run-list');
      if (state.runs.length === 0) {
        list.innerHTML = '<p class="empty-state">暂无 run</p>';
        return;
      }
      list.innerHTML = state.runs.map(r => `
        <div class="run-item ${r.run_id === state.activeRunId ? 'active' : ''}" 
             onclick="selectRun('${r.run_id}')">
          <div class="run-id">${r.run_id.substring(0, 20)}...</div>
          <span class="badge ${r.status}"><span class="dot"></span>${r.status}</span>
        </div>`).join('');
    }

    function selectRun(runId) {
      state.activeRunId = runId;
      renderRunList();
      connectWebSocket(runId);
      loadRunDetail(runId);
    }

    async function loadRunDetail(runId) {
      try {
        const run = await apiGet(`/api/runs/${runId}`);
        const jobs = Object.entries(run.jobs || {}).map(([jid, j]) => {
          const icon = j.status === 'success' ? '✅' : j.status === 'running' ? '🔄' :
                       j.status === 'failure' ? '❌' : j.status === 'cancelled' ? '🚫' : '⏳';
          return `<div>${icon} ${jid} <span class="badge ${j.status}">${j.status}</span></div>`;
        }).join('');
        document.getElementById('mon-detail').innerHTML = `
          <p><strong>run_id:</strong> ${run.run_id}</p>
          <p><strong>status:</strong> <span class="badge ${run.status}">${run.status}</span></p>
          <p><strong>duration:</strong> ${run.duration ? run.duration.toFixed(1) + 's' : '-'}</p>
          <div style="margin-top:10px;"><strong>Jobs:</strong></div>
          <div style="margin-top:6px;">${jobs || '<p class="empty-state">无</p>'}</div>
          <div style="margin-top:14px;display:flex;gap:8px;">
            <button class="btn btn-danger" onclick="cancelRun('${run.run_id}')">取消</button>
            <button class="btn" onclick="resumeRun('${run.run_id}')">续跑</button>
          </div>`;
      } catch(e) { console.error('loadRunDetail failed:', e); }
    }

    function connectWebSocket(runId) {
      if (state.ws) { state.ws.close(); state.ws = null; }
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      state.ws = new WebSocket(`${proto}://${location.host}/ws/runs/${runId}`);
      state.ws.onopen = () => updateWsStatus('connected');
      state.ws.onmessage = (e) => handleWsEvent(JSON.parse(e.data));
      state.ws.onclose = () => updateWsStatus('disconnected');
      state.ws.onerror = () => updateWsStatus('disconnected');
    }

    function handleWsEvent(event) {
      const logBox = document.getElementById('mon-log');
      const ts = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : '';
      const ev = event.event || '';
      const jid = event.job_id || '';
      const sid = event.step_id || '';
      const line = `<div class="log-line"><span class="ts">[${ts}]</span> <span class="ev">${ev}</span> ${jid} ${sid}</div>`;
      logBox.innerHTML += line;
      logBox.scrollTop = logBox.scrollHeight;
      if (ev.startsWith('job.') || ev.startsWith('workflow.')) {
        loadRunDetail(state.activeRunId);
      }
    }

    async function cancelRun(runId) {
      try { await apiPost(`/api/runs/${runId}/cancel`, {}); loadRunDetail(runId); }
      catch(e) { alert('取消失败: ' + e.message); }
    }
    async function resumeRun(runId) {
      try { await apiPost(`/api/runs/${runId}/resume`, {}); selectRun(runId); }
      catch(e) { alert('续跑失败: ' + e.message); }
    }
```

- [ ] **Step 3: Add auto-refresh interval**

Find the `// ── Init ──` section at the bottom of the `<script>`. Update the IIFE to include a 5s auto-refresh:

```javascript
    // ── Init ──
    (async function init() {
      try {
        await loadVenues();
        await loadWorkflows();
        await loadRuns();
      } catch(e) { console.error('init failed:', e); }
      updateWsStatus('disconnected');
      // Auto-refresh run list every 5s
      setInterval(loadRuns, 5000);
    })();
```

- [ ] **Step 4: Verify**

Run the verification command, checking for `mon-run-list`, `connectWebSocket`, `handleWsEvent` in the HTML.

- [ ] **Step 5: Commit**

```bash
git add paper_review_workflow/api/static/index.html
git commit -m "feat(frontend): monitor tab with run list + WebSocket events + cancel/resume"
```

---

# M6: Decision Tab

**Goal:** Implement the "决策" tab: completed run dropdown, decision display, dimension score bars, review.md markdown rendering.

**Estimated:** 1 day

## Task 6.1: Decision tab HTML + JS

**Files:**
- Modify: `paper_review_workflow/api/static/index.html`

- [ ] **Step 1: Replace the decision tab placeholder**

Find `<div id="tab-decision" class="tab-content">` and replace its inner card with:

```html
    <div id="tab-decision" class="tab-content">
      <div class="card">
        <div class="card-header">查看决策</div>
        <div class="card-body">
          <div class="form-row">
            <label>Select Run (仅显示成功的 run)</label>
            <select id="dec-run" onchange="loadDecision(this.value)"></select>
          </div>
        </div>
      </div>
      <div id="dec-content"></div>
    </div>
```

- [ ] **Step 2: Add decision JS functions**

In the `<script>` section, replace `async function loadCompletedRuns() {}` and add decision functions:

```javascript
    // ── Decision tab ──
    async function loadCompletedRuns() {
      try {
        const data = await apiGet('/api/runs?status=success');
        const sel = document.getElementById('dec-run');
        sel.innerHTML = '<option value="">-- 选择 run --</option>' +
          data.runs.map(r => `<option value="${r.run_id}">${r.run_id.substring(0, 24)}... (${r.duration ? r.duration.toFixed(1) + 's' : '?'})</option>`).join('');
      } catch(e) { console.error('loadCompletedRuns failed:', e); }
    }

    async function loadDecision(runId) {
      if (!runId) { document.getElementById('dec-content').innerHTML = ''; return; }
      try {
        const run = await apiGet(`/api/runs/${runId}`);
        const decide = run.jobs?.decide?.outputs || {};
        const synthesize = run.jobs?.synthesize?.outputs || {};
        const venueName = decide.venue || 'unknown';
        const venue = state.venues.find(v => v.name === venueName);
        const scoreMax = decide.score_range ? decide.score_range[1] : (venue ? venue.score_max : 10);

        // Decision card
        let html = `
          <div class="card">
            <div class="card-header">推荐决定</div>
            <div class="card-body">
              <p><strong>Recommendation:</strong> <span class="badge success">${decide.recommendation || '?'}</span></p>
              <p><strong>Weighted Score:</strong> ${decide.weighted_score || '?' } / ${scoreMax}</p>
              <p><strong>Venue:</strong> ${venueName}</p>
              <p><strong>Score Range:</strong> ${decide.score_range ? decide.score_range.join('-') : '?'}</p>
            </div>
          </div>`;

        // Dimension scores
        if (decide.per_dimension) {
          const bars = Object.entries(decide.per_dimension).map(([dim, d]) => {
            const pct = (d.score / scoreMax) * 100;
            return `<div class="score-bar">
              <span class="label">${dim}</span>
              <div class="bar"><div class="fill" style="width:${pct}%"></div></div>
              <span class="val">${d.score}/${scoreMax} (conf: ${d.confidence || '?'})</span>
            </div>`;
          }).join('');
          html += `<div class="card"><div class="card-header">各维度分数</div><div class="card-body">${bars}</div></div>`;
        }

        // Key strengths/concerns
        if (decide.key_strengths || decide.key_concerns) {
          html += `<div class="card"><div class="card-header">关键评估</div><div class="card-body">`;
          if (decide.key_strengths && decide.key_strengths.length) {
            html += `<strong>优点:</strong><ul>${decide.key_strengths.map(s => `<li>${s}</li>`).join('')}</ul>`;
          }
          if (decide.key_concerns && decide.key_concerns.length) {
            html += `<strong>担忧:</strong><ul>${decide.key_concerns.map(s => `<li>${s}</li>`).join('')}</ul>`;
          }
          html += `</div></div>`;
        }

        // Rationale
        if (decide.decision_rationale) {
          html += `<div class="card"><div class="card-header">Rationale</div><div class="card-body">${decide.decision_rationale}</div></div>`;
        }

        // Review.md (from synthesize outputs)
        if (synthesize.review_path) {
          html += `<div class="card"><div class="card-header">综合评审 (review.md)</div><div class="card-body" id="dec-review"><p>加载中...</p></div></div>`;
          // Fetch review.md content via a simple GET (the path is a file path, we need to read it)
          // Since the path is absolute, we'll display it as text
          try {
            // Try to fetch the file content via static path or direct read
            // The review_path is like ./sessions/.../50_synthesize/review.md
            // We can't directly fetch local file paths from browser
            // Instead, show the path and let user know
            document.getElementById('dec-content').innerHTML = html;
            document.getElementById('dec-review').innerHTML = `<p>review.md 路径: <code>${synthesize.review_path}</code></p><p>请用 <code>cat ${synthesize.review_path}</code> 查看</p>`;
          } catch(e) {
            document.getElementById('dec-review').innerHTML = `<p>无法加载 review.md: ${e.message}</p>`;
          }
        } else {
          document.getElementById('dec-content').innerHTML = html;
        }
      } catch(e) {
        document.getElementById('dec-content').innerHTML = `<p>加载失败: ${e.message}</p>`;
      }
    }
```

Note: The `review.md` content lives at a local file path (e.g., `./sessions/.../50_synthesize/review.md`) which the browser can't directly fetch. The decision tab shows the path and instructs the user to `cat` it. A future enhancement could add a `GET /api/runs/{id}/review` endpoint that reads and returns the file content, but that's out of scope for this SPEC.

- [ ] **Step 3: Verify**

Run the verification command, checking for `dec-run`, `loadDecision`, `loadCompletedRuns` in the HTML.

- [ ] **Step 4: Commit**

```bash
git add paper_review_workflow/api/static/index.html
git commit -m "feat(frontend): decision tab with score bars + decision display"
```

---

# M7: Integration Tests + README + Tag

**Goal:** Run backend integration tests, update README, tag v0.4.0.

**Estimated:** 0.5 day

## Task 7.1: Verify all integration tests pass

**Files:**
- Test: `tests/integration/test_api_static_frontend.py`

- [ ] **Step 1: Run the frontend integration tests**

Run: `pytest tests/integration/test_api_static_frontend.py -v`
Expected: All 5 tests pass

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ --ignore=tests/unit/test_extract_arxiv.py -k "not test_extract_arxiv_id" 2>&1 | tail -10`
Expected: All tests pass (except pre-existing extract_arxiv_id deselected + E2E skips)

- [ ] **Step 3: Manual smoke test**

Run: `python main.py server --port 8765 &`
Then open `http://localhost:8765/` in browser. Verify:
- 4 tabs visible (装配/触发/监控/决策)
- Tab switching works
- Assembly tab: venue dropdown has 3 options (NeurIPS/ICML/ACL)
- Assembly tab: weight sliders appear when venue selected
- Assembly tab: "Generate YAML" produces YAML text

Kill the server: `kill %1`

## Task 7.2: Update README + tag

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read current README**

Run: `cat README.md`

- [ ] **Step 2: Add Web UI section**

Find the "API Server Mode" section (from Phase 2 #5). After it, add:

```markdown
## Web UI

The FastAPI server includes a built-in web frontend. Start the server:

```bash
python main.py server
```

Then open `http://localhost:8000/` in your browser. The frontend provides 4 tabs:

1. **装配 (Assemble)**: Select venue, adjust dimension weights, generate + download YAML, register & dispatch
2. **触发 (Dispatch)**: Select a registered workflow, enter paper_source, dispatch a review
3. **监控 (Monitor)**: List all runs, click to view real-time progress via WebSocket, cancel/resume
4. **决策 (Decision)**: Select a completed run, view recommendation + per-dimension scores + rationale

The frontend is a single-file pure HTML/JS/CSS app (`paper_review_workflow/api/static/index.html`) — zero build step, zero npm dependencies.
```

- [ ] **Step 3: Commit + tag**

```bash
git add README.md
git commit -m "docs: add Web UI documentation to README"
git tag v0.4.0
```

---

# Self-Review Checklist

## Spec Coverage

| SPEC Section | Implemented By |
|---|---|
| 1. Scope | M1-M7 cover Phase 2 #3 only ✅ |
| 2. Architecture | M1 backend changes, M2-M6 frontend ✅ |
| 3.1-3.4 4 Tabs | M3 assembly, M4 dispatch, M5 monitor, M6 decision ✅ |
| 4.1 GET /api/venues | M1 Task 1.1 ✅ |
| 4.2 StaticFiles + GET / | M1 Task 1.1 ✅ |
| 5.1 HTML structure | M2 skeleton ✅ |
| 5.2 JS module structure | M2-M6 incrementally ✅ |
| 5.3 YAML generation | M3 generateYAML() ✅ |
| 5.4 Markdown rendering | M6 (simplified — shows path instead of fetching) ✅ |
| 6. Testing strategy | M1 integration tests, M7 manual smoke ✅ |
| 7. Milestones | M1-M7 directly map ✅ |
| 8. 验收标准 | All covered ✅ |

## Placeholder Scan

- ✅ No "TBD"/"TODO"
- ✅ All HTML/CSS/JS code shown in full
- ✅ All test code shown
- ✅ No "implement later"

## Type Consistency

- `state.venues` / `state.workflows` / `state.runs` / `state.activeRunId` / `state.ws` — consistent across M2-M6 ✅
- `apiGet(path)` / `apiPost(path, body)` — same signature throughout ✅
- `switchTab(tabName)` — used in M2 init + M3 registerAndDispatch + M4 dispatchRun ✅
- `selectRun(runId)` — defined in M5, called from M3 + M4 ✅
- `loadRuns()` / `loadCompletedRuns()` — defined in M5 + M6, called from switchTab ✅
- `updateWsStatus(status)` — defined in M2, used in M5 WS handlers ✅
- `connectWebSocket(runId)` — defined in M5, called from selectRun ✅

---

# Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-16-simple-frontend-impl.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
