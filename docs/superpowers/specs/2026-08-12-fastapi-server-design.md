# FastAPI Server (Phase 2 #5) — SPEC-PRD

**版本**: 1.0
**日期**: 2026-08-12
**状态**: 设计确认中
**Phase**: 2 (subsystem #5 of 7)

---

## 1. 项目定位与范围

### 1.1 一句话定位

为 paper-review-workflow 引擎增加 FastAPI HTTP API + WebSocket 实时推送,让长时运行的评审任务通过 HTTP 触发、监控、续跑,为 Phase 2 #3 简易前端铺路。

### 1.2 范围(本 SPEC 覆盖)

- 新增 `paper_review_workflow/api/` 子包,实现 FastAPI app 工厂 + 7 个核心 REST endpoints + 2 个 WebSocket endpoints
- 新增 `WSManager` 类作为 EventBus 订阅者,广播事件到所有 WS 连接(解耦同步引擎与 async WebSocket)
- 扩展 `ReviewEngine`:增加 `load_workflow_directory()` / `register_workflow()` / `recover_interrupted_runs()` / `dispatch_workflow()` / `execute_existing_run()` 方法
- 扩展 CLI `main.py`:新增 `server` 子命令;无子命令时默认启动 server(与 lwf 一致)
- 启动时扫描 `configs/` 目录自动注册 workflow + 恢复中断 run
- 关闭时取消活跃 run + 关闭 storage

### 1.3 不做(Phase 2 其他子系统)

- 简易前端(Phase 2 #3)— 仅预留 `/static` 路径,不实际服务静态文件
- venue-specific 模式(Phase 2 #1)
- 其他 LLM Provider(Phase 2 #2)
- OpenReview XML 导出(Phase 2 #4)
- 多论文批量评审(Phase 2 #6)
- 认证(本地使用,无认证)
- CI/CD + PyPI 发布(Phase 2 #7)

---

## 2. 整体架构

### 2.1 目录结构

```
paper_review_workflow/
├── api/                                  # ★ 新增子包
│   ├── __init__.py                       # 导出 create_app + ReviewServer
│   ├── server.py                         # FastAPI app 工厂 + endpoints + WS
│   ├── dependencies.py                  # FastAPI 依赖注入(get_engine)
│   ├── schemas.py                        # Pydantic 请求/响应模型
│   ├── ws_manager.py                     # WebSocket 连接管理 + 事件广播
│   └── static/                           # 预留(Phase 2 #3 用,本次空)
│       └── .gitkeep
├── main.py                               # 改造:增加 server 子命令 + 默认行为
├── paper_review_workflow/
│   ├── engine.py                         # 改造:加 5 个新方法
│   ├── cli.py                            # 改造:加 _cmd_server + 默认启动
│   ├── core/
│   │   └── event_bus.py                  # 不变(WSManager 是外部订阅者)
│   └── ...
└── tests/
    ├── integration/
    │   ├── test_api_endpoints.py         # 新增:7 endpoints
    │   ├── test_api_websocket.py         # 新增:WS 端点 + 事件推送
    │   ├── test_api_lifecycle.py         # 新增:startup 恢复 + shutdown 取消
    │   └── test_api_register.py          # 新增:动态注册 YAML
    └── unit/
        ├── test_ws_manager.py           # 新增
        └── test_engine_api_ext.py       # 新增(引擎扩展)
```

### 2.2 架构层次

```
┌────────────────────────────────────────────────────────┐
│  HTTP Client / Browser (Phase 2 #3)                   │
└──────────────┬──────────────────────────────────────────┘
               │ REST + WebSocket
               ▼
┌────────────────────────────────────────────────────────┐
│  FastAPI App  (api/server.py)                          │
│   - REST endpoints (7 个)                              │
│   - WebSocket endpoints (2 个: /ws + /ws/runs/{id})    │
│   - startup: load configs/ + recover runs              │
│   - shutdown: cancel active + close storage           │
└──────────────┬──────────────────────────────────────────┘
               │ 依赖注入(get_engine)
               ▼
┌────────────────────────────────────────────────────────┐
│  ReviewEngine  (扩展)                                  │
│   - 现有:run_workflow / resume_run / cancel_run        │
│   - 新增:load_workflow_directory                       │
│   - 新增:register_workflow                              │
│   - 新增:dispatch_workflow(只创建 run,不执行)         │
│   - 新增:execute_existing_run(后台执行入口)            │
│   - 新增:recover_interrupted_runs                      │
└──────────────┬──────────────────────────────────────────┘
               │ EventBus 订阅
               ▼
┌────────────────────────────────────────────────────────┐
│  EventBus + WSManager  (api/ws_manager.py)            │
│   - WSManager.attach(loop) 在 startup 调用              │
│   - 每个事件 → JSON 广播到所有 WS 连接                  │
│   - 历史缓存(最近 1000 个事件),新连接 catch-up        │
│   - FilteredWS 包装类:过滤 /ws/runs/{run_id}           │
└────────────────────────────────────────────────────────┘
```

### 2.3 关键设计原则

1. **App 工厂模式**: `create_app(storage=None, configs_dir="configs")` 返回 FastAPI 实例,便于测试时创建独立实例。
2. **依赖注入**: `get_engine()` 用 FastAPI `Depends`,测试时可以 mock engine。
3. **EventBus 解耦**: 不直接在 EventBus 里写 WS 代码,而是用 `WSManager` 作为普通订阅者。EventBus 保持纯同步(无 asyncio 依赖),WSManager 处理 async 广播。
4. **线程安全**: 引擎由 ThreadPoolExecutor 驱动(EventBus 同步回调在子线程),WSManager 用 `asyncio.run_coroutine_threadsafe` 调度到 FastAPI event loop。
5. **CLI 与 API 共享引擎**: `server` 子命令是 `run`/`resume` 的并行通道,共享 `ReviewEngine` 类,不重复实现。

---

## 3. EventBus + WSManager 设计

### 3.1 WSManager 类(api/ws_manager.py)

```python
class WSManager:
    """WebSocket 连接管理 + 事件广播。
    
    作为 EventBus 的订阅者运行,收到事件后异步广播到所有 WS 连接。
    解耦 EventBus(synchronous) 与 WebSocket(async)。"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._connections: set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._unsubscribe: Optional[Callable] = None
        self._history: list[WorkflowEvent] = []
        self._history_limit = 1000

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        """在 FastAPI startup 时调用。订阅 EventBus + 缓存 event loop。"""
        self._loop = loop
        self._unsubscribe = self.event_bus.subscribe(self._on_event)

    def detach(self) -> None:
        """在 shutdown 时调用。取消订阅 + 关闭所有连接。"""
        if self._unsubscribe:
            self._unsubscribe()
        for ws in list(self._connections):
            asyncio.run_coroutine_threadsafe(ws.close(), self._loop)
        self._connections.clear()

    def _on_event(self, event: WorkflowEvent) -> None:
        """EventBus 同步回调(在引擎线程)。把事件扔到 event loop 异步广播。"""
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]
        if self._loop and self._connections:
            asyncio.run_coroutine_threadsafe(
                self._broadcast(event), self._loop
            )

    async def _broadcast(self, event: WorkflowEvent) -> None:
        """异步广播事件到所有连接。"""
        message = event.to_json()
        dead = []
        for ws in list(self._connections):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.discard(ws)

    async def add_connection(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)

    def remove_connection(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def send_history(self, ws: WebSocket, run_id: Optional[str] = None) -> None:
        """新连接时发送历史事件(可选过滤 run_id)。"""
        for evt in self._history:
            if run_id and evt.run_id != run_id:
                continue
            try:
                await ws.send_text(evt.to_json())
            except Exception:
                break
```

### 3.2 FilteredWS 包装(在 server.py 内联)

```python
class FilteredWS:
    """包装 WebSocket,只发送匹配 run_id 的事件 + heartbeat/pong"""
    def __init__(self, ws: WebSocket, target_run_id: str):
        self._ws = ws
        self._run_id = target_run_id

    async def send_text(self, message: str) -> None:
        try:
            data = json.loads(message)
            if data.get("run_id") == self._run_id or data.get("event") in ("heartbeat", "pong"):
                await self._ws.send_text(message)
        except Exception:
            await self._ws.send_text(message)

    async def close(self) -> None:
        await self._ws.close()
```

### 3.3 关键点

1. **线程安全**: `_on_event` 从同步引擎线程调用(EventBus 由 WorkflowExecutor 在 ThreadPoolExecutor 里调用),用 `asyncio.run_coroutine_threadsafe` 调度到 FastAPI event loop。
2. **历史缓存**: 保留最近 1000 个事件,新 WS 连接可以 catch-up。
3. **过滤 endpoint**: `/ws/runs/{run_id}` 用 `FilteredWS` 包装类过滤只发该 run 的事件。
4. **EventBus 不变**: `EventBus` 保持纯同步,无需任何修改。WSManager 只是一个普通订阅者。

---

## 4. 7 个 API 端点契约

### 4.1 请求/响应 Schemas(api/schemas.py)

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class DispatchRequest(BaseModel):
    """POST /api/runs 触发评审"""
    workflow_name: Optional[str] = None  # 引用 configs/ 中已注册的 workflow
    yaml_content: Optional[str] = None   # 直接传 YAML 内容(二选一)
    inputs: Dict[str, Any] = Field(default_factory=dict)

class RegisterWorkflowRequest(BaseModel):
    """POST /api/workflows/register 动态注册 YAML"""
    yaml_content: str
    name: Optional[str] = None

class ResumeRequest(BaseModel):
    """POST /api/runs/{run_id}/resume"""
    rerun_components: Optional[List[str]] = None
    rerun_all: bool = False

class RunResponse(BaseModel):
    run_id: str
    workflow_name: str
    status: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[float] = None
    jobs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

class WorkflowSummary(BaseModel):
    name: str
    file_path: Optional[str] = None
    jobs: List[str] = Field(default_factory=list)
    dispatch_inputs: Dict[str, Any] = Field(default_factory=dict)

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    active_runs: int = 0
    total_runs: int = 0
```

### 4.2 端点契约表

| 方法 | 路径 | 请求 | 响应 | 行为 |
|---|---|---|---|---|
| **GET** | `/api/health` | - | `HealthResponse` | 健康检查,返回活跃 run 数 + 总 run 数 |
| **GET** | `/api/workflows` | - | `{total, workflows: [WorkflowSummary]}` | 列出已注册 workflow(启动扫描 + POST 注册的) |
| **POST** | `/api/workflows/register` | `RegisterWorkflowRequest` | `{message, name, jobs}` | 动态注册 YAML,返回 name |
| **POST** | `/api/runs` | `DispatchRequest` | `{run_id, workflow_name, status, message}` 202 | 触发评审,立即返回 run_id,后台执行 |
| **GET** | `/api/runs` | `?status=&limit=50&offset=0` | `{total, runs: [RunResponse]}` | 列出历史 run,支持状态过滤 + 分页 |
| **GET** | `/api/runs/{run_id}` | - | `RunResponse` 404 if not found | 查看单个 run 详情(含 jobs/steps 状态) |
| **POST** | `/api/runs/{run_id}/cancel` | - | `{run_id, status: "cancelled"}` | 取消运行中的 run |
| **POST** | `/api/runs/{run_id}/resume` | `ResumeRequest` | `{run_id, status, message}` | 续跑(支持 rerun_components / rerun_all) |

### 4.3 关键行为

**`POST /api/runs` 触发流程**:

```python
@app.post("/api/runs", status_code=202)
async def dispatch_run(req: DispatchRequest, background_tasks: BackgroundTasks):
    # 1. 解析 workflow 定义
    if req.workflow_name:
        wf_def = engine.get_workflow_def(req.workflow_name)
        if not wf_def:
            raise HTTPException(404, f"workflow '{req.workflow_name}' not registered")
    elif req.yaml_content:
        wf_def = engine.parser.parse_string(req.yaml_content)
    else:
        raise HTTPException(400, "must provide workflow_name or yaml_content")

    # 2. 预分配 run(PENDING 状态,存入 storage)
    run = engine.dispatch_workflow(wf_def.name, req.inputs)

    # 3. 调度后台执行
    background_tasks.add_task(_run_in_background, run.id)

    return {
        "run_id": run.id,
        "workflow_name": wf_def.name,
        "status": run.status.value,
        "message": "review dispatched, see GET /api/runs/{run_id} for status",
    }

async def _run_in_background(run_id: str) -> None:
    """后台执行:加载 run,扔到线程池跑同步引擎"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: engine.execute_existing_run(run_id))
```

**`GET /api/runs/{run_id}` 响应示例**:

```json
{
  "run_id": "20260812-...",
  "workflow_name": "normal-paper-review",
  "status": "running",
  "start_time": "2026-08-12T14:30:22Z",
  "end_time": null,
  "duration": 15.3,
  "jobs": {
    "extract": {"status": "success", "outputs": {...}, "duration": 12.4},
    "dimensions_novelty": {"status": "running", "outputs": {}, "duration": 5.2},
    ...
    "synthesize": {"status": "pending", ...},
    "decide": {"status": "pending", ...}
  }
}
```

**`POST /api/runs/{run_id}/resume` 流程**:

```python
@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str, req: ResumeRequest, background_tasks: BackgroundTasks):
    run = engine.storage.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"run not found: {run_id}")

    if req.rerun_all:
        engine._reset_all_components(run)
    elif req.rerun_components:
        engine._mark_for_rerun(run, req.rerun_components)

    background_tasks.add_task(_run_in_background, run_id)
    return {"run_id": run_id, "status": "pending", "message": "resume scheduled"}
```

---

## 5. WebSocket 端点

### 5.1 全局 `/ws`

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """全局 WebSocket,接收所有工作流事件"""
    await ws_manager.add_connection(websocket)
    try:
        await ws_manager.send_history(websocket)  # catch-up 历史事件
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text('{"event":"pong"}')
            except asyncio.TimeoutError:
                await websocket.send_text('{"event":"heartbeat"}')
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.remove_connection(websocket)
```

### 5.2 单 run `/ws/runs/{run_id}`

```python
@app.websocket("/ws/runs/{run_id}")
async def websocket_run_endpoint(websocket: WebSocket, run_id: str):
    """针对指定 run 的 WebSocket,只推送该 run_id 相关事件"""
    filtered = FilteredWS(websocket, run_id)
    await ws_manager.add_connection(filtered)  # 注:FilteredWS 也实现 WebSocket 接口
    try:
        await ws_manager.send_history(filtered, run_id=run_id)
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text('{"event":"pong"}')
            except asyncio.TimeoutError:
                await websocket.send_text('{"event":"heartbeat"}')
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.remove_connection(filtered)
```

### 5.3 事件格式(沿用 EventBus 的 `to_json()`)

```json
{
  "event": "step.completed",
  "timestamp": "2026-08-12T14:30:35+00:00",
  "run_id": "20260812-143022-a1b2",
  "job_id": "dimensions_novelty",
  "step_id": "step-uuid",
  "data": {
    "status": "success",
    "duration": 8.2,
    "outputs": {"score": 4, "confidence": 0.85}
  }
}
```

---

## 6. CLI 集成 — `server` 子命令

### 6.1 命令结构

```bash
# Phase 2 新增
python main.py server [options]
  --host TEXT        监听地址 (默认 127.0.0.1)
  --port INT         监听端口 (默认 8000)
  --reload           开发模式自动重载
  --configs-dir PATH configs/ 目录路径 (默认 ./configs,可被 PAPER_REVIEW_CONFIGS_DIR 环境变量覆盖)

# Phase 1 已有(不变)
python main.py run <yaml> ...
python main.py resume <run_id> ...
python main.py list-runs ...
python main.py show-run <run_id>

# 默认行为(与 lwf 一致)
python main.py         # 等价于 python main.py server
```

### 6.2 cli.py 改造

```python
def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(...)
    # ... 现有全局参数 ...

    sub = parser.add_subparsers(dest="command")

    # ── 新增:server 子命令 ──
    server_p = sub.add_parser("server", help="启动 FastAPI 服务器")
    server_p.add_argument("--host", default="127.0.0.1")
    server_p.add_argument("--port", type=int, default=8000)
    server_p.add_argument("--reload", action="store_true")
    server_p.add_argument("--configs-dir", default=None)

    # ... 其他子命令不变 ...

    args = parser.parse_args(argv)

    if args.command == "server":
        return _cmd_server(args)
    elif args.command == "run":
        return _cmd_run(engine, args)
    # ... 其他分发不变 ...
    elif args.command is None:
        # ★ 默认启动 server(与 lwf 一致)
        return _cmd_server(args)
    return 0

def _cmd_server(args) -> int:
    """启动 FastAPI 服务器"""
    if args.configs_dir:
        os.environ["PAPER_REVIEW_CONFIGS_DIR"] = args.configs_dir

    try:
        import uvicorn
    except ImportError:
        print("[Error] uvicorn not installed. Run: pip install uvicorn[standard]", file=sys.stderr)
        return 3

    from .api import create_app

    app = create_app(
        sessions_root=args.storage_dir,
        configs_dir=args.configs_dir or os.environ.get("PAPER_REVIEW_CONFIGS_DIR", "./configs"),
    )

    print(f"""
╔══════════════════════════════════════════════════════╗
║  Paper Review Workflow API Server                    ║
║                                                      ║
║  API docs:  http://{args.host}:{args.port}/docs      ║
║  WebSocket: ws://{args.host}:{args.port}/ws          ║
║  Health:    http://{args.host}:{args.port}/api/health║
╚══════════════════════════════════════════════════════╝
""")

    try:
        uvicorn.run(app, host=args.host, port=args.port,
                    log_level=args.log_level.lower())
    except KeyboardInterrupt:
        return 130
    return 0
```

### 6.3 依赖更新

`pyproject.toml` 增加 `fastapi` + `uvicorn`:

```toml
[project]
dependencies = [
    # ... 现有 ...
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "websockets>=13.0",
]
```

### 6.4 信号处理

`server` 子命令**不复用** Phase 1 的全局 SIGINT/SIGTERM 处理(避免和 uvicorn 的信号管理冲突),改为靠 FastAPI 的 `@app.on_event("shutdown")` 回调清理资源:

```python
@app.on_event("startup")
async def startup_event():
    """FastAPI 启动:加载 configs + 恢复中断 run + attach WSManager"""
    loop = asyncio.get_event_loop()
    ws_manager.attach(loop)
    engine.load_workflow_directory(configs_dir)
    recovered = engine.recover_interrupted_runs()
    if recovered:
        logger.warning(f"[API] recovered {recovered} interrupted runs (marked as cancelled)")

@app.on_event("shutdown")
async def shutdown_event():
    """FastAPI 关闭:取消活跃 run + detach WSManager + 关闭 storage"""
    ws_manager.detach()
    cancelled = engine.shutdown(timeout=10.0)
    if cancelled:
        logger.info(f"[API] cancelled {cancelled} active runs on shutdown")
    engine.storage.close()
```

---

## 7. Engine 扩展

`ReviewEngine` 新增 5 个方法:

```python
class ReviewEngine:
    def __init__(self, ...):
        # ... 现有 ...
        self._workflow_defs: Dict[str, WorkflowDef] = {}  # 已注册的 workflow 定义

    # ── 配置发现 ──

    def load_workflow_directory(self, configs_dir: str) -> Dict[str, WorkflowDef]:
        """扫描目录,加载所有 *.yaml/*.yml 文件。返回 {name: WorkflowDef}"""
        path = Path(configs_dir)
        if not path.exists():
            logger.warning(f"[Engine] configs dir not found: {configs_dir}")
            return {}
        loaded = {}
        for yml_file in sorted(path.glob("**/*.y*ml")):
            try:
                wf_def = self.parser.parse_file(str(yml_file))
                self._register_workflow_def(wf_def)
                loaded[wf_def.name] = wf_def
            except Exception as e:
                logger.warning(f"[Engine] failed to load {yml_file}: {e}")
        return loaded

    def register_workflow(self, yaml_content: str, name: Optional[str] = None) -> WorkflowDef:
        """动态注册 YAML 字符串(POST /api/workflows/register 用)"""
        wf_def = self.parser.parse_string(yaml_content)
        if name:
            wf_def.name = name
        self._register_workflow_def(wf_def)
        return wf_def

    def _register_workflow_def(self, wf_def: WorkflowDef) -> None:
        self._workflow_defs[wf_def.name] = wf_def

    def get_workflow_defs(self) -> Dict[str, WorkflowDef]:
        return dict(self._workflow_defs)

    def get_workflow_def(self, name: str) -> Optional[WorkflowDef]:
        return self._workflow_defs.get(name)

    # ── 崩溃恢复 ──

    def recover_interrupted_runs(self) -> int:
        """启动时扫描 storage,把 status=PENDING/RUNNING 的 run 标记为 CANCELLED"""
        try:
            interrupted = self.storage.list_runs(limit=1000)
        except Exception as e:
            logger.warning(f"[Engine] recover scan failed: {e}")
            return 0
        recovered = 0
        for run in interrupted:
            if run.status in (WorkflowStatus.PENDING, WorkflowStatus.RUNNING):
                run.status = WorkflowStatus.CANCELLED
                run.end_time = datetime.now()
                self.storage.save_run(run)
                recovered += 1
        return recovered

    # ── Dispatch 入口(API 用,同步入口 run_workflow 不变) ──

    def dispatch_workflow(self, workflow_name: str, inputs: Dict[str, Any]) -> WorkflowRun:
        """供 API dispatch 端点用:按 name 创建 PENDING run + 存 storage,不执行。
        实际执行由 _run_in_background 调 execute_existing_run 完成。"""
        wf_def = self.get_workflow_def(workflow_name)
        if wf_def is None:
            raise ValueError(f"workflow not registered: {workflow_name}")
        run = WorkflowRun(
            workflow_def=wf_def,
            trigger_type="workflow_dispatch",
            trigger_payload={"inputs": inputs},
            env=dict(wf_def.env) if wf_def.env else {},
        )
        if wf_def.file_path:
            run.env["__workflow_file__"] = wf_def.file_path
        run.env["__workflow_name__"] = wf_def.name
        self.storage.save_run(run)
        self.event_bus.publish(WorkflowEvent(
            event_type=EventType.WORKFLOW_CREATED, run_id=run.id,
            data={"workflow_name": wf_def.name, "trigger_type": "workflow_dispatch"},
        ))
        return run

    def execute_existing_run(self, run_id: str) -> WorkflowRun:
        """加载已存在的 run 并执行(用于后台 task)"""
        run = self.storage.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        if run.workflow_def is None and run.env.get("__workflow_file__"):
            run.workflow_def = self.parser.parse_file(run.env["__workflow_file__"])
        return self._do_execute(run)
```

### 7.1 关键点

1. **`dispatch_workflow()` 与 `run_workflow()` 的区别**:
   - `run_workflow()`:Phase 1 用,**同步**执行完返回(阻塞)
   - `dispatch_workflow()`:Phase 2 API 用,**只创建 PENDING run 不执行**,实际执行由 `_run_in_background` 异步触发
2. **`recover_interrupted_runs()`** 在 startup 调用,修正上次进程崩溃遗留的 PENDING/RUNNING 状态为 CANCELLED
3. **`execute_existing_run()`** 是 `_do_execute` 的薄包装,加 `workflow_def` 重载逻辑
4. **CLI 模式不变**: `run`/`resume`/`list-runs`/`show-run` 完全不变

---

## 8. 测试策略

### 8.1 测试金字塔

```
        ┌─────────────┐
        │  E2E (1%)    │  真实 arXiv 论文 + 真实 FastAPI server(可选)
        └─────────────┘
       ┌───────────────┐
       │ Integration   │  TestClient + mock LLM + 启动/关闭/恢复
       │    (50%)      │
       └───────────────┘
     ┌───────────────────┐
     │     Unit (49%)     │  WSManager / schemas / engine 扩展
     └───────────────────┘
```

### 8.2 测试目录

```
tests/
├── unit/
│   ├── test_ws_manager.py        # 新增
│   └── test_engine_api_ext.py    # 新增
├── integration/
│   ├── test_api_endpoints.py     # 新增:7 endpoints
│   ├── test_api_websocket.py    # 新增:WS 端点 + 事件推送
│   ├── test_api_lifecycle.py    # 新增:startup 恢复 + shutdown 取消
│   └── test_api_register.py     # 新增:动态注册 YAML
└── e2e/
    └── test_api_e2e_arxiv.py    # 新增:真实 arXiv + 真实 API
```

### 8.3 关键测试场景

**单元测试**:
- `test_ws_manager.py`:attach 后 publish 事件应进 `_history`;`_history_limit` 裁剪正确;`FilteredWS` 只透传匹配 run_id 的事件
- `test_engine_api_ext.py`:`load_workflow_directory` 扫描 configs/;`register_workflow` 动态加载;`recover_interrupted_runs` 标记 PENDING/RUNNING 为 CANCELLED;`dispatch_workflow` 创建 PENDING run 不执行;`execute_existing_run` 加载 run 后执行

**集成测试**(用 `fastapi.testclient.TestClient` + mock LLM):
- `test_api_endpoints.py`:7 个端点的 happy path + 错误路径(404/400/422)
- `test_api_websocket.py`:连接 `/ws` 后 dispatch 一个 run,应收到 `workflow.started` / `step.completed` 等事件;`/ws/runs/{run_id}` 只收该 run 的事件
- `test_api_lifecycle.py`:预置 RUNNING 状态的 run 到 storage,创建 app 触发 startup,验证 run 被标记 cancelled;dispatch 后立刻关闭,验证 active run 被 cancel
- `test_api_register.py`:`POST /api/workflows/register` 上传 YAML,然后 `GET /api/workflows` 应包含该 name,然后 `POST /api/runs` 用该 name 触发

**E2E 测试**(marked,跳过默认):
- `test_api_e2e_arxiv.py`:用真实 ANTHROPIC_API_KEY 启动 server,dispatch arXiv 论文,WS 收事件,验证 decision.json 内容

### 8.4 覆盖率目标

| 模块 | 目标 |
|---|---|
| `api/server.py` | 85%+ |
| `api/ws_manager.py` | 90%+ |
| `api/schemas.py` | 100% |
| `api/dependencies.py` | 100% |
| `engine.py`(新增方法) | 80%+ |

---

## 9. 实施阶段划分

| 里程碑 | 范围 | 预估工时 |
|---|---|---|
| **M1: 依赖 + api 子包骨架** | pyproject.toml 加 fastapi/uvicorn;创建 api/ 空骨架 + .gitkeep | 0.5 天 |
| **M2: WSManager + 单元测试** | api/ws_manager.py + FilteredWS + test_ws_manager.py | 1 天 |
| **M3: Engine 扩展** | load_workflow_directory / register_workflow / recover / dispatch / execute_existing + test_engine_api_ext.py | 1 天 |
| **M4: FastAPI app + 7 endpoints** | api/server.py + api/schemas.py + api/dependencies.py + test_api_endpoints.py | 1.5 天 |
| **M5: WebSocket endpoints** | /ws + /ws/runs/{id} + test_api_websocket.py | 1 天 |
| **M6: Lifecycle + register** | startup/shutdown events + crash recovery + test_api_lifecycle.py + test_api_register.py | 1 天 |
| **M7: CLI server 子命令 + 默认行为 + E2E** | cli.py 改造 + main.py 默认 server + e2e 测试 + 文档 | 1 天 |

**Phase 2 #5 总预估: ~7 天**

---

## 10. 验收标准

- [ ] `python main.py server` 启动 FastAPI 服务器,监听 127.0.0.1:8000
- [ ] `python main.py`(无参数)默认启动 server
- [ ] 启动时自动扫描 `configs/` 目录,`GET /api/workflows` 返回已注册的 workflow
- [ ] 启动时 storage 中的 PENDING/RUNNING run 被标记为 CANCELLED
- [ ] `POST /api/runs` 立即返回 202 + run_id,后台执行评审
- [ ] `GET /api/runs/{run_id}` 返回 run 状态(含 jobs/steps)
- [ ] `POST /api/runs/{run_id}/cancel` 取消运行中的 run
- [ ] `POST /api/runs/{run_id}/resume` 续跑(支持 rerun_components / rerun_all)
- [ ] `POST /api/workflows/register` 动态注册 YAML
- [ ] `GET /api/health` 返回 `{status, version, active_runs, total_runs}`
- [ ] `/ws` 推送所有 EventBus 事件;`/ws/runs/{run_id}` 只推送该 run 的事件
- [ ] 新连接 WS 时收到历史事件 catch-up
- [ ] shutdown 时取消活跃 run + 关闭 storage
- [ ] 单元 + 集成测试覆盖率达标,`pytest` 全过
- [ ] E2E 测试(真实 API + 真实 arXiv)可选启用

---

## 附录 A: lwf 参考模块

**lwf 项目路径**: `/Users/dengyunxi/workspace/for-staging/lwf/workflow_engine/api/server.py`(1539 行)

**复用模式**:
- `BackgroundTasks` + `asyncio.run_in_executor` 模式(line 645, 1498-1505)
- 双 WS endpoint 模式(line 1393, 1417)
- `FilteredWS` 包装类(line 1423-1435)
- startup_event/shutdown_event(line 317-356)
- `_preallocate_run` 模式(line 1464-1495)

**不复用**:
- Webhook 路由(push/PR/release)
- Schedule trigger 后台线程
- Lotus/Supabase/SSO 认证
- Cookie 管理
- Settings router
- DefinitionStore(流程定义版本管理)
- StaticFiles 服务

---

**END OF SPEC**
