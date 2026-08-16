"""CLI dispatcher."""
import argparse
import json
import logging
import signal
import sys
from typing import Optional, List

from .engine import ReviewEngine
from .storage import MemoryStorage, JsonFileStorage


def _build_storage(args):
    storage_type = getattr(args, "storage", "json")
    if storage_type == "memory":
        return MemoryStorage()
    return JsonFileStorage(data_dir=getattr(args, "storage_dir", "./sessions"))


def _cmd_run(engine: ReviewEngine, args) -> int:
    try:
        payload = json.loads(args.payload) if args.payload else {}
    except json.JSONDecodeError as e:
        print(f"[Error] payload JSON parse failed: {e}", file=sys.stderr)
        return 2

    # Parse --env KEY=VALUE overrides
    extra_env: dict = {}
    for kv in args.env or []:
        if "=" not in kv:
            print(f"[Error] invalid --env {kv}, expected KEY=VALUE", file=sys.stderr)
            return 2
        k, _, v = kv.partition("=")
        extra_env[k] = v

    run = engine.run_from_file(args.yaml, payload=payload, extra_env=extra_env)
    print(f"\n final status: {run.status.value}")
    if run.duration:
        print(f" total duration: {run.duration:.2f}s")
    return 0 if run.status.value == "success" else 1


def _cmd_resume(engine: ReviewEngine, args) -> int:
    rerun = args.rerun.split(",") if args.rerun else None
    run = engine.resume_run(args.run_id, rerun_components=rerun, rerun_all=args.rerun_all)
    print(f"\n final status: {run.status.value}")
    return 0 if run.status.value == "success" else 1


def _cmd_list_runs(engine: ReviewEngine, args) -> int:
    from .core.models import WorkflowStatus
    status = WorkflowStatus(args.status) if args.status else None
    runs = engine.list_runs(paper_id=args.paper_id, status=status, limit=args.limit)
    print(f"{'RUN_ID':<24} {'STATUS':<10} {'WORKFLOW':<20} {'DURATION'}")
    for r in runs:
        dur = f"{r.duration:.1f}s" if r.duration else "-"
        wf = (r.workflow_def.name if r.workflow_def else "")[:20]
        print(f"{r.id:<24} {r.status.value:<10} {wf:<20} {dur}")
    return 0


def _cmd_show_run(engine: ReviewEngine, args) -> int:
    run = engine.get_run(args.run_id)
    if run is None:
        print(f"run not found: {args.run_id}", file=sys.stderr)
        return 1
    print(json.dumps(run.to_dict(), indent=2, default=str))
    return 0


def _cmd_export(engine: ReviewEngine, args) -> int:
    from pathlib import Path
    from .exporters.openreview import OpenReviewExporter
    exporter = OpenReviewExporter()
    try:
        xml = exporter.export(args.run_id, engine.storage)
    except ValueError as e:
        print(f"[Error] {e}", file=sys.stderr)
        return 1
    if args.output:
        Path(args.output).write_text(xml, encoding="utf-8")
        print(f"Exported to {args.output}")
    else:
        print(xml)
    return 0


def _cmd_server(args) -> int:
    """启动 FastAPI 服务器"""
    import os

    if args.configs_dir:
        os.environ["PAPER_REVIEW_CONFIGS_DIR"] = args.configs_dir

    try:
        import uvicorn
    except ImportError:
        print("[Error] uvicorn not installed. Run: pip install uvicorn[standard]",
              file=sys.stderr)
        return 3

    from .api import create_app

    app = create_app(
        storage=_build_storage(args),
        sessions_root=args.storage_dir,
        configs_dir=args.configs_dir or os.environ.get(
            "PAPER_REVIEW_CONFIGS_DIR", "./configs"),
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


def main(argv: Optional[list] = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--storage", default="json", choices=["memory", "json"])
    common.add_argument("--storage-dir", default="./sessions")
    common.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # For subparsers: use SUPPRESS defaults so global args set before the
    # subcommand are not overwritten by the subparser's defaults.
    common_sub = argparse.ArgumentParser(add_help=False)
    common_sub.add_argument("--storage", default=argparse.SUPPRESS,
                           choices=["memory", "json"])
    common_sub.add_argument("--storage-dir", default=argparse.SUPPRESS)
    common_sub.add_argument("--log-level", default=argparse.SUPPRESS,
                            choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    parser = argparse.ArgumentParser(description="Paper review workflow",
                                     parents=[common])
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="run a YAML review workflow",
                           parents=[common_sub])
    run_p.add_argument("yaml")
    run_p.add_argument("--trigger", default="workflow_dispatch")
    run_p.add_argument("--payload", default="{}")
    run_p.add_argument("--env", action="append", default=[])

    res_p = sub.add_parser("resume", help="resume a run", parents=[common_sub])
    res_p.add_argument("run_id")
    res_p.add_argument("--rerun", default=None, help="comma-separated component names")
    res_p.add_argument("--rerun-all", action="store_true")

    list_p = sub.add_parser("list-runs", help="list historical runs",
                            parents=[common_sub])
    list_p.add_argument("--paper-id", default=None)
    list_p.add_argument("--status", default=None,
                        choices=["pending", "running", "success", "failure", "cancelled"])
    list_p.add_argument("--limit", type=int, default=50)

    show_p = sub.add_parser("show-run", help="show run details", parents=[common_sub])
    show_p.add_argument("run_id")

    # export subcommand (Phase 2 #4)
    export_p = sub.add_parser("export", help="导出评审结果为 XML")
    export_p.add_argument("run_id", help="要导出的 run ID")
    export_p.add_argument("--format", default="xml", choices=["xml"], help="导出格式")
    export_p.add_argument("--output", default=None, help="输出文件路径(不指定则 stdout)")

    # ── server 子命令 (Phase 2 #5) ──
    server_p = sub.add_parser("server", help="启动 FastAPI 服务器",
                               parents=[common_sub])
    server_p.add_argument("--host", default="127.0.0.1", help="监听地址")
    server_p.add_argument("--port", type=int, default=8000, help="监听端口")
    server_p.add_argument("--reload", action="store_true", help="开发模式自动重载")
    server_p.add_argument("--configs-dir", default=None,
                          help="configs/ 目录路径(默认 PAPER_REVIEW_CONFIGS_DIR 或 ./configs)")

    args = parser.parse_args(argv)
    # Apply fallback defaults for global args when subparser used SUPPRESS
    # (i.e., when the user did not pass them after the subcommand)
    if not hasattr(args, "storage"):
        args.storage = "json"
    if not hasattr(args, "storage_dir"):
        args.storage_dir = "./sessions"
    if not hasattr(args, "log_level"):
        args.log_level = "INFO"
    # Apply fallback defaults for server subcommand args when no subcommand
    # is given (default-to-server behavior needs these to exist on args)
    if not hasattr(args, "host"):
        args.host = "127.0.0.1"
    if not hasattr(args, "port"):
        args.port = 8000
    if not hasattr(args, "reload"):
        args.reload = False
    if not hasattr(args, "configs_dir"):
        args.configs_dir = None
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    if args.command == "server":
        return _cmd_server(args)

    if args.command is None:
        # ★ 默认启动 server (与 lwf 一致)
        return _cmd_server(args)

    engine = ReviewEngine(storage=_build_storage(args),
                          sessions_root=args.storage_dir)

    def _sig(signum, frame):
        print("\n Received interrupt, shutting down gracefully...")
        cancelled = engine.shutdown(timeout=30)
        print(f"Cancelled {cancelled} active run(s). State saved.")
        sys.exit(130)
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    if args.command == "run":
        return _cmd_run(engine, args)
    elif args.command == "resume":
        return _cmd_resume(engine, args)
    elif args.command == "list-runs":
        return _cmd_list_runs(engine, args)
    elif args.command == "show-run":
        return _cmd_show_run(engine, args)
    elif args.command == "export":
        return _cmd_export(engine, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
