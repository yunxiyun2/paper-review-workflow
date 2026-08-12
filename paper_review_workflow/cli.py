"""CLI dispatcher."""
import argparse
import json
import logging
import signal
import sys

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


def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--storage", default="json", choices=["memory", "json"])
    common.add_argument("--storage-dir", default="./sessions")
    common.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    parser = argparse.ArgumentParser(description="Paper review workflow",
                                     parents=[common])
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="run a YAML review workflow",
                           parents=[common])
    run_p.add_argument("yaml")
    run_p.add_argument("--trigger", default="workflow_dispatch")
    run_p.add_argument("--payload", default="{}")
    run_p.add_argument("--env", action="append", default=[])

    res_p = sub.add_parser("resume", help="resume a run", parents=[common])
    res_p.add_argument("run_id")
    res_p.add_argument("--rerun", default=None, help="comma-separated component names")
    res_p.add_argument("--rerun-all", action="store_true")

    list_p = sub.add_parser("list-runs", help="list historical runs",
                            parents=[common])
    list_p.add_argument("--paper-id", default=None)
    list_p.add_argument("--status", default=None,
                        choices=["pending", "running", "success", "failure", "cancelled"])
    list_p.add_argument("--limit", type=int, default=50)

    show_p = sub.add_parser("show-run", help="show run details", parents=[common])
    show_p.add_argument("run_id")

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    if args.command is None:
        parser.print_help()
        return 0

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
