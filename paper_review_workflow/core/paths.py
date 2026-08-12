"""Session directory and ID generation utilities."""
import hashlib
import secrets
import time
from pathlib import Path


def generate_paper_id(title: str) -> str:
    """8-char hex of sha256(title)."""
    return hashlib.sha256(title.encode("utf-8")).hexdigest()[:8]


def generate_run_id() -> str:
    """Format: YYYYMMDD-HHMMSS-xxxx (4 hex chars)."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(2)
    return f"{ts}-{suffix}"


def get_session_dir(sessions_root: str, paper_id: str, run_id: str) -> Path:
    return Path(sessions_root) / paper_id / run_id


def generate_pending_paper_id(run_id: str) -> str:
    """Pre-extract paper_id placeholder."""
    return f"pending-{run_id}"
