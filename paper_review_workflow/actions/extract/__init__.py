"""ExtractAction: parse a local PDF into structured artifacts.

PDF-only by design: the review flow accepts uploaded PDF files, and the
arXiv/URL import path has been removed (routing ambiguity + stale-artifact
hazards outweighed the convenience).
"""
import hashlib
import json
import shutil
import logging
from pathlib import Path

from ..base import BaseAction, ActionResult
from ..registry import ActionRegistry
from .pdf import parse_pdf

logger = logging.getLogger(__name__)


class ExtractAction(BaseAction):
    """Parse a local PDF into metadata + full text."""

    @property
    def description(self) -> str:
        return "Extract paper content from a local PDF"

    def run(self, params, env, context, log_callback=None):
        source = params.get("source", "").strip()
        if not source:
            return ActionResult(success=False, message="source is required")

        path = Path(source)
        # Strict routing: a local PDF file, nothing else. (parse_arxiv_id used
        # re.search on the raw path, so local files like papers/2401.1.pdf were
        # silently hijacked into the arXiv branch — that path no longer exists.)
        if path.suffix.lower() != ".pdf" or not path.is_file():
            return ActionResult(
                success=False,
                message=f"only local PDF files are supported (got: {source})",
            )

        try:
            return self._extract_pdf(path, Path(params["session_dir"]), log_callback)
        except Exception as e:
            logger.exception("extract failed")
            return ActionResult(success=False, message=f"extract failed: {e}")

    def _extract_pdf(self, pdf_path: Path, session_dir: Path, log_callback) -> ActionResult:
        if log_callback:
            log_callback(f"parsing local PDF: {pdf_path}")

        parsed = parse_pdf(str(pdf_path))
        out_dir = session_dir / "00_extract"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy the PDF to artifacts (session dirs are per-run; copy overwrites
        # any stale artifact from a resumed run)
        artifacts = out_dir / "artifacts"
        artifacts.mkdir(exist_ok=True)
        shutil.copy(pdf_path, artifacts / "paper.pdf")

        return self._write_outputs(out_dir, parsed, log_callback)

    def _write_outputs(self, out_dir: Path, parsed: dict, log_callback) -> ActionResult:
        metadata = parsed["metadata"]
        paper_id = hashlib.sha256(metadata["title"].encode("utf-8")).hexdigest()[:8]

        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "full_text.md").write_text(parsed["full_text"], encoding="utf-8")

        if log_callback:
            log_callback(f"extracted: {metadata['title'][:80]}")

        return ActionResult(
            success=True,
            outputs={
                "paper_id": paper_id,
                "full_text_path": str(out_dir / "full_text.md"),
                "metadata_path": str(out_dir / "metadata.json"),
            },
            log_lines=[f"extracted paper_id={paper_id}"],
        )


def register_extract_action(registry: ActionRegistry) -> None:
    registry.register("paper-review/extract@v1", ExtractAction())
