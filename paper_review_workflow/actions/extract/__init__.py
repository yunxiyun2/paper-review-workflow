"""ExtractAction: parse PDF or arXiv paper into structured artifacts."""
import hashlib
import json
import shutil
import logging
from pathlib import Path
from typing import Optional

from ..base import BaseAction, ActionResult
from ..registry import ActionRegistry
from .pdf import parse_pdf
from .arxiv import parse_arxiv_id, fetch_arxiv, ArxivFetchError

logger = logging.getLogger(__name__)


class ExtractAction(BaseAction):
    """Parse a paper (PDF path or arXiv ID) into metadata/sections/full_text/references."""

    @property
    def description(self) -> str:
        return "Extract paper content (PDF or arXiv)"

    def run(self, params, env, context, log_callback=None):
        source = params.get("source", "").strip()
        session_dir = Path(params["session_dir"])

        if not source:
            return ActionResult(success=False, message="source is required")

        try:
            if self._is_arxiv(source):
                return self._extract_arxiv(source, session_dir, log_callback)
            elif source.endswith(".pdf") and Path(source).is_file():
                return self._extract_pdf(source, session_dir, log_callback)
            else:
                return ActionResult(
                    success=False,
                    message=f"unsupported source: {source} (must be arXiv ID/URL or PDF path)"
                )
        except Exception as e:
            logger.exception("extract failed")
            return ActionResult(success=False, message=f"extract failed: {e}")

    def _is_arxiv(self, source: str) -> bool:
        try:
            parse_arxiv_id(source)
            return True
        except ValueError:
            return False

    def _extract_pdf(self, pdf_path: str, session_dir: Path, log_callback) -> ActionResult:
        if log_callback:
            log_callback(f"parsing local PDF: {pdf_path}")

        parsed = parse_pdf(pdf_path)
        out_dir = session_dir / "00_extract"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy the PDF to artifacts
        artifacts = out_dir / "artifacts"
        artifacts.mkdir(exist_ok=True)
        shutil.copy(pdf_path, artifacts / "paper.pdf")

        return self._write_outputs(out_dir, parsed, arxiv_id=None, log_callback=log_callback)

    def _extract_arxiv(self, source: str, session_dir: Path, log_callback) -> ActionResult:
        arxiv_id = parse_arxiv_id(source)
        if log_callback:
            log_callback(f"fetching arXiv paper: {arxiv_id}")

        try:
            fetch_result = fetch_arxiv(arxiv_id, session_dir)
        except ArxivFetchError as e:
            return ActionResult(success=False, message=str(e))

        if not fetch_result.get("pdf_path"):
            return ActionResult(success=False, message=f"could not fetch any PDF for {arxiv_id}")

        if log_callback:
            log_callback(f"parsing PDF: {fetch_result['pdf_path']}")

        parsed = parse_pdf(fetch_result["pdf_path"])
        parsed["metadata"]["arxiv_id"] = arxiv_id

        out_dir = session_dir / "00_extract"
        return self._write_outputs(out_dir, parsed, arxiv_id=arxiv_id, log_callback=log_callback)

    def _write_outputs(self, out_dir: Path, parsed: dict,
                       arxiv_id: Optional[str], log_callback) -> ActionResult:
        metadata = parsed["metadata"]
        if arxiv_id:
            metadata["arxiv_id"] = arxiv_id

        # Compute paper_id from title
        paper_id = hashlib.sha256(metadata["title"].encode("utf-8")).hexdigest()[:8]

        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )
        (out_dir / "sections.json").write_text(
            json.dumps(parsed["sections"], ensure_ascii=False, indent=2)
        )
        (out_dir / "full_text.md").write_text(parsed["full_text"])
        (out_dir / "references.json").write_text(
            json.dumps(parsed["references"], ensure_ascii=False, indent=2)
        )

        if log_callback:
            log_callback(f"extracted: {metadata['title'][:80]}")
            log_callback(f"sections: {len(parsed['sections'])}, references: {len(parsed['references'])}")

        return ActionResult(
            success=True,
            outputs={
                "paper_id": paper_id,
                "full_text_path": str(out_dir / "full_text.md"),
                "metadata_path": str(out_dir / "metadata.json"),
                "sections_path": str(out_dir / "sections.json"),
                "references_path": str(out_dir / "references.json"),
            },
            log_lines=[f"extracted paper_id={paper_id}"],
        )


def register_extract_action(registry: ActionRegistry) -> None:
    registry.register("paper-review/extract@v1", ExtractAction())
