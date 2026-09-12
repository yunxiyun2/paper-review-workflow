"""PDF parsing using PyMuPDF (fitz).

Produces exactly what downstream consumers use: metadata (title/authors/
abstract feed the review prompts) and full_text (the LLM's paper text).
Sections/references extraction was removed — it was line-heuristic based,
unreliable on real PDFs, and consumed by nothing.
"""
import re
from pathlib import Path
from typing import Dict, List

import pymupdf as fitz  # PyMuPDF


def parse_pdf(pdf_path: str) -> Dict:
    """Parse a PDF file into metadata and full text.

    Returns:
        {
            "metadata": {"title", "authors", "abstract", "doi", "arxiv_id", "keywords"},
            "full_text": str,
        }

    Raises:
        ValueError: if the PDF yields (almost) no text — likely a scanned/
            image-only document that would need OCR.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(path))
    try:
        # sort=True orders blocks top-to-bottom/left-to-right, which keeps
        # single-column layouts and two-column layouts substantially more
        # readable than the raw insertion order.
        pages_text = [page.get_text("text", sort=True) for page in doc]
        full_text = "\n\n".join(p.strip() for p in pages_text if p.strip())

        total_chars = sum(len(p) for p in pages_text)
        if total_chars < max(200, 50 * len(pages_text)):
            raise ValueError(
                f"PDF contains almost no extractable text "
                f"({total_chars} chars across {len(pages_text)} pages) — "
                "it is likely a scanned/image-only document and would need OCR"
            )

        metadata = _extract_metadata(doc, pages_text)
        return {"metadata": metadata, "full_text": full_text}
    finally:
        doc.close()


def _extract_metadata(doc, pages_text: List[str]) -> Dict:
    # Prefer PDF metadata, fallback to first-page heuristics
    pdf_meta = doc.metadata or {}
    title = pdf_meta.get("title", "").strip()
    authors_str = pdf_meta.get("author", "").strip()

    if not title:
        # First non-empty line of page 1, skip "arXiv:" prelude
        first_page_lines = [l.strip() for l in pages_text[0].splitlines() if l.strip()]
        for line in first_page_lines:
            if line.lower().startswith("arxiv"):
                continue
            title = line
            break

    if not authors_str:
        # Heuristic: line(s) after title, before "Abstract"
        lines = pages_text[0].splitlines() if pages_text else []
        author_lines = []
        in_authors = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith("abstract"):
                break
            if title and stripped == title:
                in_authors = True
                continue
            if in_authors:
                author_lines.append(stripped)
        authors_str = ", ".join(author_lines[:3])  # cap at 3

    authors = [a.strip() for a in re.split(r"[,;]|\band\b", authors_str) if a.strip()]
    abstract = _extract_abstract(pages_text)

    return {
        "title": title or "Untitled",
        "authors": authors,
        "abstract": abstract,
        "doi": None,
        "arxiv_id": None,
        "keywords": [],
    }


def _extract_abstract(pages_text: List[str]) -> str:
    # Anchor on the first page only — searching the whole document can hit an
    # "Abstract" heading in the body or references.
    first_page = pages_text[0] if pages_text else ""
    m = re.search(r"Abstract[:\s]*(.+?)(?=\n\s*(?:1\.?\s+)?(?:Introduction|Keywords|I\.\s))",
                  first_page, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()[:2000]
    return ""
