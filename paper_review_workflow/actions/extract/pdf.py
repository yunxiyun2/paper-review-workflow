"""PDF parsing using PyMuPDF (fitz)."""
import re
from pathlib import Path
from typing import Dict, List

import pymupdf as fitz  # PyMuPDF


SECTION_TITLE_RE = re.compile(
    r"^(?:\d+\.?\d*\.?\d*\s+)?(Abstract|Introduction|Background|Related Work|"
    r"Method(?:s)?|Approach|Model|Experiments?|Results?|Evaluation|"
    r"Discussion|Conclusion[s]?|References|Acknowledgments?)\s*$",
    re.IGNORECASE,
)


def parse_pdf(pdf_path: str) -> Dict:
    """Parse a PDF file into metadata, sections, full_text, references.

    Returns:
        {
            "metadata": {"title", "authors", "abstract", "doi", "arxiv_id", "keywords"},
            "sections": [{"title", "level", "text", "page_start", "page_end"}],
            "full_text": str (markdown),
            "references": [{"raw", "page"}],
        }
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(path))
    try:
        pages_text = [page.get_text("text") for page in doc]
        full_text = "\n\n".join(pages_text)

        metadata = _extract_metadata(doc, pages_text)
        sections = _extract_sections(pages_text)
        references = _extract_references(pages_text)

        return {
            "metadata": metadata,
            "sections": sections,
            "full_text": full_text,
            "references": references,
        }
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
    full = "\n".join(pages_text)
    m = re.search(r"Abstract[:\s]*(.+?)(?=\n\s*(?:1\.?\s+)?(?:Introduction|Keywords|I\.\s))",
                  full, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()[:2000]
    return ""


def _extract_sections(pages_text: List[str]) -> List[Dict]:
    sections = []
    current = None

    for page_idx, page_text in enumerate(pages_text):
        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            m = SECTION_TITLE_RE.match(stripped)
            if m:
                if current:
                    current["page_end"] = page_idx
                    sections.append(current)
                current = {
                    "title": stripped,
                    "level": 1 if not stripped[0].isdigit() else
                             (len(stripped.split()[0].rstrip(".").split("."))),
                    "text": "",
                    "page_start": page_idx,
                    "page_end": page_idx,
                }
            elif current:
                current["text"] += stripped + "\n"

    if current:
        sections.append(current)
    return sections


def _extract_references(pages_text: List[str]) -> List[Dict]:
    refs = []
    in_refs = False
    for page_idx, page_text in enumerate(pages_text):
        for line in page_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^References\s*$", stripped, re.IGNORECASE):
                in_refs = True
                continue
            if in_refs:
                if re.match(r"^\[\d+\]", stripped) or stripped[0:1].isupper():
                    refs.append({"raw": stripped, "page": page_idx})
    return refs
