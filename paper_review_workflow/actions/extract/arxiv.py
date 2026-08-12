"""arXiv paper fetching: latex source first, PDF fallback."""
import re
import logging
from pathlib import Path
from typing import Union

import httpx

logger = logging.getLogger(__name__)


ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)


class ArxivFetchError(Exception):
    pass


def parse_arxiv_id(source: str) -> str:
    """Extract arXiv ID from various input formats."""
    source = source.strip()
    m = ARXIV_ID_RE.search(source)
    if not m:
        # Try pure ID without URL
        if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", source):
            return re.sub(r"v\d+$", "", source)
        raise ValueError(f"cannot parse arxiv id from: {source}")
    return m.group(1)


def fetch_arxiv_source(arxiv_id: str, dest: Union[str, Path],
                       timeout: float = 60.0) -> bool:
    """Fetch latex source tarball. Returns True if successful."""
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, follow_redirects=True)
            resp.raise_for_status()
            Path(dest).write_bytes(resp.content)
        return True
    except Exception as e:
        logger.warning(f"arxiv source fetch failed for {arxiv_id}: {e}")
        return False


def fetch_arxiv_pdf(arxiv_id: str, dest: Union[str, Path],
                    timeout: float = 60.0) -> None:
    """Fetch PDF. Raises ArxivFetchError on failure."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, follow_redirects=True)
            resp.raise_for_status()
            Path(dest).write_bytes(resp.content)
    except Exception as e:
        raise ArxivFetchError(f"failed to fetch PDF for {arxiv_id}: {e}")


def fetch_arxiv(arxiv_id: str, session_dir: Path) -> dict:
    """Fetch arxiv paper, prefer latex source, fallback to PDF.

    Returns:
        {"pdf_path": str, "latex_tarball": Optional[str], "arxiv_id": str}
    """
    artifacts = session_dir / "00_extract" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    latex_path = artifacts / "source.tar.gz"
    pdf_path = artifacts / "paper.pdf"

    if fetch_arxiv_source(arxiv_id, latex_path):
        logger.info(f"got latex source for {arxiv_id}")
        # Still try to get PDF for fallback parsing
        try:
            fetch_arxiv_pdf(arxiv_id, pdf_path)
        except ArxivFetchError:
            pass
        return {"pdf_path": str(pdf_path) if pdf_path.exists() else None,
                "latex_tarball": str(latex_path), "arxiv_id": arxiv_id}

    # Fallback to PDF only
    fetch_arxiv_pdf(arxiv_id, pdf_path)
    return {"pdf_path": str(pdf_path), "latex_tarball": None, "arxiv_id": arxiv_id}
