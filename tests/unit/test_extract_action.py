import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from paper_review_workflow.actions.extract import ExtractAction
from paper_review_workflow.actions.base import ActionResult


def test_extract_local_pdf_success(tmp_path):
    fixture = Path("tests/fixtures/sample_paper.pdf")
    if not fixture.exists():
        pytest.skip("fixture missing")

    session_dir = tmp_path / "session"
    session_dir.mkdir()

    action = ExtractAction()
    result = action.run(
        params={"source": str(fixture), "session_dir": str(session_dir)},
        env={}, context={}, log_callback=lambda x: None,
    )

    assert result.success
    out_dir = session_dir / "00_extract"
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "sections.json").exists()
    assert (out_dir / "full_text.md").exists()
    assert (out_dir / "references.json").exists()

    meta = json.loads((out_dir / "metadata.json").read_text())
    assert "title" in meta
    assert isinstance(meta["authors"], list)

    assert "paper_id" in result.outputs
    assert len(result.outputs["paper_id"]) == 8
    assert result.outputs["full_text_path"].endswith("full_text.md")


def test_extract_unsupported_source(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    action = ExtractAction()
    result = action.run(
        params={"source": "not-a-pdf-or-arxiv-id", "session_dir": str(session_dir)},
        env={}, context={}, log_callback=lambda x: None,
    )
    assert not result.success
    assert "unsupported source" in result.message


@patch("paper_review_workflow.actions.extract.arxiv.fetch_arxiv")
@patch("paper_review_workflow.actions.extract.pdf.parse_pdf")
def test_extract_arxiv_id(mock_parse, mock_fetch, tmp_path):
    # Setup mocks
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    pdf_path = session_dir / "00_extract" / "artifacts" / "paper.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"fake pdf")

    mock_fetch.return_value = {
        "pdf_path": str(pdf_path), "latex_tarball": None, "arxiv_id": "2402.12098"
    }
    mock_parse.return_value = {
        "metadata": {"title": "Test Paper", "authors": ["A"], "abstract": "abs",
                     "doi": None, "arxiv_id": None, "keywords": []},
        "sections": [{"title": "Intro", "level": 1, "text": "...", "page_start": 0, "page_end": 0}],
        "full_text": "Intro ...",
        "references": [],
    }

    action = ExtractAction()
    result = action.run(
        params={"source": "2402.12098", "session_dir": str(session_dir)},
        env={}, context={}, log_callback=lambda x: None,
    )

    assert result.success
    meta = json.loads((session_dir / "00_extract" / "metadata.json").read_text())
    assert meta["arxiv_id"] == "2402.12098"
