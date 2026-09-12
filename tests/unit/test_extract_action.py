import json
import pytest
from pathlib import Path

from paper_review_workflow.actions.extract import ExtractAction


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
    assert (out_dir / "full_text.md").exists()
    # sections.json / references.json were removed (no downstream consumer)
    assert not (out_dir / "sections.json").exists()
    assert not (out_dir / "references.json").exists()

    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "title" in meta
    assert isinstance(meta["authors"], list)

    assert "paper_id" in result.outputs
    assert len(result.outputs["paper_id"]) == 8
    assert result.outputs["full_text_path"].endswith("full_text.md")


def test_extract_rejects_non_pdf_path(tmp_path):
    """A local file that merely *contains* an arXiv-like number must NOT be
    hijacked into any other branch — only real .pdf files are accepted."""
    fake = tmp_path / "papers" / "2401.12345.txt"
    fake.parent.mkdir(parents=True)
    fake.write_text("not a pdf")

    action = ExtractAction()
    result = action.run(
        params={"source": str(fake), "session_dir": str(tmp_path / "s")},
        env={}, context={}, log_callback=lambda x: None,
    )
    assert not result.success
    assert "only local PDF files" in result.message


def test_extract_rejects_arxiv_id():
    """arXiv IDs / URLs are no longer a supported import path."""
    action = ExtractAction()
    result = action.run(
        params={"source": "2402.12098", "session_dir": "/tmp/x"},
        env={}, context={}, log_callback=lambda x: None,
    )
    assert not result.success
    assert "only local PDF files" in result.message


def test_extract_rejects_missing_pdf(tmp_path):
    action = ExtractAction()
    result = action.run(
        params={"source": str(tmp_path / "missing.pdf"), "session_dir": str(tmp_path / "s")},
        env={}, context={}, log_callback=lambda x: None,
    )
    assert not result.success
    assert "only local PDF files" in result.message


def test_extract_rejects_empty_source():
    action = ExtractAction()
    result = action.run(
        params={"source": "  ", "session_dir": "/tmp/x"},
        env={}, context={}, log_callback=lambda x: None,
    )
    assert not result.success
    assert "source is required" in result.message
