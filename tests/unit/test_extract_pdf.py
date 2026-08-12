import pytest
from pathlib import Path
from paper_review_workflow.actions.extract.pdf import parse_pdf


def test_parse_pdf_returns_metadata_sections_fulltext(tmp_path):
    fixture = Path("tests/fixtures/sample_paper.pdf")
    if not fixture.exists():
        pytest.skip("fixture missing")

    result = parse_pdf(str(fixture))

    assert "metadata" in result
    assert "sections" in result
    assert "full_text" in result
    assert "references" in result
    assert isinstance(result["metadata"]["title"], str)
    assert len(result["metadata"]["title"]) > 0
    assert len(result["sections"]) >= 1
    assert "Introduction" in result["full_text"] or "introduction" in result["full_text"].lower()


def test_parse_pdf_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        parse_pdf("/nonexistent/path/to/file.pdf")


def test_parse_pdf_corrupted(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(Exception):
        parse_pdf(str(bad))
