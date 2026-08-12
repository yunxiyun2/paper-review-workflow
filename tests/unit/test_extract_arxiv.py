import pytest
from unittest.mock import patch, MagicMock
from paper_review_workflow.actions.extract.arxiv import (
    parse_arxiv_id, fetch_arxiv_source, fetch_arxiv_pdf, ArxivFetchError
)


def test_parse_arxiv_id_from_numeric():
    assert parse_arxiv_id("2402.12098") == "2402.12098"


def test_parse_arxiv_id_from_abs_url():
    assert parse_arxiv_id("https://arxiv.org/abs/2402.12098") == "2402.12098"


def test_parse_arxiv_id_from_pdf_url():
    assert parse_arxiv_id("https://arxiv.org/pdf/2402.12098.pdf") == "2402.12098"


def test_parse_arxiv_id_from_v_version():
    assert parse_arxiv_id("2402.12098v2") == "2402.12098"


def test_parse_arxiv_id_invalid():
    with pytest.raises(ValueError):
        parse_arxiv_id("not-an-arxiv-id")


def _make_mock_client(mock_resp):
    """Build a MagicMock client whose context-manager returns itself,
    so `with httpx.Client(...) as client:` yields the same mock."""
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.get.return_value = mock_resp
    return mock_client


@patch("paper_review_workflow.actions.extract.arxiv.httpx.get")
def test_fetch_arxiv_pdf_success(mock_get, tmp_path):
    """Test fetch_arxiv_pdf with patched httpx.get (sync)"""
    import paper_review_workflow.actions.extract.arxiv as arxiv_mod
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"%PDF-1.4 fake pdf"
    mock_resp.raise_for_status = MagicMock()
    mock_client = _make_mock_client(mock_resp)
    with patch.object(arxiv_mod.httpx, "Client", return_value=mock_client):
        out = tmp_path / "paper.pdf"
        fetch_arxiv_pdf("2402.12098", out)
        assert out.read_bytes() == b"%PDF-1.4 fake pdf"


@patch("paper_review_workflow.actions.extract.arxiv.httpx.Client")
def test_fetch_arxiv_pdf_failure_raises(mock_client_cls, tmp_path):
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.raise_for_status.side_effect = Exception("404")
    mock_client_cls.return_value = _make_mock_client(mock_resp)

    out = tmp_path / "paper.pdf"
    with pytest.raises(ArxivFetchError):
        fetch_arxiv_pdf("9999.99999", out)


@patch("paper_review_workflow.actions.extract.arxiv.httpx.Client")
def test_fetch_arxiv_source_latex_tarball(mock_client_cls, tmp_path):
    import tarfile
    import io
    # Build a fake tarball with a .tex file
    tar_path = tmp_path / "src.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tex_content = b"\\title{Fake Paper}\n\\section{Intro}\nHello"
        info = tarfile.TarInfo(name="main.tex")
        info.size = len(tex_content)
        tf.addfile(info, io.BytesIO(tex_content))

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = tar_path.read_bytes()
    mock_resp.raise_for_status = MagicMock()
    mock_client_cls.return_value = _make_mock_client(mock_resp)

    out = tmp_path / "src.tar.gz"
    has_latex = fetch_arxiv_source("2402.12098", out)
    assert has_latex is True
    assert out.exists()
