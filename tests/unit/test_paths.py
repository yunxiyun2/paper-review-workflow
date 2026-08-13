from paper_review_workflow.core.paths import generate_paper_id, generate_run_id, get_session_dir


def test_paper_id_is_sha256_prefix():
    pid = generate_paper_id("Attention Is All You Need")
    assert len(pid) == 8
    assert all(c in "0123456789abcdef" for c in pid)


def test_paper_id_deterministic():
    assert generate_paper_id("Same Title") == generate_paper_id("Same Title")


def test_paper_id_different_titles():
    assert generate_paper_id("Title A") != generate_paper_id("Title B")


def test_run_id_format():
    rid = generate_run_id()
    # 20260812-143022-a1b2c3d4 (8 hex chars suffix for collision resistance)
    assert len(rid) == 24
    assert rid[8] == "-"
    assert rid[15] == "-"


def test_run_id_unique():
    ids = {generate_run_id() for _ in range(100)}
    assert len(ids) == 100  # no collisions


def test_session_dir_structure(tmp_path):
    d = get_session_dir(str(tmp_path), "abcd1234", "20260812-143022-a1b2")
    assert d == tmp_path / "abcd1234" / "20260812-143022-a1b2"
