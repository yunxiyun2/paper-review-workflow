import pytest
from paper_review_workflow.core.models import WorkflowRun, WorkflowStatus
from paper_review_workflow.storage.memory import MemoryStorage
from paper_review_workflow.storage.json_file import JsonFileStorage


@pytest.fixture
def sample_run():
    return WorkflowRun()


def test_memory_save_and_get(sample_run):
    s = MemoryStorage()
    s.open()
    s.save_run(sample_run)
    loaded = s.get_run(sample_run.id)
    assert loaded is not None
    assert loaded.id == sample_run.id


def test_memory_get_nonexistent():
    s = MemoryStorage()
    s.open()
    assert s.get_run("nonexistent") is None


def test_memory_list_runs(sample_run):
    s = MemoryStorage()
    s.open()
    s.save_run(sample_run)
    runs = s.list_runs(limit=10)
    assert len(runs) == 1


def test_json_file_save_and_get(sample_run, tmp_path):
    s = JsonFileStorage(data_dir=str(tmp_path))
    s.open()
    s.save_run(sample_run)
    loaded = s.get_run(sample_run.id)
    assert loaded is not None
    assert loaded.id == sample_run.id


def test_json_file_persists_after_close(sample_run, tmp_path):
    s1 = JsonFileStorage(data_dir=str(tmp_path))
    s1.open()
    s1.save_run(sample_run)
    s1.close()

    s2 = JsonFileStorage(data_dir=str(tmp_path))
    s2.open()
    loaded = s2.get_run(sample_run.id)
    assert loaded is not None
