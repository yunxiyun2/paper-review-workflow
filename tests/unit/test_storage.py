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


def test_save_job_instance_applies_update(tmp_path):
    """C1 fix: save_job_instance must actually persist the job update"""
    from paper_review_workflow.core.models import JobInstance, JobDef, JobStatus
    s = MemoryStorage()
    s.open()
    run = WorkflowRun()
    job = JobInstance(job_def=JobDef(id="j1", name="J", runs_on="local"))
    job.status = JobStatus.PENDING
    run.jobs["j1"] = job
    s.save_run(run)

    # Mutate and save just the job
    job.status = JobStatus.SUCCESS
    s.save_job_instance(run.id, job)

    loaded = s.get_run(run.id)
    assert loaded.jobs["j1"].status == JobStatus.SUCCESS


def test_path_traversal_rejected(tmp_path):
    """C2 fix: run_id with path separators must be rejected"""
    s = JsonFileStorage(data_dir=str(tmp_path))
    s.open()
    with pytest.raises(ValueError):
        s.get_run("../../etc/passwd")


def test_memory_list_runs_returns_copies():
    """I1 fix: list_runs should return deep copies, not references"""
    s = MemoryStorage()
    s.open()
    run = WorkflowRun()
    s.save_run(run)
    runs = s.list_runs(limit=10)
    runs[0].status = WorkflowStatus.FAILURE  # mutate
    loaded = s.get_run(run.id)
    assert loaded.status == WorkflowStatus.PENDING  # unchanged


def test_corrupt_index_rebuilds(tmp_path):
    """I3 fix: corrupt index should trigger rebuild from run files"""
    s1 = JsonFileStorage(data_dir=str(tmp_path))
    s1.open()
    run = WorkflowRun()
    s1.save_run(run)
    s1.close()

    # Corrupt the index
    (tmp_path / "index.json").write_text("not valid json")

    s2 = JsonFileStorage(data_dir=str(tmp_path))
    s2.open()
    loaded = s2.get_run(run.id)
    assert loaded is not None  # should still find via rebuilt index
