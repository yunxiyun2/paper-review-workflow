import pytest
from pathlib import Path
from datetime import datetime
from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.core.models import WorkflowRun, WorkflowStatus
from paper_review_workflow.storage.memory import MemoryStorage
from paper_review_workflow.storage.json_file import JsonFileStorage


@pytest.fixture
def engine():
    return ReviewEngine(storage=MemoryStorage())


@pytest.fixture
def configs_dir(tmp_path):
    """Create a tmp configs/ dir with 2 YAML files + 1 bad"""
    d = tmp_path / "configs"
    d.mkdir()
    (d / "wf1.yaml").write_text("""
name: wf-one
on: {workflow_dispatch: {}}
jobs:
  j:
    runs-on: local
    steps:
      - uses: paper-review/echo@v1
""")
    (d / "wf2.yml").write_text("""
name: wf-two
on: {workflow_dispatch: {}}
jobs:
  j:
    runs-on: local
    steps:
      - uses: paper-review/echo@v1
""")
    (d / "bad.yml").write_text("not: valid: yaml: [")
    return str(d)


def test_load_workflow_directory_loads_yamls(engine, configs_dir):
    loaded = engine.load_workflow_directory(configs_dir)
    assert "wf-one" in loaded
    assert "wf-two" in loaded
    assert len(loaded) == 2  # bad.yml skipped


def test_load_workflow_directory_nonexistent_dir(engine, tmp_path):
    loaded = engine.load_workflow_directory(str(tmp_path / "nonexistent"))
    assert loaded == {}


def test_get_workflow_defs_returns_registered(engine, configs_dir):
    engine.load_workflow_directory(configs_dir)
    defs = engine.get_workflow_defs()
    assert "wf-one" in defs
    assert "wf-two" in defs


def test_get_workflow_def_by_name(engine, configs_dir):
    engine.load_workflow_directory(configs_dir)
    wf = engine.get_workflow_def("wf-one")
    assert wf is not None
    assert wf.name == "wf-one"
    assert "j" in wf.jobs


def test_get_workflow_def_unknown_returns_none(engine):
    assert engine.get_workflow_def("nonexistent") is None


def test_register_workflow_via_yaml_string(engine):
    yaml_content = """
name: wf-dynamic
on: {workflow_dispatch: {}}
jobs:
  j:
    runs-on: local
    steps: []
"""
    wf = engine.register_workflow(yaml_content)
    assert wf.name == "wf-dynamic"
    # Verify it's in the registry
    assert engine.get_workflow_def("wf-dynamic") is not None


def test_register_workflow_with_name_override(engine):
    yaml_content = """
name: original
on: {workflow_dispatch: {}}
jobs:
  j:
    runs-on: local
    steps: []
"""
    wf = engine.register_workflow(yaml_content, name="overridden")
    assert wf.name == "overridden"
    assert engine.get_workflow_def("overridden") is not None
    assert engine.get_workflow_def("original") is None


def test_recover_interrupted_runs_marks_running_as_cancelled(tmp_path):
    storage = JsonFileStorage(data_dir=str(tmp_path))
    storage.open()
    # 3 runs: 1 running, 1 pending, 1 success
    running = WorkflowRun(status=WorkflowStatus.RUNNING, start_time=datetime.now())
    pending = WorkflowRun(status=WorkflowStatus.PENDING)
    success = WorkflowRun(status=WorkflowStatus.SUCCESS)
    storage.save_run(running)
    storage.save_run(pending)
    storage.save_run(success)
    storage.close()

    # Create new engine pointing to same storage
    engine = ReviewEngine(storage=JsonFileStorage(data_dir=str(tmp_path)))
    recovered = engine.recover_interrupted_runs()
    assert recovered == 2  # running + pending

    # Verify states
    r = engine.storage.get_run(running.id)
    assert r.status == WorkflowStatus.CANCELLED
    p = engine.storage.get_run(pending.id)
    assert p.status == WorkflowStatus.CANCELLED
    s = engine.storage.get_run(success.id)
    assert s.status == WorkflowStatus.SUCCESS  # unchanged


def test_recover_interrupted_runs_empty_storage(engine):
    """No runs in empty storage"""
    recovered = engine.recover_interrupted_runs()
    assert recovered == 0


def test_recover_interrupted_runs_all_terminal(engine, tmp_path):
    """All runs already terminal — nothing to recover"""
    storage = JsonFileStorage(data_dir=str(tmp_path))
    storage.open()
    success = WorkflowRun(status=WorkflowStatus.SUCCESS)
    failure = WorkflowRun(status=WorkflowStatus.FAILURE)
    storage.save_run(success)
    storage.save_run(failure)
    storage.close()

    engine = ReviewEngine(storage=JsonFileStorage(data_dir=str(tmp_path)))
    recovered = engine.recover_interrupted_runs()
    assert recovered == 0


def test_dispatch_workflow_creates_pending_run_without_executing(engine, configs_dir):
    """dispatch_workflow must create PENDING run in storage but NOT execute."""
    engine.load_workflow_directory(configs_dir)

    run = engine.dispatch_workflow("wf-one", inputs={"msg": "hi"})
    assert run.status == WorkflowStatus.PENDING
    assert run.trigger_payload == {"inputs": {"msg": "hi"}}
    assert run.env["__workflow_name__"] == "wf-one"
    # Verify saved to storage
    loaded = engine.storage.get_run(run.id)
    assert loaded is not None
    assert loaded.status == WorkflowStatus.PENDING


def test_dispatch_workflow_unknown_raises(engine):
    with pytest.raises(ValueError, match="workflow not registered"):
        engine.dispatch_workflow("nonexistent", inputs={})


def test_dispatch_workflow_stores_file_path(engine, configs_dir):
    """dispatch_workflow stores __workflow_file__ for later reload by execute_existing_run"""
    engine.load_workflow_directory(configs_dir)
    run = engine.dispatch_workflow("wf-one", inputs={})
    assert "__workflow_file__" in run.env
    assert run.env["__workflow_file__"].endswith("wf1.yaml")


def test_execute_existing_run_loads_and_executes(engine, configs_dir):
    """execute_existing_run loads a run from storage and runs it."""
    engine.load_workflow_directory(configs_dir)

    # Dispatch a run (PENDING, not executed)
    run = engine.dispatch_workflow("wf-one", inputs={})
    assert run.status == WorkflowStatus.PENDING

    # Execute it
    executed = engine.execute_existing_run(run.id)
    assert executed.status == WorkflowStatus.SUCCESS
    # Verify step output persisted
    assert "j" in executed.jobs
    assert executed.jobs["j"].status.value == "success"


def test_execute_existing_run_unknown_id_raises(engine):
    with pytest.raises(ValueError, match="run not found"):
        engine.execute_existing_run("nonexistent-run-id")


def test_execute_existing_run_reloads_workflow_def_from_env(engine, configs_dir, tmp_path):
    """execute_existing_run reloads workflow_def from __workflow_file__ if None"""
    engine.load_workflow_directory(configs_dir)
    # Create a run, then null out workflow_def (simulate reload from storage)
    run = engine.dispatch_workflow("wf-one", inputs={})
    run.workflow_def = None
    engine.storage.save_run(run)

    # Create a NEW engine (simulates fresh process)
    engine2 = ReviewEngine(storage=engine.storage)
    engine2.load_workflow_directory(configs_dir)
    executed = engine2.execute_existing_run(run.id)
    assert executed.status == WorkflowStatus.SUCCESS
