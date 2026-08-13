import pytest
from pathlib import Path
from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.memory import MemoryStorage


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
