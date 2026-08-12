import pytest
from paper_review_workflow.core.parser import WorkflowParser


@pytest.fixture
def minimal_yaml(tmp_path):
    yaml_content = """
name: minimal-test
on:
  workflow_dispatch:
    inputs:
      message:
        required: true
        default: hello
        type: string
env:
  GREETING: hi
jobs:
  echo:
    name: Echo
    runs-on: local
    steps:
      - id: echo-step
        uses: paper-review/echo@v1
        with:
          message: ${{ inputs.message }}
"""
    p = tmp_path / "minimal.yaml"
    p.write_text(yaml_content)
    return str(p)


def test_parse_minimal_yaml(minimal_yaml):
    parser = WorkflowParser()
    wf = parser.parse_file(minimal_yaml)

    assert wf.name == "minimal-test"
    assert wf.env["GREETING"] == "hi"
    assert "echo" in wf.jobs
    assert wf.jobs["echo"].runs_on == "local"
    assert len(wf.jobs["echo"].steps) == 1
    step = wf.jobs["echo"].steps[0]
    assert step.uses == "paper-review/echo@v1"
    assert step.with_params["message"] == "${{ inputs.message }}"


def test_parse_dispatch_inputs(minimal_yaml):
    parser = WorkflowParser()
    wf = parser.parse_file(minimal_yaml)
    assert wf.on.workflow_dispatch is not None
    inputs = wf.on.workflow_dispatch["inputs"]
    assert "message" in inputs
    assert inputs["message"]["default"] == "hello"


def test_parse_needs_list():
    yaml = """
name: t
on: {workflow_dispatch: {}}
jobs:
  a:
    runs-on: local
    steps: [{uses: paper-review/echo@v1}]
  b:
    needs: a
    runs-on: local
    steps: []
  c:
    needs: [a, b]
    runs-on: local
    steps: []
"""
    parser = WorkflowParser()
    wf = parser.parse_string(yaml)
    assert wf.jobs["b"].needs == ["a"]
    assert wf.jobs["c"].needs == ["a", "b"]


def test_parse_matrix_strategy():
    yaml = """
name: t
on: {workflow_dispatch: {}}
jobs:
  dims:
    runs-on: local
    strategy:
      matrix:
        dimension: [novelty, soundness]
    steps: []
"""
    parser = WorkflowParser()
    wf = parser.parse_string(yaml)
    assert wf.jobs["dims"].strategy["matrix"]["dimension"] == ["novelty", "soundness"]


def test_parse_job_env_coerces_to_string():
    yaml = """
name: t
on: {workflow_dispatch: {}}
jobs:
  j:
    runs-on: local
    env:
      PORT: 8080
      DEBUG: true
    steps: []
"""
    parser = WorkflowParser()
    wf = parser.parse_string(yaml)
    assert wf.jobs["j"].env["PORT"] == "8080"
    assert wf.jobs["j"].env["DEBUG"] == "True"


def test_parse_directory_collects_yml_and_yaml(tmp_path):
    """parse_directory should pick up both .yml and .yaml files by stem."""
    (tmp_path / "a.yml").write_text(
        "name: a\non: {workflow_dispatch: {}}\njobs:\n  j:\n    runs-on: local\n    steps: []\n"
    )
    (tmp_path / "b.yaml").write_text(
        "name: b\non: {workflow_dispatch: {}}\njobs:\n  j:\n    runs-on: local\n    steps: []\n"
    )
    parser = WorkflowParser()
    result = parser.parse_directory(str(tmp_path))
    assert "a" in result
    assert "b" in result
    assert result["a"].name == "a"
    assert result["b"].name == "b"


def test_parse_directory_skips_invalid_files(tmp_path, capsys):
    """parse_directory should print and skip files that fail to parse."""
    (tmp_path / "good.yml").write_text(
        "name: good\non: {workflow_dispatch: {}}\njobs:\n  j:\n    runs-on: local\n    steps: []\n"
    )
    (tmp_path / "bad.yml").write_text("::: not valid yaml :::")
    (tmp_path / "bad2.yaml").write_text("::: also not valid :::")
    parser = WorkflowParser()
    result = parser.parse_directory(str(tmp_path))
    assert "good" in result
    assert "bad" not in result
    assert "bad2" not in result
    captured = capsys.readouterr()
    assert "[Parser]" in captured.out


def test_parse_trigger_no_workflow_dispatch_key():
    """When `on` is a dict but lacks workflow_dispatch, default to {}."""
    yaml = "name: t\non: {}\njobs: {}\n"
    parser = WorkflowParser()
    wf = parser.parse_string(yaml)
    assert wf.on.workflow_dispatch == {}


def test_parse_trigger_non_dict_on():
    """When `on` is a string (non-dict), workflow_dispatch defaults to {}."""
    yaml = "name: t\non: push\njobs: {}\n"
    parser = WorkflowParser()
    wf = parser.parse_string(yaml)
    assert wf.on.workflow_dispatch == {}


def test_parse_needs_unexpected_type_returns_empty():
    """When needs is an unexpected type (e.g. int), return empty list."""
    yaml = """
name: t
on: {workflow_dispatch: {}}
jobs:
  j:
    needs: 123
    runs-on: local
    steps: []
"""
    parser = WorkflowParser()
    wf = parser.parse_string(yaml)
    assert wf.jobs["j"].needs == []


def test_parse_uses_yaml_bool_true_key_for_on():
    """YAML 1.1 parses bare `on:` to bool True; parser should still resolve it."""
    yaml = """
on:
  workflow_dispatch: {}
name: bool-key
jobs: {}
"""
    parser = WorkflowParser()
    wf = parser.parse_string(yaml)
    assert wf.name == "bool-key"
    assert wf.on.workflow_dispatch == {}
