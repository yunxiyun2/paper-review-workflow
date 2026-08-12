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
