import json
from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.memory import MemoryStorage


def test_engine_runs_minimal_yaml(tmp_path):
    yaml_path = tmp_path / "minimal.yaml"
    yaml_path.write_text("""
name: minimal
on: {workflow_dispatch: {}}
jobs:
  echo:
    runs-on: local
    steps:
      - id: e
        uses: paper-review/echo@v1
        with:
          message: hi
""")
    engine = ReviewEngine(storage=MemoryStorage())
    run = engine.run_from_file(str(yaml_path), payload={})
    assert run.status.value == "success"
    assert run.jobs["echo"].steps[0].outputs["message"] == "hi"
