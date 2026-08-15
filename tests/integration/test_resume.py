import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.json_file import JsonFileStorage
from paper_review_workflow.llm.schemas import SynthesisResult
from paper_review_workflow.llm.base import LLMResponse
from paper_review_workflow.llm.client import LLMClient
from paper_review_workflow.actions.registry import ActionRegistry
from paper_review_workflow.actions.synthesize import SynthesizeAction


@pytest.fixture
def fake_paper():
    return str(Path("tests/fixtures/sample_paper.pdf"))


def test_resume_skips_completed_and_reruns_failed(fake_paper, tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    LLMClient.reset()

    fake_score_obj = MagicMock()
    fake_score_obj.score = 7
    fake_score_obj.confidence = 0.8
    fake_score_obj.strengths = ["a"]
    fake_score_obj.weaknesses = ["b"]
    fake_score_obj.justification = "x" * 200
    fake_score_obj.evidence = []

    fake_synth = SynthesisResult(
        summary="x" * 250, key_strengths=["s"], key_weaknesses=["w"],
        questions_for_authors=["q"], overall_assessment="ok",
    )

    fake_dim_response = MagicMock(spec=LLMResponse)
    fake_dim_response.structured = fake_score_obj
    fake_dim_response.usage = {"input_tokens": 100, "output_tokens": 50,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 30000}
    fake_dim_response.model = "test-model"

    fake_synth_response = MagicMock(spec=LLMResponse)
    fake_synth_response.structured = fake_synth
    fake_synth_response.usage = {"input_tokens": 500, "output_tokens": 200,
                                  "cache_creation_input_tokens": 0,
                                  "cache_read_input_tokens": 0}
    fake_synth_response.model = "test-model"

    session_dir = tmp_path / "session"

    # Use a plain (non-f-string) template with sentinel placeholders, then
    # .replace() them.  This avoids the f-string brace-escaping trap where
    # ${{ expr }} would need ${{{{ expr }}}} and is error-prone.  The existing
    # test_full_review_mocked.py uses the same approach for the same reason.
    yaml_content = """\
name: test-resume
on: {workflow_dispatch: {}}
env:
  LLM_PROVIDER: anthropic
  LLM_MODEL: test-model
  SESSIONS_ROOT: __SESSIONS_ROOT__
  VENUE: neurips
jobs:
  extract:
    runs-on: local
    outputs:
      paper_id: ${{ steps.extract.outputs.paper_id }}
      full_text_path: ${{ steps.extract.outputs.full_text_path }}
      metadata_path: ${{ steps.extract.outputs.metadata_path }}
    steps:
      - id: extract
        uses: paper-review/extract@v1
        with:
          source: __FAKE_PAPER__
          session_dir: __SESSION_DIR__
  dimensions:
    needs: extract
    strategy:
      matrix:
        dimension: [soundness, presentation]
      max-parallel: 2
    runs-on: local
    steps:
      - uses: paper-review/dim_score@v1
        with:
          dimension: ${{ matrix.dimension }}
          session_dir: __SESSION_DIR__
          full_text_path: ${{ needs.extract.outputs.full_text_path }}
          metadata_path: ${{ needs.extract.outputs.metadata_path }}
  synthesize:
    needs: dimensions
    runs-on: local
    steps:
      - id: synthesize
        uses: paper-review/synthesize@v1
        with:
          session_dir: __SESSION_DIR__
  decide:
    needs: synthesize
    runs-on: local
    steps:
      - uses: paper-review/decide@v1
        with:
          session_dir: __SESSION_DIR__
          scores_path: ${{ needs.synthesize.outputs.scores_path }}
"""
    yaml_content = yaml_content.replace("__FAKE_PAPER__", fake_paper)
    yaml_content = yaml_content.replace("__SESSION_DIR__", str(session_dir))
    yaml_content = yaml_content.replace("__SESSIONS_ROOT__", str(tmp_path))

    yaml = tmp_path / "test_resume.yaml"
    yaml.write_text(yaml_content)

    storage = JsonFileStorage(data_dir=str(tmp_path / "storage"))

    # Reset singleton so builtin actions are re-registered cleanly.
    ActionRegistry._instance = None

    # The test workflow only has 2 dimensions (soundness, presentation) instead
    # of the full 3.  SynthesizeAction.MIN_DIMENSIONS defaults to 6, which would
    # reject a 2-dimension run.  Patch it down to 2 so synthesis can proceed
    # with the reduced test fixture.
    with patch.object(SynthesizeAction, "MIN_DIMENSIONS", 2):
        # First run: let one dimension fail
        call_count = {"dim": 0}

        def first_run_side_effect(**kwargs):
            schema = kwargs.get("response_schema")
            if schema is SynthesisResult:
                raise RuntimeError("should not reach synthesize on failed run")
            call_count["dim"] += 1
            # Fail the second dim call (presentation in matrix expansion order)
            if call_count["dim"] == 2:
                raise RuntimeError("simulated API timeout")
            return fake_dim_response

        with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
            mock_client = MagicMock()
            mock_client.model = "test-model"
            mock_client.complete.side_effect = first_run_side_effect
            mock_from_env.return_value = mock_client

            engine = ReviewEngine(storage=storage)
            run1 = engine.run_from_file(str(yaml), payload={})
            assert run1.status.value == "failure"

        # Second run: resume, fix mock
        with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
            mock_client = MagicMock()
            mock_client.model = "test-model"
            def second_run_side_effect(**kwargs):
                schema = kwargs.get("response_schema")
                if schema is SynthesisResult:
                    return fake_synth_response
                return fake_dim_response
            mock_client.complete.side_effect = second_run_side_effect
            mock_from_env.return_value = mock_client

            engine2 = ReviewEngine(storage=storage)
            run2 = engine2.resume_run(run1.id)

    assert run2.status.value == "success"
    # Verify decision.json exists
    assert (session_dir / "60_decision" / "decision.json").exists()
