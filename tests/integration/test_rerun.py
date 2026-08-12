import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.json_file import JsonFileStorage
from paper_review_workflow.llm.schemas import DimensionScore, SynthesisResult
from paper_review_workflow.llm.base import LLMResponse
from paper_review_workflow.llm.client import LLMClient
from paper_review_workflow.actions.registry import ActionRegistry


@pytest.fixture
def fake_paper():
    return str(Path("tests/fixtures/sample_paper.pdf"))


def test_rerun_dim_cascades_to_downstream(fake_paper, tmp_path, monkeypatch):
    """--rerun dimensions_novelty should cascade to synthesize + decide"""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    LLMClient.reset()

    fake_dim = DimensionScore(
        score=4, confidence=0.8, strengths=["a"], weaknesses=["b"],
        justification="x" * 200, evidence=[],
    )
    fake_synth = SynthesisResult(
        summary="x" * 250, key_strengths=["s"], key_weaknesses=["w"],
        questions_for_authors=["q"], overall_assessment="ok",
    )

    fake_dim_response = MagicMock(spec=LLMResponse)
    fake_dim_response.structured = fake_dim
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

    # Use sentinel placeholders for f-string safety.  The YAML expression
    # syntax ${{ expr }} collides with f-string brace escaping, so we write
    # a plain string with __PLACEHOLDER__ sentinels and .replace() them --
    # the same approach used by test_full_review_mocked.py and test_resume.py.
    yaml_template = """\
name: test-rerun
on: {workflow_dispatch: {}}
env:
  LLM_PROVIDER: anthropic
  LLM_MODEL: test-model
  SESSIONS_ROOT: __SESSIONS_ROOT__
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
        dimension: [novelty, soundness]
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
    yaml_content = yaml_template
    yaml_content = yaml_content.replace("__SESSIONS_ROOT__", str(tmp_path))
    yaml_content = yaml_content.replace("__SESSION_DIR__", str(session_dir))
    yaml_content = yaml_content.replace("__FAKE_PAPER__", fake_paper)

    yaml = tmp_path / "test_rerun.yaml"
    yaml.write_text(yaml_content)

    storage = JsonFileStorage(data_dir=str(tmp_path / "storage"))

    # Reset singleton so builtin actions (real DimensionAction etc.) are
    # re-registered cleanly before either run.
    ActionRegistry._instance = None

    # First run succeeds.  Patch MIN_DIMENSIONS down to 2 since the test
    # workflow only has 2 dimensions (novelty, soundness) instead of 8.
    with patch("paper_review_workflow.actions.synthesize.SynthesizeAction.MIN_DIMENSIONS", 2):
        with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
            mock_client = MagicMock()
            mock_client.model = "test-model"

            def first_side_effect(**kw):
                schema = kw.get("response_schema")
                if schema is SynthesisResult:
                    return fake_synth_response
                return fake_dim_response

            mock_client.complete.side_effect = first_side_effect
            mock_from_env.return_value = mock_client

            engine = ReviewEngine(storage=storage)
            run1 = engine.run_from_file(str(yaml), payload={})
            assert run1.status.value == "success", (
                f"first run should succeed, got {run1.status.value}"
            )

        # Now rerun dimensions_novelty -- should cascade to synthesize + decide.
        # Note: because the matrix is re-expanded as a unit when the parent
        # "dimensions" job is reset, sibling dimensions_soundness is also
        # re-run -- a known limitation of matrix rerun.  The test asserts
        # the cascade reaches synthesize + decide, not that siblings are
        # preserved.
        with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
            mock_client = MagicMock()
            mock_client.model = "test-model"

            # Track which schemas get called so we can assert synthesize
            # was actually re-invoked (proving the cascade worked).
            called_schemas = []

            def rerun_side_effect(**kw):
                schema = kw.get("response_schema")
                called_schemas.append(schema)
                if schema is SynthesisResult:
                    return fake_synth_response
                return fake_dim_response

            mock_client.complete.side_effect = rerun_side_effect
            mock_from_env.return_value = mock_client

            engine2 = ReviewEngine(storage=storage)
            run2 = engine2.resume_run(
                run1.id, rerun_components=["dimensions_novelty"]
            )

    assert run2.status.value == "success", (
        f"rerun should succeed, got {run2.status.value}"
    )

    # synthesize should have been re-executed (cascaded rerun).  The
    # SynthesisResult schema should appear in called_schemas at least once.
    assert SynthesisResult in called_schemas, (
        f"synthesize should have been rerun (cascaded), got {called_schemas}"
    )

    # decision.json should exist after the rerun.
    assert (session_dir / "60_decision" / "decision.json").exists(), (
        "decision.json should exist after rerun"
    )

    # extract was not in the rerun set and has no upstream dependencies on
    # the rerun component, so it must remain SUCCESS (genuinely untouched).
    assert "extract" in run2.jobs, "extract job should exist"
    assert run2.jobs["extract"].status.value == "success", (
        f"extract should remain SUCCESS (untouched), got "
        f"{run2.jobs['extract'].status.value}"
    )

    # dimensions_soundness: the matrix re-expands as a unit when the parent
    # "dimensions" job is reset, so this sibling is also re-run.  With the
    # mock returning a valid DimensionScore, it ends up SUCCESS again.
    # We assert it is SUCCESS (not that it was preserved).
    assert "dimensions_soundness" in run2.jobs, "dimensions_soundness job should exist"
    assert run2.jobs["dimensions_soundness"].status.value == "success", (
        f"dimensions_soundness should be SUCCESS after rerun, got "
        f"{run2.jobs['dimensions_soundness'].status.value}"
    )

    # dimensions_novelty was reset and re-run; it must be SUCCESS now.
    assert "dimensions_novelty" in run2.jobs, "dimensions_novelty job should exist"
    assert run2.jobs["dimensions_novelty"].status.value == "success", (
        f"dimensions_novelty should be SUCCESS after rerun, got "
        f"{run2.jobs['dimensions_novelty'].status.value}"
    )

    # synthesize and decide were reset + re-run; they must be SUCCESS.
    assert run2.jobs["synthesize"].status.value == "success", (
        f"synthesize should be SUCCESS after cascaded rerun, got "
        f"{run2.jobs['synthesize'].status.value}"
    )
    assert run2.jobs["decide"].status.value == "success", (
        f"decide should be SUCCESS after cascaded rerun, got "
        f"{run2.jobs['decide'].status.value}"
    )
