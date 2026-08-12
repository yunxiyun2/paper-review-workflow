import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.memory import MemoryStorage
from paper_review_workflow.llm.schemas import DimensionScore, SynthesisResult
from paper_review_workflow.llm.base import LLMResponse
from paper_review_workflow.llm.client import LLMClient
from paper_review_workflow.actions.registry import ActionRegistry


@pytest.fixture
def fake_paper():
    return str(Path("tests/fixtures/sample_paper.pdf"))


def test_full_review_workflow_mocked(fake_paper, tmp_path, monkeypatch):
    """End-to-end: extract -> 8 dims -> synthesize -> decide. LLM mocked."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    LLMClient.reset()

    fake_dim_score = DimensionScore(
        score=4, confidence=0.8,
        strengths=["a strength"], weaknesses=["a weakness"],
        justification="x" * 250,
        evidence=[],
    )
    fake_synth = SynthesisResult(
        summary="x" * 250,
        key_strengths=["strength"],
        key_weaknesses=["weakness"],
        questions_for_authors=["question?"],
        overall_assessment="good paper overall",
    )

    fake_dim_response = MagicMock(spec=LLMResponse)
    fake_dim_response.structured = fake_dim_score
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

    # Reset singleton so builtin actions (real DimensionAction etc.) are
    # re-registered -- previous tests may have overridden dim_score@v1.
    ActionRegistry._instance = None

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.model = "test-model"

        def complete_side_effect(**kwargs):
            schema = kwargs.get("response_schema")
            if schema is SynthesisResult:
                return fake_synth_response
            return fake_dim_response

        mock_client.complete.side_effect = complete_side_effect
        mock_from_env.return_value = mock_client

        # Build YAML with placeholders to avoid f-string brace escaping.
        # ${{ }} is the expression syntax; inside an f-string each { becomes {{
        # and each } becomes }}, so ${ expr } -> ${{{{ expr }}}} -- error-prone.
        # Using .replace() on a plain string sidesteps this entirely.
        sessions_root = tmp_path / "sessions"
        yaml_content = """\
name: test-review
on: {workflow_dispatch: {}}
env:
  LLM_PROVIDER: anthropic
  LLM_MODEL: test-model
  LLM_MAX_TOKENS: "4096"
  LLM_TEMPERATURE: "0.0"
  SESSIONS_ROOT: __SESSIONS_ROOT__
  WEIGHT_NOVELTY: "1.0"
  WEIGHT_SOUNDESS: "1.2"
  WEIGHT_SIGNIFICANCE: "1.2"
  WEIGHT_CLARITY: "0.8"
  WEIGHT_REPRODUCIBILITY: "1.0"
  WEIGHT_RELATED_WORK: "0.8"
  WEIGHT_POSITIONING: "0.8"
  WEIGHT_PRESENTATION: "0.8"
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
        dimension: [novelty, soundness, significance, clarity,
                    reproducibility, related_work, positioning, presentation]
      max-parallel: 8
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
    outputs:
      review_path: ${{ steps.synthesize.outputs.review_path }}
      scores_path: ${{ steps.synthesize.outputs.scores_path }}
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
        yaml_content = yaml_content.replace("__SESSIONS_ROOT__", str(sessions_root))

        yaml = tmp_path / "test_review.yaml"
        yaml.write_text(yaml_content)

        engine = ReviewEngine(storage=MemoryStorage())
        run = engine.run_from_file(str(yaml), payload={})

    assert run.status.value == "success"

    assert (session_dir / "00_extract" / "full_text.md").exists()
    for dim in ["novelty", "soundness", "significance", "clarity",
                "reproducibility", "related_work", "positioning", "presentation"]:
        assert (session_dir / f"10_dim_{dim}" / "score.json").exists(), f"missing {dim}"
    assert (session_dir / "50_synthesize" / "review.md").exists()
    assert (session_dir / "60_decision" / "decision.json").exists()

    decision = json.loads((session_dir / "60_decision" / "decision.json").read_text())
    assert decision["recommendation"] in [
        "strong_accept", "accept", "weak_accept", "borderline",
        "weak_reject", "reject", "strong_reject",
    ]
    assert 1.0 <= decision["weighted_score"] <= 5.0
