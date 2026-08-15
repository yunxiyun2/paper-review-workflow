import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.memory import MemoryStorage
from paper_review_workflow.llm.client import LLMClient
from paper_review_workflow.llm.base import LLMResponse
from paper_review_workflow.actions.registry import ActionRegistry
from paper_review_workflow.actions.synthesize import SynthesizeAction


@pytest.fixture
def mock_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    LLMClient.reset()

    # ACL score range 1-4
    fake_score = MagicMock()
    fake_score.score = 3
    fake_score.confidence = 0.85
    fake_score.strengths = ["rigorous methodology"]
    fake_score.weaknesses = ["limited baselines"]
    fake_score.justification = "x" * 250
    fake_score.evidence = [{"section": "3.2", "quote": "we show", "page": 5}]

    fake_synth = MagicMock()
    fake_synth.summary = "x" * 250
    fake_synth.key_strengths = ["strong"]
    fake_synth.key_weaknesses = ["weak"]
    fake_synth.questions_for_authors = ["why?"]
    fake_synth.overall_assessment = "good paper"

    fake_dim_resp = MagicMock(spec=LLMResponse)
    fake_dim_resp.structured = fake_score
    fake_dim_resp.usage = {"input_tokens": 100, "output_tokens": 50,
                           "cache_creation_input_tokens": 0,
                           "cache_read_input_tokens": 30000}
    fake_dim_resp.model = "test-model"

    fake_synth_resp = MagicMock(spec=LLMResponse)
    fake_synth_resp.structured = fake_synth
    fake_synth_resp.usage = {"input_tokens": 500, "output_tokens": 200,
                             "cache_creation_input_tokens": 0,
                             "cache_read_input_tokens": 0}
    fake_synth_resp.model = "test-model"

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.model = "test-model"

        def side_effect(**kw):
            schema = kw.get("response_schema")
            if hasattr(schema, "__name__") and "Synthesis" in schema.__name__:
                return fake_synth_resp
            return fake_dim_resp

        mock_client.complete.side_effect = side_effect
        mock_from_env.return_value = mock_client
        yield mock_client


def test_acl_venue_full_review(mock_llm, tmp_path):
    """End-to-end: ACL venue produces soundness/excitement/reproducibility/overall scores + decision."""
    # Reset registry singleton so builtin actions re-register cleanly.
    ActionRegistry._instance = None

    engine = ReviewEngine(storage=MemoryStorage())

    session_dir = tmp_path / "session"
    yaml_path = tmp_path / "acl_test.yaml"
    yaml_content = """\
name: acl-test
on: {workflow_dispatch: {}}
env:
  LLM_PROVIDER: anthropic
  LLM_MODEL: test-model
  VENUE: acl
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
          source: "__FAKE_PAPER__"
          session_dir: "__SESSION_DIR__"
  dimensions:
    needs: extract
    strategy:
      matrix:
        dimension: [soundness, excitement, reproducibility, overall]
      max-parallel: 4
    runs-on: local
    steps:
      - uses: paper-review/dim_score@v1
        with:
          dimension: ${{ matrix.dimension }}
          session_dir: "__SESSION_DIR__"
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
          session_dir: "__SESSION_DIR__"
  decide:
    needs: synthesize
    runs-on: local
    steps:
      - uses: paper-review/decide@v1
        with:
          session_dir: "__SESSION_DIR__"
          scores_path: ${{ needs.synthesize.outputs.scores_path }}
"""
    yaml_content = yaml_content.replace("__SESSIONS_ROOT__", str(tmp_path))
    yaml_content = yaml_content.replace("__FAKE_PAPER__", str(Path("tests/fixtures/sample_paper.pdf")))
    yaml_content = yaml_content.replace("__SESSION_DIR__", str(session_dir))
    yaml_path.write_text(yaml_content)

    with patch.object(SynthesizeAction, "MIN_DIMENSIONS", 2):
        run = engine.run_from_file(str(yaml_path), payload={})

    assert run.status.value == "success", f"run failed: {run.status.value}"

    # Verify 4 ACL dims present
    for dim in ["soundness", "excitement", "reproducibility", "overall"]:
        assert (session_dir / f"10_dim_{dim}" / "score.json").exists(), f"missing {dim}"
        score_json = json.loads((session_dir / f"10_dim_{dim}" / "score.json").read_text())
        assert score_json["venue"] == "acl"
        assert score_json["dimension"] == dim
        assert 1 <= score_json["score"] <= 4  # ACL range

    # Verify decision
    decision = json.loads((session_dir / "60_decision" / "decision.json").read_text())
    assert decision["venue"] == "acl"
    assert decision["score_range"] == [1, 4]
    assert decision["recommendation"] in [
        "strong_accept", "accept", "weak_accept", "borderline",
        "weak_reject", "reject", "strong_reject",
    ]
