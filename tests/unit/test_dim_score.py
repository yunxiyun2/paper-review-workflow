import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_review_workflow.actions.dimensions import DimensionAction
from paper_review_workflow.llm.schemas import DimensionScore
from paper_review_workflow.llm.base import LLMResponse


@pytest.fixture
def fake_full_text(tmp_path):
    p = tmp_path / "full_text.md"
    p.write_text("# Sample Paper\n\nThis is the paper content. " * 50)
    return str(p)


@pytest.fixture
def fake_metadata(tmp_path):
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps({
        "title": "Sample Paper",
        "authors": ["A"],
        "abstract": "...",
        "doi": None, "arxiv_id": None, "keywords": [],
    }))
    return str(p)


def _make_llm_response(score: DimensionScore) -> LLMResponse:
    """Build a fake LLMResponse with usage info."""
    return LLMResponse(
        text=None,
        structured=score,
        usage={
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_creation_input_tokens": 800,
            "cache_read_input_tokens": 200,
        },
        model="test-model",
    )


def test_dimension_action_scores_novelty(tmp_path, fake_full_text, fake_metadata):
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    action = DimensionAction()

    fake_score = DimensionScore(
        score=4, confidence=0.85,
        strengths=["new method X"], weaknesses=["unclear scope"],
        justification="x" * 250,
        evidence=[{"section": "3.2", "quote": "we propose", "page": 5}],
    )

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.complete.return_value = _make_llm_response(fake_score)
        mock_client.model = "test-model"
        mock_from_env.return_value = mock_client

        result = action.run(
            params={
                "dimension": "novelty",
                "session_dir": str(session_dir),
                "full_text_path": fake_full_text,
                "metadata_path": fake_metadata,
            },
            env={}, context={}, log_callback=lambda x: None,
        )

    assert result.success
    assert result.outputs["score"] == 4
    assert result.outputs["confidence"] == 0.85

    out_dir = session_dir / "10_dim_novelty"
    score_json = json.loads((out_dir / "score.json").read_text())
    assert score_json["dimension"] == "novelty"
    assert score_json["score"] == 4
    assert "cache_read_input_tokens" in score_json["usage"]

    review_md = (out_dir / "review.md").read_text()
    assert "novelty" in review_md.lower()
    assert "score: 4" in review_md.lower() or "score: 4/5" in review_md.lower()


def test_dimension_action_passes_paper_as_cached_context(tmp_path, fake_full_text, fake_metadata):
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    action = DimensionAction()
    fake_score = DimensionScore(
        score=3, confidence=0.7,
        strengths=["a"], weaknesses=["b"],
        justification="y" * 200,
    )

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.complete.return_value = _make_llm_response(fake_score)
        mock_from_env.return_value = mock_client

        action.run(
            params={
                "dimension": "soundness",
                "session_dir": str(session_dir),
                "full_text_path": fake_full_text,
                "metadata_path": fake_metadata,
            },
            env={}, context={}, log_callback=lambda x: None,
        )

    call_kwargs = mock_client.complete.call_args.kwargs
    # cached_context should contain the paper text
    assert "Sample Paper" in call_kwargs["cached_context"]
    # system prompt should mention "Soundness"
    assert "Soundness" in call_kwargs["system"] or "soundness" in call_kwargs["system"].lower()


def test_dimension_action_unknown_dimension(tmp_path, fake_full_text, fake_metadata):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    action = DimensionAction()
    result = action.run(
        params={
            "dimension": "unknown_dim",
            "session_dir": str(session_dir),
            "full_text_path": fake_full_text,
            "metadata_path": fake_metadata,
        },
        env={}, context={}, log_callback=lambda x: None,
    )
    assert not result.success
    assert "unknown dimension" in result.message.lower() or "no prompt" in result.message.lower()
