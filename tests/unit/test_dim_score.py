import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_review_workflow.actions.dimensions import DimensionAction


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


@pytest.fixture
def neurips_env():
    return {"VENUE": "neurips"}


def test_dimension_action_scores_soundness_neurips(tmp_path, fake_full_text, fake_metadata, neurips_env):
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    action = DimensionAction()

    # NeurIPS soundness schema is 1-10
    fake_score_data = {
        "score": 7,
        "confidence": 0.85,
        "strengths": ["rigorous proof"],
        "weaknesses": ["unclear scope"],
        "justification": "x" * 250,
        "evidence": [{"section": "3.2", "quote": "we prove", "page": 5}],
    }

    mock_response = MagicMock()
    mock_response.structured = MagicMock(**fake_score_data)
    mock_response.usage = {"input_tokens": 100, "output_tokens": 50,
                           "cache_creation_input_tokens": 0,
                           "cache_read_input_tokens": 30000}
    mock_response.model = "test-model"

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.complete.return_value = mock_response
        mock_client.model = "test-model"
        mock_from_env.return_value = mock_client

        result = action.run(
            params={
                "dimension": "soundness",
                "session_dir": str(session_dir),
                "full_text_path": fake_full_text,
                "metadata_path": fake_metadata,
            },
            env=neurips_env, context={}, log_callback=lambda x: None,
        )

    assert result.success
    assert result.outputs["score"] == 7
    assert result.outputs["confidence"] == 0.85

    out_dir = session_dir / "10_dim_soundness"
    score_json = json.loads((out_dir / "score.json").read_text())
    assert score_json["venue"] == "neurips"
    assert score_json["dimension"] == "soundness"
    assert score_json["score"] == 7

    review_md = (out_dir / "review.md").read_text()
    assert "soundness" in review_md.lower()
    assert "Score: 7/10" in review_md  # NeurIPS max is 10


def test_dimension_action_passes_paper_as_cached_context(tmp_path, fake_full_text, fake_metadata, neurips_env):
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    action = DimensionAction()
    fake_score_data = {
        "score": 5,
        "confidence": 0.7,
        "strengths": ["a"],
        "weaknesses": ["b"],
        "justification": "y" * 200,
        "evidence": [],
    }

    mock_response = MagicMock()
    mock_response.structured = MagicMock(**fake_score_data)
    mock_response.usage = {}
    mock_response.model = "test-model"

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.complete.return_value = mock_response
        mock_from_env.return_value = mock_client

        action.run(
            params={
                "dimension": "presentation",
                "session_dir": str(session_dir),
                "full_text_path": fake_full_text,
                "metadata_path": fake_metadata,
            },
            env=neurips_env, context={}, log_callback=lambda x: None,
        )

    call_kwargs = mock_client.complete.call_args.kwargs
    assert "Sample Paper" in call_kwargs["cached_context"]


def test_dimension_action_unknown_dimension_for_neurips(tmp_path, fake_full_text, fake_metadata, neurips_env):
    """'novelty' is not a NeurIPS dimension"""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    action = DimensionAction()
    result = action.run(
        params={
            "dimension": "novelty",  # Not in NeurIPS dimensions
            "session_dir": str(session_dir),
            "full_text_path": fake_full_text,
            "metadata_path": fake_metadata,
        },
        env=neurips_env, context={}, log_callback=lambda x: None,
    )
    assert not result.success
    assert "not in venue" in result.message.lower()


def test_dimension_action_unknown_venue(tmp_path, fake_full_text, fake_metadata):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    action = DimensionAction()
    result = action.run(
        params={
            "dimension": "soundness",
            "session_dir": str(session_dir),
            "full_text_path": fake_full_text,
            "metadata_path": fake_metadata,
        },
        env={"VENUE": "nonexistent"}, context={}, log_callback=lambda x: None,
    )
    assert not result.success
    assert "venue not found" in result.message.lower()


def test_dimension_action_icml_venue(tmp_path, fake_full_text, fake_metadata):
    """ICML uses 1-4 scale, 4 dimensions"""
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    action = DimensionAction()
    fake_score_data = {
        "score": 3,  # ICML max is 4
        "confidence": 0.7,
        "strengths": ["a"],
        "weaknesses": ["b"],
        "justification": "y" * 200,
        "evidence": [],
    }

    mock_response = MagicMock()
    mock_response.structured = MagicMock(**fake_score_data)
    mock_response.usage = {}
    mock_response.model = "test-model"

    with patch("paper_review_workflow.llm.client.LLMClient.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.complete.return_value = mock_response
        mock_from_env.return_value = mock_client

        result = action.run(
            params={
                "dimension": "significance",
                "session_dir": str(session_dir),
                "full_text_path": fake_full_text,
                "metadata_path": fake_metadata,
            },
            env={"VENUE": "icml"}, context={}, log_callback=lambda x: None,
        )

    assert result.success
    score_json = json.loads((session_dir / "10_dim_significance" / "score.json").read_text())
    assert score_json["venue"] == "icml"
    assert score_json["dimension"] == "significance"
