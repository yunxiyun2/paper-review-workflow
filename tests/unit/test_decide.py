import json
import pytest
from pathlib import Path

from paper_review_workflow.actions.decide import DecideAction


@pytest.fixture
def session_with_scores(tmp_path):
    session_dir = tmp_path / "session"
    syn_dir = session_dir / "50_synthesize"
    syn_dir.mkdir(parents=True)
    scores = {
        dim: {"score": 4, "confidence": 0.8} for dim in [
            "novelty", "soundness", "significance", "clarity",
            "reproducibility", "related_work", "positioning", "presentation",
        ]
    }
    (syn_dir / "scores.json").write_text(json.dumps(scores))
    return session_dir


def test_decide_all_5_strong_accept(session_with_scores):
    # Modify to all 5s
    syn_dir = session_with_scores / "50_synthesize"
    scores = {dim: {"score": 5, "confidence": 1.0} for dim in [
        "novelty", "soundness", "significance", "clarity",
        "reproducibility", "related_work", "positioning", "presentation",
    ]}
    (syn_dir / "scores.json").write_text(json.dumps(scores))

    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_scores),
                "scores_path": str(syn_dir / "scores.json")},
        env={}, context={}, log_callback=lambda x: None,
    )

    assert result.success
    decision = json.loads((session_with_scores / "60_decision" / "decision.json").read_text())
    assert decision["recommendation"] == "strong_accept"
    assert decision["weighted_score"] == 5.0


def test_decide_all_1_strong_reject(session_with_scores):
    syn_dir = session_with_scores / "50_synthesize"
    scores = {dim: {"score": 1, "confidence": 1.0} for dim in [
        "novelty", "soundness", "significance", "clarity",
        "reproducibility", "related_work", "positioning", "presentation",
    ]}
    (syn_dir / "scores.json").write_text(json.dumps(scores))

    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_scores),
                "scores_path": str(syn_dir / "scores.json")},
        env={}, context={}, log_callback=lambda x: None,
    )

    decision = json.loads((session_with_scores / "60_decision" / "decision.json").read_text())
    assert decision["recommendation"] == "strong_reject"
    assert decision["weighted_score"] == 1.0


def test_decide_weighted_average_uses_weights(session_with_scores):
    """soundness (1.2) should weight high vs presentation (0.8)"""
    syn_dir = session_with_scores / "50_synthesize"
    scores = {
        "novelty": {"score": 5}, "soundness": {"score": 1},  # high-weight low score
        "significance": {"score": 5}, "clarity": {"score": 5},
        "reproducibility": {"score": 5}, "related_work": {"score": 5},
        "positioning": {"score": 5}, "presentation": {"score": 5},
    }
    (syn_dir / "scores.json").write_text(json.dumps(scores))

    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_scores),
                "scores_path": str(syn_dir / "scores.json")},
        env={}, context={}, log_callback=lambda x: None,
    )

    decision = json.loads((session_with_scores / "60_decision" / "decision.json").read_text())
    # Weighted avg should be lower than simple avg (4.625) due to soundness weight
    assert decision["weighted_score"] < 4.625
    assert decision["weighted_score"] > 4.0


def test_decide_missing_dim_treated_as_na(session_with_scores):
    syn_dir = session_with_scores / "50_synthesize"
    scores = {
        "novelty": {"score": 4}, "soundness": {"score": 4},
        "significance": {"score": 4}, "clarity": {"score": 4},
        "reproducibility": {"score": 4}, "related_work": {"score": 4},
        "positioning": {"score": 4},
        # presentation missing
    }
    (syn_dir / "scores.json").write_text(json.dumps(scores))

    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_scores),
                "scores_path": str(syn_dir / "scores.json")},
        env={}, context={}, log_callback=lambda x: None,
    )

    assert result.success
    decision = json.loads((session_with_scores / "60_decision" / "decision.json").read_text())
    assert decision["weighted_score"] == 4.0  # remaining 7 all 4


def test_decide_description_property():
    """DecideAction.description returns a human-readable string."""
    action = DecideAction()
    assert isinstance(action.description, str)
    assert "recommendation" in action.description.lower()


def test_decide_missing_scores_file_returns_failure(tmp_path):
    """When scores_path does not exist, run should return failure."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_dir),
                "scores_path": str(tmp_path / "missing.json")},
        env={}, context={}, log_callback=lambda x: None,
    )
    assert not result.success
    assert "cannot read scores" in result.message


def test_decide_invalid_scores_json_returns_failure(tmp_path):
    """When scores.json contains invalid JSON, run should return failure."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("::: not json :::")
    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_dir),
                "scores_path": str(bad_json)},
        env={}, context={}, log_callback=lambda x: None,
    )
    assert not result.success
    assert "cannot read scores" in result.message


def test_decide_score_is_none_skips_dimension(session_with_scores):
    """When score is None for a dimension, that dimension should be skipped."""
    syn_dir = session_with_scores / "50_synthesize"
    scores = {
        "novelty": {"score": None},  # None score skipped
        "soundness": {"score": 4},
        "significance": {"score": 4},
        "clarity": {"score": 4},
        "reproducibility": {"score": 4},
        "related_work": {"score": 4},
        "positioning": {"score": 4},
        "presentation": {"score": 4},
    }
    (syn_dir / "scores.json").write_text(json.dumps(scores))
    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_scores),
                "scores_path": str(syn_dir / "scores.json")},
        env={}, context={}, log_callback=lambda x: None,
    )
    assert result.success
    decision = json.loads((session_with_scores / "60_decision" / "decision.json").read_text())
    # 7 dimensions scored 4 -- weighted score 4.0
    assert decision["weighted_score"] == 4.0


def test_decide_empty_scores_returns_failure(tmp_path):
    """Empty scores dict should return failure with 'no dimensions to score'."""
    session_dir = tmp_path / "session"
    syn_dir = session_dir / "50_synthesize"
    syn_dir.mkdir(parents=True)
    (syn_dir / "scores.json").write_text("{}")
    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_dir)},
        env={}, context={}, log_callback=lambda x: None,
    )
    assert not result.success
    assert "no dimensions to score" in result.message


def test_decide_invalid_weight_env_value_ignored(session_with_scores):
    """Invalid WEIGHT_ env values should be silently ignored."""
    syn_dir = session_with_scores / "50_synthesize"
    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_scores),
                "scores_path": str(syn_dir / "scores.json")},
        env={"WEIGHT_NOVELTY": "not-a-number"}, context={}, log_callback=lambda x: None,
    )
    assert result.success
    # Default weight for novelty (1.0) should be used
    decision = json.loads((session_with_scores / "60_decision" / "decision.json").read_text())
    assert decision["weights_used"]["novelty"] == 1.0


def test_decide_map_to_recommendation_fallback():
    """Negative scores should fall through thresholds to strong_reject."""
    action = DecideAction()
    assert action._map_to_recommendation(-1.0) == "strong_reject"


def test_decide_extract_key_concerns_with_weaknesses(session_with_scores):
    """Concerns should be extracted from low-score dimensions with weaknesses."""
    syn_dir = session_with_scores / "50_synthesize"
    scores = {
        "novelty": {"score": 2, "weaknesses": ["unclear contribution"]},
        "soundness": {"score": 4},
        "significance": {"score": 4},
        "clarity": {"score": 4},
        "reproducibility": {"score": 4},
        "related_work": {"score": 4},
        "positioning": {"score": 4},
        "presentation": {"score": 4},
    }
    (syn_dir / "scores.json").write_text(json.dumps(scores))
    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_scores),
                "scores_path": str(syn_dir / "scores.json")},
        env={}, context={}, log_callback=lambda x: None,
    )
    assert result.success
    decision = json.loads((session_with_scores / "60_decision" / "decision.json").read_text())
    assert any("novelty" in c for c in decision["key_concerns"])


def test_decide_uses_default_scores_path_when_not_provided(session_with_scores):
    """When scores_path is not provided, default to session_dir/50_synthesize/scores.json."""
    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_scores)},
        env={}, context={}, log_callback=lambda x: None,
    )
    assert result.success


def test_decide_final_report_copied_when_review_exists(session_with_scores):
    """final_report.md should be created when 50_synthesize/review.md exists."""
    syn_dir = session_with_scores / "50_synthesize"
    (syn_dir / "review.md").write_text("# Review\n\nThis is a review.")
    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_scores)},
        env={}, context={}, log_callback=lambda x: None,
    )
    assert result.success
    final_report = session_with_scores / "final_report.md"
    assert final_report.exists()
    assert "Review" in final_report.read_text()
