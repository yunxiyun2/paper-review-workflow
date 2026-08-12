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
