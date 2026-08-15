import json
import pytest
from pathlib import Path

from paper_review_workflow.actions.decide import DecideAction


@pytest.fixture
def session_with_neurips_scores(tmp_path):
    """3 dims (soundness/presentation/contribution), all score 9 (NeurIPS 1-10)"""
    session_dir = tmp_path / "session"
    syn_dir = session_dir / "50_synthesize"
    syn_dir.mkdir(parents=True)
    scores = {
        "soundness": {"score": 9, "confidence": 0.9},
        "presentation": {"score": 9, "confidence": 0.8},
        "contribution": {"score": 9, "confidence": 0.85},
    }
    (syn_dir / "scores.json").write_text(json.dumps(scores))
    return session_dir


def test_decide_neurips_all_9_strong_accept(session_with_neurips_scores):
    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_neurips_scores)},
        env={"VENUE": "neurips"}, context={}, log_callback=lambda x: None,
    )

    assert result.success
    decision = json.loads((session_with_neurips_scores / "60_decision" / "decision.json").read_text())
    assert decision["venue"] == "neurips"
    assert decision["recommendation"] == "strong_accept"
    assert decision["weighted_score"] == 9.0
    assert decision["score_range"] == [1, 10]


def test_decide_neurips_all_1_strong_reject(session_with_neurips_scores):
    syn_dir = session_with_neurips_scores / "50_synthesize"
    scores = {dim: {"score": 1, "confidence": 1.0}
              for dim in ["soundness", "presentation", "contribution"]}
    (syn_dir / "scores.json").write_text(json.dumps(scores))

    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_neurips_scores)},
        env={"VENUE": "neurips"}, context={}, log_callback=lambda x: None,
    )

    decision = json.loads((session_with_neurips_scores / "60_decision" / "decision.json").read_text())
    assert decision["recommendation"] == "strong_reject"
    assert decision["weighted_score"] == 1.0


def test_decide_neurips_weighted_average_uses_weights(session_with_neurips_scores):
    """contribution (1.4) should weight higher than presentation (0.8)"""
    syn_dir = session_with_neurips_scores / "50_synthesize"
    scores = {
        "soundness": {"score": 5},
        "presentation": {"score": 1},  # low-weight low score
        "contribution": {"score": 9},  # high-weight high score
    }
    (syn_dir / "scores.json").write_text(json.dumps(scores))

    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_neurips_scores)},
        env={"VENUE": "neurips"}, context={}, log_callback=lambda x: None,
    )

    decision = json.loads((session_with_neurips_scores / "60_decision" / "decision.json").read_text())
    # Weighted avg: (5*1.3 + 1*0.8 + 9*1.4) / (1.3+0.8+1.4) = (6.5+0.8+12.6)/3.5 = 19.9/3.5 ≈ 5.69
    assert 5.5 < decision["weighted_score"] < 6.0


def test_decide_icml_uses_icml_thresholds(tmp_path):
    """ICML 1-4 scale uses ICML thresholds (3.75 → strong_accept)"""
    session_dir = tmp_path / "session"
    syn_dir = session_dir / "50_synthesize"
    syn_dir.mkdir(parents=True)
    scores = {
        "soundness": {"score": 4},
        "significance": {"score": 4},
        "originality": {"score": 4},
        "clarity": {"score": 4},
    }
    (syn_dir / "scores.json").write_text(json.dumps(scores))

    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_dir)},
        env={"VENUE": "icml"}, context={}, log_callback=lambda x: None,
    )

    decision = json.loads((session_dir / "60_decision" / "decision.json").read_text())
    assert decision["venue"] == "icml"
    assert decision["recommendation"] == "strong_accept"
    assert decision["score_range"] == [1, 4]
    assert decision["weighted_score"] == 4.0


def test_decide_acl_uses_acl_thresholds(tmp_path):
    """ACL 1-4 scale with ACL dimensions"""
    session_dir = tmp_path / "session"
    syn_dir = session_dir / "50_synthesize"
    syn_dir.mkdir(parents=True)
    scores = {
        "soundness": {"score": 1},
        "excitement": {"score": 1},
        "reproducibility": {"score": 1},
        "overall": {"score": 1},
    }
    (syn_dir / "scores.json").write_text(json.dumps(scores))

    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_dir)},
        env={"VENUE": "acl"}, context={}, log_callback=lambda x: None,
    )

    decision = json.loads((session_dir / "60_decision" / "decision.json").read_text())
    assert decision["venue"] == "acl"
    assert decision["recommendation"] == "strong_reject"


def test_decide_missing_dim_skipped(session_with_neurips_scores):
    """Missing dim is skipped, not zeroed"""
    syn_dir = session_with_neurips_scores / "50_synthesize"
    scores = {
        "soundness": {"score": 9},
        "presentation": {"score": 9},
        # contribution missing
    }
    (syn_dir / "scores.json").write_text(json.dumps(scores))

    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_neurips_scores)},
        env={"VENUE": "neurips"}, context={}, log_callback=lambda x: None,
    )

    assert result.success
    decision = json.loads((session_with_neurips_scores / "60_decision" / "decision.json").read_text())
    assert decision["weighted_score"] == 9.0  # remaining 2 dims all 9


def test_decide_env_weight_override(session_with_neurips_scores):
    """WEIGHT_SOUNDNESS env var overrides venue_config weight"""
    action = DecideAction()
    result = action.run(
        params={"session_dir": str(session_with_neurips_scores)},
        env={"VENUE": "neurips", "WEIGHT_SOUNDNESS": "5.0"},
        context={}, log_callback=lambda x: None,
    )
    assert result.success
    decision = json.loads((session_with_neurips_scores / "60_decision" / "decision.json").read_text())
    # soundness weighted at 5.0 instead of 1.3
    assert decision["weights_used"]["soundness"] == 5.0
