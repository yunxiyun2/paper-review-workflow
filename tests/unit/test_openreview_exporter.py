import json
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from paper_review_workflow.exporters.openreview import OpenReviewExporter
from paper_review_workflow.core.models import WorkflowRun, WorkflowStatus, JobInstance, JobStatus


def _make_run_with_decision(tmp_path, venue="neurips"):
    """Create a mock run with decision.json + review.md"""
    session_dir = tmp_path / "session"
    decision_dir = session_dir / "60_decision"
    synth_dir = session_dir / "50_synthesize"
    decision_dir.mkdir(parents=True)
    synth_dir.mkdir(parents=True)

    decision = {
        "venue": venue,
        "recommendation": "weak_accept",
        "weighted_score": 6.5,
        "score_range": [1, 10],
        "per_dimension": {
            "soundness": {"score": 7, "confidence": 0.85},
            "presentation": {"score": 5, "confidence": 0.70},
            "contribution": {"score": 8, "confidence": 0.90},
        },
        "decision_rationale": "Weighted average is 6.5",
    }
    (decision_dir / "decision.json").write_text(json.dumps(decision))

    review_md = "# Peer Review\n\nThis paper proposes a novel approach."
    (synth_dir / "review.md").write_text(review_md)

    run = WorkflowRun(status=WorkflowStatus.SUCCESS)
    run.env["__paper_id__"] = "abcd1234"
    decide_job = JobInstance(status=JobStatus.SUCCESS)
    decide_job.outputs = {"decision_path": str(decision_dir / "decision.json")}
    synth_job = JobInstance(status=JobStatus.SUCCESS)
    synth_job.outputs = {"review_path": str(synth_dir / "review.md")}
    run.jobs = {"decide": decide_job, "synthesize": synth_job}
    return run


def test_export_returns_xml_string(tmp_path):
    storage = MagicMock()
    run = _make_run_with_decision(tmp_path)
    storage.get_run.return_value = run

    exporter = OpenReviewExporter()
    xml = exporter.export("test-run-id", storage)

    assert '<?xml version="1.0"' in xml
    assert '<note>' in xml
    assert '</note>' in xml


def test_export_has_5_content_fields(tmp_path):
    storage = MagicMock()
    run = _make_run_with_decision(tmp_path)
    storage.get_run.return_value = run

    exporter = OpenReviewExporter()
    xml = exporter.export("test-run-id", storage)

    assert 'name="recommendation"' in xml
    assert 'name="confidence"' in xml
    assert 'name="review"' in xml
    assert 'name="soundness"' in xml
    assert 'name="contribution"' in xml


def test_export_recommendation_value(tmp_path):
    storage = MagicMock()
    run = _make_run_with_decision(tmp_path)
    storage.get_run.return_value = run

    exporter = OpenReviewExporter()
    xml = exporter.export("test-run-id", storage)

    assert "weak_accept" in xml


def test_export_confidence_is_1_to_5(tmp_path):
    storage = MagicMock()
    run = _make_run_with_decision(tmp_path)
    storage.get_run.return_value = run

    exporter = OpenReviewExporter()
    xml = exporter.export("test-run-id", storage)

    # avg confidence = (0.85 + 0.70 + 0.90) / 3 ≈ 0.817 × 5 ≈ 4.08 → 4
    assert 'name="confidence"' in xml
    # Extract the value
    import re
    m = re.search(r'name="confidence">(\d+)', xml)
    assert m is not None
    val = int(m.group(1))
    assert 1 <= val <= 5


def test_export_has_metadata(tmp_path):
    storage = MagicMock()
    run = _make_run_with_decision(tmp_path)
    run.id = "test-run-id"
    storage.get_run.return_value = run

    exporter = OpenReviewExporter()
    xml = exporter.export("test-run-id", storage)

    assert "<metadata>" in xml
    assert "<venue>neurips</venue>" in xml
    assert "<paper_id>abcd1234</paper_id>" in xml
    assert "test-run-id" in xml


def test_export_review_md_content(tmp_path):
    storage = MagicMock()
    run = _make_run_with_decision(tmp_path)
    storage.get_run.return_value = run

    exporter = OpenReviewExporter()
    xml = exporter.export("test-run-id", storage)

    assert "This paper proposes a novel approach." in xml


def test_export_icml_maps_significance(tmp_path):
    """ICML uses 'significance' as contribution-equivalent"""
    storage = MagicMock()
    run = _make_run_with_decision(tmp_path, venue="icml")
    # Override per_dimension for ICML
    decision_path = run.jobs["decide"].outputs["decision_path"]
    decision = json.loads(Path(decision_path).read_text())
    decision["per_dimension"] = {
        "soundness": {"score": 3, "confidence": 0.8},
        "significance": {"score": 4, "confidence": 0.9},
        "originality": {"score": 3, "confidence": 0.7},
        "clarity": {"score": 4, "confidence": 0.8},
    }
    Path(decision_path).write_text(json.dumps(decision))

    storage.get_run.return_value = run
    exporter = OpenReviewExporter()
    xml = exporter.export("test-run-id", storage)

    # contribution field should have significance value (4)
    import re
    m = re.search(r'name="contribution">(\d+)', xml)
    assert m is not None
    assert int(m.group(1)) == 4


def test_export_nonexistent_run_raises():
    storage = MagicMock()
    storage.get_run.return_value = None
    exporter = OpenReviewExporter()
    with pytest.raises(ValueError, match="run not found"):
        exporter.export("nonexistent", storage)
