# OpenReview XML Export (Phase 2 #4) Implementation Plan

**Goal:** Export review results as standard OpenReview note XML via CLI + API.

**Architecture:** `OpenReviewExporter` class reads decision.json + review.md from session dir, maps to 5 OpenReview fields, generates XML. CLI `export` subcommand + API `GET /api/runs/{id}/export` endpoint both call the exporter.

---

## M1: OpenReviewExporter + unit tests

### Task 1.1: Create exporter

**Files:** Create `paper_review_workflow/exporters/__init__.py` + `paper_review_workflow/exporters/openreview.py`, Test `tests/unit/test_openreview_exporter.py`

Create `paper_review_workflow/exporters/__init__.py` (empty).

Create `paper_review_workflow/exporters/openreview.py`:

```python
"""OpenReview XML exporter — converts review results to standard OpenReview note XML."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from ..storage import StorageBackend


class OpenReviewExporter:
    """Export a run's review results as OpenReview note XML."""

    # NeurIPS has "contribution", ICML has "significance", ACL has "overall"
    CONTRIBUTION_FIELD_MAP = {
        "neurips": "contribution",
        "icml": "significance",
        "acl": "overall",
    }

    def export(self, run_id: str, storage: StorageBackend) -> str:
        """Generate OpenReview XML for a completed run."""
        run = storage.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")

        decision = self._load_decision(run)
        review_md = self._load_review_md(run)

        venue = decision.get("venue", "neurips")
        recommendation = decision.get("recommendation", "borderline")
        confidence = self._calc_confidence(decision)
        soundness = self._get_score(decision, "soundness")
        contribution = self._get_contribution(decision, venue)

        paper_id = run.env.get("__paper_id__", "")
        run_id_val = run.id

        return self._build_xml(
            recommendation, confidence, review_md,
            soundness, contribution, venue, paper_id, run_id_val
        )

    def _load_decision(self, run) -> dict:
        """Load decision.json from run's decide job outputs."""
        decide_job = run.jobs.get("decide")
        if not decide_job:
            return {}
        decision_path = decide_job.outputs.get("decision_path")
        if decision_path and Path(decision_path).exists():
            return json.loads(Path(decision_path).read_text())
        # Fallback: check outputs dict directly
        return decide_job.outputs or {}

    def _load_review_md(self, run) -> str:
        """Load review.md from synthesize job outputs."""
        synth_job = run.jobs.get("synthesize")
        if not synth_job:
            return ""
        review_path = synth_job.outputs.get("review_path")
        if review_path and Path(review_path).exists():
            return Path(review_path).read_text()
        return ""

    def _calc_confidence(self, decision: dict) -> int:
        """Calculate overall confidence (1-5) from per-dimension confidences."""
        per_dim = decision.get("per_dimension", {})
        if not per_dim:
            return 3
        confs = [d.get("confidence", 0.5) for d in per_dim.values() if d.get("confidence") is not None]
        if not confs:
            return 3
        avg = sum(confs) / len(confs)
        return max(1, min(5, round(avg * 5)))

    def _get_score(self, decision: dict, dim_name: str) -> int:
        """Get a dimension's score from decision."""
        per_dim = decision.get("per_dimension", {})
        dim_data = per_dim.get(dim_name, {})
        return dim_data.get("score", 0)

    def _get_contribution(self, decision: dict, venue: str) -> int:
        """Get contribution-equivalent score based on venue."""
        field_name = self.CONTRIBUTION_FIELD_MAP.get(venue, "contribution")
        return self._get_score(decision, field_name)

    def _build_xml(self, recommendation, confidence, review_md,
                   soundness, contribution, venue, paper_id, run_id) -> str:
        """Build OpenReview note XML string."""
        note = ET.Element("note")
        content = ET.SubElement(note, "content")

        ET.SubElement(content, "field", name="recommendation").text = str(recommendation)
        ET.SubElement(content, "field", name="confidence").text = str(confidence)
        ET.SubElement(content, "field", name="review").text = review_md
        ET.SubElement(content, "field", name="soundness").text = str(soundness)
        ET.SubElement(content, "field", name="contribution").text = str(contribution)

        metadata = ET.SubElement(note, "metadata")
        ET.SubElement(metadata, "venue").text = venue
        ET.SubElement(metadata, "paper_id").text = paper_id
        ET.SubElement(metadata, "run_id").text = run_id

        # Pretty print
        ET.indent(note, space="  ")
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(note, encoding="unicode")
```

Create `tests/unit/test_openreview_exporter.py`:

```python
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
```

Run: `pytest tests/unit/test_openreview_exporter.py -v`
Commit: `git add paper_review_workflow/exporters/ tests/unit/test_openreview_exporter.py && git commit -m "feat(exporters): OpenReview XML exporter with 5 fields + metadata"`

---

## M2: CLI + API integration

### Task 2.1: CLI export subcommand

**Files:** Modify `paper_review_workflow/cli.py`

In `cli.py`, add export subparser + `_cmd_export`:

```python
    # export subcommand
    export_p = sub.add_parser("export", help="导出评审结果为 XML")
    export_p.add_argument("run_id", help="要导出的 run ID")
    export_p.add_argument("--format", default="xml", choices=["xml"], help="导出格式")
    export_p.add_argument("--output", default=None, help="输出文件路径(不指定则 stdout)")
```

Add dispatch: `elif args.command == "export": return _cmd_export(engine, args)`

Add function:
```python
def _cmd_export(engine, args):
    from .exporters.openreview import OpenReviewExporter
    exporter = OpenReviewExporter()
    try:
        xml = exporter.export(args.run_id, engine.storage)
    except ValueError as e:
        print(f"[Error] {e}", file=sys.stderr)
        return 1
    if args.output:
        Path(args.output).write_text(xml, encoding="utf-8")
        print(f"Exported to {args.output}")
    else:
        print(xml)
    return 0
```

### Task 2.2: API endpoint

**Files:** Modify `paper_review_workflow/api/server.py`

In `create_app()`, after the resume endpoint, add:

```python
    @app.get("/api/runs/{run_id}/export")
    async def export_run(run_id: str, format: str = "xml"):
        run = engine.storage.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
        from ..exporters.openreview import OpenReviewExporter
        exporter = OpenReviewExporter()
        try:
            xml = exporter.export(run_id, engine.storage)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        from fastapi.responses import Response
        return Response(content=xml, media_type="application/xml")
```

### Task 2.3: Integration test

Create `tests/integration/test_api_export.py`:

```python
import pytest
from fastapi.testclient import TestClient
from paper_review_workflow.api import create_app
from paper_review_workflow.storage.memory import MemoryStorage


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    app = create_app(storage=MemoryStorage(), configs_dir="configs")
    with TestClient(app) as c:
        yield c


def test_export_nonexistent_returns_404(client):
    r = client.get("/api/runs/nonexistent/export?format=xml")
    assert r.status_code == 404


def test_export_returns_xml_content_type():
    """Test with a mock run that has decision + review"""
    from paper_review_workflow.core.models import WorkflowRun, WorkflowStatus, JobInstance, JobStatus
    from paper_review_workflow.storage.memory import MemoryStorage
    import json
    from pathlib import Path

    storage = MemoryStorage()
    storage.open()
    
    # Create a mock completed run with decision
    run = WorkflowRun(status=WorkflowStatus.SUCCESS)
    run.env["__paper_id__"] = "test1234"
    
    decide_job = JobInstance(status=JobStatus.SUCCESS)
    decide_job.outputs = {"decision_path": "/tmp/test_decision.json"}
    synth_job = JobInstance(status=JobStatus.SUCCESS)
    synth_job.outputs = {"review_path": "/tmp/test_review.md"}
    run.jobs = {"decide": decide_job, "synthesize": synth_job}
    
    # Create the files
    decision = {
        "venue": "neurips", "recommendation": "weak_accept",
        "per_dimension": {"soundness": {"score": 7, "confidence": 0.85},
                         "contribution": {"score": 8, "confidence": 0.9},
                         "presentation": {"score": 5, "confidence": 0.7}},
    }
    Path("/tmp/test_decision.json").write_text(json.dumps(decision))
    Path("/tmp/test_review.md").write_text("# Review\n\nGood paper.")
    
    storage.save_run(run)
    
    # Use a fresh app with this storage
    app = create_app(storage=storage, configs_dir="configs")
    with TestClient(app) as c:
        r = c.get(f"/api/runs/{run.id}/export?format=xml")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/xml"
        assert "weak_accept" in r.text
        assert "<note>" in r.text
        assert "Good paper." in r.text
    
    # Cleanup
    Path("/tmp/test_decision.json").unlink(missing_ok=True)
    Path("/tmp/test_review.md").unlink(missing_ok=True)
```

Run: `pytest tests/integration/test_api_export.py -v`
Commit: `git add paper_review_workflow/cli.py paper_review_workflow/api/server.py tests/integration/test_api_export.py && git commit -m "feat(api+cli): export review results as OpenReview XML"`

---

## M3: README + tag

Update README with "Export" section. Tag v0.6.0 + push.

```markdown
## Export

Export review results as OpenReview XML:

```bash
# CLI
python main.py export <run_id> --format xml
python main.py export <run_id> --output report.xml

# API
curl http://localhost:8000/api/runs/<run_id>/export?format=xml > report.xml
```

The XML follows the standard OpenReview note format with 5 fields: recommendation, confidence, review, soundness, contribution.
```

Commit + tag: `git commit -m "docs: add export documentation" && git tag v0.6.0 && git push origin main && git push --tags`
