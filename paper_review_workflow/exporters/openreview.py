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
