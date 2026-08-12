"""DecideAction: pure-rule weighted scoring -> OpenReview 7-tier recommendation."""
import json
import logging
from pathlib import Path
from typing import Dict, List

from .base import BaseAction, ActionResult
from .registry import ActionRegistry

logger = logging.getLogger(__name__)


DEFAULT_WEIGHTS: Dict[str, float] = {
    "novelty": 1.0,
    "soundness": 1.2,
    "significance": 1.2,
    "clarity": 0.8,
    "reproducibility": 1.0,
    "related_work": 0.8,
    "positioning": 0.8,
    "presentation": 0.8,
}


THRESHOLDS: List[tuple] = [
    (4.5, "strong_accept"),
    (4.0, "accept"),
    (3.5, "weak_accept"),
    (3.0, "borderline"),
    (2.0, "weak_reject"),
    (1.5, "reject"),
    (0.0, "strong_reject"),
]


VALID_RECOMMENDATIONS = [label for _, label in THRESHOLDS]


class DecideAction(BaseAction):
    """Compute weighted average of dimension scores and map to 7-tier recommendation."""

    @property
    def description(self) -> str:
        return "Decide final recommendation (pure rule, no LLM)"

    def run(self, params, env, context, log_callback=None):
        session_dir = Path(params["session_dir"])
        scores_path = params.get("scores_path") or str(
            session_dir / "50_synthesize" / "scores.json"
        )

        try:
            scores: Dict[str, dict] = json.loads(Path(scores_path).read_text())
        except Exception as e:
            return ActionResult(success=False, message=f"cannot read scores: {e}")

        weights = self._resolve_weights(env)
        per_dimension = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for dim, weight in weights.items():
            if dim not in scores:
                if log_callback:
                    log_callback(f"dimension {dim} missing, skipping (weight={weight})")
                continue
            score = scores[dim].get("score")
            confidence = scores[dim].get("confidence", 0.0)
            if score is None:
                continue
            weighted = score * weight
            per_dimension[dim] = {
                "score": score,
                "confidence": confidence,
                "weighted": weighted / weight,  # back to raw for display
            }
            weighted_sum += weighted
            total_weight += weight

        if total_weight == 0:
            return ActionResult(success=False, message="no dimensions to score")

        weighted_score = weighted_sum / total_weight
        recommendation = self._map_to_recommendation(weighted_score)

        decision = {
            "schema_version": "1.0",
            "recommendation": recommendation,
            "weighted_score": round(weighted_score, 2),
            "per_dimension": per_dimension,
            "decision_rationale": self._generate_rationale(recommendation, weighted_score, per_dimension),
            "key_concerns": self._extract_key_concerns(scores),
            "key_strengths": self._extract_key_strengths(scores),
            "weights_used": weights,
        }

        out_dir = session_dir / "60_decision"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "decision.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2)
        )

        # Also create final_report.md as a copy of synthesize/review.md
        final_report = session_dir / "final_report.md"
        review_path = session_dir / "50_synthesize" / "review.md"
        if review_path.exists():
            final_report.write_text(review_path.read_text())

        if log_callback:
            log_callback(f"recommendation: {recommendation} (score={weighted_score:.2f})")

        return ActionResult(
            success=True,
            outputs={
                "decision_path": str(out_dir / "decision.json"),
                "recommendation": recommendation,
                "weighted_score": round(weighted_score, 2),
            },
        )

    def _resolve_weights(self, env: Dict[str, str]) -> Dict[str, float]:
        weights = dict(DEFAULT_WEIGHTS)
        for dim in list(weights.keys()):
            env_key = f"WEIGHT_{dim.upper()}"
            if env_key in env:
                try:
                    weights[dim] = float(env[env_key])
                except ValueError:
                    pass
        return weights

    def _map_to_recommendation(self, score: float) -> str:
        for threshold, label in THRESHOLDS:
            if score >= threshold:
                return label
        return "strong_reject"

    def _generate_rationale(self, recommendation: str, score: float,
                            per_dimension: dict) -> str:
        top_dim = max(per_dimension.items(), key=lambda kv: kv[1]["weighted"])
        bot_dim = min(per_dimension.items(), key=lambda kv: kv[1]["weighted"])
        return (
            f"Weighted average across {len(per_dimension)} dimensions is {score:.2f}/5, "
            f"mapping to '{recommendation}'. "
            f"Strongest dimension: {top_dim[0]} ({top_dim[1]['weighted']:.1f}). "
            f"Weakest dimension: {bot_dim[0]} ({bot_dim[1]['weighted']:.1f})."
        )

    def _extract_key_concerns(self, scores: dict) -> list:
        concerns = []
        for dim, data in scores.items():
            score = data.get("score")
            if score is not None and score <= 2:
                weaknesses = data.get("weaknesses", [])
                if weaknesses:
                    concerns.append(f"{dim}: {weaknesses[0]}")
        return concerns[:3]

    def _extract_key_strengths(self, scores: dict) -> list:
        strengths = []
        for dim, data in scores.items():
            score = data.get("score")
            if score is not None and score >= 4:
                s = data.get("strengths", [])
                if s:
                    strengths.append(f"{dim}: {s[0]}")
        return strengths[:3]


def register_decide_action(registry: ActionRegistry) -> None:
    registry.register("paper-review/decide@v1", DecideAction())
