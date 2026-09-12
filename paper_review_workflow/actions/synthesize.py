"""SynthesizeAction: combine dimension scores into unified review (venue-driven)."""
import json
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .base import BaseAction, ActionResult
from .registry import ActionRegistry
from ..core.venue_config import VenueConfig
from ..llm.client import LLMClient
from ..llm.schemas import SynthesisResult

logger = logging.getLogger(__name__)


_PROMPTS_DIR = Path(__file__).parent / "prompts"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
)


class SynthesizeAction(BaseAction):
    """Read dim score.json files, call LLM, write review.md + scores.json."""

    MIN_DIMENSIONS = 2  # absolute floor when venue config is unavailable

    @property
    def description(self) -> str:
        return "Synthesize dimension scores into unified review"

    def run(self, params, env, context, log_callback=None):
        session_dir = Path(params["session_dir"])

        venue_name = env.get("VENUE", "neurips")
        score_min = score_max = None
        try:
            venue_config = VenueConfig.load(venue_name)
            dimensions = venue_config.dimensions
            score_min = venue_config.score_min
            score_max = venue_config.score_max
        except ValueError:
            dimensions = []

        dim_results = {}
        missing = []
        # If venue config loaded, use its dimensions; otherwise scan for 10_dim_* dirs
        if dimensions:
            for dim in dimensions:
                score_path = session_dir / f"10_dim_{dim}" / "score.json"
                if score_path.exists():
                    dim_results[dim] = json.loads(score_path.read_text())
                else:
                    missing.append(dim)
        else:
            for d in sorted(session_dir.glob("10_dim_*")):
                score_path = d / "score.json"
                if score_path.exists():
                    dim = d.name.replace("10_dim_", "")
                    dim_results[dim] = json.loads(score_path.read_text())

        if log_callback:
            log_callback(f"loaded {len(dim_results)} dimensions, missing: {missing}")

        # Require all of the venue's dimensions; fall back to a small floor
        # only when no venue config could be loaded.
        min_required = len(dimensions) if dimensions else self.MIN_DIMENSIONS
        if len(dim_results) < min_required:
            return ActionResult(
                success=False,
                message=f"too many missing dimensions ({len(missing)} missing, need >= {min_required})",
            )

        prompt = self._render_prompt(dim_results, venue_name=venue_name,
                                     score_min=score_min, score_max=score_max)
        dim_summary = json.dumps(dim_results, ensure_ascii=False, default=str)

        client = LLMClient.from_env(env)
        response = client.complete(
            system=prompt,
            messages=[{"role": "user", "content": dim_summary}],
            response_schema=SynthesisResult,
        )
        synthesis = response.structured

        out_dir = session_dir / "50_synthesize"
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "scores.json").write_text(
            json.dumps(dim_results, ensure_ascii=False, indent=2)
        )
        self._write_review_md(out_dir / "review.md", synthesis, dim_results)

        if log_callback:
            log_callback(f"synthesis complete: {len(synthesis.key_strengths)} strengths, "
                        f"{len(synthesis.key_weaknesses)} weaknesses")
            # Stream the synthesized meta-review so users see it in real time
            summary = (synthesis.summary or "").strip()
            for para in summary.split("\n"):
                if para.strip():
                    log_callback(f"📝 [综合评审] {para.strip()}")
            if synthesis.key_strengths:
                log_callback("✅ [综合评审] 优点: " + "；".join(synthesis.key_strengths))
            if synthesis.key_weaknesses:
                log_callback("⚠️ [综合评审] 不足: " + "；".join(synthesis.key_weaknesses))
            if synthesis.questions_for_authors:
                log_callback("❓ [综合评审] 作者需回应的问题: " + "；".join(synthesis.questions_for_authors))
            if synthesis.overall_assessment:
                log_callback(f"🎯 [综合评审] 总体评价: {synthesis.overall_assessment.strip()}")

        return ActionResult(
            success=True,
            outputs={
                "review_path": str(out_dir / "review.md"),
                "scores_path": str(out_dir / "scores.json"),
            },
        )

    def _render_prompt(self, dim_results: dict, venue_name: str,
                       score_min: int, score_max: int) -> str:
        template = _jinja_env.get_template("synthesize.j2")
        return template.render(dimensions=dim_results, venue_name=venue_name,
                               score_min=score_min, score_max=score_max)

    def _write_review_md(self, path: Path, synthesis: SynthesisResult,
                         dim_results: dict) -> None:
        lines = ["# Peer Review Synthesis", "", "## Summary", synthesis.summary, ""]
        lines.extend(["## Key Strengths", *[f"- {s}" for s in synthesis.key_strengths], ""])
        lines.extend(["## Key Weaknesses", *[f"- {w}" for w in synthesis.key_weaknesses], ""])
        lines.extend(["## Questions for Authors", *[f"- {q}" for q in synthesis.questions_for_authors], ""])
        lines.extend(["## Overall Assessment", synthesis.overall_assessment, ""])
        lines.extend(["## Per-Dimension Scores"])
        for dim, result in dim_results.items():
            lines.append(f"- **{dim}**: {result.get('score', '?')} (conf={result.get('confidence', '?')})")
        path.write_text("\n".join(lines))


def register_synthesize_action(registry: ActionRegistry) -> None:
    registry.register("paper-review/synthesize@v1", SynthesizeAction())
