"""SynthesizeAction: combine 8 dimension scores into unified review."""
import json
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .base import BaseAction, ActionResult
from .registry import ActionRegistry
from .dimensions import ALL_DIMENSIONS
from ..llm.client import LLMClient
from ..llm.schemas import SynthesisResult

logger = logging.getLogger(__name__)


_PROMPTS_DIR = Path(__file__).parent / "prompts"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
)


class SynthesizeAction(BaseAction):
    """Read 8 dim score.json files, call LLM, write review.md + scores.json."""

    MIN_DIMENSIONS = 6  # require at least 6/8 to synthesize

    @property
    def description(self) -> str:
        return "Synthesize 8 dimension scores into unified review"

    def run(self, params, env, context, log_callback=None):
        session_dir = Path(params["session_dir"])

        dim_results = {}
        missing = []
        for dim in ALL_DIMENSIONS:
            score_path = session_dir / f"10_dim_{dim}" / "score.json"
            if score_path.exists():
                dim_results[dim] = json.loads(score_path.read_text())
            else:
                missing.append(dim)

        if log_callback:
            log_callback(f"loaded {len(dim_results)}/8 dimensions, missing: {missing}")

        if len(dim_results) < self.MIN_DIMENSIONS:
            return ActionResult(
                success=False,
                message=f"too many missing dimensions ({len(missing)} missing, need >= {self.MIN_DIMENSIONS})",
            )

        prompt = self._render_prompt(dim_results)
        dim_summary = json.dumps(dim_results, ensure_ascii=False, default=str)

        client = LLMClient.from_env()
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

        return ActionResult(
            success=True,
            outputs={
                "review_path": str(out_dir / "review.md"),
                "scores_path": str(out_dir / "scores.json"),
            },
        )

    def _render_prompt(self, dim_results: dict) -> str:
        template = _jinja_env.get_template("synthesize.j2")
        return template.render(dimensions=dim_results)

    def _write_review_md(self, path: Path, synthesis: SynthesisResult,
                         dim_results: dict) -> None:
        lines = ["# Peer Review Synthesis", "", "## Summary", synthesis.summary, ""]
        lines.extend(["## Key Strengths", *[f"- {s}" for s in synthesis.key_strengths], ""])
        lines.extend(["## Key Weaknesses", *[f"- {w}" for w in synthesis.key_weaknesses], ""])
        lines.extend(["## Questions for Authors", *[f"- {q}" for q in synthesis.questions_for_authors], ""])
        lines.extend(["## Overall Assessment", synthesis.overall_assessment, ""])
        lines.extend(["## Per-Dimension Scores"])
        for dim, result in dim_results.items():
            lines.append(f"- **{dim}**: {result.get('score', '?')}/5 (conf={result.get('confidence', '?')})")
        path.write_text("\n".join(lines))


def register_synthesize_action(registry: ActionRegistry) -> None:
    registry.register("paper-review/synthesize@v1", SynthesizeAction())
