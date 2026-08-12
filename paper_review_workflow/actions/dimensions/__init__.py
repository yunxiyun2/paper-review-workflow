"""DimensionAction: scores one paper dimension via LLM, used in matrix strategy."""
import json
import logging
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..base import BaseAction, ActionResult
from ..registry import ActionRegistry
from ...llm.client import LLMClient
from ...llm.schemas import DimensionScore

logger = logging.getLogger(__name__)


ALL_DIMENSIONS = [
    "novelty", "soundness", "significance", "clarity",
    "reproducibility", "related_work", "positioning", "presentation",
]


_PROMPTS_DIR = Path(__file__).parent / "prompts"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
)


class DimensionAction(BaseAction):
    """Score one dimension of a paper. Driven by `with.dimension` param."""

    @property
    def description(self) -> str:
        return "Score one dimension of a paper (matrix-driven)"

    def run(self, params, env, context, log_callback=None):
        dimension = params.get("dimension")
        if not dimension:
            return ActionResult(success=False, message="dimension param required")
        if dimension not in ALL_DIMENSIONS:
            return ActionResult(
                success=False,
                message=f"unknown dimension: {dimension}; must be one of {ALL_DIMENSIONS}",
            )

        session_dir = Path(params["session_dir"])
        full_text_path = params["full_text_path"]

        try:
            paper_text = Path(full_text_path).read_text(encoding="utf-8")
        except Exception as e:
            return ActionResult(success=False, message=f"cannot read paper: {e}")

        # Load metadata (optional, for context in prompt)
        metadata = {}
        if "metadata_path" in params:
            try:
                metadata = json.loads(Path(params["metadata_path"]).read_text())
            except Exception:
                pass

        prompt = self._render_prompt(dimension, metadata)

        client = LLMClient.from_env()
        response = client.complete(
            system=prompt,
            messages=[{"role": "user", "content": f"Score the {dimension} dimension of this paper."}],
            response_schema=DimensionScore,
            cached_context=paper_text,
        )
        score = response.structured

        out_dir = session_dir / f"10_dim_{dimension}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write_score_json(
            out_dir / "score.json", score, dimension, response.model, response.usage
        )
        self._write_review_md(out_dir / "review.md", score, dimension)

        if log_callback:
            log_callback(
                f"[{dimension}] score={score.score} conf={score.confidence:.2f}"
            )

        return ActionResult(
            success=True,
            outputs={
                "score": score.score,
                "confidence": score.confidence,
                "score_path": str(out_dir / "score.json"),
                "review_path": str(out_dir / "review.md"),
            },
            log_lines=[f"[{dimension}] score={score.score}"],
        )

    def _render_prompt(self, dimension: str, metadata: dict) -> str:
        template = _jinja_env.get_template(f"{dimension}.j2")
        return template.render(metadata=metadata, dimension=dimension)

    def _write_score_json(
        self,
        path: Path,
        score: DimensionScore,
        dimension: str,
        model: str,
        usage: dict,
    ) -> None:
        payload = {
            "schema_version": "1.0",
            "dimension": dimension,
            "score": score.score,
            "confidence": score.confidence,
            "strengths": score.strengths,
            "weaknesses": score.weaknesses,
            "justification": score.justification,
            "evidence": score.evidence,
            "model_used": model,
            "usage": usage,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def _write_review_md(self, path: Path, score: DimensionScore,
                         dimension: str) -> None:
        lines = [
            f"# {dimension.title()} — Score: {score.score}/5 (confidence: {score.confidence:.2f})",
            "",
            "## Strengths",
            *[f"- {s}" for s in score.strengths],
            "",
            "## Weaknesses",
            *[f"- {w}" for w in score.weaknesses],
            "",
            "## Justification",
            score.justification,
        ]
        if score.evidence:
            lines.extend(["", "## Evidence"])
            for ev in score.evidence:
                lines.append(
                    f"- §{ev.get('section', '?')}, p.{ev.get('page', '?')}: "
                    f"{ev.get('quote', '')[:100]}"
                )
        path.write_text("\n".join(lines))


def register_dimension_action(registry: ActionRegistry) -> None:
    registry.register("paper-review/dim_score@v1", DimensionAction())
