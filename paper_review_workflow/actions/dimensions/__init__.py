"""DimensionAction: scores one paper dimension via LLM, driven by venue config."""
import json
import logging
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..base import BaseAction, ActionResult
from ..registry import ActionRegistry
from ...core.venue_config import VenueConfig
from ...llm.client import LLMClient

logger = logging.getLogger(__name__)


_PACKAGE_ROOT = Path(__file__).parent.parent.parent  # paper_review_workflow/
_jinja_env = Environment(
    loader=FileSystemLoader(str(_PACKAGE_ROOT / "actions" / "dimensions" / "prompts")),
    autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
)

DIM_LABELS_ZH = {
    "soundness": "严谨性",
    "presentation": "表达清晰度",
    "contribution": "贡献性",
    "excitement": "创新兴奋度",
    "reproducibility": "可复现性",
    "overall": "总体评价",
    "significance": "重要性",
    "originality": "原创性",
    "clarity": "清晰性",
}


class DimensionAction(BaseAction):
    """Score one dimension of a paper. Driven by `with.dimension` + `env.VENUE` params."""

    @property
    def description(self) -> str:
        return "Score one dimension of a paper (venue-driven)"

    def run(self, params, env, context, log_callback=None):
        venue_name = env.get("VENUE", "neurips")
        dimension = params.get("dimension")

        if not dimension:
            return ActionResult(success=False, message="dimension param required")

        try:
            venue_config = VenueConfig.load(venue_name)
        except ValueError as e:
            return ActionResult(success=False, message=str(e))

        if dimension not in venue_config.dimensions:
            return ActionResult(
                success=False,
                message=f"dimension {dimension} not in venue {venue_name}'s dimensions: {venue_config.dimensions}",
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

        # Get venue-specific schema (dynamic score range)
        schema = venue_config.get_dimension_score_schema()

        # Render venue-specific prompt
        try:
            prompt = self._render_prompt(venue_config.prompts_dir, dimension, metadata)
        except Exception as e:
            return ActionResult(success=False, message=f"prompt render failed: {e}")

        # Call LLM with venue's schema
        client = LLMClient.from_env(env)
        response = client.complete(
            system=prompt,
            messages=[{"role": "user", "content": f"Score the {dimension} dimension of this paper."}],
            response_schema=schema,
            cached_context=paper_text,
            log_callback=log_callback,
        )
        score = response.structured
        llm_usage = self._llm_usage(response)

        # Write score.json (with venue field) + review.md
        out_dir = session_dir / f"10_dim_{dimension}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write_score_json(out_dir / "score.json", score, dimension, venue_name,
                                response.model, response.usage)
        self._write_review_md(out_dir / "review.md", score, dimension, venue_config)

        if log_callback:
            log_callback(f"[{venue_name}/{dimension}] score={score.score} conf={score.confidence:.2f}")
            # Stream the model's actual review so users see it in real time
            label = DIM_LABELS_ZH.get(dimension, dimension)
            justification = (score.justification or "").strip()
            if justification:
                for para in justification.split("\n"):
                    if para.strip():
                        log_callback(f"💬 [{label}] {para.strip()}")
            if score.strengths:
                log_callback(f"✅ [{label}] 优点: " + "；".join(score.strengths))
            if score.weaknesses:
                log_callback(f"⚠️ [{label}] 不足: " + "；".join(score.weaknesses))

        return ActionResult(
            success=True,
            outputs={
                "score": score.score,
                "confidence": score.confidence,
                "score_path": str(out_dir / "score.json"),
                "review_path": str(out_dir / "review.md"),
                "llm_usage": llm_usage,
            },
            log_lines=[f"[{venue_name}/{dimension}] score={score.score}"],
        )

    @staticmethod
    def _llm_usage(response) -> dict:
        """Normalize the provider usage (GLM/OpenAI: prompt/completion_tokens;
        anthropic: input/output_tokens) into a flat token-usage record."""
        usage = getattr(response, "usage", None) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        return {
            "model": getattr(response, "model", ""),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def _render_prompt(self, prompts_dir: str, dimension: str, metadata: dict) -> str:
        """Render venue-specific prompt template."""
        # prompts_dir is relative to PACKAGE_ROOT/actions/dimensions/prompts/
        template_path = f"{prompts_dir}/{dimension}.j2"
        template = _jinja_env.get_template(template_path)
        return template.render(metadata=metadata, dimension=dimension)

    @staticmethod
    def _evidence_dicts(score) -> list:
        """Evidence entries as plain dicts (EvidenceItem models from the LLM
        schema are not JSON serializable; old score.json dicts pass through)."""
        return [e.model_dump() if hasattr(e, "model_dump") else e
                for e in (score.evidence or [])]

    def _write_score_json(self, path: Path, score, dimension: str, venue: str,
                          model: str, usage: dict) -> None:
        payload = {
            "schema_version": "1.0",
            "venue": venue,
            "dimension": dimension,
            "score": score.score,
            "confidence": score.confidence,
            "strengths": score.strengths,
            "weaknesses": score.weaknesses,
            "justification": score.justification,
            "evidence": self._evidence_dicts(score),
            "model_used": model,
            "usage": usage,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def _write_review_md(self, path: Path, score, dimension: str, venue_config: VenueConfig) -> None:
        score_range = f"{venue_config.score_min}-{venue_config.score_max}"
        lines = [
            f"# {dimension.title()} — Score: {score.score}/{venue_config.score_max} (confidence: {score.confidence:.2f})",
            "",
            f"_Venue: {venue_config.display_name} ({venue_config.name})_",
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
            for ev in self._evidence_dicts(score):
                location = (ev.get("location") or ev.get("section") or ev.get("claim") or "").strip()
                quote = (ev.get("quote") or ev.get("evidence") or "").strip()
                if not quote:
                    continue  # skip placeholder entries like "§?, p.?:"
                lines.append(f"- {location or '?'}: {quote[:160]}")
        path.write_text("\n".join(lines))


def register_dimension_action(registry: ActionRegistry) -> None:
    registry.register("paper-review/dim_score@v1", DimensionAction())
