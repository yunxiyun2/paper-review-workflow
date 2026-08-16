# Paper Review Workflow

AI-powered academic paper review workflow engine. Runs a venue-specific LLM-based peer review (NeurIPS 3-dimension, ICML/ACL 4-dimension) on a PDF or arXiv paper, producing structured scores and a final recommendation.

## Installation

```bash
git clone <repo>
cd paper-review-workflow
pip install -e ".[dev]"
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quick Start

Review an arXiv paper:

```bash
python main.py run configs/neurips_review.yaml \
    --payload '{"paper_source": "2402.12098"}'
```

Review a local PDF:

```bash
python main.py run configs/neurips_review.yaml \
    --payload '{"paper_source": "/path/to/paper.pdf"}'
```

## API Server Mode

Start the FastAPI HTTP API + WebSocket server:

```bash
# Default (no subcommand starts server)
python main.py

# Explicit
python main.py server --host 0.0.0.0 --port 8000

# Dev mode with auto-reload
python main.py server --reload
```

API docs at `http://localhost:8000/docs`. Key endpoints:

- `GET /api/health` — health check
- `GET /api/workflows` — list registered workflows
- `POST /api/workflows/register` — register a YAML workflow
- `POST /api/runs` — dispatch a review (returns run_id immediately)
- `GET /api/runs/{run_id}` — get run status
- `POST /api/runs/{run_id}/cancel` — cancel a run
- `POST /api/runs/{run_id}/resume` — resume a failed/interrupted run

WebSocket endpoints:
- `ws://localhost:8000/ws` — all events
- `ws://localhost:8000/ws/runs/{run_id}` — filtered to single run

Example: dispatch a review via curl:

```bash
curl -X POST http://localhost:8000/api/runs \
  -H "Content-Type: application/json" \
  -d '{"workflow_name": "neurips-paper-review", "inputs": {"paper_source": "2402.12098"}}'
```

## Web UI

The FastAPI server includes a built-in web frontend. Start the server:

```bash
python main.py server
```

Then open `http://localhost:8000/` in your browser. The frontend provides 4 tabs:

1. **装配 (Assemble)**: Select venue, adjust dimension weights, generate + download YAML, register & dispatch
2. **触发 (Dispatch)**: Select a registered workflow, enter paper_source, dispatch a review
3. **监控 (Monitor)**: List all runs, click to view real-time progress via WebSocket, cancel/resume
4. **决策 (Decision)**: Select a completed run, view recommendation + per-dimension scores + rationale

The frontend is a single-file pure HTML/JS/CSS app (`paper_review_workflow/api/static/index.html`) — zero build step, zero npm dependencies.

## Output Structure

Each run creates a session directory:

```
sessions/<paper_id>/<run_id>/
  run.json                    # Full run state (lwf storage)
  run_manifest.json           # Static archive
  00_extract/                 # Parsed paper
    metadata.json
    sections.json
    full_text.md
    references.json
  10_dim_soundness/            # 3 dimension scores (parallel)
    score.json
    review.md
  ...
  50_synthesize/              # Meta-review
    review.md
    scores.json
  60_decision/                # Final recommendation
    decision.json
  final_report.md
```

## Resume / Rerun

If a run fails or is interrupted:

```bash
python main.py resume <run_id>
```

Force rerun a specific component (cascades to downstream):

```bash
python main.py resume <run_id> --rerun dimensions_soundness
```

## List Past Runs

```bash
python main.py list-runs
python main.py show-run <run_id>
```

## Venue-Specific Mode

The engine supports three venue configurations, selected via the `VENUE` env var (default `neurips`). Each venue defines its own dimensions, score range, weights, and recommendation thresholds; venue configs live in `configs/venues/<name>.yaml`.

| Venue | Dimensions | Score range |
| --- | --- | --- |
| **NeurIPS** (default) | Soundness / Presentation / Contribution (3) | 1-10 |
| **ICML** | Soundness / Significance / Originality / Clarity (4) | 1-4 |
| **ACL** | Soundness / Excitement / Reproducibility / Overall (4) | 1-4 |

`DimensionAction` builds a dynamic Pydantic schema per venue (e.g. `DimensionScore_neurips` enforces 1-10, `DimensionScore_icml` enforces 1-4). `DecideAction` reads the venue's weights and thresholds to compute the weighted average and map it to an OpenReview-style 7-tier recommendation (`strong_accept` ... `strong_reject`).

To dispatch a review under a specific venue, set `VENUE` in the workflow `env` block (or pass it via the API). The example `configs/neurips_review.yaml` defaults to `neurips`; override with `VENUE: icml` or `VENUE: acl` to switch venues.

## Configuration

See `configs/neurips_review.yaml` for the default NeurIPS 3-dimension review config. Venue configs (dimensions, score ranges, weights, thresholds) are in `configs/venues/`.

## Multi-Provider LLM Support

The engine supports 3 LLM providers:

| Provider | Env Var | Default Model | Structured Output |
|----------|---------|---------------|-------------------|
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` | tool use (strict) |
| OpenAI | `OPENAI_API_KEY` | (user must set `LLM_MODEL`) | json_schema (strict) |
| DeepSeek | `DEEPSEEK_API_KEY` | (user must set `LLM_MODEL`) | json_object + prompt schema |

Switch providers via environment variables:

```bash
# OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-4o  # user specifies current model name

# DeepSeek
export LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=sk-...
export LLM_MODEL=deepseek-chat

# Anthropic (default)
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
# LLM_MODEL defaults to claude-sonnet-4-6
```

All providers share the same retry strategy (3x exponential backoff) and schema validation (`SchemaValidationError` on invalid LLM output).

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

## Testing

```bash
# Unit + integration (no API key needed)
pytest

# E2E (requires API key, costs ~$0.5)
pytest --run-e2e -m e2e
```

## Architecture

This project reuses the lwf workflow engine skeleton (GitHub Actions-style YAML, three-layer state machine, matrix parallelism) and adds paper-review-specific actions:

- `extract`: PDF / arXiv parsing
- `dim_score` (matrix × 3 NeurIPS / × 4 ICML/ACL): LLM scoring per dimension
- `synthesize`: LLM meta-review
- `decide`: pure-rule weighted scoring

See `docs/superpowers/specs/2026-08-12-paper-review-workflow-design.md` for the full design.

## License

MIT
