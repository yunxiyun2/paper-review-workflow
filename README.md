# Paper Review Workflow

AI-powered academic paper review workflow engine. Runs an 8-dimension LLM-based peer review on a PDF or arXiv paper, producing structured scores and a final recommendation.

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
python main.py run configs/normal_review.yaml \
    --payload '{"paper_source": "2402.12098"}'
```

Review a local PDF:

```bash
python main.py run configs/normal_review.yaml \
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
  -d '{"workflow_name": "normal-paper-review", "inputs": {"paper_source": "2402.12098"}}'
```

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
  10_dim_novelty/             # 8 dimension scores (parallel)
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
python main.py resume <run_id> --rerun dim_novelty
```

## List Past Runs

```bash
python main.py list-runs
python main.py show-run <run_id>
```

## Configuration

See `configs/normal_review.yaml` for the default 8-dimension review config. Customize weights via `WEIGHT_*` env vars.

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
- `dim_score` (matrix × 8): LLM scoring per dimension
- `synthesize`: LLM meta-review
- `decide`: pure-rule weighted scoring

See `docs/superpowers/specs/2026-08-12-paper-review-workflow-design.md` for the full design.

## License

MIT
