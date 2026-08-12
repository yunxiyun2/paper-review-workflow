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
