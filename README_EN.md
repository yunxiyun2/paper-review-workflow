# Paper Review Workflow

简体中文 | [English](./README.md)

An LLM-powered academic paper review workflow engine. Upload a PDF, pick a target venue (NeurIPS / ICML / ACL) and dimension weights, and a multi-dimension LLM pipeline produces structured scores, a synthesized meta-review, and a final recommendation.

## Features

- **Venue-aware review**: NeurIPS (3 dimensions, 1-10), ICML (4 dimensions, 1-4), ACL (4 dimensions, 1-4) — each venue has its own dimensions, weights, score range and recommendation thresholds
- **Multi-provider LLMs**: Zhipu GLM, OpenAI, DeepSeek, Anthropic Claude — bring a per-request API key, or configure via environment variables
- **Ephemeral review mode**: PDF and API key live in server memory only; the key can be deleted at any time and a whole task can be removed with one click. No persistence by design
- **Structured real-time logs**: terminal-style output showing model call parameters, token usage, model thinking, and per-stage state transitions
- **Resume from failure**: re-run from the breakpoint; completed stages are never re-billed
- **OpenReview export**: export results as OpenReview XML
- **Dual themes**: Paper (light) / Deep Space (dark), switchable from the bottom-right corner

## Quick Start

```bash
git clone <repo>
cd paper-review-workflow
pip install -e .
python main.py server
```

Open <http://localhost:8000/>, go to "发起评审" (New Review), enter a task name, pick venue and weights, upload a PDF, choose provider and model, paste your API key, and start.

### Configuring a model via environment (optional)

```bash
export LLM_PROVIDER=zhipu          # zhipu / openai / deepseek / anthropic
export ZHIPU_API_KEY=...           # provider-specific env var
# Default models: zhipu→glm-4.6, anthropic→claude-sonnet-4-6; others require LLM_MODEL
```

## Web UI

| Tab | Purpose |
| --- | --- |
| New Review | Task name, venue + dimension weights, PDF upload, provider + model, API key |
| Monitor | Run list (by task name), run details, token usage stats, live logs, cancel / resume / delete API key / delete task |
| Decision | Recommendation, per-dimension scores, key assessment, rationale, synthesized review (copyable) |

Theme switch: the sun/moon button at the bottom-right; the choice persists.

## CLI

```bash
python main.py server                                   # start API + web server
python main.py run configs/neurips_review.yaml \
    --payload '{"paper_source": "/path/to/paper.pdf"}'  # run a review
python main.py list-runs                                # list past runs
python main.py show-run <run_id>                        # run details
python main.py resume <run_id>                          # resume from failure
python main.py export <run_id> --format xml             # export OpenReview XML
```

## HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/review` | Ephemeral review: multipart PDF + api_key + model + venue + weights + task_name |
| GET | `/api/runs` | List runs (filterable by status) |
| GET | `/api/runs/{id}` | Run details (step logs, token usage) |
| POST | `/api/runs/{id}/cancel` | Cancel |
| POST | `/api/runs/{id}/resume` | Resume (optional api_key) |
| POST | `/api/runs/{id}/delete-key` | Drop the in-memory API key |
| DELETE | `/api/runs/{id}` | Delete task (key, session dir, run record) |
| GET | `/api/runs/{id}/decision` | Decision payload |
| GET | `/api/runs/{id}/review` | Synthesized review (Markdown) |
| GET | `/api/runs/{id}/export` | OpenReview XML |
| GET | `/api/venues` | Venue configs |
| WS | `/ws` | Live event stream |

Interactive docs: <http://localhost:8000/docs>

## Architecture

Built on the lwf workflow engine skeleton (GitHub Actions-style YAML, three-layer state machine, matrix parallelism) with review-specific actions:

- `extract`: PDF parsing (local PDF only) → metadata + full text
- `dim_score`: per-dimension scoring in parallel via venue matrix (max-parallel: 2), structured output with evidence citations
- `synthesize`: cross-dimension meta-review
- `decide`: rule-based weighted scoring mapped to the OpenReview 7-tier recommendation

Session layout:

```
sessions/<run_id>/
  upload/paper.pdf          # uploaded PDF
  00_extract/               # metadata, full text
  10_dim_<dimension>/       # score.json + review.md per dimension
  50_synthesize/            # meta-review
  60_decision/decision.json # final decision
  final_report.md
sessions/runs/<run_id>.json # run state (JSON file storage, no database)
```

## LLM Providers

| Provider | Env Var | Default Model | Structured Output |
| --- | --- | --- | --- |
| Zhipu GLM | `ZHIPU_API_KEY` | glm-4.6 | json_object + schema in prompt |
| OpenAI | `OPENAI_API_KEY` | requires `LLM_MODEL` | json_schema |
| DeepSeek | `DEEPSEEK_API_KEY` | requires `LLM_MODEL` | json_object + schema in prompt |
| Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4-6 | tool use |

All providers share: 3-5x exponential backoff retries, schema-validation self-repair (validation errors are fed back to the model), and long backoff on rate limits (429).

## Privacy & Data Safety

- API keys live only in server-process memory and are never written to disk (storage masks `*_API_KEY` fields as a second line of defense)
- "Delete API-Key" drops the in-memory key immediately; "Delete Task" removes the PDF, artifacts and run record together
- Uploaded PDFs and review artifacts are kept until explicitly deleted

## Testing

```bash
pytest                    # unit + integration (no API key needed)
pytest --run-e2e -m e2e   # end-to-end (requires a real key)
```

## License

MIT
