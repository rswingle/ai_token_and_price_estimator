# ai_token_and_price_estimator

Estimate token usage and rank **stateless API** LLM providers by cost for any AI project.

Given a free-form project prompt, the tool:

1. Estimates total tokens (input + output + reasoning buffer) using `tiktoken`.
2. Fetches per-million-token pricing (live web refresh with curated offline fallback).
3. Filters out **chat / session / stateful** products and deprecated models.
4. Computes project cost per provider with the standard formula.
5. Recommends the best option via a weighted score: cost 40% / capability 30% / context 20% / latency 10%.
6. Emits both a human-readable report and a structured JSON document.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
.venv/bin/python -m src "Build an agent that browses the web and writes a 1500-word report" --format both
```

Pass the prompt via stdin when the prompt is large:

```bash
cat my_prompt.txt | .venv/bin/python -m src --format json
```

### Flags

| Flag                         | Description                                                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `--target-output N`          | Override heuristic; supply an exact target output token count.                                                   |
| `--multiplier X`             | Override the output/input ratio (skips heuristic).                                                               |
| `--refresh`                  | Attempt a live scrape of provider marketing pages (short timeout, best-effort).                                  |
| `--use-openrouter`           | Merge live USD prices from OpenRouter's public `/api/v1/models` (opt-in; falls back to baseline on any failure). |
| `--no-cache`                 | Skip the on-disk cache; use the curated baseline directly.                                                       |
| `--include-deprecated`       | Include deprecated models in the evaluation set.                                                                 |
| `--format human\|json\|both` | Output format (default `both`).                                              |
| `--excel PATH`             | Also write a multi-sheet `.xlsx` workbook to PATH (5 sheets: token estimate, recommendation, ranked pricing comparison, all providers, excluded). |
| `--top N`                  | Number of recommendations to return (default 3).                             |
| `--weight-cost X`            | Override cost weight (default 0.40).                                                                             |
| `--weight-capability X`      | Override capability weight (default 0.30).                                                                       |
| `--weight-context X`         | Override context-window weight (default 0.20).                                                                   |
| `--weight-latency X`         | Override latency weight (default 0.10).                                                                          |

## Output

The `human` format prints a bordered ASCII report with sections for:

- Project prompt
- Token estimate (input, output, reasoning buffer, total, multiplier, notes)
- Pricing comparison table (ranked by total cost)
- Recommendation (best + top alternatives + weighted score + reason)
- Excluded records (with reasons)
- Warnings & assumptions

The `json` format is the full machine-readable document:

```json
{
  "generated_at": "2026-06-03T18:29:06Z",
  "project_prompt": "...",
  "token_estimate": { "input_tokens": ..., "output_tokens_estimated": ..., ... },
  "pricing_source": { "source": "baseline|cache|live", "stale": true|false, "fetched_at": "...", "errors": [] },
  "providers": [ { "provider": "...", "model": "...", "input_cost_per_1m": ..., "output_cost_per_1m": ..., "context_window": ..., "source_url": "...", "fetched_at": "..." } ],
  "excluded": [ { "provider": "...", "model": "...", "reasons": [...] } ],
  "cost_comparison": [ { "provider": "...", "model": "...", "total_cost": ..., "fits_context_window": ... } ],
  "recommendation": { "best_model": "...", "best_provider": "...", "best_total_cost_usd": ..., "best_weighted_score": ..., "reason": "...", "alternatives": [...] }
}
```

## Architecture

```
src/
  __main__.py          CLI entry point
  token_estimator.py   Module 1: classify + count + estimate
  pricing_research.py  Module 2: web refresh + curated baseline + cache
  model_filter.py      Module 3: stateless-only + deprecation filter
  cost_engine.py       Module 4: cost formula + ranking
  recommender.py       Module 5: weighted scoring
  report.py            Output formatters (human + JSON)
tests/
  test_token_estimator.py
  test_cost_engine.py
  test_model_filter.py
  test_recommender.py
```

The five modules are pure / side-effect-bounded: each can be imported and tested
in isolation. Pricing is the only network-bound module; everything else runs
offline once a snapshot is loaded.

## Pricing strategy

1. `--refresh` attempts a live scrape of the official provider pricing pages
   (OpenAI, Anthropic, Google, Mistral, Groq, Together, Fireworks, DeepSeek,
   OpenRouter). Each request has a 6-second timeout; total refresh is capped at
   20 seconds.
2. If at least one record was updated, the result is treated as `live` and
   saved to `cache/pricing_snapshot.json`.
3. Otherwise the tool falls back to the in-repo **curated baseline** (20
   widely-known models across 8 providers). Each record carries its official
   `source_url` and `fetched_at` timestamp.
4. The JSON output always tags the snapshot as `stale=true` when it came from
   the cache or baseline.

If a provider's pricing page is unreachable or returns no extractable values
(common: bot-protected 403 responses), the baseline pricing is used and the
field is reported as `stale`. The tool never guesses pricing values.

## Heuristics

Token estimation uses a 5-class classifier and task-specific multipliers:

| Task kind | Multiplier range | Reasoning buffer |
| --------- | ---------------- | ---------------- |
| Q&A       | 0.5x – 1.0x      | 10%              |
| Analysis  | 1.0x – 2.0x      | 20%              |
| General   | 1.0x – 2.0x      | 20%              |
| Long-form | 2.0x – 5.0x      | 15%              |
| Code      | 2.0x – 4.0x      | 30%              |
| Agentic   | 3.0x – 6.0x      | 50%              |

A flat 80-token system-prompt overhead is added to every input.

## Run tests

```bash
.venv/bin/python -m pytest tests/ -v
```

56 tests cover classification, token counting, deterministic estimation, the
cost formula, context-window fit, filter rules, ranking, and the weighted
recommendation algorithm.
