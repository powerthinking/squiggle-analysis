# squiggle-analysis

Post-training analysis pipeline for Squiggle runs.

This package consumes artifacts written by `squiggle-experiments` and produces interpretable summaries of model behavior.

## Pipeline Overview

For a given `run_id`, analysis proceeds in order:

1. **Geometry computation** — Extract basis-invariant descriptors from captures
2. **Event detection** — Identify significant geometric changes over time
3. **Report generation** — Produce human-readable summaries

Each step writes a stable artifact and can be re-run independently.

## Geometry State

Geometry is computed from captured activations and written to:

- `geometry_state/<run_id>.parquet`

Current metrics:
- `effective_rank` — Dimensionality utilization
- `sv_entropy` — Singular value distribution entropy
- `topk_mass_k8` — Mass concentration in top-k singular values

The geometry state is treated as *ground truth* for downstream analysis.

## Event Detection

Events identify statistically unusual geometric changes. Key features:

### Adaptive Thresholding

Thresholds computed per (layer, metric) series:

```
threshold = median(deltas) + k × MAD(deltas)
```

This identifies unusual changes relative to each metric's own distribution.

### Peak Selection with Suppression

Non-maximum suppression prevents detecting the same transition multiple times:

- Sort candidates by magnitude
- Select top candidate
- Suppress nearby candidates within `suppression_radius` steps
- Repeat until budget exhausted

### Warmup-Aware Budgeting

Separate budgets for pre-warmup and post-warmup candidates:

```
max_peaks_post = max_events_per_series - max_pre_warmup
max_peaks_pre = max_pre_warmup
```

This prevents early training transients from consuming the event budget.

### Detection Summary

Retention statistics written to `detection_summary/<run_id>.parquet`:
- Raw candidates per series
- Selected vs skipped counts
- Skip reasons (suppression, budget, warmup cap)
- Pre/post warmup breakdown

Events are written to:

- `events_candidates/<run_id>.parquet`

Supported event types:
- `change_point` (single metric)
- `change_point_composite` (multi-metric; `metric = __composite__`)

## Reporting

Each run produces a human-readable report:

- `runs/<run_id>/reports/report.md`

The report summarizes:
- Training scalars
- Capture inventory
- Geometry metrics
- Top detected events
- Detection retention analysis
- Event distribution diagnostics

## CLI

Run full analysis:

```bash
python -m squiggle_analysis --run-id <RUN_ID>
```

Force recompute:

```bash
python -m squiggle_analysis --run-id <RUN_ID> --force
```

Event detection options:

```bash
python -m squiggle_analysis --run-id <RUN_ID> \
    --adaptive-k 2.5 \
    --suppression-radius 15 \
    --warmup-fraction 0.1 \
    --max-pre-warmup 1
```

## Cross-Run Comparison

Compare multiple runs for seed invariance:

```bash
python -m squiggle_analysis --compare RUN_ID1 RUN_ID2 RUN_ID3 --output comparison.md
```

The comparison includes:
- Common events across runs
- Trajectory correlation
- Retention metrics comparison
- Phase distribution analysis

## LLM Qualitative Analysis

Optionally send reports to an LLM (GPT-4o or Claude) for expert interpretation:

```bash
# Single-run with LLM analysis
python -m squiggle_analysis --run-id <RUN_ID> --llm-analysis

# Comparison with LLM analysis and specific question
python -m squiggle_analysis --compare RUN1 RUN2 \
    --output comparison.md \
    --llm-analysis \
    --llm-question "Are these runs seed-invariant post-warmup?"
```

### LLM Options

| Option | Default | Description |
|--------|---------|-------------|
| `--llm-analysis` | off | Enable LLM qualitative analysis |
| `--llm-backend` | openai | Backend: `openai` or `anthropic` |
| `--llm-model` | gpt-4o | Model to use |
| `--llm-question` | (none) | Specific question to ask |

### Output

LLM analysis writes to a separate JSON file with structured output:
- `runs/<run_id>/reports/llm_analysis.json` (single-run)
- `<output>.llm_analysis.json` (comparison)

The output includes headline summary, key findings, hypotheses with confidence scores, recommended actions with effort/impact ratings, and suggested improvements.

### Installation

Requires the `llm` optional dependency:

```bash
pip install squiggle-analysis[llm]
```

Set your API key:

```bash
export OPENAI_API_KEY=sk-...    # For OpenAI
export ANTHROPIC_API_KEY=sk-... # For Anthropic
```
