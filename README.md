# squiggle-analysis — **Geometry, events, reports**

### `squiggle-analysis/README.md`

```md
# squiggle-analysis

Post-training analysis pipeline for Squiggle runs.

This repo consumes artifacts written by `squiggle-experiments`
and produces interpretable summaries of model behavior.

## Pipeline Overview

For a given `run_id`, analysis proceeds in order:

1. **Geometry computation**
2. **Event detection**
3. **Report generation**

Each step writes a stable artifact and can be re-run independently.

## Geometry State

Geometry is computed from captured activations and written to:

- `geometry_state/<run_id>.parquet`

Current metrics include:
- `effective_rank`
- `topk_mass_k8`

The geometry state is treated as *ground truth* for downstream analysis.

## Events

Events detect significant geometric change over time.

- Interval-based (`start_step`, `end_step`)
- Metric-aware thresholds
- Layer-specific

Event candidates are scored using the shared scoring implementation in `squiggle_core.scoring`.

Supported event types:
- `change_point` (single metric)
- `change_point_composite` (multi-metric; `metric = __composite__`)

Events are written to:

- `events_candidates/<run_id>.parquet`

### Adaptive Thresholding

By default, event detection uses **adaptive thresholds** computed per (layer, metric) as:

```
threshold = median(deltas) + k × MAD(deltas)
```

Where MAD is the Median Absolute Deviation. This identifies statistically unusual changes relative to each metric's own distribution, rather than using fixed thresholds that may not scale across different models.

Parameters (in `detect_events()`):
- `adaptive_threshold: bool = True` — Enable adaptive thresholding (default: on)
- `adaptive_k: float = 2.5` — Multiplier for MAD (higher = fewer events)
- `adaptive_min_threshold: float = 0.01` — Floor to prevent too-low thresholds

To use fixed thresholds instead (legacy behavior):
```python
detect_events(run_id, adaptive_threshold=False, rank_threshold=0.2, mass_threshold=0.03)
```

### Baseline Usage

Optional baseline usage:
- score against a designated baseline run (`baseline_run_id`) or a persisted baseline artifact (`baseline_id`)


## Reporting

Each run produces a human-readable report:

- `runs/<run_id>/reports/report.md`


The report summarizes:
- Training scalars
- Probe performance
- Capture inventory
- Geometry metrics
- Top detected events

Reports are intended to be:
- Shareable
- Diffable
- Durable

## CLI

Run full analysis:

python -m squiggle_analysis --run-id <RUN_ID>


Force recompute:
python -m squiggle_analysis --run-id <RUN_ID> --force
