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

Events are written to:

- `events/<run_id>.parquet` 


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
