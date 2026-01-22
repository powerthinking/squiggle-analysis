from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from squiggle_core import paths
from squiggle_core.scoring.squiggle_scoring import (
    ScoringConfig,
    build_baselines_from_samples,
    compute_event_score,
 )


def detect_events(
    run_id: str,
    *,
    analysis_id: str = "analysis@2.0",
    baseline_run_id: str | None = None,
    rank_threshold: float = 0.2,
    mass_threshold: float = 0.03,
) -> None:
    geom_path = paths.geometry_state_path(run_id)
    if not geom_path.exists():
        raise FileNotFoundError(
            f"Geometry state parquet not found for run_id='{run_id}'. Expected: {geom_path}\n"
            f"Run geometry computation first."
        )

    geom = pd.read_parquet(geom_path)

    required = {
        "run_id",
        "analysis_id",
        "schema_version",
        "created_at_utc",
        "step",
        "layer",
        "metric",
        "value",
    }
    missing = required - set(geom.columns)
    if missing:
        raise ValueError(
            f"Geometry state is missing required columns: {sorted(missing)}\n"
            f"Found columns: {list(geom.columns)}"
        )

    out = paths.events_candidates_path(run_id)
    out.parent.mkdir(parents=True, exist_ok=True)

    created_at_utc = datetime.now(timezone.utc)
    schema_version = "events_candidates@2.0"

    # Define the schema we always write (even if empty)
    out_cols = [
        "run_id",
        "analysis_id",
        "schema_version",
        "created_at_utc",
        "event_id",
        "layer",
        "metric",
        "step",        # canonical event timestamp (validator/report-friendly)
        "start_step",  # interval context
        "end_step",
        "event_type",
        "score",

        "magnitude",
        "structure_modifier",
        "magnitude_eff",
        "coherence",
        "novelty",

        "metric_size",
        "metric_z",
        "baseline_median",
        "baseline_mad",

        "volatility_event",
        "volatility_baseline",
        "volatility_ratio",
        "volatility_ratio_agg",
    ]

    if geom.empty:
        pd.DataFrame(columns=out_cols).to_parquet(out, index=False)
        return

    baseline_geom = geom
    if baseline_run_id is not None:
        baseline_geom_path = paths.geometry_state_path(baseline_run_id)
        if not baseline_geom_path.exists():
            raise FileNotFoundError(
                f"Geometry state parquet not found for baseline_run_id='{baseline_run_id}'. Expected: {baseline_geom_path}"
            )
        baseline_geom = pd.read_parquet(baseline_geom_path)

    size_samples: dict[str, list[float]] = {}
    for (_, metric), g in baseline_geom.groupby(["layer", "metric"], sort=True):
        g = g.sort_values("step")
        values = g["value"].to_numpy()
        if len(values) < 2:
            continue
        ds = [float(abs(values[i] - values[i - 1])) for i in range(1, len(values))]
        if not ds:
            continue
        key = str(metric)
        size_samples.setdefault(key, []).extend(ds)

    baselines = build_baselines_from_samples(size_samples)
    scoring_cfg = ScoringConfig(use_structure_modifier=False)

    def _threshold_for_metric(metric_name: str) -> float:
        if metric_name == "effective_rank":
            return float(rank_threshold)
        if metric_name.startswith("topk_mass_"):
            return float(mass_threshold)
        # default fallback for future metrics
        return float(rank_threshold)

    events: list[dict] = []
    event_id = 0

    # Group per (layer, metric) so thresholds apply cleanly
    for (layer, metric), g in geom.groupby(["layer", "metric"], sort=True):
        g = g.sort_values("step")
        values = g["value"].to_numpy()
        steps = g["step"].to_numpy()

        if len(values) < 2:
            continue

        thr = _threshold_for_metric(str(metric))

        for i in range(1, len(values)):
            delta = float(values[i] - values[i - 1])
            if abs(delta) > thr:
                start_step = int(steps[i - 1])
                end_step = int(steps[i])

                # Canonical event step: choose END by default
                # (means "the change is observed at end_step")
                step = end_step

                metric_key = str(metric)
                metric_size = float(abs(delta))

                breakdown = None
                baseline_med = None
                baseline_mad = None
                metric_z = None

                if metric_key in baselines:
                    b = baselines[metric_key]
                    baseline_med = float(b.median)
                    baseline_mad = float(b.mad)
                    breakdown = compute_event_score(
                        metric_sizes={metric_key: metric_size},
                        baselines={metric_key: b},
                        cfg=scoring_cfg,
                    )
                    score = float(breakdown.score)
                    metric_z = float(breakdown.metric_z.get(metric_key, 0.0))
                else:
                    score = metric_size

                events.append(
                    {
                        "run_id": run_id,
                        "analysis_id": analysis_id,
                        "schema_version": schema_version,
                        "created_at_utc": created_at_utc,
                        "event_id": f"e{event_id}",
                        "layer": int(layer),
                        "metric": str(metric),
                        "step": step,
                        "start_step": start_step,
                        "end_step": end_step,
                        "event_type": "change_point",

                        "score": score,

                        "magnitude": (float(breakdown.magnitude) if breakdown else None),
                        "structure_modifier": (float(breakdown.structure_modifier) if breakdown else None),
                        "magnitude_eff": (float(breakdown.magnitude_eff) if breakdown else None),
                        "coherence": (float(breakdown.coherence) if breakdown else None),
                        "novelty": (float(breakdown.novelty) if breakdown else None),

                        "metric_size": metric_size,
                        "metric_z": metric_z,
                        "baseline_median": baseline_med,
                        "baseline_mad": baseline_mad,

                        "volatility_event": None,
                        "volatility_baseline": None,
                        "volatility_ratio": None,
                        "volatility_ratio_agg": None,
                    }
                )
                event_id += 1

    df = pd.DataFrame(events)

    # Always write with the same columns, even if no events were found
    if df.empty:
        df = pd.DataFrame(columns=out_cols)
    else:
        # Ensure column set/order is stable
        for c in out_cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[out_cols]

        # Deterministic ordering (nice for diffs + reports)
        df = df.sort_values(["score", "layer", "metric", "step"], ascending=[False, True, True, True]).reset_index(drop=True)

        # Reassign event_id after sorting so IDs correspond to report ordering (optional)
        df["event_id"] = [f"e{i}" for i in range(len(df))]

    df.to_parquet(out, index=False)
