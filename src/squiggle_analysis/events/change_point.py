from __future__ import annotations

import pandas as pd
from squiggle_core import paths


def detect_events(run_id: str, rank_threshold: float = 0.2, mass_threshold: float = 0.03) -> None:
    geom_path = paths.geometry_state_long_path(run_id)
    if not geom_path.exists():
        raise FileNotFoundError(
            f"Geometry state parquet not found for run_id='{run_id}'. Expected: {geom_path}\n"
            f"Run geometry computation first."
        )

    geom = pd.read_parquet(geom_path)

    required = {"run_id", "step", "layer", "metric", "value"}
    missing = required - set(geom.columns)
    if missing:
        raise ValueError(
            f"Geometry state is missing required columns: {sorted(missing)}\n"
            f"Found columns: {list(geom.columns)}"
        )

    out = paths.events_path(run_id)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Define the schema we always write (even if empty)
    out_cols = [
        "run_id",
        "event_id",
        "layer",
        "metric",
        "step",        # canonical event timestamp (validator/report-friendly)
        "start_step",  # interval context
        "end_step",
        "score",
        "event_type",
    ]

    if geom.empty:
        pd.DataFrame(columns=out_cols).to_parquet(out, index=False)
        return

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

                events.append(
                    {
                        "run_id": run_id,
                        "event_id": f"e{event_id}",
                        "layer": int(layer),
                        "metric": str(metric),
                        "step": step,
                        "start_step": start_step,
                        "end_step": end_step,
                        "score": abs(delta),
                        "event_type": "change_point",
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
