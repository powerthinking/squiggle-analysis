import pandas as pd
from squiggle_core import paths


def detect_events(run_id: str, threshold: float = 0.2) -> None:
    geom_path = paths.geometry_state_path(run_id)
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

    if geom.empty:
        # Write an empty events parquet (nice behavior) and return
        out = paths.events_path(run_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "run_id",
                "event_id",
                "layer",
                "metric",
                "start_step",
                "end_step",
                "score",
                "event_type",
            ]
        ).to_parquet(out)
        return

    events = []
    event_id = 0

    # You can optionally filter to one metric for thin slice
    geom = geom[geom["metric"] == "effective_rank"].copy()

    for layer, g in geom.groupby("layer"):
        g = g.sort_values("step")
        values = g["value"].to_numpy()
        steps = g["step"].to_numpy()

        for i in range(1, len(values)):
            delta = float(values[i] - values[i - 1])
            if abs(delta) > threshold:
                events.append(
                    {
                        "run_id": run_id,
                        "event_id": f"e{event_id}",
                        "layer": int(layer),
                        "metric": "effective_rank",
                        "start_step": int(steps[i - 1]),
                        "end_step": int(steps[i]),
                        "score": abs(delta),
                        "event_type": "change_point",
                    }
                )
                event_id += 1

    out = paths.events_path(run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(events).to_parquet(out)
