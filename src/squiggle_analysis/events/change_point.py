import pandas as pd
from squiggle_core import paths


def detect_events(run_id: str, threshold: float = 0.2):
    geom = pd.read_parquet(paths.geometry_state_path(run_id))

    events = []
    event_id = 0

    for layer, g in geom.groupby("layer"):
        g = g.sort_values("step")
        values = g["value"].values

        for i in range(1, len(values)):
            delta = values[i] - values[i - 1]
            if abs(delta) > threshold:
                events.append(
                    {
                        "run_id": run_id,
                        "event_id": f"e{event_id}",
                        "layer": layer,
                        "metric": "effective_rank",
                        "start_step": int(g.iloc[i - 1]["step"]),
                        "end_step": int(g.iloc[i]["step"]),
                        "score": float(abs(delta)),
                        "event_type": "change_point",
                    }
                )
                event_id += 1

    df = pd.DataFrame(events)
    paths.events_path(run_id).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.events_path(run_id))
