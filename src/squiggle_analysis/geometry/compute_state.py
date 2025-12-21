import pandas as pd
from squiggle_core import paths
from squiggle_core.geometry.state import compute_effective_rank


def compute_geometry_state(run_id: str):
    rows = []

    samples_root = paths.samples_dir(run_id)

    for step_dir in sorted(samples_root.glob("step_*")):
        step = int(step_dir.name.split("_")[1])

        for tensor_path in step_dir.glob("*.pt"):
            layer = parse_layer(tensor_path.name)

            rank = compute_effective_rank(tensor_path)

            rows.append(
                {
                    "run_id": run_id,
                    "step": step,
                    "layer": layer,
                    "metric": "effective_rank",
                    "value": rank,
                }
            )

    df = pd.DataFrame(rows)
    paths.geometry_state_path(run_id).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.geometry_state_path(run_id))


def parse_layer(filename: str) -> int:
    # example: resid_layer_03.pt
    for part in filename.split("_"):
        if part.isdigit():
            return int(part)
    return -1
