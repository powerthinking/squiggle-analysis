import pandas as pd
from squiggle_core import paths
from squiggle_core.geometry.state import compute_effective_rank, compute_topk_mass



def compute_geometry_state(run_id: str) -> None:
    captures_root = paths.captures_dir(run_id)

    if not captures_root.exists():
        raise FileNotFoundError(
            f"No captures directory for run_id='{run_id}'. Expected: {captures_root}\n"
            f"Run the scout training first so captures get written."
        )

    step_dirs = sorted(captures_root.glob("step_*"))
    if not step_dirs:
        raise FileNotFoundError(
            f"Captures directory exists but has no step_* folders for run_id='{run_id}'.\n"
            f"Expected something like: {captures_root}/step_000050/"
        )

    rows = []

    for step_dir in step_dirs:
        try:
            step = int(step_dir.name.split("_")[1])
        except Exception as e:
            raise ValueError(f"Unexpected step directory name: {step_dir.name}") from e

        tensor_files = list(step_dir.glob("*.pt"))
        if not tensor_files:
            # Not fatal, but note: this step had no tensors
            continue

        for tensor_path in tensor_files:
            layer = parse_layer(tensor_path.name)
            rank = compute_effective_rank(tensor_path)
            topk = compute_topk_mass(tensor_path, k=8)

            rows.append(
                {
                    "run_id": run_id,
                    "step": step,
                    "layer": layer,
                    "metric": "effective_rank",
                    "value": rank,
                }
            )
            rows.append(
                {
                    "run_id": run_id,
                    "step": step,
                    "layer": layer,
                    "metric": "topk_mass_k8",
                    "value": topk,
                }
            )

    if not rows:
        raise RuntimeError(
            f"Found step directories for run_id='{run_id}' but no tensors were processed.\n"
            f"Check that instrumentation is writing .pt files under: {captures_root}/step_*/"
        )

    df = pd.DataFrame(rows)

    out_path = paths.geometry_state_long_path(run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)


def parse_layer(filename: str) -> int:
    """
    For thin slice, we support filenames like:
      - resid_layer_03.pt
      - layer_3_resid.pt
    If no layer is found, return -1.
    """
    parts = filename.replace(".pt", "").split("_")
    for i, p in enumerate(parts):
        if p.lower() == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
        if p.isdigit():
            return int(p)
    return -1

