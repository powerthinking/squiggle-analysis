from datetime import datetime, timezone

import pandas as pd
from squiggle_core import paths
from squiggle_core.geometry.state import compute_effective_rank, compute_topk_mass



def compute_geometry_state(
    run_id: str,
    *,
    analysis_id: str = "analysis@2.0",
    components: list[str] | None = None,
) -> None:
    """
    Compute geometry state metrics for a training run.

    Args:
        run_id: The run to analyze
        analysis_id: Version identifier for this analysis
        components: List of tensor components to analyze. If None, defaults to
                   residual stream tensors (x_out, resid). Pass ["all"] to analyze
                   all available components.
    """
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

    # Default to residual stream components
    if components is None:
        target_components = {DEFAULT_COMPONENT} | LEGACY_COMPONENTS
    elif components == ["all"]:
        target_components = None  # Process all
    else:
        target_components = set(components)

    rows = []

    created_at_utc = datetime.now(timezone.utc)
    schema_version = "geometry_state@2.0"

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
            component = parse_component(tensor_path.name)

            # Filter by component if specified
            if target_components is not None and component not in target_components:
                continue

            layer = parse_layer(tensor_path.name)
            rank = compute_effective_rank(tensor_path)
            topk = compute_topk_mass(tensor_path, k=8)

            rows.append(
                {
                    "run_id": run_id,
                    "analysis_id": analysis_id,
                    "schema_version": schema_version,
                    "created_at_utc": created_at_utc,
                    "step": step,
                    "layer": layer,
                    "metric": "effective_rank",
                    "value": rank,
                }
            )
            rows.append(
                {
                    "run_id": run_id,
                    "analysis_id": analysis_id,
                    "schema_version": schema_version,
                    "created_at_utc": created_at_utc,
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

    out_path = paths.geometry_state_path(run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)


def parse_layer(filename: str) -> int:
    """
    For thin slice, we support filenames like:
      - resid_layer_03.pt
      - layer_3_resid.pt
      - layer_00_x_out.pt
    If no layer is found, return -1.
    """
    parts = filename.replace(".pt", "").split("_")
    for i, p in enumerate(parts):
        if p.lower() == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
        if p.isdigit():
            return int(p)
    return -1


def parse_component(filename: str) -> str:
    """
    Extract the component type from tensor filename.
    Examples:
      - resid_layer_03.pt -> "resid"
      - layer_00_x_out.pt -> "x_out"
      - layer_00_attn_out.pt -> "attn_out"
      - layer_00_mlp_out.pt -> "mlp_out"
      - embed.pt -> "embed"
    """
    name = filename.replace(".pt", "")
    parts = name.split("_")

    # Handle "embed" special case
    if name == "embed":
        return "embed"

    # Find layer index and extract component after it
    for i, p in enumerate(parts):
        if p.lower() == "layer" and i + 1 < len(parts) and parts[i + 1].isdigit():
            # Component is everything after layer number
            if i + 2 < len(parts):
                return "_".join(parts[i + 2:])
            return "resid"  # Default if nothing follows layer number

    # Legacy: component before layer (e.g., resid_layer_03)
    if "layer" in [p.lower() for p in parts]:
        layer_idx = next(i for i, p in enumerate(parts) if p.lower() == "layer")
        if layer_idx > 0:
            return "_".join(parts[:layer_idx])

    return "unknown"


# Default to x_out (residual stream) for geometry analysis
DEFAULT_COMPONENT = "x_out"
LEGACY_COMPONENTS = {"resid", "resid_layer"}  # Older naming conventions

