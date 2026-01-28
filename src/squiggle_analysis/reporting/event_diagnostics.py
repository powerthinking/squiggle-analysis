"""Event distribution diagnostics for squiggle analysis reports.

This module provides functions to analyze and visualize the temporal and spatial
distribution of detected events, helping to identify wave propagation patterns
and temporal clustering.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from squiggle_core import paths


def detect_capture_cadence(geom: pd.DataFrame) -> list[int]:
    """Extract the step grid from geometry_state.

    This returns the actual capture points (steps where data was recorded),
    which may not be uniformly spaced. Histogram bins should align to this grid.

    Args:
        geom: Geometry state DataFrame with 'step' column

    Returns:
        Sorted list of unique step values
    """
    if geom is None or geom.empty or "step" not in geom.columns:
        return []
    return sorted(geom["step"].unique().tolist())


def compute_step_histogram(
    events: pd.DataFrame,
    step_grid: list[int],
) -> pd.DataFrame:
    """Compute event counts binned by step.

    Creates a histogram showing event counts at each capture step, both globally
    and per-metric.

    Args:
        events: Events DataFrame with 'step' and 'metric' columns
        step_grid: Step values to use as bin centers (from geometry_state)

    Returns:
        DataFrame with columns: step, all, <metric1>, <metric2>, ...
    """
    if events is None or events.empty or not step_grid:
        return pd.DataFrame(columns=["step", "all"])

    # Get unique metrics (excluding __composite__)
    metrics = sorted([m for m in events["metric"].unique() if m != "__composite__"])

    # Initialize histogram
    rows = []
    for step in step_grid:
        row = {"step": step, "all": 0}
        for m in metrics:
            row[m] = 0
        rows.append(row)

    hist_df = pd.DataFrame(rows)
    step_to_idx = {s: i for i, s in enumerate(step_grid)}

    # Count events per step
    for _, ev in events.iterrows():
        ev_step = ev["step"]
        # Find closest step in grid
        closest_step = min(step_grid, key=lambda s: abs(s - ev_step))
        idx = step_to_idx.get(closest_step)
        if idx is not None:
            hist_df.loc[idx, "all"] += 1
            metric = ev.get("metric", "")
            if metric in hist_df.columns:
                hist_df.loc[idx, metric] += 1

    return hist_df


def compute_wavefront_matrix(
    events: pd.DataFrame,
    metric: str,
    step_grid: list[int],
    n_layers: int | None = None,
) -> pd.DataFrame:
    """Compute layer-by-step event count matrix for a specific metric.

    Creates a matrix showing event presence at each (layer, step) combination,
    useful for visualizing wave propagation patterns.

    Args:
        events: Events DataFrame with 'step', 'layer', and 'metric' columns
        metric: The metric to filter events by
        step_grid: Step values for columns (from geometry_state)
        n_layers: Number of layers (auto-detected if None)

    Returns:
        DataFrame with layers as rows (descending) and steps as columns
    """
    if events is None or events.empty or not step_grid:
        return pd.DataFrame()

    # Filter to specific metric
    metric_events = events[events["metric"] == metric]
    if metric_events.empty:
        return pd.DataFrame()

    # Detect number of layers
    if n_layers is None:
        n_layers = int(events["layer"].max()) + 1

    # Initialize matrix
    matrix = pd.DataFrame(
        0,
        index=range(n_layers - 1, -1, -1),  # Descending order (upper layers first)
        columns=step_grid,
    )
    matrix.index.name = "layer"

    # Count events per (layer, step)
    for _, ev in metric_events.iterrows():
        layer = int(ev["layer"])
        ev_step = ev["step"]
        # Find closest step in grid
        closest_step = min(step_grid, key=lambda s: abs(s - ev_step))
        if layer in matrix.index and closest_step in matrix.columns:
            matrix.loc[layer, closest_step] += 1

    return matrix


def format_step_histogram_md(histogram: pd.DataFrame) -> str:
    """Format step histogram as Markdown table.

    Args:
        histogram: DataFrame from compute_step_histogram()

    Returns:
        Markdown table string
    """
    if histogram is None or histogram.empty:
        return "_No histogram data available._"

    # Limit to reasonable size for display
    max_rows = 30
    if len(histogram) > max_rows:
        histogram = histogram.head(max_rows)

    return histogram.to_markdown(index=False)


def format_wavefront_md(
    wavefront: pd.DataFrame,
    metric: str,
    max_cols: int = 15,
    max_rows: int = 12,
) -> str:
    """Format wavefront matrix as truncated Markdown table.

    Args:
        wavefront: DataFrame from compute_wavefront_matrix()
        metric: Metric name for header
        max_cols: Maximum number of step columns to show
        max_rows: Maximum number of layer rows to show

    Returns:
        Markdown table string with truncation note
    """
    if wavefront is None or wavefront.empty:
        return f"_No wavefront data for {metric}._"

    total_rows = len(wavefront)
    total_cols = len(wavefront.columns)

    # Truncate if needed
    truncated = wavefront.iloc[:max_rows, :max_cols].copy()

    # Add layer column from index
    truncated = truncated.reset_index()

    # Build header note
    if total_rows > max_rows or total_cols > max_cols:
        note = f"### {metric} wavefront (truncated: {min(total_rows, max_rows)}/{total_rows} layers, {min(total_cols, max_cols)}/{total_cols} steps)\n\n"
    else:
        note = f"### {metric} wavefront\n\n"

    return note + truncated.to_markdown(index=False)


def save_wavefront_artifact(
    wavefront: pd.DataFrame,
    run_id: str,
    metric: str,
) -> Path:
    """Save full wavefront matrix as parquet artifact.

    Args:
        wavefront: DataFrame from compute_wavefront_matrix()
        run_id: The run ID
        metric: Metric name for filename

    Returns:
        Path to saved artifact
    """
    reports_dir = paths.reports_dir(run_id)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize metric name for filename
    safe_metric = metric.replace("/", "_").replace(":", "_")
    artifact_path = reports_dir / f"wavefront_{safe_metric}.parquet"

    # Reset index to include layer as column
    wavefront_out = wavefront.reset_index()
    wavefront_out.to_parquet(artifact_path, index=False)

    return artifact_path


def compute_composite_stats(events: pd.DataFrame) -> dict:
    """Compute summary statistics for composite events.

    Args:
        events: Events DataFrame

    Returns:
        Dictionary with composite event statistics
    """
    if events is None or events.empty:
        return {
            "total_composites": 0,
            "by_metric_count": {},
            "step_distribution": [],
        }

    composite_events = events[events["event_type"] == "change_point_composite"]
    if composite_events.empty:
        return {
            "total_composites": 0,
            "by_metric_count": {},
            "step_distribution": [],
        }

    total = len(composite_events)

    # Count by number of participating metrics
    by_metric_count = {}
    for _, ev in composite_events.iterrows():
        # Parse metric_sizes_json to count participating metrics
        try:
            sizes_json = ev.get("metric_sizes_json", "{}")
            if sizes_json and isinstance(sizes_json, str):
                sizes = json.loads(sizes_json)
                n_metrics = len(sizes)
                key = f"{n_metrics}-metric"
                by_metric_count[key] = by_metric_count.get(key, 0) + 1
        except (json.JSONDecodeError, TypeError):
            pass

    # Step distribution (steps where composites occur)
    step_distribution = sorted(composite_events["step"].unique().tolist())

    # Participating metric combinations
    metric_combinations = []
    for _, ev in composite_events.iterrows():
        try:
            sizes_json = ev.get("metric_sizes_json", "{}")
            if sizes_json and isinstance(sizes_json, str):
                sizes = json.loads(sizes_json)
                combo = " + ".join(sorted(sizes.keys()))
                metric_combinations.append(combo)
        except (json.JSONDecodeError, TypeError):
            pass

    # Count combinations
    combo_counts = {}
    for combo in metric_combinations:
        combo_counts[combo] = combo_counts.get(combo, 0) + 1

    return {
        "total_composites": total,
        "by_metric_count": by_metric_count,
        "step_distribution": step_distribution,
        "metric_combinations": combo_counts,
    }


def format_composite_section_md(stats: dict) -> str:
    """Format composite event statistics as Markdown section.

    Args:
        stats: Dictionary from compute_composite_stats()

    Returns:
        Markdown section string
    """
    lines = ["## Composite Events\n"]

    total = stats.get("total_composites", 0)
    if total == 0:
        lines.append("_No composite events detected._")
        return "\n".join(lines)

    lines.append(f"- Total composites: **{total}**")

    # By metric count
    by_count = stats.get("by_metric_count", {})
    if by_count:
        count_str = ", ".join(f"{k}: {v}" for k, v in sorted(by_count.items()))
        lines.append(f"- By metric count: {count_str}")

    # Step distribution summary
    steps = stats.get("step_distribution", [])
    if steps:
        lines.append(f"- Steps with composites: {len(steps)} unique steps")
        if len(steps) <= 10:
            lines.append(f"  - Steps: {steps}")
        else:
            lines.append(f"  - First 10: {steps[:10]}")

    # Metric combinations
    combos = stats.get("metric_combinations", {})
    if combos:
        lines.append("\n**Participating metric combinations:**\n")
        lines.append("| Combination | Count |")
        lines.append("|-------------|-------|")
        for combo, count in sorted(combos.items(), key=lambda x: -x[1]):
            lines.append(f"| {combo} | {count} |")

    return "\n".join(lines)


def generate_event_diagnostics(
    events: pd.DataFrame,
    geom: pd.DataFrame,
    run_id: str,
    save_artifacts: bool = True,
) -> str:
    """Generate complete event diagnostics section for report.

    This is the main entry point for adding diagnostics to reports.

    Args:
        events: Events DataFrame
        geom: Geometry state DataFrame
        run_id: The run ID
        save_artifacts: Whether to save wavefront parquets

    Returns:
        Markdown section string
    """
    lines = ["\n## Event Distribution\n"]

    # Detect capture cadence
    step_grid = detect_capture_cadence(geom)
    if not step_grid:
        lines.append("_Cannot compute distribution: no geometry data._")
        return "\n".join(lines)

    lines.append(
        f"_Capture cadence: {len(step_grid)} steps, range {min(step_grid)}-{max(step_grid)}_\n"
    )

    # Step histogram
    histogram = compute_step_histogram(events, step_grid)
    lines.append("### Event Step Histogram\n")
    lines.append(format_step_histogram_md(histogram))
    lines.append("")

    # Wavefront views per metric
    if events is not None and not events.empty:
        metrics = sorted([m for m in events["metric"].unique() if m != "__composite__"])
        n_layers = int(events["layer"].max()) + 1 if "layer" in events.columns else None

        artifact_paths = []
        for metric in metrics[:3]:  # Limit to top 3 metrics for brevity
            wavefront = compute_wavefront_matrix(events, metric, step_grid, n_layers)
            if not wavefront.empty:
                lines.append("")
                lines.append(format_wavefront_md(wavefront, metric))

                if save_artifacts:
                    artifact_path = save_wavefront_artifact(wavefront, run_id, metric)
                    artifact_paths.append(str(artifact_path))

        if artifact_paths:
            lines.append("")
            lines.append("**Wavefront artifacts saved:**")
            for p in artifact_paths:
                lines.append(f"- `{p}`")

    # Composite event statistics
    lines.append("")
    comp_stats = compute_composite_stats(events)
    lines.append(format_composite_section_md(comp_stats))

    return "\n".join(lines)
