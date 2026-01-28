"""Event distribution diagnostics for squiggle analysis reports.

This module provides functions to analyze and visualize the temporal and spatial
distribution of detected events, helping to identify wave propagation patterns
and temporal clustering.
"""

from __future__ import annotations

import json
from bisect import bisect_left
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
        Sorted list of unique step values (ints)
    """
    if geom is None or geom.empty or "step" not in geom.columns:
        return []
    # Force int for consistent downstream comparisons / formatting
    return sorted([int(s) for s in geom["step"].unique().tolist()])


def _snap_step_to_grid(ev_step: int, step_grid: list[int]) -> int:
    """Snap an event step to the closest capture step in step_grid.

    Uses binary search (O(log n)). Assumes step_grid is sorted ascending.
    """
    if not step_grid:
        return ev_step
    # If exact match, return immediately
    i = bisect_left(step_grid, ev_step)
    if i == 0:
        return step_grid[0]
    if i >= len(step_grid):
        return step_grid[-1]
    before = step_grid[i - 1]
    after = step_grid[i]
    return before if abs(ev_step - before) <= abs(after - ev_step) else after


def compute_step_histogram(
    events: pd.DataFrame,
    step_grid: list[int],
    *,
    snap_to_grid: bool = True,
) -> pd.DataFrame:
    """Compute event counts binned by step.

    Creates a histogram showing event counts at each capture step, both globally
    and per-metric.

    Args:
        events: Events DataFrame with 'step' and 'metric' columns
        step_grid: Step values to use as bin centers (from geometry_state)
        snap_to_grid: If True, snap event steps to nearest step_grid value.
                      If False, only count events whose step is exactly in step_grid.

    Returns:
        DataFrame with columns: step, all, <metric1>, <metric2>, ...
    """
    if events is None or events.empty or not step_grid:
        return pd.DataFrame(columns=["step", "all"])

    if "step" not in events.columns or "metric" not in events.columns:
        return pd.DataFrame(columns=["step", "all"])

    # Normalize grid to ints
    step_grid_int = [int(s) for s in step_grid]
    step_set = set(step_grid_int)

    # Get unique metrics (excluding __composite__)
    metrics = sorted([m for m in events["metric"].unique() if m != "__composite__"])

    # Initialize histogram rows
    rows = []
    for step in step_grid_int:
        row = {"step": step, "all": 0}
        for m in metrics:
            row[m] = 0
        rows.append(row)

    hist_df = pd.DataFrame(rows)
    step_to_idx = {s: i for i, s in enumerate(step_grid_int)}

    # Count events per step
    for _, ev in events.iterrows():
        try:
            ev_step = int(ev["step"])
        except Exception:
            continue

        if snap_to_grid:
            binned_step = _snap_step_to_grid(ev_step, step_grid_int)
        else:
            if ev_step not in step_set:
                continue
            binned_step = ev_step

        idx = step_to_idx.get(binned_step)
        if idx is None:
            continue

        hist_df.loc[idx, "all"] += 1
        metric = ev.get("metric", "")
        if metric in hist_df.columns:
            hist_df.loc[idx, metric] += 1

    return hist_df


def compute_window_coverage_histogram(
    events: pd.DataFrame,
    step_grid: list[int],
) -> pd.DataFrame:
    """Compute coverage histogram counting all steps within event windows.

    Unlike compute_step_histogram (which counts only the peak step), this counts
    every grid step in [start_step, end_step] for each event. Useful for detecting
    whether "spiky" peak histograms are artifacts of choosing one representative step.

    Args:
        events: Events DataFrame with 'step', 'start_step', 'end_step', 'metric' columns
        step_grid: Capture step grid (from geometry_state)

    Returns:
        DataFrame with columns: step, all, <metric1>, <metric2>, ...
    """
    if events is None or events.empty or not step_grid:
        return pd.DataFrame(columns=["step", "all"])

    step_grid_int = [int(s) for s in step_grid]
    step_set = set(step_grid_int)
    grid_spacing = step_grid_int[1] - step_grid_int[0] if len(step_grid_int) > 1 else 1

    metrics = sorted([m for m in events["metric"].unique() if m != "__composite__"])

    hist = pd.DataFrame({"step": step_grid_int})
    hist["all"] = 0
    for m in metrics:
        hist[m] = 0

    step_to_idx = {s: i for i, s in enumerate(step_grid_int)}

    for _, ev in events.iterrows():
        metric = ev.get("metric", "")
        try:
            peak = int(ev["step"])
            s = int(ev.get("start_step", peak))
            e = int(ev.get("end_step", peak))
        except Exception:
            continue

        if e < s:
            s, e = e, s

        # Count each grid step in the window
        for st in range(s, e + 1, grid_spacing):
            if st in step_set:
                idx = step_to_idx.get(st)
                if idx is not None:
                    hist.loc[idx, "all"] += 1
                    if metric in hist.columns:
                        hist.loc[idx, metric] += 1

    return hist


def compute_series_diversity(
    events: pd.DataFrame,
    step_grid: list[int] | None = None,
    suppression_radius: int = 15,
) -> dict:
    """Compute per-series window overlap and peak clustering metrics.

    Metrics per series:
    - n_events: total events emitted
    - covered_steps: unique grid steps across all event windows
    - overlap_ratio: 1 - (covered_steps / total_window_steps)
    - min_gap: minimum gap between consecutive peak steps
    - median_gap: median gap between consecutive peak steps
    - n_close_pairs: count of peak pairs within suppression_radius

    Interpretation:
    - High overlap: multiple events describe the same transition region (over-segmentation)
    - Low overlap: events represent distinct regions (or suppression already collapsed near-dupes)
    - Small min_gap: series wanted clustered peaks (suppression may have collapsed some)
    - Large min_gap: peaks are naturally well-separated

    Note: True "suppression retention" requires storing candidate counts during detection.

    Args:
        events: Events DataFrame with step, start_step, end_step columns
        step_grid: Optional step grid for computing covered steps precisely
        suppression_radius: The suppression radius used during detection (for gap analysis)

    Returns:
        Dictionary with overlap and clustering statistics
    """
    if events is None or events.empty:
        return {"n_series": 0, "series_data": [], "summary": {}}

    # Determine grid spacing
    grid_spacing = 1
    if step_grid and len(step_grid) > 1:
        grid_spacing = int(step_grid[1]) - int(step_grid[0])

    # Group events by series, collecting windows and peaks
    series_data: dict[str, dict] = {}

    for _, ev in events.iterrows():
        # Get series ID
        if "series_id" in ev and ev.get("series_id"):
            sid = str(ev["series_id"])
        else:
            try:
                layer = int(ev["layer"])
                metric = str(ev["metric"])
                sid = f"{layer}:{metric}"
            except Exception:
                continue

        # Get window bounds and peak
        try:
            peak = int(ev["step"])
            start = int(ev.get("start_step", peak))
            end = int(ev.get("end_step", peak))
        except Exception:
            continue

        if end < start:
            start, end = end, start

        if sid not in series_data:
            series_data[sid] = {"windows": [], "peaks": [], "n_events": 0}

        series_data[sid]["windows"].append((start, end))
        series_data[sid]["peaks"].append(peak)
        series_data[sid]["n_events"] += 1

    # Compute per-series metrics
    series_stats = []
    for sid, data in series_data.items():
        n_events = data["n_events"]
        windows = data["windows"]
        peaks = sorted(data["peaks"])

        # Compute covered steps (union of all windows) - in grid units
        covered: set[int] = set()
        total_window_steps = 0
        for start, end in windows:
            # Count grid points in [start, end]
            window_steps = set(range(start, end + 1, grid_spacing))
            covered.update(window_steps)
            total_window_steps += len(window_steps)

        covered_steps = len(covered)

        # Overlap ratio
        if total_window_steps > 0:
            overlap_ratio = 1.0 - (covered_steps / total_window_steps)
        else:
            overlap_ratio = 0.0

        # Compute gaps between consecutive peaks
        gaps = []
        n_close_pairs = 0
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                gap = peaks[i + 1] - peaks[i]
                gaps.append(gap)
                if gap <= suppression_radius:
                    n_close_pairs += 1

        min_gap = min(gaps) if gaps else None
        median_gap = sorted(gaps)[len(gaps) // 2] if gaps else None

        # Span (distance from first to last peak) and coverage ratio
        if peaks and len(peaks) >= 2:
            span = peaks[-1] - peaks[0]
            # span_steps = number of grid points in span (inclusive)
            span_steps = (span // grid_spacing) + 1 if grid_spacing > 0 else span + 1
            span_coverage = covered_steps / span_steps if span_steps > 0 else 0.0
        else:
            span = 0
            span_steps = 0
            span_coverage = 0.0

        series_stats.append({
            "series_id": sid,
            "n_events": n_events,
            "covered_steps": covered_steps,
            "overlap_ratio": round(overlap_ratio, 2),
            "min_gap": min_gap,
            "median_gap": median_gap,
            "n_close_pairs": n_close_pairs,
            "span": span,
            "span_steps": span_steps,
            "span_coverage": round(span_coverage, 2),
        })

    # Build distribution: (n_events, overlap_bucket) -> count
    def _overlap_bucket(r: float) -> str:
        if r < 0.1:
            return "0-10%"
        elif r < 0.3:
            return "10-30%"
        elif r < 0.5:
            return "30-50%"
        else:
            return "50%+"

    distribution: dict[tuple[int, str], int] = {}
    for s in series_stats:
        key = (s["n_events"], _overlap_bucket(s["overlap_ratio"]))
        distribution[key] = distribution.get(key, 0) + 1

    # Summary stats
    if series_stats:
        avg_events = sum(s["n_events"] for s in series_stats) / len(series_stats)
        avg_covered = sum(s["covered_steps"] for s in series_stats) / len(series_stats)
        avg_overlap = sum(s["overlap_ratio"] for s in series_stats) / len(series_stats)

        # Gap stats (only for series with 2+ events)
        min_gaps = [s["min_gap"] for s in series_stats if s["min_gap"] is not None]
        median_gaps = [s["median_gap"] for s in series_stats if s["median_gap"] is not None]
        close_pairs = [s["n_close_pairs"] for s in series_stats]
        span_coverages = [s["span_coverage"] for s in series_stats if s["span_coverage"] > 0]

        avg_min_gap = sum(min_gaps) / len(min_gaps) if min_gaps else None
        avg_median_gap = sum(median_gaps) / len(median_gaps) if median_gaps else None
        total_close_pairs = sum(close_pairs)
        series_with_close = sum(1 for s in series_stats if s["n_close_pairs"] > 0)
        avg_span_coverage = sum(span_coverages) / len(span_coverages) if span_coverages else None
    else:
        avg_events = avg_covered = avg_overlap = 0
        avg_min_gap = avg_median_gap = avg_span_coverage = None
        total_close_pairs = series_with_close = 0

    return {
        "n_series": len(series_data),
        "distribution": distribution,
        "suppression_radius": suppression_radius,
        "summary": {
            "avg_events_per_series": round(avg_events, 2),
            "avg_covered_steps": round(avg_covered, 2),
            "avg_overlap_ratio": round(avg_overlap, 2),
            "avg_min_gap": round(avg_min_gap, 1) if avg_min_gap else None,
            "avg_median_gap": round(avg_median_gap, 1) if avg_median_gap else None,
            "total_close_pairs": total_close_pairs,
            "series_with_close_pairs": series_with_close,
            "avg_span_coverage": round(avg_span_coverage, 2) if avg_span_coverage else None,
        },
    }


def format_coverage_histogram_md(histogram: pd.DataFrame, top_n: int = 15) -> str:
    """Format coverage histogram as Markdown table."""
    if histogram is None or histogram.empty:
        return "_No coverage data available._"

    if "all" not in histogram.columns:
        return "_No coverage data available._"

    non_empty = histogram[histogram["all"] > 0].copy()
    if non_empty.empty:
        return "_No event windows to display._"

    # Sort by coverage count descending, show top N
    sorted_hist = non_empty.sort_values("all", ascending=False).head(top_n)
    sorted_hist = sorted_hist.sort_values("step")  # Re-sort chronologically

    total_coverage = int(histogram["all"].sum())
    steps_covered = len(non_empty)

    header = (
        f"_Top {min(top_n, len(sorted_hist))} steps by window coverage "
        f"({total_coverage} step-events across {steps_covered} steps):_\n\n"
    )

    return header + sorted_hist.to_markdown(index=False)


def format_series_diversity_md(diversity: dict) -> str:
    """Format series diversity as Markdown with window overlap and gap metrics."""
    n_series = diversity.get("n_series", 0)
    if n_series == 0:
        return "_No series data._"

    summary = diversity.get("summary", {})
    dist = diversity.get("distribution", {})
    suppression_radius = diversity.get("suppression_radius", 15)

    avg_events = summary.get("avg_events_per_series", 0)
    avg_covered = summary.get("avg_covered_steps", 0)
    avg_overlap = summary.get("avg_overlap_ratio", 0)
    avg_min_gap = summary.get("avg_min_gap")
    avg_median_gap = summary.get("avg_median_gap")
    total_close = summary.get("total_close_pairs", 0)
    series_with_close = summary.get("series_with_close_pairs", 0)
    avg_span_coverage = summary.get("avg_span_coverage")

    lines = [
        f"_Series: {n_series} | "
        f"Avg events: {avg_events} | "
        f"Avg covered steps: {avg_covered} | "
        f"Window overlap: {avg_overlap:.0%}_\n"
    ]

    # Gap and span metrics
    if avg_min_gap is not None:
        span_cov_str = f", span coverage: {avg_span_coverage:.0%}" if avg_span_coverage else ""
        lines.append(
            f"_Peak gaps (raw steps): min={avg_min_gap}, median={avg_median_gap}{span_cov_str} | "
            f"Close pairs (≤{suppression_radius}): {series_with_close}/{n_series} series_\n"
        )

    # Show distribution table
    if dist:
        lines.append("| Events | Window Overlap | Series Count |")
        lines.append("|--------|----------------|--------------|")
        for (n_events, overlap_bucket), count in sorted(dist.items()):
            lines.append(f"| {n_events} | {overlap_bucket} | {count} |")

    # Interpretation hints (corrected)
    lines.append("\n**Interpretation:**")
    if avg_overlap > 0.3:
        lines.append(
            "- High overlap (>30%): Multiple events describe same transition region (over-segmentation)"
        )
    elif avg_overlap < 0.1:
        lines.append(
            "- Low overlap (<10%): Events represent distinct regions "
            "(or suppression already collapsed near-duplicates)"
        )
    else:
        lines.append(
            f"- Moderate overlap ({avg_overlap:.0%}): Some events may share transition regions"
        )

    if avg_min_gap is not None:
        if avg_min_gap <= suppression_radius:
            lines.append(
                f"- Small avg min gap ({avg_min_gap}≤{suppression_radius}): "
                "Series wanted clustered peaks; suppression may have collapsed some candidates"
            )
        else:
            lines.append(
                f"- Large avg min gap ({avg_min_gap}>{suppression_radius}): "
                "Peaks naturally well-separated; suppression had less to collapse"
            )

    if total_close > 0:
        lines.append(
            f"- {total_close} close peak pairs found: These survived suppression "
            "(radius may need tuning, or these are distinct peaks)"
        )

    # Span coverage interpretation
    if avg_span_coverage is not None:
        if avg_span_coverage > 0.7 and avg_overlap < 0.1:
            lines.append(
                f"- High span coverage ({avg_span_coverage:.0%}) + low overlap: "
                "Events tile the span (good distribution)"
            )
        elif avg_span_coverage < 0.3 and avg_overlap < 0.1:
            lines.append(
                f"- Low span coverage ({avg_span_coverage:.0%}) + low overlap: "
                "Few isolated transitions across a wide range"
            )
        elif avg_span_coverage > 0.5 and avg_overlap > 0.3:
            lines.append(
                f"- High span coverage ({avg_span_coverage:.0%}) + high overlap: "
                "Over-segmentation across a broad region"
            )

    return "\n".join(lines)


def compute_wavefront_matrix(
    events: pd.DataFrame,
    metric: str,
    step_grid: list[int],
    n_layers: int | None = None,
    *,
    snap_to_grid: bool = True,
    mode: str = "peak",
) -> pd.DataFrame:
    """Compute layer-by-step event count matrix for a specific metric.

    Creates a matrix showing event presence at each (layer, step) combination,
    useful for visualizing wave propagation patterns.

    Args:
        events: Events DataFrame with 'step', 'layer', and 'metric' columns
        metric: The metric to filter events by
        step_grid: Step values for columns (from geometry_state)
        n_layers: Number of layers (auto-detected if None)
        snap_to_grid: If True, snap event steps to nearest step_grid value.
                      If False, only count events whose step is exactly in step_grid.
        mode: "peak" counts only peak step, "coverage" fills [start_step, end_step] window

    Returns:
        DataFrame with layers as rows (descending) and steps as columns
    """
    if events is None or events.empty or not step_grid:
        return pd.DataFrame()

    required_cols = {"step", "layer", "metric"}
    if not required_cols.issubset(set(events.columns)):
        missing = sorted(list(required_cols - set(events.columns)))
        print(f"[wavefront] Missing required columns: {missing}")
        return pd.DataFrame()

    # Filter to specific metric
    metric_events = events[events["metric"] == metric]
    if metric_events.empty:
        print(
            f"[wavefront] No events for metric '{metric}' - "
            f"available: {events['metric'].unique().tolist()}"
        )
        return pd.DataFrame()

    # Normalize grid to ints
    step_grid_int = [int(s) for s in step_grid]
    step_set = set(step_grid_int)
    grid_spacing = step_grid_int[1] - step_grid_int[0] if len(step_grid_int) > 1 else 1

    # Detect number of layers (prefer metric-specific max)
    if n_layers is None:
        try:
            n_layers = int(metric_events["layer"].max()) + 1
        except Exception:
            # Fall back to global if metric is weird
            n_layers = int(events["layer"].max()) + 1

    # Initialize matrix
    matrix = pd.DataFrame(
        0,
        index=range(n_layers - 1, -1, -1),  # Descending order (upper layers first)
        columns=step_grid_int,
    )
    matrix.index.name = "layer"

    # Count events per (layer, step)
    events_counted = 0
    dropped_not_on_grid = 0
    dropped_bad_layer = 0

    for _, ev in metric_events.iterrows():
        try:
            layer = int(ev["layer"])
            ev_step = int(ev["step"])
        except Exception:
            continue

        if layer not in matrix.index:
            dropped_bad_layer += 1
            continue

        if mode == "coverage":
            # Fill all grid steps in [start_step, end_step] window
            try:
                start = int(ev.get("start_step", ev_step))
                end = int(ev.get("end_step", ev_step))
            except Exception:
                start = end = ev_step
            if end < start:
                start, end = end, start

            # Iterate by grid spacing for efficiency
            for st in range(start, end + 1, grid_spacing):
                if st in step_set and st in matrix.columns:
                    matrix.loc[layer, st] += 1
                    events_counted += 1
        else:
            # Peak mode: only count the peak step
            if snap_to_grid:
                closest_step = _snap_step_to_grid(ev_step, step_grid_int)
            else:
                if ev_step not in step_set:
                    dropped_not_on_grid += 1
                    continue
                closest_step = ev_step

            if closest_step not in matrix.columns:
                dropped_not_on_grid += 1
                continue

            matrix.loc[layer, closest_step] += 1
            events_counted += 1

    # Debug output if matrix is all zeros despite having events
    total_count = int(matrix.values.sum())
    if total_count == 0 and len(metric_events) > 0:
        print(
            f"[wavefront] WARNING: Matrix all zeros for '{metric}' despite "
            f"{len(metric_events)} events. events_counted={events_counted}, "
            f"dropped_bad_layer={dropped_bad_layer}, dropped_not_on_grid={dropped_not_on_grid}"
        )

    return matrix


def format_step_histogram_md(histogram: pd.DataFrame, top_n: int = 15) -> str:
    """Format step histogram as Markdown table, showing top steps by event count."""
    if histogram is None or histogram.empty:
        return "_No histogram data available._"

    # Filter to steps with at least one event
    if "all" not in histogram.columns:
        return "_No histogram data available._"

    non_empty = histogram[histogram["all"] > 0].copy()
    if non_empty.empty:
        return "_No events to display._"

    # Sort by event count descending, show top N
    sorted_hist = non_empty.sort_values("all", ascending=False).head(top_n)

    # Re-sort by step for display (chronological within top N)
    sorted_hist = sorted_hist.sort_values("step")

    total_steps_with_events = len(non_empty)
    total_events = int(histogram["all"].sum())

    header = (
        f"_Top {min(top_n, len(sorted_hist))} steps by event count "
        f"({total_events} events across {total_steps_with_events} steps):_\n\n"
    )

    # Add caveat about potential binning artifacts
    caveat = (
        "\n\n_Note: Apparent clusters may reflect detection parameters "
        "(suppression radius, top-k) or grid snapping, not just true signal._"
    )

    return header + sorted_hist.to_markdown(index=False) + caveat


def _select_interesting_columns(
    wavefront: pd.DataFrame,
    max_cols: int = 21,
    padding: int = 1,
) -> list:
    """Select columns around steps with events (interesting region)."""
    all_cols = list(wavefront.columns)
    if not all_cols:
        return []

    # Find columns with at least one event
    event_cols = [col for col in all_cols if wavefront[col].sum() > 0]
    if not event_cols:
        # No events - return first max_cols
        return all_cols[:max_cols]

    # Build set of interesting columns (event columns + padding)
    col_to_idx = {col: i for i, col in enumerate(all_cols)}
    interesting_idxs: set[int] = set()

    for col in event_cols:
        idx = col_to_idx[col]
        for offset in range(-padding, padding + 1):
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < len(all_cols):
                interesting_idxs.add(neighbor_idx)

    # Sort and limit to max_cols (keep densest region if too many)
    sorted_idxs = sorted(interesting_idxs)

    if len(sorted_idxs) <= max_cols:
        return [all_cols[i] for i in sorted_idxs]

    # Too many columns - find the densest contiguous window of max_cols,
    # where density is the count of "true" event columns (not padded neighbors).
    best_start = 0
    best_count = -1
    event_idx_set = {col_to_idx[c] for c in event_cols}

    for start in range(len(sorted_idxs) - max_cols + 1):
        window_idxs = sorted_idxs[start : start + max_cols]
        count = sum(1 for i in window_idxs if i in event_idx_set)
        if count > best_count:
            best_count = count
            best_start = start

    selected_idxs = sorted_idxs[best_start : best_start + max_cols]
    return [all_cols[i] for i in selected_idxs]


def _select_rows_by_activity(
    wavefront: pd.DataFrame,
    max_rows: int,
    *,
    strategy: str = "top_layers",
) -> pd.DataFrame:
    """Select which layer rows to display.

    strategy:
      - "top_layers": keep highest layers (current behavior)
      - "most_active": keep layers with highest row-sum within the selected columns
    """
    if wavefront is None or wavefront.empty:
        return wavefront

    if strategy == "most_active":
        row_sums = wavefront.sum(axis=1)
        keep = row_sums.sort_values(ascending=False).head(max_rows).index
        # Keep displayed in descending layer order
        keep_sorted = sorted(keep, reverse=True)
        return wavefront.loc[keep_sorted]

    # Default: top layers (descending index already)
    return wavefront.iloc[:max_rows]


def format_wavefront_md(
    wavefront: pd.DataFrame,
    metric: str,
    max_cols: int = 21,
    max_rows: int = 12,
    *,
    row_strategy: str = "most_active",
) -> str:
    """Format wavefront matrix as Markdown table, showing columns around events."""
    if wavefront is None or wavefront.empty:
        return f"_No wavefront data for {metric}._"

    total_rows = len(wavefront)
    total_cols = len(wavefront.columns)

    # Select columns around interesting steps (where events occur)
    selected_cols = _select_interesting_columns(wavefront, max_cols=max_cols, padding=1)
    if not selected_cols:
        return f"_No wavefront data for {metric}._"

    # Filter to selected columns
    view = wavefront[selected_cols].copy()

    # Select rows
    view = _select_rows_by_activity(view, max_rows, strategy=row_strategy)

    # Count events in view
    events_in_view = int(view.values.sum())
    total_events = int(wavefront.values.sum())

    # Add layer column from index
    out = view.reset_index()

    # Build header note
    step_range = f"steps {min(selected_cols)}-{max(selected_cols)}"
    shown_layers = len(view)
    if total_rows > shown_layers or total_cols > len(selected_cols):
        note = (
            f"### {metric} wavefront ({step_range}, "
            f"{shown_layers}/{total_rows} layers, "
            f"{events_in_view}/{total_events} events shown)\n\n"
        )
    else:
        note = f"### {metric} wavefront ({step_range})\n\n"

    return note + out.to_markdown(index=False)


def save_wavefront_artifact(
    wavefront: pd.DataFrame,
    run_id: str,
    metric: str,
) -> Path:
    """Save full wavefront matrix as parquet artifact."""
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
    """Compute summary statistics for composite events."""
    if events is None or events.empty:
        return {
            "total_composites": 0,
            "by_metric_count": {},
            "step_distribution": [],
        }

    if "event_type" not in events.columns:
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
    by_metric_count: dict[str, int] = {}
    for _, ev in composite_events.iterrows():
        try:
            sizes_json = ev.get("metric_sizes_json", "{}")
            if sizes_json and isinstance(sizes_json, str):
                sizes = json.loads(sizes_json)
                n_metrics = len(sizes)
                key = f"{n_metrics}-metric"
                by_metric_count[key] = by_metric_count.get(key, 0) + 1
        except (json.JSONDecodeError, TypeError):
            continue

    # Step distribution (steps where composites occur)
    try:
        step_distribution = sorted([int(s) for s in composite_events["step"].unique().tolist()])
    except Exception:
        step_distribution = sorted(composite_events["step"].unique().tolist())

    # Participating metric combinations
    metric_combinations: list[str] = []
    for _, ev in composite_events.iterrows():
        try:
            sizes_json = ev.get("metric_sizes_json", "{}")
            if sizes_json and isinstance(sizes_json, str):
                sizes = json.loads(sizes_json)
                combo = " + ".join(sorted(sizes.keys()))
                metric_combinations.append(combo)
        except (json.JSONDecodeError, TypeError):
            continue

    # Count combinations
    combo_counts: dict[str, int] = {}
    for combo in metric_combinations:
        combo_counts[combo] = combo_counts.get(combo, 0) + 1

    return {
        "total_composites": total,
        "by_metric_count": by_metric_count,
        "step_distribution": step_distribution,
        "metric_combinations": combo_counts,
    }


def format_composite_section_md(stats: dict) -> str:
    """Format composite event statistics as Markdown section."""
    lines = ["## Composite Events\n"]

    total = stats.get("total_composites", 0)
    if total == 0:
        lines.append("_No composite events detected._")
        return "\n".join(lines)

    lines.append(f"- Total composites: **{total}**")

    # By metric count
    by_count = stats.get("by_metric_count", {})
    if by_count:
        # sort "2-metric", "3-metric", ...
        def _k(x: str) -> int:
            try:
                return int(x.split("-", 1)[0])
            except Exception:
                return 10**9

        count_str = ", ".join(f"{k}: {v}" for k, v in sorted(by_count.items(), key=lambda kv: _k(kv[0])))
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


def load_detection_summary(run_id: str) -> pd.DataFrame | None:
    """Load detection summary parquet if it exists."""
    summary_path = paths.detection_summary_path(run_id)
    if summary_path.exists():
        return pd.read_parquet(summary_path)
    return None


def format_retention_summary_md(summary: pd.DataFrame | None) -> str:
    """Format detection retention summary as Markdown section."""
    if summary is None or summary.empty:
        return "_No detection summary available (run analysis with --force to generate)._"

    n_series = len(summary)
    lines = ["## Detection Retention Summary\n"]
    lines.append(
        "_True retention: how many raw candidates (above threshold) survived suppression + top-k._\n"
    )

    # Overall stats (single-metric events only; composites are derived separately)
    avg_retention = summary["retention_ratio"].mean()
    avg_suppression_skip = summary["suppression_skip_ratio"].mean()
    avg_topk_skip = summary["topk_skip_ratio"].mean()
    total_candidates = int(summary["n_candidates_raw"].sum())
    total_selected = int(summary["n_selected_final"].sum())
    total_skipped_supp = int(summary["n_skipped_suppression"].sum())
    total_skipped_topk = int(summary["n_skipped_topk"].sum())

    lines.append(f"**Overall (single-metric):** {total_selected}/{total_candidates} candidates "
                 f"retained ({avg_retention:.0%} retention)")

    # Pre/post warmup breakdown with retention RATES and skip breakdowns
    if "n_candidates_pre" in summary.columns:
        total_pre = int(summary["n_candidates_pre"].sum())
        total_post = int(summary["n_candidates_post"].sum())
        selected_pre = int(summary["n_selected_pre"].sum())
        selected_post = int(summary["n_selected_post"].sum())
        pre_rate = selected_pre / max(total_pre, 1)
        post_rate = selected_post / max(total_post, 1)
        total_prewarm_cap = int(summary["n_skipped_pre_warmup_cap"].sum())

        # Pre/post breakdown for suppression and topk
        supp_pre = int(summary.get("n_skipped_suppression_pre", pd.Series([0])).sum())
        supp_post = int(summary.get("n_skipped_suppression_post", pd.Series([0])).sum())
        topk_pre = int(summary.get("n_skipped_topk_pre", pd.Series([0])).sum())
        topk_post = int(summary.get("n_skipped_topk_post", pd.Series([0])).sum())

        if total_pre > 0 or total_post > 0:
            lines.append(f"- Pre-warmup: {selected_pre}/{total_pre} retained ({pre_rate:.0%})")
            lines.append(f"- Post-warmup: {selected_post}/{total_post} retained ({post_rate:.0%})")
            lines.append(f"- Suppression pre/post: {supp_pre} / {supp_post}")
            lines.append(f"- Top-k pre/post: {topk_pre} / {topk_post}")
            if total_prewarm_cap > 0:
                lines.append(f"- Pre-warmup cap skips: {total_prewarm_cap}")

            # Per-phase mean candidates per series (quantifies pre/post story)
            mean_raw_pre = total_pre / n_series
            mean_raw_post = total_post / n_series
            mean_sel_pre = selected_pre / n_series
            mean_sel_post = selected_post / n_series
            lines.append(
                f"- Mean raw/series: pre={mean_raw_pre:.1f}, post={mean_raw_post:.1f}"
            )
            lines.append(
                f"- Mean selected/series: pre={mean_sel_pre:.1f}, post={mean_sel_post:.1f}"
            )

    lines.append("")

    # Distribution of retention ratios
    lines.append("**Retention distribution:**\n")
    lines.append("| Retention | Series Count |")
    lines.append("|-----------|--------------|")

    def _bucket(r: float) -> str:
        if r >= 1.0:
            return "100%"
        elif r >= 0.8:
            return "80-99%"
        elif r >= 0.5:
            return "50-79%"
        elif r >= 0.2:
            return "20-49%"
        else:
            return "<20%"

    buckets: dict[str, int] = {}
    for r in summary["retention_ratio"]:
        b = _bucket(r)
        buckets[b] = buckets.get(b, 0) + 1

    # Show all bins for easy cross-run comparison (including zeros)
    for bucket in ["100%", "80-99%", "50-79%", "20-49%", "<20%"]:
        count = buckets.get(bucket, 0)
        lines.append(f"| {bucket} | {count} |")

    # Top 5 worst retention series with full skip breakdown
    lines.append("\n**Lowest retention series (most collapsed):**\n")
    worst = summary.nsmallest(5, "retention_ratio")
    if not worst.empty:
        lines.append("| Layer | Metric | Raw | Final | Retention | Supp | Pre-cap | Top-k |")
        lines.append("|-------|--------|-----|-------|-----------|------|---------|-------|")
        for _, row in worst.iterrows():
            layer = int(row["layer"])
            metric = row["metric"]
            raw = int(row["n_candidates_raw"])
            final = int(row["n_selected_final"])
            ret = row["retention_ratio"]
            supp = int(row["n_skipped_suppression"])
            precap = int(row.get("n_skipped_pre_warmup_cap", 0))
            topk = int(row["n_skipped_topk"])
            lines.append(f"| {layer} | {metric} | {raw} | {final} | {ret:.0%} | {supp} | {precap} | {topk} |")

    # Interpretation based on skip patterns
    interpretations = []

    # Get pre/post breakdown for skip interpretation
    supp_pre = int(summary.get("n_skipped_suppression_pre", pd.Series([0])).sum())
    supp_post = int(summary.get("n_skipped_suppression_post", pd.Series([0])).sum())
    topk_pre = int(summary.get("n_skipped_topk_pre", pd.Series([0])).sum())
    topk_post = int(summary.get("n_skipped_topk_post", pd.Series([0])).sum())
    total_supp = supp_pre + supp_post
    total_topk = topk_pre + topk_post

    # Suppression interpretation with pre/post breakdown
    if total_supp > 0:
        if supp_post > supp_pre * 2:
            interpretations.append(
                f"- Suppression mostly post-warmup ({supp_post}/{total_supp}): "
                "post-warmup candidates cluster heavily (local bursts)."
            )
        elif supp_pre > supp_post * 2:
            interpretations.append(
                f"- Suppression mostly pre-warmup ({supp_pre}/{total_supp}): "
                "early transients cluster heavily."
            )
        elif avg_suppression_skip > 0.2:
            interpretations.append(
                f"- Suppression ({avg_suppression_skip:.0%}): "
                f"Clustered candidates (pre/post: {supp_pre}/{supp_post})."
            )

    # Top-k interpretation with pre/post breakdown
    if total_topk > 0:
        if topk_post > topk_pre * 2:
            interpretations.append(
                f"- Top-k mostly post-warmup ({topk_post}/{total_topk}): "
                "many post-warmup candidates exceed budget."
            )
        elif topk_pre > topk_post * 2:
            interpretations.append(
                f"- Top-k mostly pre-warmup ({topk_pre}/{total_topk}): "
                "early candidates compete for limited budget."
            )

    # Pre-cap interpretation
    if "n_skipped_pre_warmup_cap" in summary.columns:
        total_prewarm_cap = int(summary["n_skipped_pre_warmup_cap"].sum())
        total_pre = int(summary["n_candidates_pre"].sum())
        if total_pre > 0 and total_prewarm_cap > 0:
            prewarm_cap_rate = total_prewarm_cap / total_pre
            if prewarm_cap_rate > 0.3:
                interpretations.append(
                    f"- Pre-warmup cap ({prewarm_cap_rate:.0%} of pre-warmup): "
                    "Many early candidates capped - warmup region is noisy."
                )

    # Pre vs post comparison
    if "n_candidates_pre" in summary.columns:
        total_pre = int(summary["n_candidates_pre"].sum())
        total_post = int(summary["n_candidates_post"].sum())
        selected_pre = int(summary["n_selected_pre"].sum())
        selected_post = int(summary["n_selected_post"].sum())
        pre_rate = selected_pre / max(total_pre, 1)
        post_rate = selected_post / max(total_post, 1)
        if total_pre > 0 and total_post > 0 and abs(pre_rate - post_rate) > 0.15:
            if pre_rate < post_rate:
                interpretations.append(
                    f"- Pre vs post: Pre-warmup retention ({pre_rate:.0%}) << post ({post_rate:.0%}) "
                    "- warmup is noisier than main training."
                )
            else:
                interpretations.append(
                    f"- Pre vs post: Pre-warmup retention ({pre_rate:.0%}) >> post ({post_rate:.0%}) "
                    "- early training has clearer signals."
                )

    if interpretations:
        lines.append("\n**Interpretation:**")
        lines.extend(interpretations)
    else:
        lines.append("\n_Retention patterns within normal ranges._")

    return "\n".join(lines)


def generate_event_diagnostics(
    events: pd.DataFrame,
    geom: pd.DataFrame,
    run_id: str,
    save_artifacts: bool = True,
    *,
    snap_to_grid: bool = True,
    row_strategy: str = "top_layers",
    max_metrics: int = 3,
) -> str:
    """Generate complete event diagnostics section for report.

    Args:
        events: Events DataFrame
        geom: Geometry state DataFrame
        run_id: The run ID
        save_artifacts: Whether to save wavefront parquets
        snap_to_grid: Snap event steps to nearest capture steps
        row_strategy: "top_layers" or "most_active"
        max_metrics: Limit wavefront rendering to first N metrics (sorted)

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

    # Step histogram (peak steps only)
    histogram = compute_step_histogram(events, step_grid, snap_to_grid=snap_to_grid)
    lines.append("### Event Step Histogram (Peak Steps)\n")
    lines.append(format_step_histogram_md(histogram))
    lines.append("")

    # Coverage histogram (all steps in event windows)
    coverage = compute_window_coverage_histogram(events, step_grid)
    lines.append("\n### Event Window Coverage\n")
    lines.append(
        "_Counts each grid step within [start_step, end_step] windows. "
        "If coverage is smoother than peaks, clustering may be a reporting artifact._\n"
    )
    lines.append(format_coverage_histogram_md(coverage))
    lines.append("")

    # Series diversity (detect window overlap)
    diversity = compute_series_diversity(events, step_grid=step_grid)
    lines.append("\n### Per-Series Window Overlap\n")
    lines.append(format_series_diversity_md(diversity))
    lines.append("")

    # Wavefront views per metric (both peak and coverage modes)
    if events is not None and not events.empty:
        metrics = sorted([m for m in events["metric"].unique() if m != "__composite__"])
        # n_layers: infer from events if possible
        n_layers = int(events["layer"].max()) + 1 if "layer" in events.columns else None

        artifact_paths: list[str] = []

        # Show coverage wavefront first (smoother, shows transition regions)
        lines.append("\n### Coverage Wavefronts (Window Fill)\n")
        lines.append(
            "_Shows all steps within event windows - reveals transition regions "
            "rather than discrete peaks._\n"
        )
        for metric in metrics[:max_metrics]:
            wavefront_cov = compute_wavefront_matrix(
                events,
                metric,
                step_grid,
                n_layers=n_layers,
                snap_to_grid=snap_to_grid,
                mode="coverage",
            )
            if not wavefront_cov.empty:
                lines.append("")
                lines.append(
                    format_wavefront_md(
                        wavefront_cov,
                        f"{metric} (coverage)",
                        row_strategy=row_strategy,
                    )
                )

        # Then show peak wavefront (traditional view)
        lines.append("\n### Peak Wavefronts (Peak Step Only)\n")
        lines.append("_Shows only peak step per event - may overstate discrete fronts._\n")
        for metric in metrics[:max_metrics]:
            wavefront_peak = compute_wavefront_matrix(
                events,
                metric,
                step_grid,
                n_layers=n_layers,
                snap_to_grid=snap_to_grid,
                mode="peak",
            )
            if not wavefront_peak.empty:
                lines.append("")
                lines.append(
                    format_wavefront_md(
                        wavefront_peak,
                        f"{metric} (peak)",
                        row_strategy=row_strategy,
                    )
                )

                if save_artifacts:
                    artifact_path = save_wavefront_artifact(wavefront_peak, run_id, metric)
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

    # Detection retention summary (true retention from candidates)
    detection_summary = load_detection_summary(run_id)
    if detection_summary is not None and not detection_summary.empty:
        lines.append("\n")
        lines.append(format_retention_summary_md(detection_summary))

    return "\n".join(lines)
