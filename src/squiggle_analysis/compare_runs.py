"""Multi-run comparison for seed invariance testing.

Compares training runs with different seeds to identify events that appear
consistently across runs, indicating genuine learning dynamics rather than
random fluctuations.

Usage:
    python -m squiggle_analysis.compare_runs RUN_ID1 RUN_ID2 [RUN_ID3 ...]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from squiggle_core import paths

from .metrics.event_diversity import EventDiversityMetrics, compute_event_diversity
from .trajectories import extract_metric_trajectories, compare_run_trajectories


@dataclass
class RunData:
    """Container for all analysis data from a single run."""

    run_id: str
    geometry_df: pd.DataFrame
    events_df: pd.DataFrame
    scalars_df: Optional[pd.DataFrame]
    meta: Dict
    diversity: EventDiversityMetrics


def load_run_data(run_id: str) -> RunData:
    """
    Load all relevant analysis data for a single run.

    Args:
        run_id: The run identifier

    Returns:
        RunData with geometry, events, scalars, metadata, and diversity metrics
    """
    # Load geometry state
    geometry_path = paths.geometry_state_path(run_id)
    if not geometry_path.exists():
        raise FileNotFoundError(f"Geometry state not found for {run_id}: {geometry_path}")
    geometry_df = pd.read_parquet(geometry_path)

    # Load events candidates
    events_path = paths.events_candidates_path(run_id)
    if not events_path.exists():
        raise FileNotFoundError(f"Events candidates not found for {run_id}: {events_path}")
    events_df = pd.read_parquet(events_path)

    # Load scalars (optional)
    scalars_path = paths.metrics_scalar_path(run_id)
    scalars_df = None
    if scalars_path.exists():
        scalars_df = pd.read_parquet(scalars_path)

    # Load metadata
    meta_path = paths.run_dir(run_id) / "meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    # Compute diversity metrics
    diversity = compute_event_diversity(events_df)

    return RunData(
        run_id=run_id,
        geometry_df=geometry_df,
        events_df=events_df,
        scalars_df=scalars_df,
        meta=meta,
        diversity=diversity,
    )


def _find_matching_event(
    events_df: pd.DataFrame,
    signature: tuple,
    target_step: int,
    tolerance: int,
) -> Optional[pd.Series]:
    """
    Find an event matching the signature within step tolerance.

    Args:
        events_df: Events DataFrame to search
        signature: (layer, metric, event_type) tuple
        target_step: Step to match
        tolerance: Maximum step difference

    Returns:
        Matching event row or None
    """
    layer, metric, event_type = signature

    matches = events_df[
        (events_df["layer"] == layer)
        & (events_df["metric"] == metric)
        & (events_df["event_type"] == event_type)
        & (abs(events_df["step"] - target_step) <= tolerance)
    ]

    if matches.empty:
        return None

    # Return the closest match
    closest_idx = (abs(matches["step"] - target_step)).idxmin()
    return matches.loc[closest_idx]


def find_common_events(
    runs: List[RunData],
    step_tolerance: int = 5,
) -> pd.DataFrame:
    """
    Find events that appear in ALL runs at similar steps.

    Args:
        runs: List of RunData objects to compare
        step_tolerance: Maximum step difference for matching events

    Returns:
        DataFrame with common events and their statistics
    """
    if len(runs) < 2:
        raise ValueError("Need at least 2 runs to compare")

    reference_run = runs[0]
    results = []
    seen_signatures = set()

    for _, event in reference_run.events_df.iterrows():
        sig = (event["layer"], event["metric"], event["event_type"])

        # Skip duplicates (same signature, different step in reference run)
        if sig in seen_signatures:
            continue

        matches = [event]

        for other_run in runs[1:]:
            match = _find_matching_event(
                other_run.events_df, sig, event["step"], step_tolerance
            )
            if match is not None:
                matches.append(match)
            else:
                break  # Event not in this run

        if len(matches) == len(runs):
            # Event appears in ALL runs
            seen_signatures.add(sig)
            steps = [m["step"] for m in matches]
            scores = [m["score"] for m in matches]

            results.append({
                "layer": sig[0],
                "metric": sig[1],
                "event_type": sig[2],
                "mean_step": np.mean(steps),
                "std_step": np.std(steps),
                "min_step": min(steps),
                "max_step": max(steps),
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "n_runs": len(runs),
            })

    return pd.DataFrame(results)


def compute_trajectory_correlation(
    runs: List[RunData],
    metric: str = "effective_rank",
    layers: List[int] = [0, 12, 23],
) -> pd.DataFrame:
    """
    Compute pairwise correlation of metric trajectories across runs.

    Args:
        runs: List of RunData objects
        metric: Geometry metric to compare
        layers: Layers to compute correlation for

    Returns:
        Correlation matrix (run_id x run_id) averaged across layers
    """
    n_runs = len(runs)
    run_ids = [r.run_id for r in runs]

    # Extract trajectories for each run
    trajectories = {}
    for run in runs:
        traj = extract_metric_trajectories(run.geometry_df, layers=layers, metric=metric)
        # Pivot to wide format: step as index, layer columns
        pivoted = traj.pivot(index="step", columns="layer", values="value")
        trajectories[run.run_id] = pivoted

    # Compute pairwise correlations
    correlations = np.zeros((n_runs, n_runs))

    for i, run_i in enumerate(runs):
        for j, run_j in enumerate(runs):
            if i == j:
                correlations[i, j] = 1.0
            elif i < j:
                traj_i = trajectories[run_i.run_id]
                traj_j = trajectories[run_j.run_id]

                # Align on common steps
                common_steps = traj_i.index.intersection(traj_j.index)
                if len(common_steps) < 3:
                    correlations[i, j] = np.nan
                    correlations[j, i] = np.nan
                    continue

                # Compute correlation for each layer, then average
                layer_corrs = []
                for layer in layers:
                    if layer in traj_i.columns and layer in traj_j.columns:
                        vals_i = traj_i.loc[common_steps, layer].values
                        vals_j = traj_j.loc[common_steps, layer].values
                        if len(vals_i) > 2 and np.std(vals_i) > 0 and np.std(vals_j) > 0:
                            corr = np.corrcoef(vals_i, vals_j)[0, 1]
                            layer_corrs.append(corr)

                if layer_corrs:
                    avg_corr = np.mean(layer_corrs)
                    correlations[i, j] = avg_corr
                    correlations[j, i] = avg_corr
                else:
                    correlations[i, j] = np.nan
                    correlations[j, i] = np.nan

    return pd.DataFrame(correlations, index=run_ids, columns=run_ids)


def analyze_event_phases(run: RunData) -> Dict[str, any]:
    """
    Analyze event distribution across training phases.

    Uses event_phase column if available (shaping/transition/locking),
    otherwise falls back to LR-based phases (warmup/high_lr/floor).

    Returns dict with:
    - phase_counts: events per phase
    - boundary_events: events within 5 steps of phase boundaries
    - schedule_artifact_warning: True if most events at boundaries
    """
    meta = run.meta
    events_df = run.events_df

    # Get phase boundaries from meta (if available)
    phase_boundaries = meta.get("phase_boundaries", {})
    warmup_end = phase_boundaries.get("warmup_end", 0)
    decay_start = phase_boundaries.get("decay_start", 0)
    floor_start = phase_boundaries.get("floor_start", meta.get("steps", 10000))
    total_steps = phase_boundaries.get("total_steps", meta.get("steps", 10000))

    # Also check for epoch boundaries
    steps_per_epoch = meta.get("steps_per_epoch", 0)
    epochs = meta.get("epochs", 1)

    if events_df.empty:
        return {
            "phase_counts": {"warmup": 0, "high_lr": 0, "floor": 0},
            "boundary_events": 0,
            "schedule_artifact_warning": False,
            "phase_boundaries": phase_boundaries,
        }

    events_df = events_df.copy()

    # Use new event_phase column if available, otherwise compute from step
    if "event_phase" in events_df.columns and events_df["event_phase"].notna().any():
        # Map event_phase to display format
        phase_counts = events_df["event_phase"].value_counts().to_dict()
    else:
        # Fallback: classify events into LR-based phases
        def get_phase(step: int) -> str:
            if step < warmup_end:
                return "warmup"
            elif step < floor_start:
                return "high_lr"
            else:
                return "floor"

        events_df["phase"] = events_df["step"].apply(get_phase)
        phase_counts = events_df["phase"].value_counts().to_dict()

    # Check for boundary events (within 5 steps of any boundary)
    boundary_tolerance = 5
    boundaries = [warmup_end, floor_start]
    if steps_per_epoch > 0:
        # Add epoch boundaries
        for e in range(1, epochs + 1):
            boundaries.append(e * steps_per_epoch)

    boundary_events = 0
    for _, event in events_df.iterrows():
        step = event["step"]
        for boundary in boundaries:
            if abs(step - boundary) <= boundary_tolerance:
                boundary_events += 1
                break

    # Warning if >50% of events are at boundaries
    total_events = len(events_df)
    schedule_artifact_warning = (boundary_events / max(1, total_events)) > 0.5

    return {
        "phase_counts": phase_counts,
        "boundary_events": boundary_events,
        "total_events": total_events,
        "boundary_fraction": boundary_events / max(1, total_events),
        "schedule_artifact_warning": schedule_artifact_warning,
        "phase_boundaries": phase_boundaries,
    }


def _extract_seed_from_run_id(run_id: str, meta: Dict) -> str:
    """Extract seed from run_id or metadata."""
    # Try metadata first
    if "seed" in meta:
        return str(meta["seed"])

    # Try parsing from run_id (format: YYYYMMDD_HHMMSS_name_sXXX)
    parts = run_id.split("_")
    for part in parts:
        if part.startswith("s") and part[1:].isdigit():
            return part[1:]

    return "?"


def _get_final_loss(scalars_df: Optional[pd.DataFrame]) -> str:
    """Get final training loss from scalars."""
    if scalars_df is None:
        return "N/A"

    # Handle wide format (columns: step, loss, lr, etc.)
    if "loss" in scalars_df.columns:
        final_row = scalars_df.loc[scalars_df["step"].idxmax()]
        return f"{final_row['loss']:.4f}"

    # Handle long format (columns: metric_name, value, step)
    if "metric_name" in scalars_df.columns:
        loss_df = scalars_df[scalars_df["metric_name"] == "train/loss"]
        if loss_df.empty:
            return "N/A"
        final_loss = loss_df.loc[loss_df["step"].idxmax(), "value"]
        return f"{final_loss:.4f}"

    return "N/A"


def generate_comparison_report(
    run_ids: List[str],
    output_path: Optional[Path] = None,
    step_tolerance: int = 5,
    generate_plots: bool = True,
    plots_dir: Optional[Path] = None,
    layers: List[int] = [0, 12, 23],
) -> str:
    """
    Generate a markdown comparison report for multiple runs.

    Args:
        run_ids: List of run IDs to compare
        output_path: If provided, write report to this file
        step_tolerance: Step tolerance for common event detection
        generate_plots: Whether to generate trajectory plots
        plots_dir: Directory for plots (default: same dir as output_path or cwd)
        layers: Layers to include in trajectory plots

    Returns:
        Markdown report string
    """
    # Load all runs
    runs = [load_run_data(run_id) for run_id in run_ids]

    # Find common events
    common_events = find_common_events(runs, step_tolerance=step_tolerance)

    # Compute trajectory correlation
    corr_matrix = compute_trajectory_correlation(runs, layers=layers)

    # Generate trajectory plots
    plot_paths: Dict[str, Path] = {}
    if generate_plots:
        if plots_dir is None:
            if output_path:
                plots_dir = Path(output_path).parent / "plots"
            else:
                plots_dir = Path.cwd() / "comparison_plots"

        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Build geometry dict for compare_run_trajectories
        run_geometry_dfs = {run.run_id: run.geometry_df for run in runs}

        for metric in ["effective_rank", "sv_entropy"]:
            compare_run_trajectories(
                run_geometry_dfs,
                layers=layers,
                metric=metric,
                output_dir=plots_dir,
            )
            # Track plot paths
            for layer in layers:
                plot_key = f"{metric}_layer_{layer}"
                plot_paths[plot_key] = plots_dir / f"{metric}_layer_{layer}_comparison.png"

    # Build report
    lines = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines.append("# Seed Invariance Comparison Report")
    lines.append("")
    lines.append(f"**Runs compared:** {len(runs)}")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Step tolerance:** {step_tolerance}")
    lines.append("")

    # Run Summary Table
    lines.append("## Run Summary")
    lines.append("")
    lines.append("| Run ID | Seed | Final Loss | Events | Diversity |")
    lines.append("|--------|------|------------|--------|-----------|")

    for run in runs:
        seed = _extract_seed_from_run_id(run.run_id, run.meta)
        final_loss = _get_final_loss(run.scalars_df)
        n_events = len(run.events_df)
        diversity = run.diversity.diversity_score

        # Truncate run_id for display
        short_id = run.run_id[:20] + "..." if len(run.run_id) > 23 else run.run_id

        lines.append(f"| {short_id} | {seed} | {final_loss} | {n_events} | {diversity:.3f} |")

    lines.append("")

    # Common Events Section
    lines.append("## Common Events (Seed-Invariant)")
    lines.append("")

    if common_events.empty:
        lines.append("No common events found across all runs.")
    else:
        lines.append(f"Found **{len(common_events)}** events appearing in all {len(runs)} runs.")
        lines.append("")
        lines.append("| Layer | Metric | Type | Mean Step | Step std | Mean Score |")
        lines.append("|-------|--------|------|-----------|----------|------------|")

        # Sort by score
        sorted_events = common_events.sort_values("mean_score", ascending=False)

        for _, row in sorted_events.head(30).iterrows():
            lines.append(
                f"| {row['layer']} | {row['metric']} | {row['event_type']} | "
                f"{row['mean_step']:.1f} | {row['std_step']:.2f} | {row['mean_score']:.3f} |"
            )

    lines.append("")

    # Trajectory Correlation Section
    lines.append("## Trajectory Correlation (effective_rank)")
    lines.append("")

    if corr_matrix.isna().all().all():
        lines.append("Insufficient data to compute trajectory correlation.")
    else:
        # Build correlation table
        header = "|  | " + " | ".join(
            [r[:8] + "..." if len(r) > 11 else r for r in corr_matrix.columns]
        ) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(corr_matrix.columns) + 1))

        for idx, row in corr_matrix.iterrows():
            short_idx = idx[:8] + "..." if len(idx) > 11 else idx
            vals = " | ".join([f"{v:.2f}" if not np.isnan(v) else "N/A" for v in row])
            lines.append(f"| {short_idx} | {vals} |")

        lines.append("")

        # Compute mean correlation (excluding diagonal)
        mask = ~np.eye(len(corr_matrix), dtype=bool)
        off_diag = corr_matrix.values[mask]
        mean_corr = np.nanmean(off_diag)
        lines.append(f"**Mean pairwise correlation:** {mean_corr:.3f}")

        if mean_corr > 0.95:
            lines.append("")
            lines.append("High correlation (>0.95) indicates consistent learning dynamics.")

    lines.append("")

    # Trajectory Plots Section
    if generate_plots and plot_paths:
        lines.append("## Trajectory Plots")
        lines.append("")
        lines.append("Overlay plots showing metric trajectories across all runs.")
        lines.append("")

        for metric in ["effective_rank", "sv_entropy"]:
            lines.append(f"### {metric.replace('_', ' ').title()}")
            lines.append("")
            for layer in layers:
                plot_key = f"{metric}_layer_{layer}"
                if plot_key in plot_paths:
                    plot_file = plot_paths[plot_key]
                    # Use relative path if output_path is set
                    if output_path:
                        try:
                            rel_path = plot_file.relative_to(Path(output_path).parent)
                        except ValueError:
                            rel_path = plot_file
                    else:
                        rel_path = plot_file
                    lines.append(f"**Layer {layer}:**")
                    lines.append(f"![{metric} Layer {layer}]({rel_path})")
                    lines.append("")

    lines.append("")

    # Diversity Comparison Section
    lines.append("## Diversity Comparison")
    lines.append("")

    diversity_scores = [run.diversity.diversity_score for run in runs]
    mean_div = np.mean(diversity_scores)
    std_div = np.std(diversity_scores)

    lines.append(f"**Mean diversity:** {mean_div:.3f}")
    lines.append(f"**Diversity std:** {std_div:.3f}")
    lines.append("")

    if std_div < 0.1:
        lines.append("Consistent diversity scores across runs.")
    else:
        lines.append("Note: High variance in diversity scores - runs may have different event patterns.")

    lines.append("")

    # Phase Analysis Section
    lines.append("## Event Phase Analysis")
    lines.append("")

    phase_analyses = [analyze_event_phases(run) for run in runs]
    any_schedule_warning = any(pa["schedule_artifact_warning"] for pa in phase_analyses)

    # Detect which phase naming convention is used
    first_phase_counts = phase_analyses[0]["phase_counts"]
    use_new_phases = "shaping" in first_phase_counts or "locking" in first_phase_counts

    if use_new_phases:
        lines.append("Events classified by training phase (shaping, transition, locking).")
        lines.append("")
        lines.append("| Run | Shaping | Transition | Locking | Boundary Events | Warning |")
        lines.append("|-----|---------|------------|---------|-----------------|---------|")

        for run, pa in zip(runs, phase_analyses):
            short_id = run.run_id[:15] + "..." if len(run.run_id) > 18 else run.run_id
            shaping_count = pa["phase_counts"].get("shaping", 0)
            transition_count = pa["phase_counts"].get("transition", 0)
            locking_count = pa["phase_counts"].get("locking", 0)
            boundary_pct = pa["boundary_fraction"] * 100
            warning = "Yes" if pa["schedule_artifact_warning"] else "-"
            lines.append(f"| {short_id} | {shaping_count} | {transition_count} | {locking_count} | {boundary_pct:.0f}% | {warning} |")
    else:
        lines.append("Events classified by training phase (warmup, high_lr, floor).")
        lines.append("")
        lines.append("| Run | Warmup | High LR | Floor | Boundary Events | Warning |")
        lines.append("|-----|--------|---------|-------|-----------------|---------|")

        for run, pa in zip(runs, phase_analyses):
            short_id = run.run_id[:15] + "..." if len(run.run_id) > 18 else run.run_id
            warmup_count = pa["phase_counts"].get("warmup", 0)
            high_lr_count = pa["phase_counts"].get("high_lr", 0)
            floor_count = pa["phase_counts"].get("floor", 0)
            boundary_pct = pa["boundary_fraction"] * 100
            warning = "Yes" if pa["schedule_artifact_warning"] else "-"
            lines.append(f"| {short_id} | {warmup_count} | {high_lr_count} | {floor_count} | {boundary_pct:.0f}% | {warning} |")

    lines.append("")

    if any_schedule_warning:
        lines.append("**Warning:** >50% of events are at phase boundaries - likely schedule artifacts.")
        lines.append("")
        lines.append("Consider:")
        lines.append("- Running with constant LR to isolate data-driven events")
        lines.append("- Filtering events near warmup_end and LR floor transitions")
    else:
        lines.append("Events are distributed across phases - good evidence of data-driven dynamics.")

    lines.append("")

    # Conclusion Section
    lines.append("## Conclusion")
    lines.append("")

    n_common = len(common_events)
    mean_corr_val = np.nanmean(corr_matrix.values[~np.eye(len(corr_matrix), dtype=bool)])

    if n_common > 0:
        lines.append(f"- **{n_common} common events** detected across all seeds")
    else:
        lines.append("- **No common events** - seed invariance not established")

    if not np.isnan(mean_corr_val):
        if mean_corr_val > 0.95:
            lines.append(f"- **High trajectory correlation** (mean: {mean_corr_val:.2f})")
        elif mean_corr_val > 0.8:
            lines.append(f"- **Moderate trajectory correlation** (mean: {mean_corr_val:.2f})")
        else:
            lines.append(f"- **Low trajectory correlation** (mean: {mean_corr_val:.2f})")

    if std_div < 0.1:
        lines.append(f"- **Consistent diversity scores** (std: {std_div:.2f})")

    lines.append("")

    if n_common > 5 and mean_corr_val > 0.9:
        lines.append("The runs show **strong seed invariance**.")
    elif n_common > 0 or mean_corr_val > 0.8:
        lines.append("The runs show **moderate seed invariance**.")
    else:
        lines.append("The runs show **weak seed invariance** - investigate differences.")

    lines.append("")

    report = "\n".join(lines)

    # Write to file if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report written to: {output_path}")

    return report


def main():
    """CLI entry point for compare_runs."""
    parser = argparse.ArgumentParser(
        description="Compare multiple training runs for seed invariance"
    )
    parser.add_argument(
        "run_ids",
        nargs="+",
        help="Run IDs to compare (at least 2)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path for report (default: print to stdout)",
    )
    parser.add_argument(
        "--step-tolerance", "-t",
        type=int,
        default=5,
        help="Step tolerance for matching events (default: 5)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip trajectory plot generation",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory for trajectory plots (default: plots/ next to output)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 12, 23],
        help="Layers to include in trajectory plots (default: 0 12 23)",
    )

    args = parser.parse_args()

    if len(args.run_ids) < 2:
        parser.error("Need at least 2 run IDs to compare")

    report = generate_comparison_report(
        args.run_ids,
        output_path=args.output,
        step_tolerance=args.step_tolerance,
        generate_plots=not args.no_plots,
        plots_dir=args.plots_dir,
        layers=args.layers,
    )

    if args.output is None:
        print(report)


if __name__ == "__main__":
    main()
