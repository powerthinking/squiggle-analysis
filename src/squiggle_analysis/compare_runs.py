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
class DetectionConfig:
    """Detection parameters extracted from detection_summary."""

    adaptive_k: Optional[float] = None
    suppression_radius: Optional[int] = None
    max_peaks: Optional[int] = None
    warmup_end_step: Optional[int] = None
    max_pre_warmup: Optional[int] = None

    def fingerprint(self) -> str:
        """Short fingerprint string for display."""
        parts = []
        if self.adaptive_k is not None:
            parts.append(f"k={self.adaptive_k}")
        if self.suppression_radius is not None:
            parts.append(f"supp={self.suppression_radius}")
        if self.max_peaks is not None:
            parts.append(f"max={self.max_peaks}")
        if self.warmup_end_step is not None:
            parts.append(f"warmup={self.warmup_end_step}")
        if self.max_pre_warmup is not None:
            parts.append(f"pre={self.max_pre_warmup}")
        return ", ".join(parts) if parts else "unknown"

    def config_hash(self) -> str:
        """Short hash of config for quick comparison."""
        import hashlib
        s = f"{self.adaptive_k}:{self.suppression_radius}:{self.max_peaks}:{self.warmup_end_step}:{self.max_pre_warmup}"
        return hashlib.md5(s.encode()).hexdigest()[:8]


@dataclass
class RetentionMetrics:
    """Retention metrics from detection summary."""

    n_candidates: int = 0
    n_selected: int = 0
    retention_rate: float = 0.0
    pre_retention_rate: float = 0.0
    post_retention_rate: float = 0.0
    suppression_rate: float = 0.0
    suppression_pre: int = 0
    suppression_post: int = 0
    topk_post: int = 0
    pre_warmup_cap: int = 0
    # Per-series density metrics
    mean_candidates_per_series: float = 0.0
    mean_candidates_post_per_series: float = 0.0


@dataclass
class RunData:
    """Container for all analysis data from a single run."""

    run_id: str
    analysis_id: Optional[str]
    geometry_df: pd.DataFrame
    events_df: pd.DataFrame
    scalars_df: Optional[pd.DataFrame]
    meta: Dict
    diversity: EventDiversityMetrics
    retention: Optional[RetentionMetrics] = None
    detection_config: Optional[DetectionConfig] = None


def load_run_data(run_id: str, analysis_id: Optional[str] = None) -> RunData:
    """
    Load all relevant analysis data for a single run.

    Args:
        run_id: The run identifier
        analysis_id: Optional analysis version. If provided, loads versioned events/detection_summary.
                     If None, tries to find the latest analysis or falls back to legacy paths.

    Returns:
        RunData with geometry, events, scalars, metadata, and diversity metrics
    """
    # Load geometry state (not versioned by analysis_id)
    geometry_path = paths.geometry_state_path(run_id)
    if not geometry_path.exists():
        raise FileNotFoundError(f"Geometry state not found for {run_id}: {geometry_path}")
    geometry_df = pd.read_parquet(geometry_path)

    # Determine analysis_id to use
    used_analysis_id = analysis_id
    if analysis_id is None:
        # Try to find available analysis versions
        available = paths.list_analysis_ids(run_id)
        if available:
            # Use the most recent (last alphabetically, which works for w10_p1... format)
            used_analysis_id = available[-1]
            print(f"[load_run_data] Auto-selected analysis_id: {used_analysis_id} for {run_id}")

    # Load events candidates (versioned if analysis_id provided)
    events_path = paths.events_candidates_path(run_id, used_analysis_id)
    if not events_path.exists() and used_analysis_id:
        # Fallback to legacy path
        legacy_path = paths.events_candidates_path(run_id, None)
        if legacy_path.exists():
            events_path = legacy_path
            used_analysis_id = None
            print(f"[load_run_data] Falling back to legacy events path for {run_id}")

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

    # Load detection summary (retention metrics + detection config) if available
    retention = None
    detection_config = None
    detection_summary_path = paths.detection_summary_path(run_id, used_analysis_id)
    if not detection_summary_path.exists() and used_analysis_id:
        # Fallback to legacy path
        detection_summary_path = paths.detection_summary_path(run_id, None)
    if detection_summary_path.exists():
        ds = pd.read_parquet(detection_summary_path)
        n_series = len(ds)
        n_candidates = int(ds["n_candidates_raw"].sum())
        n_selected = int(ds["n_selected_final"].sum())

        # Pre/post breakdown
        n_pre = int(ds["n_candidates_pre"].sum())
        n_post = int(ds["n_candidates_post"].sum())
        sel_pre = int(ds["n_selected_pre"].sum())
        sel_post = int(ds["n_selected_post"].sum())

        # Skip breakdowns
        supp_pre = int(ds.get("n_skipped_suppression_pre", pd.Series([0])).sum())
        supp_post = int(ds.get("n_skipped_suppression_post", pd.Series([0])).sum())
        topk_post = int(ds.get("n_skipped_topk_post", pd.Series([0])).sum())
        pre_cap = int(ds["n_skipped_pre_warmup_cap"].sum())

        retention = RetentionMetrics(
            n_candidates=n_candidates,
            n_selected=n_selected,
            retention_rate=n_selected / max(n_candidates, 1),
            pre_retention_rate=sel_pre / max(n_pre, 1),
            post_retention_rate=sel_post / max(n_post, 1),
            suppression_rate=(supp_pre + supp_post) / max(n_candidates, 1),
            suppression_pre=supp_pre,
            suppression_post=supp_post,
            topk_post=topk_post,
            pre_warmup_cap=pre_cap,
            mean_candidates_per_series=n_candidates / max(n_series, 1),
            mean_candidates_post_per_series=n_post / max(n_series, 1),
        )

        # Extract detection config from first row (should be same across all series)
        if not ds.empty:
            first_row = ds.iloc[0]
            detection_config = DetectionConfig(
                adaptive_k=float(first_row.get("adaptive_k")) if "adaptive_k" in ds.columns else None,
                suppression_radius=int(first_row.get("suppression_radius_steps")) if "suppression_radius_steps" in ds.columns else None,
                max_peaks=int(first_row.get("max_peaks")) if "max_peaks" in ds.columns else None,
                warmup_end_step=int(first_row.get("warmup_end_step")) if "warmup_end_step" in ds.columns and pd.notna(first_row.get("warmup_end_step")) else None,
                max_pre_warmup=int(first_row.get("max_pre_warmup")) if "max_pre_warmup" in ds.columns and pd.notna(first_row.get("max_pre_warmup")) else None,
            )

    return RunData(
        run_id=run_id,
        analysis_id=used_analysis_id,
        geometry_df=geometry_df,
        events_df=events_df,
        scalars_df=scalars_df,
        meta=meta,
        diversity=diversity,
        retention=retention,
        detection_config=detection_config,
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


def _event_key(row: pd.Series, step_bucket: int = 10) -> tuple:
    """Create a bucketed event key for set operations."""
    step_bucketed = (int(row["step"]) // step_bucket) * step_bucket
    return (int(row["layer"]), row["metric"], row["event_type"], step_bucketed)


def compute_event_jaccard(
    runs: List[RunData],
    step_bucket: int = 10,
) -> Dict[str, any]:
    """
    Compute Jaccard similarity and precision/recall between runs.

    Args:
        runs: List of RunData objects (must be exactly 2 for pairwise)
        step_bucket: Step bucketing for key comparison

    Returns:
        Dict with jaccard, precision_ab, recall_ab, precision_ba, recall_ba
    """
    if len(runs) != 2:
        return {}

    run_a, run_b = runs

    # Create event key sets
    keys_a = set(_event_key(row, step_bucket) for _, row in run_a.events_df.iterrows())
    keys_b = set(_event_key(row, step_bucket) for _, row in run_b.events_df.iterrows())

    intersection = keys_a & keys_b
    union = keys_a | keys_b

    jaccard = len(intersection) / max(len(union), 1)
    precision_ab = len(intersection) / max(len(keys_a), 1)  # common / A
    recall_ab = len(intersection) / max(len(keys_b), 1)     # common / B

    return {
        "jaccard": jaccard,
        "intersection": len(intersection),
        "union": len(union),
        "events_a": len(keys_a),
        "events_b": len(keys_b),
        "precision_ab": precision_ab,  # What fraction of A's events appear in B
        "recall_ab": recall_ab,         # What fraction of B's events appear in A
    }


def generate_event_raster_plot(
    runs: List[RunData],
    output_dir: Path,
    max_layer: int = 24,
) -> Dict[str, Path]:
    """
    Generate event raster plots (step x layer, colored by metric).

    Args:
        runs: List of RunData objects
        output_dir: Directory to save plots
        max_layer: Maximum layer index for y-axis

    Returns:
        Dict mapping plot names to paths
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}

    # Color map for metrics
    metric_colors = {
        "effective_rank": "blue",
        "sv_entropy": "green",
        "topk_mass_k8": "orange",
        "__composite__": "red",
    }

    for run in runs:
        fig, ax = plt.subplots(figsize=(12, 6))

        events = run.events_df
        if events.empty:
            continue

        # Filter out composites for cleaner visualization (or include with different marker)
        single_events = events[events["metric"] != "__composite__"]

        for metric, color in metric_colors.items():
            metric_events = single_events[single_events["metric"] == metric]
            if not metric_events.empty:
                ax.scatter(
                    metric_events["step"],
                    metric_events["layer"],
                    c=color,
                    s=30,
                    alpha=0.7,
                    label=metric,
                )

        ax.set_xlabel("Step")
        ax.set_ylabel("Layer")
        ax.set_ylim(-0.5, max_layer - 0.5)
        ax.set_title(f"Event Raster: {run.run_id[:30]}...")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Short filename
        short_id = run.run_id.split("_")[-1] if "_" in run.run_id else run.run_id[:10]
        plot_path = output_dir / f"raster_{short_id}.png"
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        plot_paths[f"raster_{short_id}"] = plot_path

    # Generate intersection raster (if 2 runs)
    if len(runs) == 2:
        fig, ax = plt.subplots(figsize=(12, 6))

        run_a, run_b = runs
        keys_a = {_event_key(row, 10): row for _, row in run_a.events_df.iterrows()}
        keys_b = {_event_key(row, 10): row for _, row in run_b.events_df.iterrows()}

        common_keys = set(keys_a.keys()) & set(keys_b.keys())

        if common_keys:
            common_events = [keys_a[k] for k in common_keys]
            steps = [e["step"] for e in common_events]
            layers = [e["layer"] for e in common_events]
            colors = [metric_colors.get(e["metric"], "gray") for e in common_events]

            ax.scatter(steps, layers, c=colors, s=50, alpha=0.8)

        ax.set_xlabel("Step")
        ax.set_ylabel("Layer")
        ax.set_ylim(-0.5, max_layer - 0.5)
        ax.set_title(f"Common Events ({len(common_keys)} events)")
        ax.grid(True, alpha=0.3)

        # Legend
        patches = [mpatches.Patch(color=c, label=m) for m, c in metric_colors.items() if m != "__composite__"]
        ax.legend(handles=patches, loc="upper right", fontsize=8)

        plot_path = output_dir / "raster_intersection.png"
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        plot_paths["raster_intersection"] = plot_path

    return plot_paths


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


def _get_versioned_path(base_path: Path, overwrite: bool = False) -> Path:
    """
    Get the appropriate path for a file, with versioning support.

    If overwrite=True or file doesn't exist, returns base_path.
    Otherwise, returns next available version: base_v2.ext, base_v3.ext, etc.
    """
    if overwrite or not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    version = 2
    while True:
        versioned_path = parent / f"{stem}_v{version}{suffix}"
        if not versioned_path.exists():
            return versioned_path
        version += 1


def generate_comparison_report(
    run_ids: List[str],
    output_path: Optional[Path] = None,
    step_tolerance: int = 5,
    generate_plots: bool = True,
    plots_dir: Optional[Path] = None,
    layers: List[int] = [0, 12, 23],
    analysis_ids: Optional[List[Optional[str]]] = None,
    llm_analysis: bool = False,
    llm_backend: str = "openai",
    llm_model: str = "gpt-4o",
    llm_question: Optional[str] = None,
    overwrite: bool = False,
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
        analysis_ids: Optional list of analysis_ids, one per run_id. If None or shorter than
                      run_ids, missing entries auto-select the latest available analysis.
        llm_analysis: Whether to generate LLM qualitative analysis
        llm_backend: LLM backend ("openai" or "anthropic")
        llm_model: Model to use for analysis
        llm_question: Optional specific question to ask the LLM
        overwrite: If True, overwrite existing files; otherwise create new versions

    Returns:
        Markdown report string
    """
    # Normalize analysis_ids to match run_ids length
    if analysis_ids is None:
        analysis_ids = [None] * len(run_ids)
    else:
        # Pad with None if shorter
        analysis_ids = list(analysis_ids) + [None] * (len(run_ids) - len(analysis_ids))

    # Load all runs with their analysis versions
    runs = [load_run_data(run_id, aid) for run_id, aid in zip(run_ids, analysis_ids)]

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

    # Compute Jaccard metrics (for 2-run comparison)
    jaccard_metrics = compute_event_jaccard(runs, step_bucket=step_tolerance * 2)

    # Generate event raster plots
    raster_paths: Dict[str, Path] = {}
    if generate_plots and plots_dir:
        raster_paths = generate_event_raster_plot(runs, plots_dir)

    # Build report
    lines = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines.append("# Seed Invariance Comparison Report")
    lines.append("")
    lines.append(f"**Runs compared:** {len(runs)}")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Step tolerance:** {step_tolerance}")
    lines.append("")

    # Detection Config Fingerprint Section (critical for apples-to-apples comparison)
    lines.append("## Detection Config Fingerprint")
    lines.append("")
    lines.append("_Verify runs used identical detection parameters before interpreting results._")
    lines.append("")

    config_hashes = []
    lines.append("| Run | Analysis ID | Config | Hash |")
    lines.append("|-----|-------------|--------|------|")
    for run in runs:
        short_id = run.run_id[:20] + "..." if len(run.run_id) > 23 else run.run_id
        aid = run.analysis_id or "(legacy)"
        if run.detection_config:
            fingerprint = run.detection_config.fingerprint()
            cfg_hash = run.detection_config.config_hash()
            config_hashes.append(cfg_hash)
        else:
            fingerprint = "N/A (no detection_summary)"
            cfg_hash = "?"
            config_hashes.append(None)
        lines.append(f"| {short_id} | {aid} | {fingerprint} | {cfg_hash} |")

    lines.append("")

    # Check for config mismatch
    valid_hashes = [h for h in config_hashes if h is not None]
    if valid_hashes and len(set(valid_hashes)) > 1:
        lines.append("**WARNING:** Detection configs differ between runs! Results may not be comparable.")
        lines.append("")
    elif valid_hashes:
        lines.append("Config match confirmed.")
        lines.append("")

    # Run Summary Table
    lines.append("## Run Summary")
    lines.append("")

    # Include retention column if any runs have it
    has_retention = any(run.retention is not None for run in runs)
    if has_retention:
        lines.append("| Run ID | Seed | Final Loss | Events | Diversity | Retention |")
        lines.append("|--------|------|------------|--------|-----------|-----------|")
    else:
        lines.append("| Run ID | Seed | Final Loss | Events | Diversity |")
        lines.append("|--------|------|------------|--------|-----------|")

    for run in runs:
        seed = _extract_seed_from_run_id(run.run_id, run.meta)
        final_loss = _get_final_loss(run.scalars_df)
        n_events = len(run.events_df)
        diversity = run.diversity.diversity_score

        # Truncate run_id for display
        short_id = run.run_id[:20] + "..." if len(run.run_id) > 23 else run.run_id

        if has_retention:
            ret_str = f"{run.retention.retention_rate:.0%}" if run.retention else "N/A"
            lines.append(
                f"| {short_id} | {seed} | {final_loss} | {n_events} | {diversity:.3f} | {ret_str} |"
            )
        else:
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

    # Invariance Metrics Section (Jaccard + Precision/Recall)
    if jaccard_metrics:
        lines.append("## Invariance Metrics")
        lines.append("")
        lines.append(f"_Event keys bucketed by step (bucket={step_tolerance * 2})_")
        lines.append("")
        lines.append(f"- **Jaccard similarity:** {jaccard_metrics['jaccard']:.1%}")
        lines.append(f"- **Intersection:** {jaccard_metrics['intersection']} events")
        lines.append(f"- **Union:** {jaccard_metrics['union']} unique event keys")
        lines.append("")
        lines.append("| Direction | Value | Interpretation |")
        lines.append("|-----------|-------|----------------|")
        lines.append(
            f"| A→B (precision) | {jaccard_metrics['precision_ab']:.1%} | "
            f"{jaccard_metrics['intersection']}/{jaccard_metrics['events_a']} of run A's events in B |"
        )
        lines.append(
            f"| B→A (recall) | {jaccard_metrics['recall_ab']:.1%} | "
            f"{jaccard_metrics['intersection']}/{jaccard_metrics['events_b']} of run B's events in A |"
        )
        lines.append("")

        # Interpret the metrics
        j = jaccard_metrics['jaccard']
        if j > 0.5:
            lines.append("**Strong invariance** - majority of events appear in both runs.")
        elif j > 0.25:
            lines.append("**Moderate invariance** - significant overlap but notable differences.")
        else:
            lines.append("**Weak invariance** - runs have substantially different event patterns.")
        lines.append("")

    # Event Raster Plots Section
    if raster_paths:
        lines.append("## Event Raster Plots")
        lines.append("")
        lines.append("Visual comparison of event locations (x=step, y=layer, color=metric).")
        lines.append("")
        for name, path in raster_paths.items():
            if output_path:
                try:
                    rel_path = path.relative_to(Path(output_path).parent)
                except ValueError:
                    rel_path = path
            else:
                rel_path = path
            lines.append(f"![{name}]({rel_path})")
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

    # Detection Retention Section (if detection summaries available)
    runs_with_retention = [r for r in runs if r.retention is not None]
    if runs_with_retention:
        lines.append("## Detection Retention Comparison")
        lines.append("")
        lines.append("Compares how many raw candidates (above threshold) survived peak selection.")
        lines.append("")
        lines.append("| Run | Candidates | Selected | Retention | Pre Ret. | Post Ret. | Supp | Pre-Cap |")
        lines.append("|-----|------------|----------|-----------|----------|-----------|------|---------|")

        for run in runs:
            if run.retention:
                r = run.retention
                short_id = run.run_id[:15] + "..." if len(run.run_id) > 18 else run.run_id
                lines.append(
                    f"| {short_id} | {r.n_candidates} | {r.n_selected} | "
                    f"{r.retention_rate:.0%} | {r.pre_retention_rate:.0%} | "
                    f"{r.post_retention_rate:.0%} | {r.suppression_pre + r.suppression_post} | "
                    f"{r.pre_warmup_cap} |"
                )

        lines.append("")

        # Compute retention consistency
        ret_rates = [r.retention.retention_rate for r in runs_with_retention]
        pre_rates = [r.retention.pre_retention_rate for r in runs_with_retention]
        post_rates = [r.retention.post_retention_rate for r in runs_with_retention]

        lines.append(f"**Mean retention:** {np.mean(ret_rates):.1%} (std: {np.std(ret_rates):.1%})")
        lines.append(f"**Mean pre-warmup retention:** {np.mean(pre_rates):.1%}")
        lines.append(f"**Mean post-warmup retention:** {np.mean(post_rates):.1%}")
        lines.append("")

        # Density diagnostics table
        lines.append("### Candidate Density Diagnostics")
        lines.append("")
        lines.append("_Higher density → more suppression; helps explain retention differences._")
        lines.append("")
        lines.append("| Run | Cand/Series | Post Cand/Series | Suppression | Post Ret. |")
        lines.append("|-----|-------------|------------------|-------------|-----------|")

        for run in runs_with_retention:
            r = run.retention
            short_id = run.run_id.split("_")[-1] if "_" in run.run_id else run.run_id[:8]
            lines.append(
                f"| {short_id} | {r.mean_candidates_per_series:.1f} | "
                f"{r.mean_candidates_post_per_series:.1f} | "
                f"{r.suppression_pre + r.suppression_post} | {r.post_retention_rate:.0%} |"
            )

        lines.append("")

        # Interpret retention mismatch
        if len(runs_with_retention) == 2:
            r1, r2 = runs_with_retention[0].retention, runs_with_retention[1].retention
            density_ratio = r1.mean_candidates_post_per_series / max(r2.mean_candidates_post_per_series, 0.1)

            if abs(density_ratio - 1.0) > 0.3:
                lines.append("**Density interpretation:**")
                if density_ratio > 1.0:
                    lines.append(
                        f"- Run A has {density_ratio:.1f}x more post-warmup candidates per series"
                    )
                    lines.append("- Higher density leads to more suppression (same phenomenon, different intensity)")
                else:
                    lines.append(
                        f"- Run B has {1/density_ratio:.1f}x more post-warmup candidates per series"
                    )
                    lines.append("- Higher density leads to more suppression (same phenomenon, different intensity)")
                lines.append("")

        # Check for consistent warmup behavior
        if all(pr < 0.5 for pr in pre_rates) and all(por > 0.4 for por in post_rates):
            lines.append("Warmup gate consistently filters early transients across runs.")
        elif np.std(ret_rates) > 0.1:
            lines.append("Note: Retention rates vary significantly across runs - check density table above.")

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

    # Common events
    if n_common > 0:
        lines.append(f"- **{n_common} common events** detected across all seeds")
    else:
        lines.append("- **No common events** - seed invariance not established")

    # Jaccard (if available)
    if jaccard_metrics:
        j = jaccard_metrics['jaccard']
        lines.append(f"- **Jaccard similarity:** {j:.1%} ({jaccard_metrics['intersection']}/{jaccard_metrics['union']} events)")

    # Trajectory correlation
    if not np.isnan(mean_corr_val):
        if mean_corr_val > 0.95:
            lines.append(f"- **High trajectory correlation** (mean: {mean_corr_val:.2f})")
        elif mean_corr_val > 0.8:
            lines.append(f"- **Moderate trajectory correlation** (mean: {mean_corr_val:.2f})")
        else:
            lines.append(f"- **Low trajectory correlation** (mean: {mean_corr_val:.2f})")

    if std_div < 0.1:
        lines.append(f"- **Consistent diversity scores** (std: {std_div:.2f})")

    # Final loss divergence warning
    final_losses = []
    for run in runs:
        loss_str = _get_final_loss(run.scalars_df)
        try:
            final_losses.append(float(loss_str))
        except ValueError:
            pass

    if len(final_losses) >= 2:
        loss_diff_pct = abs(final_losses[0] - final_losses[1]) / max(min(final_losses), 0.001) * 100
        if loss_diff_pct > 5:
            lines.append(f"- **Warning:** Final loss differs by {loss_diff_pct:.1f}% - different learning trajectories possible")

    lines.append("")

    # Overall assessment
    has_jaccard = jaccard_metrics and jaccard_metrics['jaccard'] > 0.25
    has_common = n_common > 5
    has_corr = not np.isnan(mean_corr_val) and mean_corr_val > 0.8

    if has_jaccard and has_common and has_corr:
        lines.append("The runs show **strong seed invariance**.")
    elif has_common or has_corr or has_jaccard:
        lines.append("The runs show **moderate seed invariance**.")
    else:
        lines.append("The runs show **weak seed invariance** - investigate differences.")

    lines.append("")

    report = "\n".join(lines)

    # Write to file if requested (with versioning)
    actual_output_path: Optional[Path] = None
    if output_path:
        output_path = Path(output_path)
        actual_output_path = _get_versioned_path(output_path, overwrite=overwrite)
        actual_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(actual_output_path, "w") as f:
            f.write(report)
        print(f"Report written to: {actual_output_path}")

    # Optional LLM analysis
    if llm_analysis:
        _run_llm_comparison_analysis(
            runs=runs,
            report_content=report,
            output_path=actual_output_path,
            plot_paths=list(plot_paths.values()) + list(raster_paths.values()),
            llm_backend=llm_backend,
            llm_model=llm_model,
            llm_question=llm_question,
            overwrite=overwrite,
        )

    return report


def _run_llm_comparison_analysis(
    runs: List[RunData],
    report_content: str,
    output_path: Optional[Path],
    plot_paths: List[Path],
    llm_backend: str,
    llm_model: str,
    llm_question: Optional[str],
    overwrite: bool = False,
) -> None:
    """Run LLM analysis on the comparison report."""
    from squiggle_analysis.llm_analysis.analyzer import (
        AnalysisRequest,
        analyze_report,
        write_analysis_result,
    )

    # Build run context with detection configs
    # Build full detection configs for each run
    detection_configs = []
    for r in runs:
        if r.detection_config:
            detection_configs.append({
                "adaptive_k": r.detection_config.adaptive_k,
                "suppression_radius": r.detection_config.suppression_radius,
                "max_peaks": r.detection_config.max_peaks,
                "warmup_end_step": r.detection_config.warmup_end_step,
                "max_pre_warmup": r.detection_config.max_pre_warmup,
            })
        else:
            detection_configs.append(None)

    run_context = {
        "analysis_mode": "comparison",
        "run_ids": [r.run_id for r in runs],
        "seeds": [r.meta.get("seed") for r in runs],
        "detection_configs": detection_configs,
        "config_fingerprints": [
            r.detection_config.fingerprint() if r.detection_config else "unknown"
            for r in runs
        ],
        "config_hashes": [
            r.detection_config.config_hash() if r.detection_config else "?"
            for r in runs
        ],
    }

    request = AnalysisRequest(
        run_context=run_context,
        primary_report=None,
        compare_report=report_content,
        artifacts=[str(p) for p in plot_paths if p.exists()],
        user_question=llm_question,
    )

    print(f"[...] Running LLM analysis with {llm_model}...")
    result = analyze_report(
        request,
        backend=llm_backend,
        model=llm_model,
    )

    # Determine output path (with versioning)
    if output_path:
        base_analysis_path = output_path.with_suffix(".llm_analysis.json")
    else:
        base_analysis_path = Path("comparison.llm_analysis.json")

    analysis_path = _get_versioned_path(base_analysis_path, overwrite=overwrite)
    write_analysis_result(result, analysis_path)

    print(f"[✓] LLM analysis written to: {analysis_path}")

    if result.validation_errors:
        print(f"    WARNING: {len(result.validation_errors)} validation errors in response")


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
        "--analysis-ids",
        nargs="+",
        help="Analysis IDs for each run (in same order as run_ids). "
             "Use 'auto' to auto-select the latest analysis for a run.",
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

    # Parse analysis_ids
    analysis_ids = None
    if args.analysis_ids:
        analysis_ids = [
            None if aid.lower() == "auto" else aid
            for aid in args.analysis_ids
        ]

    report = generate_comparison_report(
        args.run_ids,
        output_path=args.output,
        step_tolerance=args.step_tolerance,
        generate_plots=not args.no_plots,
        plots_dir=args.plots_dir,
        layers=args.layers,
        analysis_ids=analysis_ids,
    )

    if args.output is None:
        print(report)


if __name__ == "__main__":
    main()
