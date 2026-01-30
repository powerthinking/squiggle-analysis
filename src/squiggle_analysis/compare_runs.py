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


def _event_key_strict(row: pd.Series, step_tolerance: int = 5) -> tuple:
    """Create a strict event key (no bucketing, used with tolerance matching)."""
    return (int(row["layer"]), row["metric"], row["event_type"], int(row["step"]))


def _get_event_window(row: pd.Series, dilation: int = 0) -> tuple[int, int]:
    """Get event window [start, end] from row, falling back to step if not available.

    Args:
        row: Event row from DataFrame
        dilation: Number of steps to expand window on each side (for comparison)

    Returns:
        (start_step, end_step) tuple, optionally dilated
    """
    if row.get("event_type") == "change_point_composite":
        # Use composite window if available
        start = row.get("composite_start_step")
        end = row.get("composite_end_step")
        if pd.notna(start) and pd.notna(end):
            return (int(start) - dilation, int(end) + dilation)
    # Fall back to start_step/end_step
    start = row.get("start_step")
    end = row.get("end_step")
    if pd.notna(start) and pd.notna(end):
        return (int(start) - dilation, int(end) + dilation)
    # Final fallback to step with radius 0
    step = int(row["step"])
    return (step - dilation, step + dilation)


def _window_iou(w1: tuple[int, int], w2: tuple[int, int]) -> float:
    """Compute intersection-over-union of two windows."""
    inter_start = max(w1[0], w2[0])
    inter_end = min(w1[1], w2[1])
    intersection = max(0, inter_end - inter_start + 1)

    union_start = min(w1[0], w2[0])
    union_end = max(w1[1], w2[1])
    union = union_end - union_start + 1

    return intersection / max(union, 1)


def compute_event_jaccard(
    runs: List[RunData],
    step_bucket: int = 10,
) -> Dict[str, any]:
    """
    Compute Jaccard similarity and precision/recall between runs using step bucketing.

    Args:
        runs: List of RunData objects (must be exactly 2 for pairwise)
        step_bucket: Step bucketing for key comparison

    Returns:
        Dict with jaccard, precision_ab, recall_ab, and matching spec
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
        "matching_spec": f"key=(layer, metric, type, step//{step_bucket}*{step_bucket})",
    }


def compute_window_overlap_invariance(
    runs: List[RunData],
    iou_threshold: float = 0.2,
    require_same_signature: bool = True,
    compute_all_ious: bool = False,
    window_dilation: int = 0,
) -> Dict[str, any]:
    """
    Compute invariance using window overlap (IoU) instead of step bucketing.

    This is more robust to step jitter since events are matched by their
    [start_step, end_step] overlap, not just peak step.

    Args:
        runs: List of RunData objects (must be exactly 2)
        iou_threshold: Minimum IoU to consider events as matching
        require_same_signature: If True, also require (layer, metric, type) to match
        compute_all_ious: If True, compute IoU for all valid pairs (for distribution stats)
        window_dilation: Steps to expand each window for comparison (addresses IoU ceiling
                        with small detection windows). E.g., dilation=2 with radius=1 detection
                        gives effective 5-tick comparison windows.

    Returns:
        Dict with overlap-based invariance metrics
    """
    if len(runs) != 2:
        return {}

    run_a, run_b = runs

    # Build event lists with windows (optionally dilated for comparison)
    events_a = []
    for _, row in run_a.events_df.iterrows():
        events_a.append({
            "layer": int(row["layer"]),
            "metric": row["metric"],
            "event_type": row["event_type"],
            "step": int(row["step"]),
            "window": _get_event_window(row, dilation=window_dilation),
            "score": float(row.get("score", 0)),
        })

    events_b = []
    for _, row in run_b.events_df.iterrows():
        events_b.append({
            "layer": int(row["layer"]),
            "metric": row["metric"],
            "event_type": row["event_type"],
            "step": int(row["step"]),
            "window": _get_event_window(row, dilation=window_dilation),
            "score": float(row.get("score", 0)),
        })

    # Compute all IoUs for distribution stats (before filtering by threshold)
    all_ious: List[float] = []
    # Also compute upper bound per signature for greedy efficiency
    signature_counts_a: Dict[tuple, int] = {}
    signature_counts_b: Dict[tuple, int] = {}

    for ev_a in events_a:
        sig = (ev_a["layer"], ev_a["metric"], ev_a["event_type"])
        signature_counts_a[sig] = signature_counts_a.get(sig, 0) + 1

    for ev_b in events_b:
        sig = (ev_b["layer"], ev_b["metric"], ev_b["event_type"])
        signature_counts_b[sig] = signature_counts_b.get(sig, 0) + 1

    # Upper bound = sum of min(count_a, count_b) per signature
    upper_bound_matches = 0
    all_signatures = set(signature_counts_a.keys()) | set(signature_counts_b.keys())
    for sig in all_signatures:
        count_a = signature_counts_a.get(sig, 0)
        count_b = signature_counts_b.get(sig, 0)
        upper_bound_matches += min(count_a, count_b)

    if compute_all_ious:
        for ev_a in events_a:
            for ev_b in events_b:
                if require_same_signature:
                    if (ev_a["layer"] != ev_b["layer"] or
                        ev_a["metric"] != ev_b["metric"] or
                        ev_a["event_type"] != ev_b["event_type"]):
                        continue
                iou = _window_iou(ev_a["window"], ev_b["window"])
                if iou > 0:  # Only include non-zero IoUs
                    all_ious.append(iou)

    # Match events from A to B using IoU (greedy, one-to-one)
    matched_a = set()
    matched_b = set()
    matches = []

    # Sort A events by best possible IoU (descending) for better greedy matching
    a_best_ious = []
    for i, ev_a in enumerate(events_a):
        best_iou_for_a = 0.0
        for j, ev_b in enumerate(events_b):
            if require_same_signature:
                if (ev_a["layer"] != ev_b["layer"] or
                    ev_a["metric"] != ev_b["metric"] or
                    ev_a["event_type"] != ev_b["event_type"]):
                    continue
            iou = _window_iou(ev_a["window"], ev_b["window"])
            if iou > best_iou_for_a:
                best_iou_for_a = iou
        a_best_ious.append((i, best_iou_for_a))

    # Process in order of best potential match (helps greedy find better solution)
    a_best_ious.sort(key=lambda x: x[1], reverse=True)

    for i, _ in a_best_ious:
        ev_a = events_a[i]
        best_j = None
        best_iou = 0.0

        for j, ev_b in enumerate(events_b):
            if j in matched_b:
                continue

            # Check signature match if required
            if require_same_signature:
                if (ev_a["layer"] != ev_b["layer"] or
                    ev_a["metric"] != ev_b["metric"] or
                    ev_a["event_type"] != ev_b["event_type"]):
                    continue

            iou = _window_iou(ev_a["window"], ev_b["window"])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j is not None:
            matched_a.add(i)
            matched_b.add(best_j)
            matches.append({
                "layer": ev_a["layer"],
                "metric": ev_a["metric"],
                "event_type": ev_a["event_type"],
                "step_a": ev_a["step"],
                "step_b": events_b[best_j]["step"],
                "iou": best_iou,
                "score_a": ev_a["score"],
                "score_b": events_b[best_j]["score"],
            })

    n_matched_pairs = len(matches)
    n_a = len(events_a)
    n_b = len(events_b)

    # One-to-one matching: each event can match at most one partner
    # matched_a and matched_b are sets of indices that found partners
    unique_a_matched = len(matched_a)
    unique_b_matched = len(matched_b)

    # Jaccard using matched counts
    jaccard_overlap = n_matched_pairs / max(n_a + n_b - n_matched_pairs, 1)
    precision_a = unique_a_matched / max(n_a, 1)  # fraction of A events that found a match
    recall_b = unique_b_matched / max(n_b, 1)     # fraction of B events that found a match

    # Compute mean IoU of matches
    mean_iou = np.mean([m["iou"] for m in matches]) if matches else 0.0

    # Greedy efficiency: how close to upper bound?
    greedy_efficiency = n_matched_pairs / max(upper_bound_matches, 1)

    result = {
        "jaccard_overlap": jaccard_overlap,
        "n_matched_pairs": n_matched_pairs,
        "unique_a_matched": unique_a_matched,
        "unique_b_matched": unique_b_matched,
        "events_a": n_a,
        "events_b": n_b,
        "precision_a": precision_a,
        "recall_b": recall_b,
        "mean_iou": mean_iou,
        "iou_threshold": iou_threshold,
        "window_dilation": window_dilation,
        "matches": matches,  # For detailed analysis
        "matching_spec": f"key=(layer, metric, type) + IoU≥{iou_threshold}" + (f" (dilation={window_dilation})" if window_dilation else ""),
        "matching_mode": "one-to-one (greedy by IoU, sorted by best potential)",
        "upper_bound_matches": upper_bound_matches,
        "greedy_efficiency": greedy_efficiency,
    }

    # Add IoU distribution stats if computed
    if all_ious:
        all_ious_sorted = sorted(all_ious)
        n_ious = len(all_ious_sorted)
        result["iou_distribution"] = {
            "n_pairs": n_ious,
            "max": all_ious_sorted[-1] if all_ious_sorted else 0,
            "p99": all_ious_sorted[int(n_ious * 0.99)] if n_ious > 0 else 0,
            "p95": all_ious_sorted[int(n_ious * 0.95)] if n_ious > 0 else 0,
            "p75": all_ious_sorted[int(n_ious * 0.75)] if n_ious > 0 else 0,
            "median": all_ious_sorted[n_ious // 2] if n_ious > 0 else 0,
        }

    return result


def compute_invariance_curve(
    runs: List[RunData],
    iou_thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
    window_dilation: int = 0,
) -> Dict[str, any]:
    """
    Compute invariance metrics at multiple IoU thresholds.

    This creates a "stability signature" - if Jaccard collapses hard when
    threshold increases, you're mostly matching weak overlaps.

    Args:
        runs: List of RunData objects (must be exactly 2)
        iou_thresholds: List of IoU thresholds to evaluate
        window_dilation: Steps to expand windows for comparison (addresses IoU ceiling)

    Returns:
        Dict with curve data and IoU distribution stats
    """
    if len(runs) != 2:
        return {}

    # First, compute with all IoUs for distribution stats
    full_result = compute_window_overlap_invariance(
        runs, iou_threshold=0.0, compute_all_ious=True, window_dilation=window_dilation
    )

    curve = []
    for tau in iou_thresholds:
        result = compute_window_overlap_invariance(
            runs, iou_threshold=tau, window_dilation=window_dilation
        )
        if result:
            curve.append({
                "iou_threshold": tau,
                "jaccard": result["jaccard_overlap"],
                "precision": result["precision_a"],
                "recall": result["recall_b"],
                "n_matched": result["n_matched_pairs"],
                "mean_iou": result["mean_iou"],
                "greedy_efficiency": result["greedy_efficiency"],
            })

    return {
        "curve": curve,
        "iou_distribution": full_result.get("iou_distribution", {}),
        "upper_bound_matches": full_result.get("upper_bound_matches", 0),
        "window_dilation": window_dilation,
    }


def compute_composite_invariance(
    runs: List[RunData],
    iou_threshold: float = 0.3,
) -> Dict[str, any]:
    """
    Compute invariance metrics specifically for composite events.

    Args:
        runs: List of RunData objects (must be exactly 2)
        iou_threshold: Minimum IoU to consider composites as matching

    Returns:
        Dict with composite-specific invariance metrics
    """
    if len(runs) != 2:
        return {}

    run_a, run_b = runs

    # Filter to composites only
    comp_a = run_a.events_df[run_a.events_df["event_type"] == "change_point_composite"]
    comp_b = run_b.events_df[run_b.events_df["event_type"] == "change_point_composite"]

    n_comp_a = len(comp_a)
    n_comp_b = len(comp_b)

    if n_comp_a == 0 and n_comp_b == 0:
        return {"n_composites_a": 0, "n_composites_b": 0, "n_matched": 0}

    # Build composite windows
    comps_a = []
    for _, row in comp_a.iterrows():
        comps_a.append({
            "layer": int(row["layer"]),
            "step": int(row["step"]),
            "window": _get_event_window(row),
            "strength": float(row.get("composite_strength", 0)),
            "n_metrics": int(row.get("composite_n_metrics", 0)),
        })

    comps_b = []
    for _, row in comp_b.iterrows():
        comps_b.append({
            "layer": int(row["layer"]),
            "step": int(row["step"]),
            "window": _get_event_window(row),
            "strength": float(row.get("composite_strength", 0)),
            "n_metrics": int(row.get("composite_n_metrics", 0)),
        })

    # Match composites (same layer + IoU)
    matched_a = set()
    matched_b = set()
    matches = []

    for i, ca in enumerate(comps_a):
        best_j = None
        best_iou = 0.0

        for j, cb in enumerate(comps_b):
            if j in matched_b:
                continue
            if ca["layer"] != cb["layer"]:
                continue

            iou = _window_iou(ca["window"], cb["window"])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j is not None:
            matched_a.add(i)
            matched_b.add(best_j)
            matches.append({
                "layer": ca["layer"],
                "step_a": ca["step"],
                "step_b": comps_b[best_j]["step"],
                "iou": best_iou,
                "strength_a": ca["strength"],
                "strength_b": comps_b[best_j]["strength"],
            })

    n_matched = len(matches)
    unique_a_matched = len(matched_a)
    unique_b_matched = len(matched_b)
    jaccard = n_matched / max(n_comp_a + n_comp_b - n_matched, 1)

    # Coverage metrics (density-normalized)
    # coverage_a = fraction of A's composites that found a match in B
    # coverage_b = fraction of B's composites that found a match in A
    coverage_a = unique_a_matched / max(n_comp_a, 1)
    coverage_b = unique_b_matched / max(n_comp_b, 1)

    # Strength correlation for matched composites
    strength_corr = None
    if len(matches) >= 3:
        strengths_a = [m["strength_a"] for m in matches]
        strengths_b = [m["strength_b"] for m in matches]
        if np.std(strengths_a) > 0 and np.std(strengths_b) > 0:
            strength_corr = float(np.corrcoef(strengths_a, strengths_b)[0, 1])

    # Step diversity comparison
    steps_a = sorted([c["step"] for c in comps_a]) if comps_a else []
    steps_b = sorted([c["step"] for c in comps_b]) if comps_b else []
    unique_steps_a = len(set(steps_a))
    unique_steps_b = len(set(steps_b))

    # Asymmetry ratio (how unbalanced is composite generation?)
    asymmetry_ratio = max(n_comp_a, n_comp_b) / max(min(n_comp_a, n_comp_b), 1)

    return {
        "n_composites_a": n_comp_a,
        "n_composites_b": n_comp_b,
        "n_matched": n_matched,
        "unique_a_matched": unique_a_matched,
        "unique_b_matched": unique_b_matched,
        "jaccard": jaccard,
        "coverage_a": coverage_a,  # A's composites matched in B
        "coverage_b": coverage_b,  # B's composites matched in A
        "asymmetry_ratio": asymmetry_ratio,
        "strength_correlation": strength_corr,
        "unique_steps_a": unique_steps_a,
        "unique_steps_b": unique_steps_b,
        "matches": matches,
    }


def compute_neighbor_coverage(
    runs: List[RunData],
    radius_steps: int = 15,
) -> Dict[str, any]:
    """
    Compute bidirectional neighbor coverage at a given radius.

    Neighbor Coverage answers: "Do both seeds show activity in the same neighborhoods?"
    (Contrast with Peak/IoU Jaccard which answers: "Do they select the same discrete winners?")

    This is NOT a true Jaccard (no set intersection/union). It's a symmetric coverage score.

    Definitions:
        - Signature (matching key): (layer, metric, event_type)
        - Neighbor condition: ∃ event in other run with |Δstep| ≤ radius_steps
        - Metric: (A_has_neighbor + B_has_neighbor) / (|A| + |B|)

    Winner selection (for stability metrics):
        - Winner = argmax by score within matched events
        - Tiebreak: highest score → smallest step (deterministic)

    Args:
        runs: List of RunData objects (must be exactly 2)
        radius_steps: Radius for neighbor detection (typically = suppression_radius)

    Returns:
        Dict with neighbor coverage metrics and nearest-neighbor distance stats
    """
    if len(runs) != 2:
        return {}

    run_a, run_b = runs

    def get_signature(row: pd.Series) -> tuple:
        return (int(row["layer"]), row["metric"], row["event_type"])

    # Group events by signature, storing (step, score) tuples
    events_by_sig_a: Dict[tuple, List[tuple]] = {}
    for _, row in run_a.events_df.iterrows():
        sig = get_signature(row)
        score = float(row.get("score", 1.0))
        events_by_sig_a.setdefault(sig, []).append((int(row["step"]), score))

    events_by_sig_b: Dict[tuple, List[tuple]] = {}
    for _, row in run_b.events_df.iterrows():
        sig = get_signature(row)
        score = float(row.get("score", 1.0))
        events_by_sig_b.setdefault(sig, []).append((int(row["step"]), score))

    all_signatures = set(events_by_sig_a.keys()) | set(events_by_sig_b.keys())

    events_a_with_neighbor = 0
    events_b_with_neighbor = 0
    total_events_a = 0
    total_events_b = 0

    matched_neighborhoods = []

    # Collect nearest-neighbor distances for diagnostics
    nn_distances_a_to_b = []  # For each event in A, distance to nearest event in B
    nn_distances_b_to_a = []  # For each event in B, distance to nearest event in A

    for sig in all_signatures:
        events_a = events_by_sig_a.get(sig, [])
        events_b = events_by_sig_b.get(sig, [])

        total_events_a += len(events_a)
        total_events_b += len(events_b)

        if not events_a or not events_b:
            continue

        # Find events with neighbors and compute nearest-neighbor distances
        matched_a = set()
        matched_b = set()

        # Compute NN distances A→B
        for step_a, _ in events_a:
            min_dist = min(abs(step_a - step_b) for step_b, _ in events_b)
            nn_distances_a_to_b.append(min_dist)
            if min_dist <= radius_steps:
                matched_a.add(step_a)

        # Compute NN distances B→A
        for step_b, _ in events_b:
            min_dist = min(abs(step_b - step_a) for step_a, _ in events_a)
            nn_distances_b_to_a.append(min_dist)
            if min_dist <= radius_steps:
                matched_b.add(step_b)

        events_a_with_neighbor += len(matched_a)
        events_b_with_neighbor += len(matched_b)

        if matched_a and matched_b:
            # Winner = argmax by score, tiebreak by smallest step (deterministic)
            matched_events_a = [(s, sc) for s, sc in events_a if s in matched_a]
            matched_events_b = [(s, sc) for s, sc in events_b if s in matched_b]

            # Sort by (-score, step) so highest score wins, then smallest step breaks ties
            winner_a = sorted(matched_events_a, key=lambda x: (-x[1], x[0]))[0][0]
            winner_b = sorted(matched_events_b, key=lambda x: (-x[1], x[0]))[0][0]

            matched_neighborhoods.append({
                "signature": sig,
                "n_events_a_matched": len(matched_a),
                "n_events_b_matched": len(matched_b),
                "winner_a": winner_a,
                "winner_b": winner_b,
                "winner_diff": abs(winner_a - winner_b),
            })

    coverage_a = events_a_with_neighbor / max(total_events_a, 1)
    coverage_b = events_b_with_neighbor / max(total_events_b, 1)

    # Symmetric coverage (NOT Jaccard - this is bidirectional coverage)
    neighbor_coverage = (events_a_with_neighbor + events_b_with_neighbor) / max(
        total_events_a + total_events_b, 1
    )

    n_neighborhoods = len(matched_neighborhoods)
    winner_exact = sum(1 for r in matched_neighborhoods if r["winner_diff"] == 0)
    winner_close = sum(1 for r in matched_neighborhoods if r["winner_diff"] <= radius_steps // 2)

    winner_stability = winner_exact / max(n_neighborhoods, 1)
    winner_close_stability = winner_close / max(n_neighborhoods, 1)

    # Compute nearest-neighbor distance statistics
    all_nn_distances = nn_distances_a_to_b + nn_distances_b_to_a
    if all_nn_distances:
        nn_arr = np.array(all_nn_distances)
        nn_stats = {
            "median": float(np.median(nn_arr)),
            "p90": float(np.percentile(nn_arr, 90)),
            "p95": float(np.percentile(nn_arr, 95)),
            "max": float(np.max(nn_arr)),
            "within_radius": float(np.mean(nn_arr <= radius_steps)),
            "within_half_radius": float(np.mean(nn_arr <= radius_steps // 2)),
        }
    else:
        nn_stats = {}

    return {
        "neighbor_coverage": neighbor_coverage,
        "coverage_a": coverage_a,
        "coverage_b": coverage_b,
        "events_a_with_neighbor": events_a_with_neighbor,
        "events_b_with_neighbor": events_b_with_neighbor,
        "total_events_a": total_events_a,
        "total_events_b": total_events_b,
        "n_signatures_with_overlap": n_neighborhoods,
        "radius_steps": radius_steps,
        "winner_stability": winner_stability,
        "winner_close_stability": winner_close_stability,
        "nn_distance_stats": nn_stats,
        "matched_neighborhoods": matched_neighborhoods[:20],
        "interpretation": _interpret_neighbor_coverage(neighbor_coverage, winner_close_stability),
    }


# Backward compatibility alias
def compute_cluster_jaccard(runs: List[RunData], cluster_radius_steps: int = 15) -> Dict[str, any]:
    """Deprecated: Use compute_neighbor_coverage instead."""
    result = compute_neighbor_coverage(runs, radius_steps=cluster_radius_steps)
    # Map new keys to old for compatibility
    result["cluster_jaccard"] = result["neighbor_coverage"]
    result["region_coverage_a"] = result["coverage_a"]
    result["region_coverage_b"] = result["coverage_b"]
    result["cluster_radius_steps"] = result["radius_steps"]
    result["n_matched_regions"] = result["n_signatures_with_overlap"]
    result["n_matched_neighborhoods"] = result["n_signatures_with_overlap"]  # Also map old name
    result["matched_regions"] = result["matched_neighborhoods"]
    return result


def _interpret_neighbor_coverage(coverage: float, winner_close_stability: float) -> str:
    """Provide interpretation of neighbor coverage vs winner stability.

    Args:
        coverage: Fraction of events that have a neighbor within radius (NOT Jaccard)
        winner_close_stability: Fraction of matched neighborhoods where winners are within half-radius
    """
    if coverage > 0.6 and winner_close_stability < 0.5:
        return (
            "High neighbor coverage but low winner stability: "
            "both seeds show activity in the same neighborhoods, "
            "but peak selection within neighborhoods diverges."
        )
    elif coverage > 0.6 and winner_close_stability > 0.7:
        return (
            "Both neighbor coverage and winner stability are high: "
            "seeds detect similar activity in similar locations with aligned peaks."
        )
    elif coverage > 0.4 and winner_close_stability > 0.6:
        return (
            "Moderate neighbor coverage with good winner alignment: "
            "about half of events find neighbors, and when they do, peaks align well."
        )
    elif coverage < 0.3:
        return (
            "Low neighbor coverage: seeds detect events in different locations. "
            "This may indicate genuine signal differences or detection sensitivity."
        )
    else:
        return (
            f"Moderate neighbor coverage ({coverage:.0%}) with "
            f"{'aligned' if winner_close_stability > 0.5 else 'divergent'} peak selection."
        )


# Keep old name for backward compatibility
_interpret_cluster_jaccard = _interpret_neighbor_coverage


def extract_region_events(
    run: RunData,
    cluster_radius_steps: int = 15,
) -> pd.DataFrame:
    """
    Extract region-level events by clustering peaks within suppression radius.

    This transforms brittle peak-level events into stable region-level events.
    Each region is characterized by:
    - center_step: weighted centroid (by score)
    - spread_steps: std of member steps
    - peak_step: step of strongest member
    - region_strength: sum of member scores
    - n_members: count of peaks in region

    Args:
        run: RunData object
        cluster_radius_steps: Radius for clustering (typically = suppression_radius)

    Returns:
        DataFrame with region-level events
    """
    events_df = run.events_df
    if events_df.empty:
        return pd.DataFrame()

    # Group by signature
    def get_signature(row: pd.Series) -> tuple:
        return (int(row["layer"]), row["metric"], row["event_type"])

    regions = []

    for sig, group in events_df.groupby(["layer", "metric", "event_type"]):
        layer, metric, event_type = sig
        steps = group["step"].values
        scores = group["score"].values if "score" in group.columns else np.ones(len(steps))

        # Simple greedy clustering: assign each event to nearest cluster or create new
        clusters: List[Dict] = []

        for step, score in sorted(zip(steps, scores), key=lambda x: -x[1]):  # Process by score desc
            assigned = False
            for cluster in clusters:
                if abs(step - cluster["center"]) <= cluster_radius_steps:
                    # Add to existing cluster
                    cluster["steps"].append(step)
                    cluster["scores"].append(score)
                    # Update center as weighted mean
                    total_score = sum(cluster["scores"])
                    cluster["center"] = sum(s * sc for s, sc in zip(cluster["steps"], cluster["scores"])) / total_score
                    assigned = True
                    break

            if not assigned:
                # Create new cluster
                clusters.append({
                    "center": float(step),
                    "steps": [step],
                    "scores": [score],
                })

        # Convert clusters to region events
        for cluster in clusters:
            steps_arr = np.array(cluster["steps"])
            scores_arr = np.array(cluster["scores"])

            center_step = cluster["center"]
            spread_steps = float(np.std(steps_arr)) if len(steps_arr) > 1 else 0.0
            peak_step = int(steps_arr[np.argmax(scores_arr)])
            region_strength = float(np.sum(scores_arr))
            n_members = len(steps_arr)

            regions.append({
                "run_id": run.run_id,
                "layer": layer,
                "metric": metric,
                "event_type": event_type,
                "center_step": center_step,
                "spread_steps": spread_steps,
                "peak_step": peak_step,
                "start_step": int(np.min(steps_arr)),
                "end_step": int(np.max(steps_arr)),
                "region_strength": region_strength,
                "n_members": n_members,
                "member_steps": list(map(int, steps_arr)),
            })

    return pd.DataFrame(regions)


def compute_region_invariance(
    runs: List[RunData],
    cluster_radius_steps: int = 15,
) -> Dict[str, any]:
    """
    Compute invariance at the region level (the recommended canonical metric).

    This is the "correct" invariance object: regions are stable, peaks jitter.

    Args:
        runs: List of RunData objects (must be exactly 2)
        cluster_radius_steps: Radius for clustering

    Returns:
        Dict with region-level invariance metrics and diagnostics
    """
    if len(runs) != 2:
        return {}

    run_a, run_b = runs

    # Extract region events
    regions_a = extract_region_events(run_a, cluster_radius_steps)
    regions_b = extract_region_events(run_b, cluster_radius_steps)

    if regions_a.empty or regions_b.empty:
        return {"error": "No regions extracted"}

    # Match regions by signature + center proximity
    matched_regions = []
    unmatched_a = []
    unmatched_b = list(range(len(regions_b)))

    for idx_a, row_a in regions_a.iterrows():
        sig_a = (row_a["layer"], row_a["metric"], row_a["event_type"])
        center_a = row_a["center_step"]

        best_match = None
        best_dist = float("inf")

        for idx_b in unmatched_b:
            row_b = regions_b.iloc[idx_b]
            sig_b = (row_b["layer"], row_b["metric"], row_b["event_type"])

            if sig_a != sig_b:
                continue

            dist = abs(center_a - row_b["center_step"])
            if dist <= cluster_radius_steps and dist < best_dist:
                best_dist = dist
                best_match = idx_b

        if best_match is not None:
            row_b = regions_b.iloc[best_match]
            matched_regions.append({
                "layer": row_a["layer"],
                "metric": row_a["metric"],
                "event_type": row_a["event_type"],
                "center_a": row_a["center_step"],
                "center_b": row_b["center_step"],
                "center_diff": abs(row_a["center_step"] - row_b["center_step"]),
                "peak_a": row_a["peak_step"],
                "peak_b": row_b["peak_step"],
                "peak_diff": abs(row_a["peak_step"] - row_b["peak_step"]),
                "strength_a": row_a["region_strength"],
                "strength_b": row_b["region_strength"],
                "n_members_a": row_a["n_members"],
                "n_members_b": row_b["n_members"],
            })
            unmatched_b.remove(best_match)
        else:
            unmatched_a.append(idx_a)

    n_matched = len(matched_regions)
    n_a = len(regions_a)
    n_b = len(regions_b)

    # Region Jaccard
    region_jaccard = n_matched / max(n_a + n_b - n_matched, 1)

    # Center and peak alignment stats
    if matched_regions:
        center_diffs = [m["center_diff"] for m in matched_regions]
        peak_diffs = [m["peak_diff"] for m in matched_regions]

        center_mean = np.mean(center_diffs)
        center_median = np.median(center_diffs)
        center_p90 = np.percentile(center_diffs, 90)

        peak_mean = np.mean(peak_diffs)
        peak_median = np.median(peak_diffs)
        peak_p90 = np.percentile(peak_diffs, 90)

        # Strength correlation
        if n_matched >= 3:
            strengths_a = [m["strength_a"] for m in matched_regions]
            strengths_b = [m["strength_b"] for m in matched_regions]
            if np.std(strengths_a) > 0 and np.std(strengths_b) > 0:
                strength_corr = float(np.corrcoef(strengths_a, strengths_b)[0, 1])
            else:
                strength_corr = None
        else:
            strength_corr = None
    else:
        center_mean = center_median = center_p90 = 0
        peak_mean = peak_median = peak_p90 = 0
        strength_corr = None

    return {
        "region_jaccard": region_jaccard,
        "n_matched_regions": n_matched,
        "n_regions_a": n_a,
        "n_regions_b": n_b,
        "n_unmatched_a": len(unmatched_a),
        "n_unmatched_b": len(unmatched_b),
        "cluster_radius_steps": cluster_radius_steps,
        # Center alignment (region-level, stable)
        "center_diff_mean": center_mean,
        "center_diff_median": center_median,
        "center_diff_p90": center_p90,
        # Peak alignment (peak-level, jittery)
        "peak_diff_mean": peak_mean,
        "peak_diff_median": peak_median,
        "peak_diff_p90": peak_p90,
        # Strength correlation
        "strength_correlation": strength_corr,
        # Matched region details for visualization
        "matched_regions": matched_regions[:30],
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

    # Compute invariance metrics (for 2-run comparison)
    # Method 1: Step bucketing (legacy, looser)
    jaccard_bucketed = compute_event_jaccard(runs, step_bucket=step_tolerance * 2)

    # Method 2: Window overlap (IoU-based, more robust)
    jaccard_overlap = compute_window_overlap_invariance(runs, iou_threshold=0.2)

    # Method 3: Invariance curve over multiple IoU thresholds
    invariance_curve = compute_invariance_curve(runs, iou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5])

    # Method 4: Composite-specific invariance (higher threshold for broader windows)
    composite_invariance = compute_composite_invariance(runs, iou_threshold=0.3)

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

    # Common Events Section (strict matching)
    lines.append("## Common Events (Seed-Invariant)")
    lines.append("")
    lines.append(f"_Matching: same (layer, metric, type) + step within ±{step_tolerance}_")
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

    # Event Invariance Metrics Section - show BOTH methods explicitly
    lines.append("## Event Invariance Metrics")
    lines.append("")
    lines.append("_Multiple matching methods shown to clarify intersection counts:_")
    lines.append("")

    # Method 1: Window Overlap (IoU) - preferred
    if jaccard_overlap:
        lines.append("### Method 1: Window Overlap (IoU ≥ 0.2)")
        lines.append("")
        lines.append(f"_Matching: {jaccard_overlap.get('matching_spec', 'IoU-based')} ({jaccard_overlap.get('matching_mode', 'one-to-one')})_")
        lines.append("")
        lines.append(f"- **Jaccard (overlap):** {jaccard_overlap['jaccard_overlap']:.1%}")
        lines.append(f"- **Matched pairs:** {jaccard_overlap['n_matched_pairs']}")
        lines.append(f"- **Events in A that matched:** {jaccard_overlap['unique_a_matched']} / {jaccard_overlap['events_a']}")
        lines.append(f"- **Events in B that matched:** {jaccard_overlap['unique_b_matched']} / {jaccard_overlap['events_b']}")
        lines.append(f"- **Mean IoU of matches:** {jaccard_overlap['mean_iou']:.2f}")

        # Greedy efficiency
        upper_bound = jaccard_overlap.get('upper_bound_matches', 0)
        greedy_eff = jaccard_overlap.get('greedy_efficiency', 0)
        if upper_bound > 0:
            lines.append(f"- **Greedy efficiency:** {greedy_eff:.1%} ({jaccard_overlap['n_matched_pairs']}/{upper_bound} of upper bound)")

        lines.append("")
        lines.append("| Metric | Value | Interpretation |")
        lines.append("|--------|-------|----------------|")
        lines.append(f"| Precision (A→B) | {jaccard_overlap['precision_a']:.1%} | Fraction of A's events found in B |")
        lines.append(f"| Recall (B→A) | {jaccard_overlap['recall_b']:.1%} | Fraction of B's events found in A |")
        lines.append("")

    # IoU Distribution Diagnostics (validates threshold choices)
    iou_dist = invariance_curve.get("iou_distribution", {}) if invariance_curve else {}
    if iou_dist:
        lines.append("### IoU Distribution Diagnostics")
        lines.append("")
        lines.append("_Validates whether IoU thresholds are meaningful for this data._")
        lines.append("")
        lines.append("| Stat | Value |")
        lines.append("|------|-------|")
        lines.append(f"| max IoU | {iou_dist.get('max', 0):.3f} |")
        lines.append(f"| p99 | {iou_dist.get('p99', 0):.3f} |")
        lines.append(f"| p95 | {iou_dist.get('p95', 0):.3f} |")
        lines.append(f"| p75 | {iou_dist.get('p75', 0):.3f} |")
        lines.append(f"| median | {iou_dist.get('median', 0):.3f} |")
        lines.append(f"| n_pairs (IoU>0) | {iou_dist.get('n_pairs', 0)} |")
        lines.append("")

        # Interpret max IoU
        max_iou = iou_dist.get('max', 0)
        if max_iou < 0.5:
            lines.append(f"_**Note:** max IoU = {max_iou:.2f} < 0.5, so τ≥0.5 will always yield 0 matches (this is legitimate, not a bug)._")
        elif max_iou < 0.3:
            lines.append(f"_**Warning:** max IoU = {max_iou:.2f} is very low - windows may not overlap well._")
        lines.append("")

    # Invariance Curve (stability signature)
    curve_data = invariance_curve.get("curve", []) if isinstance(invariance_curve, dict) else invariance_curve
    if curve_data and len(curve_data) > 1:
        lines.append("### Invariance Stability (IoU Curve)")
        lines.append("")
        lines.append("_If Jaccard collapses when τ increases, matches are mostly weak overlaps._")
        lines.append("")
        lines.append("| IoU τ | Jaccard | Precision | Recall | Matched | Mean IoU | Greedy Eff. |")
        lines.append("|-------|---------|-----------|--------|---------|----------|-------------|")
        for pt in curve_data:
            lines.append(
                f"| {pt['iou_threshold']:.1f} | {pt['jaccard']:.1%} | "
                f"{pt['precision']:.1%} | {pt['recall']:.1%} | "
                f"{pt['n_matched']} | {pt['mean_iou']:.2f} | {pt.get('greedy_efficiency', 0):.0%} |"
            )
        lines.append("")

        # Interpret stability with specific numbers
        if len(curve_data) >= 2:
            j_01 = next((p["jaccard"] for p in curve_data if p["iou_threshold"] == 0.1), None)
            j_02 = next((p["jaccard"] for p in curve_data if p["iou_threshold"] == 0.2), None)
            j_04 = next((p["jaccard"] for p in curve_data if p["iou_threshold"] == 0.4), None)
            m_01 = next((p["mean_iou"] for p in curve_data if p["iou_threshold"] == 0.1), None)
            m_04 = next((p["mean_iou"] for p in curve_data if p["iou_threshold"] == 0.4), None)

            if j_01 is not None and j_04 is not None and j_01 > 0:
                decay_ratio = j_04 / j_01
                lines.append(f"**Curve interpretation:** From τ=0.1→0.4, Jaccard drops {j_01:.1%}→{j_04:.1%} ({decay_ratio:.0%} retained)")
                if m_01 is not None and m_04 is not None:
                    lines.append(f"Mean IoU increases {m_01:.2f}→{m_04:.2f} (stronger overlaps among survivors)")
                lines.append("")

                if decay_ratio > 0.7:
                    lines.append("_Stable curve: a core of well-aligned events persists at higher thresholds._")
                elif decay_ratio > 0.4:
                    lines.append("_Mixed overlap quality: plenty of weak-to-moderate overlaps, but a non-trivial core of strong overlaps persists._")
                else:
                    lines.append("_Fragile curve: Jaccard drops sharply, indicating mostly weak overlaps._")
                lines.append("")

    # Method 2: Step Bucketing (legacy)
    if jaccard_bucketed:
        lines.append("### Method 2: Step Bucketing (Legacy)")
        lines.append("")
        lines.append(f"_Matching: {jaccard_bucketed.get('matching_spec', f'bucket={step_tolerance * 2}')}_")
        lines.append("")
        lines.append(f"- **Jaccard (bucketed):** {jaccard_bucketed['jaccard']:.1%}")
        lines.append(f"- **Intersection:** {jaccard_bucketed['intersection']} event keys")
        lines.append(f"- **Union:** {jaccard_bucketed['union']} unique keys")
        lines.append("")

    # Interpretation (using overlap method as primary)
    if jaccard_overlap:
        j = jaccard_overlap['jaccard_overlap']
        if j > 0.5:
            interp = "**Strong event invariance** - majority of events match across runs."
        elif j > 0.25:
            interp = "**Moderate event invariance** - significant overlap but notable differences."
        else:
            interp = "**Weak event invariance** - selected events differ substantially (may be selection sensitivity, not signal difference)."
        lines.append(interp)
        lines.append("")

    # Composite Invariance Section
    if composite_invariance and (composite_invariance.get("n_composites_a", 0) > 0 or composite_invariance.get("n_composites_b", 0) > 0):
        lines.append("### Composite Event Invariance")
        lines.append("")
        lines.append("_Composite IoU threshold is higher (0.3) to reduce chance matches in broad multi-metric windows._")
        lines.append("")

        n_a = composite_invariance["n_composites_a"]
        n_b = composite_invariance["n_composites_b"]
        n_matched = composite_invariance["n_matched"]
        asymmetry = composite_invariance.get("asymmetry_ratio", 1.0)

        lines.append(f"- **Composites:** {n_a} (A) / {n_b} (B)")
        if asymmetry > 2.0:
            lines.append(f"- **Asymmetry:** {asymmetry:.1f}× (one run generates many more composites)")
        lines.append(f"- **Matched (IoU ≥ 0.3):** {n_matched}")
        lines.append(f"- **Jaccard:** {composite_invariance['jaccard']:.1%}")

        # Coverage metrics (density-normalized)
        coverage_a = composite_invariance.get("coverage_a", 0)
        coverage_b = composite_invariance.get("coverage_b", 0)
        lines.append(f"- **Coverage of A by B:** {coverage_a:.1%} ({composite_invariance.get('unique_a_matched', 0)}/{n_a} of A's composites matched)")
        lines.append(f"- **Coverage of B by A:** {coverage_b:.1%} ({composite_invariance.get('unique_b_matched', 0)}/{n_b} of B's composites matched)")

        lines.append(f"- **Unique steps:** {composite_invariance['unique_steps_a']} (A) / {composite_invariance['unique_steps_b']} (B)")

        if composite_invariance.get("strength_correlation") is not None:
            lines.append(f"- **Strength correlation:** {composite_invariance['strength_correlation']:.2f}")

        lines.append("")

        # Interpret asymmetry
        if asymmetry > 2.0:
            # Determine which run has more
            more_run = "A" if n_a > n_b else "B"
            less_run = "B" if n_a > n_b else "A"
            more_cov = coverage_a if n_a < n_b else coverage_b
            lines.append(f"**Asymmetry note:** Low Jaccard ({composite_invariance['jaccard']:.1%}) is driven by {more_run} generating {asymmetry:.1f}× more composites than {less_run}.")
            lines.append(f"Coverage of {less_run} ({more_cov:.1%}) is the more meaningful invariance metric here.")
            lines.append("")

        # Show matched composites
        matches = composite_invariance.get("matches", [])
        if matches:
            lines.append("**Matched composites:**")
            lines.append("")
            lines.append("| Layer | Step A | Step B | IoU | Strength A | Strength B |")
            lines.append("|-------|--------|--------|-----|------------|------------|")
            for m in matches[:10]:
                lines.append(
                    f"| {m['layer']} | {m['step_a']} | {m['step_b']} | "
                    f"{m['iou']:.2f} | {m['strength_a']:.3f} | {m['strength_b']:.3f} |"
                )
            if len(matches) > 10:
                lines.append(f"| ... | ({len(matches) - 10} more) | | | | |")
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

        # Density-normalized invariance explanation
        if len(runs_with_retention) == 2:
            r0, r1 = runs_with_retention[0].retention, runs_with_retention[1].retention
            density_ratio = r0.mean_candidates_post_per_series / max(r1.mean_candidates_post_per_series, 0.1)

            if abs(density_ratio - 1.0) > 0.3:
                lines.append("### Density Impact on Invariance")
                lines.append("")
                lines.append(f"Candidate density differs **{density_ratio:.1f}×** between runs.")
                lines.append("")
                lines.append("This affects event-level Jaccard because:")
                lines.append("- Higher density → more peaks above threshold")
                lines.append("- More peaks → more suppression collisions")
                lines.append("- Different survivors even if underlying 'energy landscape' is similar")
                lines.append("")
                lines.append(
                    "_To assess true invariance: compare raw candidates (before selection) "
                    "or use trajectory correlation instead of event Jaccard._"
                )
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

    # Conclusion Section - Split into Signal vs Event Invariance
    lines.append("## Conclusion")
    lines.append("")

    n_common = len(common_events)
    mean_corr_val = np.nanmean(corr_matrix.values[~np.eye(len(corr_matrix), dtype=bool)])

    # Get event invariance metrics
    event_jaccard = jaccard_overlap.get("jaccard_overlap", 0) if jaccard_overlap else 0
    event_precision = jaccard_overlap.get("precision_a", 0) if jaccard_overlap else 0
    composite_jaccard = composite_invariance.get("jaccard", 0) if composite_invariance else 0

    lines.append("### Signal Invariance (Trajectory Level)")
    lines.append("")

    # Trajectory correlation assessment
    if not np.isnan(mean_corr_val):
        if mean_corr_val > 0.95:
            signal_strength = "strong"
            lines.append(f"- **Trajectory correlation:** {mean_corr_val:.2f} (strong)")
        elif mean_corr_val > 0.8:
            signal_strength = "moderate"
            lines.append(f"- **Trajectory correlation:** {mean_corr_val:.2f} (moderate)")
        else:
            signal_strength = "weak"
            lines.append(f"- **Trajectory correlation:** {mean_corr_val:.2f} (weak)")
    else:
        signal_strength = "unknown"
        lines.append("- **Trajectory correlation:** N/A")

    if std_div < 0.1:
        lines.append(f"- **Diversity consistency:** strong (std={std_div:.2f})")

    lines.append("")
    lines.append("### Event Invariance (Selected Peaks)")
    lines.append("")

    # Event-level assessment
    if event_jaccard > 0.4:
        event_strength = "moderate"
        lines.append(f"- **Event Jaccard (IoU):** {event_jaccard:.1%} (moderate)")
    elif event_jaccard > 0.2:
        event_strength = "weak-moderate"
        lines.append(f"- **Event Jaccard (IoU):** {event_jaccard:.1%} (weak-moderate)")
    else:
        event_strength = "weak"
        lines.append(f"- **Event Jaccard (IoU):** {event_jaccard:.1%} (weak)")

    lines.append(f"- **Common events (strict):** {n_common}")
    if composite_invariance and composite_invariance.get("n_composites_a", 0) > 0:
        lines.append(f"- **Composite Jaccard:** {composite_jaccard:.1%}")

    lines.append("")

    # Explain discrepancy if signal is strong but event is weak
    if signal_strength in ("strong", "moderate") and event_strength == "weak":
        lines.append("### Interpretation")
        lines.append("")
        lines.append(
            "**Signal-Event Discrepancy:** High trajectory correlation but low event overlap "
            "typically indicates **selection sensitivity**, not signal difference."
        )
        lines.append("")
        lines.append("Likely causes:")

        # Check for density difference
        if len(runs) == 2 and runs[0].retention and runs[1].retention:
            r0, r1 = runs[0].retention, runs[1].retention
            density_ratio = r0.mean_candidates_post_per_series / max(r1.mean_candidates_post_per_series, 0.1)
            if abs(density_ratio - 1.0) > 0.3:
                lines.append(
                    f"- **Candidate density differs {density_ratio:.1f}×** — more candidates → more suppression → different survivors"
                )

        lines.append("- **Peak selection competition** — same underlying transitions, different discrete peak choices")
        lines.append("- **Step jitter** — events at step 70 vs 75 describe the same region but don't match")
        lines.append("")
        lines.append(
            "_Recommendation: Focus on common events as the 'invariant backbone'; "
            "event-level Jaccard measures selection stability, not dynamic similarity._"
        )
        lines.append("")

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
            lines.append(f"**Warning:** Final loss differs by {loss_diff_pct:.1f}% - different learning trajectories possible")
            lines.append("")

    # Overall assessment (clearer two-lane summary)
    lines.append("### Summary")
    lines.append("")
    lines.append(f"- **Signal invariance (trajectory):** {signal_strength} ({mean_corr_val:.2f})" if not np.isnan(mean_corr_val) else "- **Signal invariance:** unknown")
    lines.append(f"- **Event invariance (selected peaks):** {event_strength} (Jaccard {event_jaccard:.1%})")
    if n_common > 0:
        lines.append(f"- **Invariant backbone:** {n_common} common events")

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
