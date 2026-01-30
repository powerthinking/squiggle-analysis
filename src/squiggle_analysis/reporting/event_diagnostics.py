"""Event distribution diagnostics for squiggle analysis reports.

This module provides functions to analyze and visualize the temporal and spatial
distribution of detected events, helping to identify wave propagation patterns
and temporal clustering.
"""

from __future__ import annotations

import json
import math
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


def compute_capture_grid_summary(
    step_grid: list[int],
    suppression_radius: int = 15,
) -> dict:
    """Compute summary statistics for the capture grid.

    Args:
        step_grid: Sorted list of capture step values
        suppression_radius: Event suppression radius in steps (for risk calculation)

    Returns:
        Dictionary with grid summary statistics
    """
    if not step_grid or len(step_grid) < 2:
        return {
            "n_captures": len(step_grid) if step_grid else 0,
            "min_step": step_grid[0] if step_grid else None,
            "max_step": step_grid[-1] if step_grid else None,
            "deltas": [],
            "median_delta": None,
            "min_delta": None,
            "max_delta": None,
            "is_uniform": True,
            "uniformity_cv": 0.0,
            "resolution_ratio": 0.0,
            "quantization_risk": "LOW",
        }

    # Compute inter-step deltas
    deltas = [step_grid[i + 1] - step_grid[i] for i in range(len(step_grid) - 1)]

    n_captures = len(step_grid)
    min_step = step_grid[0]
    max_step = step_grid[-1]
    median_delta = sorted(deltas)[len(deltas) // 2]
    min_delta = min(deltas)
    max_delta = max(deltas)
    mean_delta = sum(deltas) / len(deltas)

    # Coefficient of variation for uniformity check
    if mean_delta > 0:
        delta_std = (sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)) ** 0.5
        uniformity_cv = delta_std / mean_delta
    else:
        uniformity_cv = 0.0

    # Uniform if CV < 0.1 (less than 10% variation)
    is_uniform = uniformity_cv < 0.1

    # Quantization risk based on resolution relative to suppression radius
    # resolution_ratio = median_delta / suppression_radius
    # > 0.5 → HIGH (coarse), 0.2-0.5 → MEDIUM, < 0.2 → LOW (fine-grained)
    resolution_ratio = median_delta / max(1, suppression_radius)
    if resolution_ratio > 0.5:
        quantization_risk = "HIGH"
    elif resolution_ratio > 0.2:
        quantization_risk = "MEDIUM"
    else:
        quantization_risk = "LOW"

    return {
        "n_captures": n_captures,
        "min_step": min_step,
        "max_step": max_step,
        "deltas": deltas,
        "median_delta": median_delta,
        "min_delta": min_delta,
        "max_delta": max_delta,
        "is_uniform": is_uniform,
        "uniformity_cv": round(uniformity_cv, 3),
        "resolution_ratio": round(resolution_ratio, 3),
        "quantization_risk": quantization_risk,
    }


def compute_sanity_triangle(
    events: pd.DataFrame | None,
    step_grid: list[int],
    suppression_radius: int = 15,
    warmup_buffer_points: int = 2,
) -> dict:
    """Compute the 'sanity triangle' that ties grid + suppression + windows together.

    Args:
        events: Events DataFrame
        step_grid: Capture step grid
        suppression_radius: Event suppression radius in steps
        warmup_buffer_points: Warmup buffer in grid points

    Returns:
        Dictionary with derived sanity metrics
    """
    result = {
        "median_window_len_grid": None,
        "median_window_len_steps": None,
        "suppression_ratio": None,
        "warmup_buffer_steps": None,
        "suppression_interpretation": None,
    }

    if not step_grid or len(step_grid) < 2:
        return result

    # Compute median grid spacing
    deltas = [step_grid[i + 1] - step_grid[i] for i in range(len(step_grid) - 1)]
    median_delta = sorted(deltas)[len(deltas) // 2]

    # Warmup buffer in steps
    result["warmup_buffer_steps"] = warmup_buffer_points * median_delta

    if events is None or events.empty:
        return result

    # Compute median window length from events
    if "start_step" in events.columns and "end_step" in events.columns:
        window_lens = events["end_step"] - events["start_step"]
        window_lens = window_lens[window_lens > 0]
        if not window_lens.empty:
            median_window_steps = int(window_lens.median())
            result["median_window_len_steps"] = median_window_steps

            # Convert to grid points
            median_window_grid = median_window_steps / median_delta if median_delta > 0 else 0
            result["median_window_len_grid"] = round(median_window_grid, 1)

            # Suppression ratio = suppression_radius / median_window_len
            if median_window_steps > 0:
                suppression_ratio = suppression_radius / median_window_steps
                result["suppression_ratio"] = round(suppression_ratio, 2)

                # Interpretation (calibrated: 1× means suppression ≈ window size)
                # <0.5× = weak (suppression much smaller than typical window)
                # 0.5-1.5× = moderate (suppression ≈ window size)
                # >1.5× = strong (suppression much larger than typical window)
                if suppression_ratio > 1.5:
                    result["suppression_interpretation"] = "strong"
                elif suppression_ratio >= 0.5:
                    result["suppression_interpretation"] = "moderate"
                else:
                    result["suppression_interpretation"] = "weak"

    return result


def format_sanity_triangle_md(sanity: dict) -> str:
    """Format sanity triangle as a compact summary line."""
    lines = []

    median_grid = sanity.get("median_window_len_grid")
    median_steps = sanity.get("median_window_len_steps")
    supp_ratio = sanity.get("suppression_ratio")
    warmup_steps = sanity.get("warmup_buffer_steps")
    interp = sanity.get("suppression_interpretation")

    if median_grid is not None:
        lines.append(f"- **Typical event window**: {median_grid} grid points ({median_steps} steps)")

    if supp_ratio is not None:
        lines.append(f"- **Suppression ratio**: {supp_ratio}× window length ({interp})")
        if interp == "strong":
            lines.append("  - _Suppression is larger than typical events; peaks well-separated_")
        elif interp == "weak":
            lines.append("  - _Suppression radius < event window; may see overlapping events_")

    if warmup_steps is not None:
        lines.append(f"- **Warmup buffer**: {warmup_steps} steps")

    return "\n".join(lines)


def format_capture_grid_summary_md(summary: dict, sanity: dict | None = None) -> str:
    """Format capture grid summary as Markdown section."""
    lines = ["## Capture Grid Summary\n"]

    n = summary.get("n_captures", 0)
    if n == 0:
        lines.append("_No capture grid data._")
        return "\n".join(lines)

    min_step = summary.get("min_step", 0)
    max_step = summary.get("max_step", 0)
    median_delta = summary.get("median_delta")
    min_delta = summary.get("min_delta")
    max_delta = summary.get("max_delta")
    is_uniform = summary.get("is_uniform", False)
    cv = summary.get("uniformity_cv", 0)
    resolution_ratio = summary.get("resolution_ratio", 0)
    risk = summary.get("quantization_risk", "UNKNOWN")

    lines.append(f"- **Captures**: {n} points over steps {min_step}-{max_step}")
    if median_delta is not None:
        lines.append(f"- **Step intervals**: median={median_delta}, min={min_delta}, max={max_delta}")
    lines.append(f"- **Uniformity**: {'Yes' if is_uniform else 'No'} (CV={cv:.3f})")
    lines.append(f"- **Resolution ratio**: {resolution_ratio:.3f} (median_delta / suppression_radius)")

    # Quantization risk indicator based on resolution ratio
    if risk == "HIGH":
        lines.append(
            f"- ⚠️ **Grid quantization risk**: {risk} - capture spacing is coarse relative to "
            "suppression radius (ratio > 0.5)"
        )
    elif risk == "MEDIUM":
        lines.append(f"- **Grid quantization risk**: {risk} (ratio 0.2-0.5)")
    else:
        lines.append(f"- ✓ **Grid quantization risk**: {risk} - fine-grained capture (ratio < 0.2)")

    # Add sanity triangle if provided
    if sanity:
        lines.append("")
        lines.append("### Sanity Triangle\n")
        lines.append(format_sanity_triangle_md(sanity))

    return "\n".join(lines)


def compute_peak_step_entropy(events: pd.DataFrame, step_grid: list[int]) -> dict:
    """Compute entropy of peak step distribution.

    Low entropy = peaks are concentrated at few steps (suspicious clustering).
    High entropy = peaks are spread across many steps (good diversity).

    Args:
        events: Events DataFrame with 'step' column
        step_grid: Capture step grid

    Returns:
        Dictionary with entropy metrics
    """
    base_result = {
        "n_events": 0,
        "n_unique_steps": 0,
        "entropy": 0.0,
        "max_entropy": 0.0,
        "normalized_entropy": 0.0,
        "top_step_share": 0.0,
        "top_3_step_share": 0.0,
        "concentration_warning": False,
        "aliasing_warning": False,
        "dominant_residue_share": 0.0,
    }

    if events is None or events.empty or not step_grid:
        return base_result

    if "step" not in events.columns:
        base_result["n_events"] = len(events)
        return base_result

    # Count events per step
    step_counts: dict[int, int] = {}
    peak_steps: list[int] = []
    for step in events["step"]:
        s = int(step)
        step_counts[s] = step_counts.get(s, 0) + 1
        peak_steps.append(s)

    n_events = len(events)
    n_unique_steps = len(step_counts)

    if n_events == 0 or n_unique_steps == 0:
        base_result["n_events"] = n_events
        base_result["n_unique_steps"] = n_unique_steps
        return base_result

    # Shannon entropy
    probs = [count / n_events for count in step_counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    # Max entropy = uniform distribution over all captured steps
    max_entropy = math.log2(len(step_grid)) if step_grid else math.log2(n_unique_steps)

    # Normalized entropy (0 = all at one step, 1 = uniform)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Top step share (fraction at most common step)
    sorted_counts = sorted(step_counts.values(), reverse=True)
    top_step_share = sorted_counts[0] / n_events if n_events > 0 else 0.0

    # Top 3 step share
    top_3_count = sum(sorted_counts[:3])
    top_3_step_share = top_3_count / n_events if n_events > 0 else 0.0

    # Concentration warning if normalized entropy < 0.5 OR top_step_share > 0.20
    concentration_warning = normalized_entropy < 0.5 or top_step_share > 0.20

    # Find the most common step(s)
    max_count = max(step_counts.values())
    top_steps_list = [step for step, count in step_counts.items() if count == max_count]

    # Aliasing check: compare peak residue distribution to capture grid residue distribution
    # A valid aliasing warning requires peaks to be MORE concentrated than the grid itself
    aliasing_warning = False
    dominant_residue_share = 0.0
    grid_dominant_residue_share = 0.0
    median_delta = 1

    if len(step_grid) >= 2:
        deltas = [step_grid[i + 1] - step_grid[i] for i in range(len(step_grid) - 1)]
        median_delta = sorted(deltas)[len(deltas) // 2]

    if median_delta > 1 and n_events >= 5:
        # Count residue classes for peaks
        peak_residue_counts: dict[int, int] = {}
        for s in peak_steps:
            r = s % median_delta
            peak_residue_counts[r] = peak_residue_counts.get(r, 0) + 1

        # Count residue classes for capture grid
        grid_residue_counts: dict[int, int] = {}
        for s in step_grid:
            r = s % median_delta
            grid_residue_counts[r] = grid_residue_counts.get(r, 0) + 1

        if peak_residue_counts:
            max_peak_residue = max(peak_residue_counts.values())
            dominant_residue_share = max_peak_residue / n_events

        if grid_residue_counts:
            max_grid_residue = max(grid_residue_counts.values())
            grid_dominant_residue_share = max_grid_residue / len(step_grid)

        # Only warn if peaks are SIGNIFICANTLY more concentrated than the grid
        # If grid already has 100% on one residue (uniform spacing), peaks can't be more concentrated
        # Threshold: peaks must be 20% more concentrated than grid AND > 60% on one residue
        concentration_excess = dominant_residue_share - grid_dominant_residue_share
        if median_delta >= 3 and concentration_excess > 0.2 and dominant_residue_share > 0.6:
            aliasing_warning = True

    return {
        "n_events": n_events,
        "n_unique_steps": n_unique_steps,
        "entropy": round(entropy, 3),
        "max_entropy": round(max_entropy, 3),
        "normalized_entropy": round(normalized_entropy, 3),
        "top_step_share": round(top_step_share, 3),
        "top_3_step_share": round(top_3_step_share, 3),
        "concentration_warning": concentration_warning,
        "aliasing_warning": aliasing_warning,
        "dominant_residue_share": round(dominant_residue_share, 3),
        "grid_dominant_residue_share": round(grid_dominant_residue_share, 3),
        "top_step": top_steps_list[0] if top_steps_list else None,
        "top_step_count": max_count,
        "top_step_fraction": round(max_count / n_events, 3) if n_events > 0 else 0,
        "n_grid_points": len(step_grid),
    }


def format_peak_step_entropy_md(entropy_stats: dict) -> str:
    """Format peak step entropy as Markdown section."""
    lines = ["## Peak-Step Entropy\n"]

    n_events = entropy_stats.get("n_events", 0)
    if n_events == 0:
        lines.append("_No events to analyze._")
        return "\n".join(lines)

    n_unique = entropy_stats.get("n_unique_steps", 0)
    n_grid = entropy_stats.get("n_grid_points", n_unique)
    entropy = entropy_stats.get("entropy", 0)
    max_entropy = entropy_stats.get("max_entropy", 0)
    norm_entropy = entropy_stats.get("normalized_entropy", 0)
    top_step_share = entropy_stats.get("top_step_share", 0)
    top_3_step_share = entropy_stats.get("top_3_step_share", 0)
    concentration_warning = entropy_stats.get("concentration_warning", False)
    aliasing_warning = entropy_stats.get("aliasing_warning", False)
    dominant_residue = entropy_stats.get("dominant_residue_share", 0)
    grid_dominant_residue = entropy_stats.get("grid_dominant_residue_share", 0)
    top_step = entropy_stats.get("top_step")
    top_count = entropy_stats.get("top_step_count", 0)

    # Compute expected unique steps baseline (coupon collector approximation)
    # E[unique] ≈ n_grid * (1 - (1 - 1/n_grid)^n_events)
    if n_grid > 0 and n_events > 0:
        expected_unique = n_grid * (1 - (1 - 1 / n_grid) ** n_events)
        expected_unique = min(expected_unique, n_grid)  # Can't exceed grid size
    else:
        expected_unique = 0

    lines.append(f"- **Events**: {n_events} across {n_unique} unique steps")
    lines.append(f"- **Grid points**: {n_grid} capture steps")
    lines.append(f"- **Expected unique** (uniform): {expected_unique:.1f} steps")

    # Show observed vs expected ratio
    if expected_unique > 0:
        unique_ratio = n_unique / expected_unique
        if unique_ratio < 0.7:
            lines.append(f"- **Unique ratio**: {unique_ratio:.2f}× expected (events cluster)")
        elif unique_ratio > 1.3:
            lines.append(f"- **Unique ratio**: {unique_ratio:.2f}× expected (broader than uniform)")
        else:
            lines.append(f"- **Unique ratio**: {unique_ratio:.2f}× expected (close to uniform)")

    lines.append(f"- **Entropy**: {entropy:.3f} / {max_entropy:.3f} (normalized: {norm_entropy:.3f})")
    lines.append(f"- **Concentration**: top step = {top_step_share:.1%}, top 3 steps = {top_3_step_share:.1%}")

    if top_step is not None:
        lines.append(f"- **Most common step**: {top_step} ({top_count} events)")

    if concentration_warning:
        # Explain which condition triggered the warning
        reasons = []
        if norm_entropy < 0.5:
            reasons.append(f"low entropy ({norm_entropy:.2f} < 0.5)")
        if top_step_share > 0.20:
            reasons.append(f"top step dominates ({top_step_share:.1%} > 20%)")

        lines.append(
            f"\n⚠️ **Concentration warning**: Events are clustered - {', '.join(reasons)}. "
            "This may indicate cadence/grid artifacts or a true global transition."
        )
    else:
        lines.append(f"\n✓ Events are reasonably distributed (entropy {norm_entropy:.2f}, top step {top_step_share:.1%}).")

    # Aliasing check (now with grid comparison context)
    if aliasing_warning:
        lines.append(
            f"\n⚠️ **Aliasing warning**: Peak residue concentration ({dominant_residue:.0%}) "
            f"exceeds grid baseline ({grid_dominant_residue:.0%}). "
            "Events may prefer specific grid offsets beyond what capture cadence would predict."
        )
    elif dominant_residue > 0.5 and grid_dominant_residue > 0.9:
        # No warning because grid itself is highly concentrated
        lines.append(
            f"\n_Aliasing check: Grid residues are {grid_dominant_residue:.0%} concentrated "
            "(uniform spacing), so peak residue concentration is expected._"
        )

    return "\n".join(lines)


def compute_composite_step_diversity(events: pd.DataFrame) -> dict:
    """Compute diversity of composite event steps.

    Args:
        events: Events DataFrame with 'event_type' and 'step' columns

    Returns:
        Dictionary with composite diversity metrics
    """
    if events is None or events.empty:
        return {
            "n_composites": 0,
            "n_unique_steps": 0,
            "diversity_warning": False,
            "unique_steps": [],
        }

    if "event_type" not in events.columns or "step" not in events.columns:
        return {
            "n_composites": 0,
            "n_unique_steps": 0,
            "diversity_warning": False,
            "unique_steps": [],
        }

    composites = events[events["event_type"] == "change_point_composite"]
    if composites.empty:
        return {
            "n_composites": 0,
            "n_unique_steps": 0,
            "diversity_warning": False,
            "unique_steps": [],
        }

    unique_steps = sorted([int(s) for s in composites["step"].unique()])
    n_composites = len(composites)
    n_unique_steps = len(unique_steps)

    # Warn if fewer than 3 unique steps
    diversity_warning = n_unique_steps < 3 and n_composites >= 3

    # Composite strength stats (if available)
    strength_stats = {}
    strong_composite_info = {}

    if "composite_strength" in composites.columns:
        strengths = composites["composite_strength"].dropna()
        if not strengths.empty:
            p75 = float(strengths.quantile(0.75))
            strength_floor = 0.6

            strength_stats = {
                "mean_strength": round(float(strengths.mean()), 3),
                "max_strength": round(float(strengths.max()), 3),
                "min_strength": round(float(strengths.min()), 3),
                "p75_strength": round(p75, 3),
            }

            # Strong composites = those with strength >= max(p75, floor)
            strength_threshold = max(p75, strength_floor) if n_composites >= 4 else strength_floor
            strong_mask = composites["composite_strength"] >= strength_threshold
            strong_composites = composites[strong_mask]

            strong_composite_info["strength_threshold_used"] = round(strength_threshold, 3)
            strong_composite_info["strength_floor"] = strength_floor

            if not strong_composites.empty:
                strong_steps = sorted([int(s) for s in strong_composites["step"].unique()])
                n_strong = len(strong_composites)
                n_strong_unique_steps = len(strong_steps)

                # Concentration: max fraction of strong composites at any single step
                strong_step_counts: dict[int, int] = {}
                for s in strong_composites["step"]:
                    strong_step_counts[int(s)] = strong_step_counts.get(int(s), 0) + 1
                max_strong_at_step = max(strong_step_counts.values()) if strong_step_counts else 0
                strong_step_concentration = max_strong_at_step / n_strong if n_strong > 0 else 0

                strong_composite_info.update({
                    "n_strong_composites": n_strong,
                    "n_strong_unique_steps": n_strong_unique_steps,
                    "strong_steps": strong_steps,
                    "strong_step_concentration": round(strong_step_concentration, 3),
                })

                # Additional warning if strong composites cluster at 1-2 steps
                if n_strong_unique_steps < 3 and n_strong >= 3:
                    diversity_warning = True

    return {
        "n_composites": n_composites,
        "n_unique_steps": n_unique_steps,
        "diversity_warning": diversity_warning,
        "unique_steps": unique_steps,
        **strength_stats,
        **strong_composite_info,
    }


def format_composite_step_diversity_md(diversity: dict) -> str:
    """Format composite step diversity as Markdown section."""
    lines = ["## Composite Step Diversity\n"]

    n = diversity.get("n_composites", 0)
    if n == 0:
        lines.append("_No composite events._")
        return "\n".join(lines)

    n_unique = diversity.get("n_unique_steps", 0)
    unique_steps = diversity.get("unique_steps", [])
    warning = diversity.get("diversity_warning", False)

    lines.append(f"- **Composites**: {n} across {n_unique} unique steps")

    if n_unique <= 10:
        lines.append(f"- **Steps**: {unique_steps}")

    # Strength stats if available
    if "mean_strength" in diversity:
        p75 = diversity.get("p75_strength", 0)
        lines.append(
            f"- **Strength**: mean={diversity['mean_strength']:.3f}, "
            f"range=[{diversity['min_strength']:.3f}, {diversity['max_strength']:.3f}], "
            f"p75={p75:.3f}"
        )

    # Strong composite concentration
    if "n_strong_composites" in diversity:
        n_strong = diversity["n_strong_composites"]
        n_strong_unique = diversity["n_strong_unique_steps"]
        threshold = diversity.get("strength_threshold_used", 0.6)
        floor = diversity.get("strength_floor", 0.6)
        p75 = diversity.get("p75_strength", 0)
        concentration = diversity["strong_step_concentration"]
        strong_steps = diversity.get("strong_steps", [])

        lines.append(
            f"- **Strong composites** (strength ≥ max(p75={p75:.2f}, floor={floor})={threshold:.2f}): "
            f"{n_strong} across {n_strong_unique} steps"
        )
        if n_strong_unique <= 5:
            lines.append(f"  - Steps: {strong_steps}")
        if concentration > 0.5:
            lines.append(f"  - ⚠️ Concentration: {concentration:.0%} of strong composites at single step")

    # Add note about composite repositioning and backward compatibility
    lines.append("")
    lines.append("### Backward Compatibility Note\n")
    lines.append(
        "Composite peak step is now **union-window center** (median of covered indices), "
        "not the max-delta point. Comparisons across report versions should match composites "
        "by **(start_step, end_step) overlap**, not step equality."
    )

    if warning:
        lines.append(
            f"\n⚠️ **Low diversity warning**: Only {n_unique} unique composite steps "
            f"(< 3) with {n} composites. Composites may be clustering at specific steps."
        )
    elif n_unique >= 3:
        lines.append(f"\n✓ Composites span {n_unique} distinct steps (diversity OK).")

    return "\n".join(lines)


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


def compute_composite_window_stats(events: pd.DataFrame) -> dict:
    """Compute detailed composite window statistics.

    Args:
        events: Events DataFrame with composite events

    Returns:
        Dictionary with window length, alignment, and occupancy statistics
    """
    if events is None or events.empty:
        return {}

    if "event_type" not in events.columns:
        return {}

    composites = events[events["event_type"] == "change_point_composite"]
    if composites.empty:
        return {}

    result: dict = {"n_composites": len(composites)}

    # Window length stats
    if "composite_window_len" in composites.columns:
        window_lens = composites["composite_window_len"].dropna()
        if not window_lens.empty:
            result["window_len_median"] = float(window_lens.median())
            result["window_len_min"] = float(window_lens.min())
            result["window_len_max"] = float(window_lens.max())

    # Alignment std stats (lower = tighter co-occurrence)
    if "composite_alignment_std" in composites.columns:
        alignment_stds = composites["composite_alignment_std"].dropna()
        if not alignment_stds.empty:
            result["alignment_std_median"] = float(alignment_stds.median())
            result["alignment_std_p75"] = float(alignment_stds.quantile(0.75))
            result["alignment_std_max"] = float(alignment_stds.max())

    # Min metric occupancy stats (higher = tighter participation)
    if "composite_min_metric_occupancy" in composites.columns:
        occupancies = composites["composite_min_metric_occupancy"].dropna()
        if not occupancies.empty:
            result["min_occupancy_median"] = float(occupancies.median())
            result["min_occupancy_p25"] = float(occupancies.quantile(0.25))
            result["min_occupancy_min"] = float(occupancies.min())

    # Coverage ratio stats
    if "composite_coverage_ratio" in composites.columns:
        coverages = composites["composite_coverage_ratio"].dropna()
        if not coverages.empty:
            result["coverage_ratio_median"] = float(coverages.median())
            result["coverage_ratio_p25"] = float(coverages.quantile(0.25))

    # Strength stats
    if "composite_strength" in composites.columns:
        strengths = composites["composite_strength"].dropna()
        if not strengths.empty:
            result["strength_median"] = float(strengths.median())
            result["strength_p75"] = float(strengths.quantile(0.75))
            result["strength_max"] = float(strengths.max())

    # N metrics stats
    if "composite_n_metrics" in composites.columns:
        n_metrics = composites["composite_n_metrics"].dropna()
        if not n_metrics.empty:
            result["n_metrics_median"] = float(n_metrics.median())
            result["n_metrics_max"] = int(n_metrics.max())

    return result


def format_composite_window_stats_md(stats: dict) -> str:
    """Format composite window stats as Markdown section."""
    lines = ["### Composite Window Statistics\n"]

    n = stats.get("n_composites", 0)
    if n == 0:
        lines.append("_No composite events._")
        return "\n".join(lines)

    lines.append(f"_Statistics across {n} composite events:_\n")

    # Window length
    if "window_len_median" in stats:
        lines.append(
            f"- **Union window length**: "
            f"median={stats['window_len_median']:.1f}, "
            f"range=[{stats['window_len_min']:.0f}, {stats['window_len_max']:.0f}]"
        )

    # Alignment std
    if "alignment_std_median" in stats:
        lines.append(
            f"- **Alignment std** (peak index spread): "
            f"median={stats['alignment_std_median']:.2f}, "
            f"p75={stats['alignment_std_p75']:.2f}"
        )
        if stats["alignment_std_median"] < 1.0:
            lines.append("  - _Low alignment std: metrics peak at nearly the same index (tight co-occurrence)_")
        elif stats["alignment_std_p75"] > 3.0:
            lines.append("  - _High alignment std: peak indices vary significantly (loose co-occurrence)_")

    # Min metric occupancy
    if "min_occupancy_median" in stats:
        lines.append(
            f"- **Min metric occupancy**: "
            f"median={stats['min_occupancy_median']:.2f}, "
            f"p25={stats['min_occupancy_p25']:.2f}"
        )
        if stats["min_occupancy_p25"] < 0.3:
            lines.append("  - _Some composites have weak participants (occupancy < 30%)_")

    # Coverage ratio
    if "coverage_ratio_median" in stats:
        lines.append(
            f"- **Coverage ratio**: "
            f"median={stats['coverage_ratio_median']:.2f}, "
            f"p25={stats['coverage_ratio_p25']:.2f}"
        )

    # Strength
    if "strength_median" in stats:
        lines.append(
            f"- **Composite strength**: "
            f"median={stats['strength_median']:.3f}, "
            f"p75={stats['strength_p75']:.3f}, "
            f"max={stats['strength_max']:.3f}"
        )

    # N metrics
    if "n_metrics_median" in stats:
        lines.append(
            f"- **Participating metrics**: "
            f"median={stats['n_metrics_median']:.1f}, "
            f"max={stats['n_metrics_max']}"
        )

    return "\n".join(lines)


def format_composite_sample_rows_md(events: pd.DataFrame, max_rows: int = 10) -> str:
    """Format sample composite rows showing window details."""
    lines = ["### Sample Composite Events\n"]

    if events is None or events.empty:
        lines.append("_No events._")
        return "\n".join(lines)

    if "event_type" not in events.columns:
        lines.append("_No event_type column._")
        return "\n".join(lines)

    composites = events[events["event_type"] == "change_point_composite"].copy()
    if composites.empty:
        lines.append("_No composite events._")
        return "\n".join(lines)

    # Sort by strength descending and take top N
    if "composite_strength" in composites.columns:
        composites = composites.sort_values("composite_strength", ascending=False)

    sample = composites.head(max_rows)

    # Build table with key fields
    cols_to_show = [
        ("event_id", "ID"),
        ("layer", "Layer"),
        ("composite_start_step", "Start"),
        ("composite_center_step", "Center"),
        ("composite_end_step", "End"),
        ("composite_n_metrics", "Metrics"),
        ("composite_strength", "Strength"),
        ("composite_alignment_std", "Align Std"),
        ("composite_coverage_ratio", "Coverage"),
        ("composite_min_metric_occupancy", "Min Occ"),
    ]

    # Filter to columns that exist
    available_cols = [(c, label) for c, label in cols_to_show if c in sample.columns]
    if not available_cols:
        lines.append("_Composite window fields not available._")
        return "\n".join(lines)

    # Build header
    headers = [label for _, label in available_cols]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    # Build rows
    for _, row in sample.iterrows():
        values = []
        for col, _ in available_cols:
            val = row.get(col)
            if pd.isna(val):
                values.append("-")
            elif isinstance(val, float):
                if col in ("composite_strength", "composite_alignment_std", "composite_coverage_ratio", "composite_min_metric_occupancy"):
                    values.append(f"{val:.3f}")
                else:
                    values.append(f"{val:.0f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")

    lines.append(f"\n_Showing top {len(sample)} composites by strength._")
    return "\n".join(lines)


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


def load_detection_summary(run_id: str, analysis_id: str | None = None) -> pd.DataFrame | None:
    """Load detection summary parquet if it exists."""
    summary_path = paths.detection_summary_path(run_id, analysis_id)
    if summary_path.exists():
        return pd.read_parquet(summary_path)
    return None


def load_warmup_info(run_id: str, analysis_id: str | None = None) -> dict | None:
    """Load warmup info JSON if it exists."""
    warmup_path = paths.run_dir(run_id) / "analysis" / (analysis_id or "default") / "warmup_info.json"
    if warmup_path.exists():
        return json.loads(warmup_path.read_text())
    return None


def format_warmup_summary_md(
    events: pd.DataFrame | None,
    warmup_info: dict | None,
    n_composites_filtered: int = 0,
    composite_buffer_points: int = 2,
) -> str:
    """Format warmup-aware event summary as Markdown section.

    Shows:
    - Warmup end step and first eligible capture step
    - Event counts: by config vs by eligibility
    - Composites filtered near warmup
    """
    lines = ["## Warmup-Aware Event Summary\n"]

    if warmup_info is None:
        lines.append("_No warmup info available._")
        return "\n".join(lines)

    warmup_end_step = warmup_info.get("warmup_end_step", 0)
    first_capture_after = warmup_info.get("first_capture_after_warmup")
    step_min = warmup_info.get("step_grid_min", 0)
    step_max = warmup_info.get("step_grid_max", 0)
    step_count = warmup_info.get("step_grid_count", 0)

    # Use new naming; fall back to legacy names for compatibility
    warmup_insert_idx = warmup_info.get("warmup_end_insert_idx", warmup_info.get("grid_index_of_warmup_end", 0))
    warmup_left_idx = warmup_info.get("warmup_end_left_idx")
    first_eligible_idx = warmup_info.get("first_eligible_idx", warmup_info.get("grid_index_first_after_warmup"))

    lines.append("### Warmup Period\n")
    lines.append(f"- **Warmup end step** (config): {warmup_end_step}")
    if first_capture_after is not None:
        lines.append(f"- **First eligible capture step**: {first_capture_after}")
        if first_capture_after != warmup_end_step + 1:
            lines.append(f"  - _(warmup ends between grid points)_")

    # Show grid indices with clear naming
    idx_info = []
    if warmup_left_idx is not None:
        idx_info.append(f"last_warmup_idx={warmup_left_idx}")
    idx_info.append(f"insert_idx={warmup_insert_idx}")
    if first_eligible_idx is not None:
        idx_info.append(f"first_eligible_idx={first_eligible_idx}")
    lines.append(f"- **Grid indices**: {', '.join(idx_info)}")
    lines.append(f"- Step range: {step_min} - {step_max} ({step_count} capture points)")
    lines.append("")

    if events is None or events.empty:
        lines.append("### Event Counts\n")
        lines.append("_No events to count._")
        return "\n".join(lines)

    # Compute event counts - TWO definitions
    total_events = len(events)

    # 1. By config: step > warmup_end_step
    post_warmup_config_mask = (
        events["step"] > warmup_end_step
        if "step" in events.columns
        else pd.Series([False] * len(events))
    )

    # 2. By eligibility: step >= first_capture_after_warmup
    if first_capture_after is not None:
        post_warmup_eligible_mask = (
            events["step"] >= first_capture_after
            if "step" in events.columns
            else pd.Series([False] * len(events))
        )
    else:
        post_warmup_eligible_mask = post_warmup_config_mask

    events_post_config = int(post_warmup_config_mask.sum())
    events_post_eligible = int(post_warmup_eligible_mask.sum())
    events_pre_warmup = total_events - events_post_config

    lines.append("### Event Counts\n")
    lines.append("| Scope | Total | Single-Metric | Composite |")
    lines.append("|-------|-------|---------------|-----------|")

    # Total row
    if "event_type" in events.columns:
        n_single = int((events["event_type"] == "change_point").sum())
        n_composite = int((events["event_type"] == "change_point_composite").sum())
    else:
        n_single = total_events
        n_composite = 0
    lines.append(f"| **All events** | {total_events} | {n_single} | {n_composite} |")

    # Post-warmup by config
    if "event_type" in events.columns:
        n_single_post = int(((events["event_type"] == "change_point") & post_warmup_config_mask).sum())
        n_composite_post = int(((events["event_type"] == "change_point_composite") & post_warmup_config_mask).sum())
    else:
        n_single_post = events_post_config
        n_composite_post = 0
    lines.append(f"| Post-warmup (config: step > {warmup_end_step}) | {events_post_config} | {n_single_post} | {n_composite_post} |")

    # Post-warmup by eligibility (if different)
    if first_capture_after is not None and first_capture_after > warmup_end_step + 1:
        if "event_type" in events.columns:
            n_single_elig = int(((events["event_type"] == "change_point") & post_warmup_eligible_mask).sum())
            n_composite_elig = int(((events["event_type"] == "change_point_composite") & post_warmup_eligible_mask).sum())
        else:
            n_single_elig = events_post_eligible
            n_composite_elig = 0
        lines.append(f"| Post-warmup (eligible: step ≥ {first_capture_after}) | {events_post_eligible} | {n_single_elig} | {n_composite_elig} |")

    # Pre-warmup row
    n_single_pre = n_single - n_single_post
    n_composite_pre = n_composite - n_composite_post
    lines.append(f"| Pre-warmup (step ≤ {warmup_end_step}) | {events_pre_warmup} | {n_single_pre} | {n_composite_pre} |")

    lines.append("")

    # Explicit delta counts (machine-readable for seed comparisons)
    delta_events_config_eligible = events_post_eligible - events_post_config
    if first_capture_after is not None and delta_events_config_eligible != 0:
        lines.append("### Config vs Eligibility Delta\n")
        lines.append(
            f"- `n_events_post_warmup_config`: {events_post_config}"
        )
        lines.append(
            f"- `n_events_post_warmup_eligible`: {events_post_eligible}"
        )
        lines.append(
            f"- `delta_events`: {delta_events_config_eligible} "
            f"({'more' if delta_events_config_eligible > 0 else 'fewer'} by eligibility)"
        )
        lines.append("")

    # Composites filtered near warmup
    if n_composites_filtered > 0:
        lines.append(
            f"**Composites filtered near warmup**: {n_composites_filtered} removed "
            f"(within {composite_buffer_points} grid points of warmup end)."
        )
        if n_composite == 0 and n_composites_filtered > 0:
            lines.append(
                "  - ⚠️ All composites were filtered - composite formation may be boundary-driven."
            )
        lines.append("")

    # Interpretation
    if events_post_config < total_events * 0.5:
        lines.append(
            f"⚠️ **Most events ({events_pre_warmup}/{total_events}) occur during warmup.** "
            "Consider extending warmup or reviewing detection parameters."
        )
    else:
        pct_post = events_post_config / total_events * 100 if total_events > 0 else 0
        lines.append(
            f"✓ {pct_post:.0f}% of events occur post-warmup, indicating stable detection."
        )

    return "\n".join(lines)


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
    total_candidates = int(summary["n_candidates_raw"].sum())
    total_selected = int(summary["n_selected_final"].sum())

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
    analysis_id: str | None = None,
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
        analysis_id: Analysis version ID for loading warmup info

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

    # Capture Grid Summary (new diagnostic)
    suppression_radius = 15  # Default suppression radius
    warmup_buffer_pts = 2  # Default warmup buffer points

    # Try to get actual values from warmup_info if available later
    grid_summary = compute_capture_grid_summary(step_grid, suppression_radius=suppression_radius)

    # Compute sanity triangle (ties grid + suppression + windows together)
    sanity = compute_sanity_triangle(
        events, step_grid,
        suppression_radius=suppression_radius,
        warmup_buffer_points=warmup_buffer_pts,
    )

    lines.append("\n")
    lines.append(format_capture_grid_summary_md(grid_summary, sanity=sanity))
    lines.append("")

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

    # Peak-step entropy (new diagnostic)
    lines.append("\n")
    entropy_stats = compute_peak_step_entropy(events, step_grid)
    lines.append(format_peak_step_entropy_md(entropy_stats))

    # Composite step diversity (new diagnostic)
    lines.append("\n")
    comp_diversity = compute_composite_step_diversity(events)
    lines.append(format_composite_step_diversity_md(comp_diversity))

    # Composite window statistics (detailed stats)
    lines.append("\n")
    comp_window_stats = compute_composite_window_stats(events)
    lines.append(format_composite_window_stats_md(comp_window_stats))

    # Sample composite rows (show actual data)
    lines.append("\n")
    lines.append(format_composite_sample_rows_md(events, max_rows=10))

    # Warmup-aware event summary
    warmup_info = load_warmup_info(run_id, analysis_id)
    if warmup_info is not None:
        n_composites_filtered = warmup_info.get("n_composites_filtered_near_warmup", 0)
        buffer_points = warmup_info.get("composite_warmup_buffer_points", 2)
        lines.append("\n")
        lines.append(format_warmup_summary_md(
            events,
            warmup_info,
            n_composites_filtered=n_composites_filtered,
            composite_buffer_points=buffer_points,
        ))

    # Detection retention summary (true retention from candidates)
    detection_summary = load_detection_summary(run_id, analysis_id)
    if detection_summary is not None and not detection_summary.empty:
        lines.append("\n")
        lines.append(format_retention_summary_md(detection_summary))

    return "\n".join(lines)
