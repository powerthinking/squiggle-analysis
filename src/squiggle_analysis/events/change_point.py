from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from squiggle_core import paths
from squiggle_core.scoring.baselines import (
    build_metric_baselines_from_run,
    load_baseline_with_volatility,
)
from squiggle_core.scoring.squiggle_scoring import (
    ScoringConfig,
    build_baselines_from_samples,
    compute_event_score,
)


def _merge_overlapping_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not windows:
        return []
    windows = sorted(windows)
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = windows[0]
    for s, e in windows[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _merge_windows_with_gap(windows: list[tuple[int, int]], *, gap: int) -> list[tuple[int, int]]:
    if not windows:
        return []
    windows = sorted(windows)
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = windows[0]
    gap = max(0, int(gap))
    for s, e in windows[1:]:
        if s <= cur_e + gap:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _interval_intersection(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int] | None:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    if s > e:
        return None
    return (s, e)


def _interval_len(a: tuple[int, int]) -> int:
    return max(0, int(a[1] - a[0] + 1))


def _covered_segments_in_window(
    *,
    windows: list[tuple[int, int]],
    window: tuple[int, int],
) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    for w in windows:
        inter = _interval_intersection(w, window)
        if inter is not None:
            segs.append(inter)
    return _merge_overlapping_windows(segs)


def _covered_len(segs: list[tuple[int, int]]) -> int:
    return int(sum(_interval_len(s) for s in segs))


def _overlap_len(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    total = 0
    for x in a:
        for y in b:
            inter = _interval_intersection(x, y)
            if inter is not None:
                total += _interval_len(inter)
    return int(total)


def _largest_component(nodes: list[str], edges: dict[str, set[str]]) -> list[str]:
    seen: set[str] = set()
    best: list[str] = []
    for n in nodes:
        if n in seen:
            continue
        stack = [n]
        comp: list[str] = []
        seen.add(n)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in edges.get(cur, set()):
                if nxt in seen:
                    continue
                seen.add(nxt)
                stack.append(nxt)
        if len(comp) > len(best):
            best = comp
    return best


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    return float(xs[len(xs) // 2])


def _mad(xs: list[float]) -> float:
    """Median Absolute Deviation."""
    if not xs:
        return 0.0
    med = _median(xs)
    devs = [abs(x - med) for x in xs]
    return _median(devs)


def _compute_adaptive_threshold(
    deltas: list[float],
    k: float = 2.5,
    min_threshold: float = 0.01,
) -> float:
    """Compute adaptive threshold as median + k * MAD.

    This identifies statistically unusual changes relative to the data's own
    distribution, rather than using fixed thresholds that may not scale
    across different models or metrics.

    Args:
        deltas: List of absolute delta values
        k: Multiplier for MAD (higher = less sensitive, fewer events)
        min_threshold: Floor value to prevent too-low thresholds

    Returns:
        Adaptive threshold value
    """
    if not deltas:
        return min_threshold
    med = _median(deltas)
    mad = _mad(deltas)
    # Use k * MAD, but ensure we have a reasonable floor
    threshold = med + k * mad
    return max(threshold, min_threshold)


def _compute_local_baseline(
    deltas: list[float],
    center_idx: int,
    window_size: int,
) -> tuple[float, float]:
    """Compute local baseline (median, MAD) around a center index.

    Returns (median, mad) for the local window.
    """
    n = len(deltas)
    start = max(0, center_idx - window_size)
    end = min(n, center_idx + window_size + 1)
    local_deltas = deltas[start:end]
    if not local_deltas:
        return 0.0, 1.0
    med = _median(local_deltas)
    mad = _mad(local_deltas)
    if mad < 1e-8:
        mad = 1e-8  # Prevent division by zero
    return med, mad


@dataclass
class EventEntropyMetrics:
    """Summary metrics for event distribution (entropy-based)."""

    n_events: int
    n_single_metric: int
    n_composite: int

    # Temporal entropy: variance of event steps (normalized by total steps)
    step_variance: float  # Raw variance of event steps
    step_cv: float  # Coefficient of variation (std/mean)
    temporal_entropy: float  # Entropy of step distribution (binned)

    # Layer entropy: how spread out are events across layers
    layer_entropy: float  # Shannon entropy of layer distribution
    layer_coverage: float  # Fraction of layers with at least one event

    # Metric entropy: diversity of metrics triggering events
    metric_entropy: float  # Shannon entropy of metric distribution
    n_unique_metrics: int

    # Phase distribution
    n_shaping: int
    n_transition: int
    n_locking: int


def _shannon_entropy(counts: list[int]) -> float:
    """Compute Shannon entropy from counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def compute_event_entropy(events_df: pd.DataFrame, total_steps: int, n_layers: int) -> EventEntropyMetrics:
    """Compute entropy-based summary metrics for events."""
    if events_df.empty:
        return EventEntropyMetrics(
            n_events=0,
            n_single_metric=0,
            n_composite=0,
            step_variance=0.0,
            step_cv=0.0,
            temporal_entropy=0.0,
            layer_entropy=0.0,
            layer_coverage=0.0,
            metric_entropy=0.0,
            n_unique_metrics=0,
            n_shaping=0,
            n_transition=0,
            n_locking=0,
        )

    n_events = len(events_df)
    single_metric_mask = events_df["event_type"] == "change_point"
    n_single_metric = int(single_metric_mask.sum())
    n_composite = n_events - n_single_metric

    # Temporal metrics
    steps = events_df["step"].to_numpy()
    if len(steps) > 1:
        step_mean = float(steps.mean())
        step_std = float(steps.std())
        step_variance = float(steps.var())
        step_cv = step_std / step_mean if step_mean > 0 else 0.0
    else:
        step_variance = 0.0
        step_cv = 0.0

    # Temporal entropy: bin steps into ~10 bins and compute entropy
    n_bins = min(10, max(1, total_steps // 5))
    if n_bins > 0 and total_steps > 0:
        bin_edges = [i * total_steps / n_bins for i in range(n_bins + 1)]
        bin_counts = [0] * n_bins
        for s in steps:
            bin_idx = min(int(s * n_bins / total_steps), n_bins - 1)
            bin_counts[bin_idx] += 1
        temporal_entropy = _shannon_entropy(bin_counts)
    else:
        temporal_entropy = 0.0

    # Layer metrics
    layer_counts: dict[int, int] = {}
    for layer in events_df["layer"]:
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    layer_entropy = _shannon_entropy(list(layer_counts.values()))
    layer_coverage = len(layer_counts) / n_layers if n_layers > 0 else 0.0

    # Metric metrics (exclude __composite__)
    metrics_series = events_df[events_df["metric"] != "__composite__"]["metric"]
    metric_counts: dict[str, int] = {}
    for m in metrics_series:
        metric_counts[m] = metric_counts.get(m, 0) + 1
    metric_entropy = _shannon_entropy(list(metric_counts.values()))
    n_unique_metrics = len(metric_counts)

    # Phase distribution
    n_shaping = 0
    n_transition = 0
    n_locking = 0
    if "event_phase" in events_df.columns:
        phase_counts = events_df["event_phase"].value_counts()
        n_shaping = int(phase_counts.get("shaping", 0))
        n_transition = int(phase_counts.get("transition", 0))
        n_locking = int(phase_counts.get("locking", 0))

    return EventEntropyMetrics(
        n_events=n_events,
        n_single_metric=n_single_metric,
        n_composite=n_composite,
        step_variance=step_variance,
        step_cv=step_cv,
        temporal_entropy=temporal_entropy,
        layer_entropy=layer_entropy,
        layer_coverage=layer_coverage,
        metric_entropy=metric_entropy,
        n_unique_metrics=n_unique_metrics,
        n_shaping=n_shaping,
        n_transition=n_transition,
        n_locking=n_locking,
    )


def _select_top_k_peaks(
    hit_idxs: list[int],
    deltas: list[float],
    max_peaks: int,
    suppression_radius: int,
) -> list[int]:
    """Select top-K peaks with non-maximum suppression.

    Args:
        hit_idxs: Indices where deltas exceed threshold
        deltas: All delta values
        max_peaks: Maximum number of peaks to return
        suppression_radius: Minimum index separation between peaks

    Returns:
        List of selected peak indices, sorted by position
    """
    if not hit_idxs:
        return []

    # Sort hits by delta magnitude (descending)
    sorted_hits = sorted(hit_idxs, key=lambda i: deltas[i], reverse=True)

    selected: list[int] = []
    for idx in sorted_hits:
        if len(selected) >= max_peaks:
            break
        # Check if this peak is far enough from already selected peaks
        too_close = any(abs(idx - s) < suppression_radius for s in selected)
        if not too_close:
            selected.append(idx)

    # Return sorted by position (chronological order)
    return sorted(selected)


def detect_events(
    run_id: str,
    *,
    analysis_id: str = "analysis@2.0",
    baseline_id: str | None = None,
    baseline_run_id: str | None = None,
    window_radius_steps: int = 5,
    composite_merge_gap_steps: int = 0,
    composite_quorum_k: int = 2,
    composite_min_support_frac: float = 0.2,
    composite_min_support_len: int = 1,
    composite_min_pairwise_overlap: int = 1,
    rank_threshold: float = 0.2,
    mass_threshold: float = 0.03,
    # Adaptive threshold parameters
    adaptive_threshold: bool = True,  # Use median + k*MAD instead of fixed thresholds
    adaptive_k: float = 2.5,  # Multiplier for MAD (higher = fewer events)
    adaptive_min_threshold: float = 0.01,  # Floor to prevent too-low thresholds
    # New parameters for local event windows
    local_baseline_window: int | None = 20,  # ±N steps for local baseline (None = global)
    local_baseline_fraction: float | None = 0.15,  # Alternative: ±15% of total steps
    composite_stabilization_steps: int = 10,  # Delay composite detection until this many steps
    min_event_separation: int = 5,  # Minimum steps between events of same type
    # Peak selection parameters (new)
    max_events_per_series: int = 5,  # Target 2-5 events per (layer, metric)
    peak_suppression_radius: int = 3,  # Minimum index separation between peaks
    event_window_radius: int = 1,  # Window around each peak for start_step/end_step
) -> Optional[EventEntropyMetrics]:
    geom_path = paths.geometry_state_path(run_id)
    if not geom_path.exists():
        raise FileNotFoundError(
            f"Geometry state parquet not found for run_id='{run_id}'. Expected: {geom_path}\n"
            f"Run geometry computation first."
        )

    geom = pd.read_parquet(geom_path)

    required = {
        "run_id",
        "step",
        "layer",
        "metric",
        "value",
    }
    missing = required - set(geom.columns)
    if missing:
        raise ValueError(
            f"Geometry state is missing required columns: {sorted(missing)}\n"
            f"Found columns: {list(geom.columns)}"
        )

    if "analysis_id" not in geom.columns:
        geom["analysis_id"] = analysis_id
    if "schema_version" not in geom.columns:
        geom["schema_version"] = "geometry_state@unknown"
    if "created_at_utc" not in geom.columns:
        geom["created_at_utc"] = datetime.now(timezone.utc)

    out = paths.events_candidates_path(run_id)
    out.parent.mkdir(parents=True, exist_ok=True)

    created_at_utc = datetime.now(timezone.utc)
    schema_version = "events_candidates@2.0"

    # Define the schema we always write (even if empty)
    out_cols = [
        "run_id",
        "analysis_id",
        "schema_version",
        "created_at_utc",
        "event_id",
        "layer",
        "metric",
        "step",        # canonical event timestamp (validator/report-friendly)
        "start_step",  # interval context
        "end_step",
        "event_type",
        "score",

        "magnitude",
        "structure_modifier",
        "magnitude_eff",
        "coherence",
        "novelty",

        "metric_size",
        "metric_z",
        "baseline_median",
        "baseline_mad",

        # New: local vs global baseline fields
        "local_baseline_median",
        "local_baseline_mad",
        "local_z_score",
        "baseline_scope",  # "local" or "global"

        # New: event phase classification
        "event_phase",  # "shaping" (early), "locking" (late), or "transition"
        "phase_progress",  # 0.0-1.0 indicating position in training

        "volatility_event",
        "volatility_baseline",
        "volatility_ratio",
        "volatility_ratio_agg",

        "metric_sizes_json",
        "metric_z_json",
        "baseline_median_json",
        "baseline_mad_json",
        "volatility_event_json",
        "volatility_baseline_json",
        "volatility_ratio_json",
    ]

    if geom.empty:
        pd.DataFrame(columns=out_cols).to_parquet(out, index=False)
        return None

    # Compute total steps for local baseline window calculation
    total_steps = int(geom["step"].max() - geom["step"].min()) if len(geom) > 1 else 1

    # Determine local baseline window size
    if local_baseline_fraction is not None:
        local_window = max(5, int(total_steps * local_baseline_fraction))
    elif local_baseline_window is not None:
        local_window = local_baseline_window
    else:
        local_window = None  # Use global baseline

    # Phase boundaries for event classification (default: early 30%, late 30%)
    shaping_boundary = 0.30  # First 30% is "shaping" phase
    locking_boundary = 0.70  # Last 30% is "locking" phase

    baselines = None
    volatility_baseline: dict[str, float] = {}
    if baseline_id is not None:
        _, baselines, volatility_baseline = load_baseline_with_volatility(baseline_id=baseline_id)

    if baselines is None and baseline_run_id is not None:
        baselines, volatility_baseline = build_metric_baselines_from_run(
            run_id=baseline_run_id,
            rank_threshold=rank_threshold,
            mass_threshold=mass_threshold,
            window_radius_steps=window_radius_steps,
        )

    if baselines is None:
        size_samples: dict[str, list[float]] = {}
        for (_, metric), g in geom.groupby(["layer", "metric"], sort=True):
            g = g.sort_values("step")
            values = g["value"].to_numpy()
            if len(values) < 2:
                continue
            ds = [float(abs(values[i] - values[i - 1])) for i in range(1, len(values))]
            if not ds:
                continue
            key = str(metric)
            size_samples.setdefault(key, []).extend(ds)
        baselines = build_baselines_from_samples(size_samples)

    use_structure = bool(volatility_baseline)
    scoring_cfg = ScoringConfig(use_structure_modifier=use_structure)

    def _threshold_for_metric(metric_name: str) -> float:
        if metric_name == "effective_rank":
            return float(rank_threshold)
        if metric_name.startswith("topk_mass_"):
            return float(mass_threshold)
        # default fallback for future metrics
        return float(rank_threshold)

    events: list[dict] = []
    event_id = 0

    layer_data: dict[int, dict[str, dict[str, object]]] = {}

    # Group per (layer, metric) so thresholds apply cleanly
    for (layer, metric), g in geom.groupby(["layer", "metric"], sort=True):
        g = g.sort_values("step")
        values = g["value"].to_numpy()
        steps = g["step"].to_numpy()

        if len(values) < 2:
            continue

        deltas_abs = [float(abs(values[i] - values[i - 1])) for i in range(1, len(values))]
        if not deltas_abs:
            continue

        # Compute threshold: adaptive (median + k*MAD) or fixed
        if adaptive_threshold:
            thr = _compute_adaptive_threshold(
                deltas_abs, k=adaptive_k, min_threshold=adaptive_min_threshold
            )
        else:
            thr = _threshold_for_metric(str(metric))

        hit_idxs = [i for i, d in enumerate(deltas_abs) if d > thr]
        if not hit_idxs:
            continue

        # Build windows for composite event detection (still merged)
        raw_windows: list[tuple[int, int]] = []
        for idx in hit_idxs:
            s = max(0, idx - int(window_radius_steps))
            e = min(len(deltas_abs) - 1, idx + int(window_radius_steps))
            raw_windows.append((s, e))
        merged_windows = _merge_overlapping_windows(raw_windows)

        metric_key = str(metric)
        vb = volatility_baseline.get(metric_key)

        ld = layer_data.setdefault(int(layer), {})
        ld[metric_key] = {
            "steps": steps,
            "deltas_abs": deltas_abs,
            "windows": merged_windows,  # Keep merged for composite detection
        }

        # Select top-K peaks with suppression (target 2-5 events per series)
        selected_peaks = _select_top_k_peaks(
            hit_idxs=hit_idxs,
            deltas=deltas_abs,
            max_peaks=max_events_per_series,
            suppression_radius=peak_suppression_radius,
        )

        # Create events for each selected peak with local window
        for peak_idx in selected_peaks:
            # The delta at peak_idx is between steps[peak_idx] and steps[peak_idx+1]
            metric_size = float(deltas_abs[peak_idx])

            # Context window for volatility calculation
            ctx_start = max(0, peak_idx - int(window_radius_steps))
            ctx_end = min(len(deltas_abs) - 1, peak_idx + int(window_radius_steps))
            context_deltas = deltas_abs[ctx_start : ctx_end + 1]
            volatility_event = float(_median(context_deltas))

            # Event location and context window
            # start_step = step before the change (where the delta starts)
            # step = step after the change (canonical event time)
            # end_step = window end for context
            win_end_idx = min(len(steps) - 1, peak_idx + 1 + event_window_radius)

            start_step = int(steps[peak_idx])  # Step before the change
            step = int(steps[min(peak_idx + 1, len(steps) - 1)])  # Step after the change
            end_step = int(steps[win_end_idx])  # Context window end

            # Compute phase progress and classification
            step_min = int(geom["step"].min())
            phase_progress = float((step - step_min) / max(1, total_steps))
            if phase_progress < shaping_boundary:
                event_phase = "shaping"
            elif phase_progress > locking_boundary:
                event_phase = "locking"
            else:
                event_phase = "transition"

            # Compute local baseline if configured
            local_med, local_mad, local_z = None, None, None
            baseline_scope = "global"

            if local_window is not None:
                local_med, local_mad = _compute_local_baseline(
                    deltas_abs, peak_idx, local_window
                )
                local_z = float((metric_size - local_med) / local_mad)
                baseline_scope = "local"

            breakdown = None
            baseline_med = None
            baseline_mad = None
            metric_z = None

            if metric_key in baselines:
                b = baselines[metric_key]
                baseline_med = float(b.median)
                baseline_mad = float(b.mad)
                breakdown = compute_event_score(
                    metric_sizes={metric_key: metric_size},
                    baselines={metric_key: b},
                    cfg=scoring_cfg,
                    volatility_event=({metric_key: volatility_event} if vb is not None else None),
                    volatility_baseline=({metric_key: float(vb)} if vb is not None else None),
                )
                score = float(breakdown.score)
                metric_z = float(breakdown.metric_z.get(metric_key, 0.0))
            else:
                score = metric_size

            # Use local z-score for final score if available and significant
            if local_z is not None and local_window is not None:
                # Blend: use local z-score as the primary score, global as context
                score = max(score, abs(local_z) / 3.0)  # Scale local z to similar range

            events.append(
                {
                    "run_id": run_id,
                    "analysis_id": analysis_id,
                    "schema_version": schema_version,
                    "created_at_utc": created_at_utc,
                    "event_id": f"e{event_id}",
                    "layer": int(layer),
                    "metric": metric_key,
                    "step": step,
                    "start_step": start_step,
                    "end_step": end_step,
                    "event_type": "change_point",
                    "score": score,
                    "magnitude": (float(breakdown.magnitude) if breakdown else None),
                    "structure_modifier": (float(breakdown.structure_modifier) if breakdown else None),
                    "magnitude_eff": (float(breakdown.magnitude_eff) if breakdown else None),
                    "coherence": (float(breakdown.coherence) if breakdown else None),
                    "novelty": (float(breakdown.novelty) if breakdown else None),
                    "metric_size": metric_size,
                    "metric_z": metric_z,
                    "baseline_median": baseline_med,
                    "baseline_mad": baseline_mad,
                    "local_baseline_median": local_med,
                    "local_baseline_mad": local_mad,
                    "local_z_score": local_z,
                    "baseline_scope": baseline_scope,
                    "event_phase": event_phase,
                    "phase_progress": phase_progress,
                    "volatility_event": volatility_event,
                    "volatility_baseline": (float(vb) if vb is not None else None),
                    "volatility_ratio": (float(volatility_event / (float(vb) + 1e-8)) if vb is not None else None),
                    "volatility_ratio_agg": (float(volatility_event / (float(vb) + 1e-8)) if vb is not None else None),

                    "metric_sizes_json": None,
                    "metric_z_json": None,
                    "baseline_median_json": None,
                    "baseline_mad_json": None,
                    "volatility_event_json": None,
                    "volatility_baseline_json": None,
                    "volatility_ratio_json": None,
                }
            )
            event_id += 1

    # Track single-metric event counts per layer for stabilization
    single_metric_events_per_layer: dict[int, int] = {}
    for ev in events:
        layer = ev.get("layer")
        if layer is not None and ev.get("event_type") == "change_point":
            single_metric_events_per_layer[layer] = single_metric_events_per_layer.get(layer, 0) + 1

    for layer, metrics in sorted(layer_data.items()):
        all_windows: list[tuple[int, int]] = []
        for info in metrics.values():
            ws = info.get("windows")
            if isinstance(ws, list):
                all_windows.extend(ws)
        if not all_windows:
            continue

        # Filter windows: only consider those after stabilization period
        # Stabilization means we need enough single-metric events first
        min_single_events = max(2, composite_quorum_k)
        if single_metric_events_per_layer.get(layer, 0) < min_single_events:
            continue

        comp_windows = _merge_windows_with_gap(all_windows, gap=int(composite_merge_gap_steps))

        steps_ref = None
        for info in metrics.values():
            steps_ref = info.get("steps")
            if steps_ref is not None:
                break
        if steps_ref is None:
            continue

        k_req = max(1, int(composite_quorum_k))
        min_support_len = max(1, int(composite_min_support_len))
        min_support_frac = float(composite_min_support_frac)
        min_pairwise = max(0, int(composite_min_pairwise_overlap))

        for s_idx, e_idx in comp_windows:
            cand = (int(s_idx), int(e_idx))
            cand_len = _interval_len(cand)
            if cand_len <= 0:
                continue

            covered: dict[str, list[tuple[int, int]]] = {}
            active: list[str] = []
            for metric_key, info in metrics.items():
                ws = info.get("windows")
                if not isinstance(ws, list) or not ws:
                    continue
                segs = _covered_segments_in_window(windows=ws, window=cand)
                if not segs:
                    continue
                cov_len = _covered_len(segs)
                if cov_len < 1:
                    continue
                cov_frac = float(cov_len) / float(cand_len)
                if cov_len >= min_support_len or cov_frac >= min_support_frac:
                    covered[metric_key] = segs
                    active.append(metric_key)

            if len(active) < k_req:
                continue

            edges: dict[str, set[str]] = {m: set() for m in active}
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    a = active[i]
                    b = active[j]
                    ol = _overlap_len(covered[a], covered[b])
                    if ol >= min_pairwise:
                        edges[a].add(b)
                        edges[b].add(a)

            component = _largest_component(active, edges)
            if len(component) < k_req:
                continue

            metric_sizes_all: dict[str, float] = {}
            volatility_event_all: dict[str, float] = {}
            for metric_key in component:
                info = metrics.get(metric_key)
                if info is None:
                    continue
                deltas_abs = info.get("deltas_abs")
                if not isinstance(deltas_abs, list) or not deltas_abs:
                    continue
                if s_idx < 0 or e_idx >= len(deltas_abs):
                    continue
                window = deltas_abs[s_idx : e_idx + 1]
                if not window:
                    continue
                metric_sizes_all[metric_key] = float(max(window))
                volatility_event_all[metric_key] = float(_median(window))

            if len(metric_sizes_all) < k_req:
                continue

            # Find the peak across all metrics in this composite window
            # Use the position of the largest change as the canonical event location
            peak_idx = s_idx
            max_delta = 0.0
            for metric_key in component:
                info = metrics.get(metric_key)
                if info is None:
                    continue
                deltas_abs = info.get("deltas_abs")
                if not isinstance(deltas_abs, list) or not deltas_abs:
                    continue
                for i in range(s_idx, min(e_idx + 1, len(deltas_abs))):
                    if deltas_abs[i] > max_delta:
                        max_delta = deltas_abs[i]
                        peak_idx = i

            # step = peak location, start_step/end_step = window boundaries for context
            step = int(steps_ref[min(peak_idx + 1, len(steps_ref) - 1)])
            start_step = int(steps_ref[s_idx])
            end_step = int(steps_ref[min(e_idx + 1, len(steps_ref) - 1)])

            baselines_scored: dict[str, object] = {}
            metric_sizes_scored: dict[str, float] = {}
            baseline_median_map: dict[str, float] = {}
            baseline_mad_map: dict[str, float] = {}
            for mk, mv in metric_sizes_all.items():
                if mk not in baselines:
                    continue
                b = baselines[mk]
                baselines_scored[mk] = b
                metric_sizes_scored[mk] = float(mv)
                baseline_median_map[mk] = float(b.median)
                baseline_mad_map[mk] = float(b.mad)

            vb_map: dict[str, float] = {}
            ve_map: dict[str, float] = {}
            for mk in metric_sizes_scored.keys():
                vb = volatility_baseline.get(mk)
                if vb is None:
                    continue
                vb_map[mk] = float(vb)
                ve = volatility_event_all.get(mk)
                if ve is not None:
                    ve_map[mk] = float(ve)

            breakdown = compute_event_score(
                metric_sizes=metric_sizes_scored,
                baselines=baselines_scored,  # type: ignore[arg-type]
                cfg=scoring_cfg,
                volatility_event=(ve_map if ve_map else None),
                volatility_baseline=(vb_map if vb_map else None),
            )

            vr_map: dict[str, float] = {}
            for mk, ve in ve_map.items():
                vb = vb_map.get(mk)
                if vb is None:
                    continue
                vr_map[mk] = float(ve / (vb + 1e-8))

            # Compute phase for composite event
            step_min = int(geom["step"].min())
            comp_phase_progress = float((step - step_min) / max(1, total_steps))
            if comp_phase_progress < shaping_boundary:
                comp_event_phase = "shaping"
            elif comp_phase_progress > locking_boundary:
                comp_event_phase = "locking"
            else:
                comp_event_phase = "transition"

            # Skip composite events that are too early (stabilization period)
            if step < step_min + composite_stabilization_steps:
                continue

            events.append(
                {
                    "run_id": run_id,
                    "analysis_id": analysis_id,
                    "schema_version": schema_version,
                    "created_at_utc": created_at_utc,
                    "event_id": f"e{event_id}",
                    "layer": int(layer),
                    "metric": "__composite__",
                    "step": step,
                    "start_step": start_step,
                    "end_step": end_step,
                    "event_type": "change_point_composite",
                    "score": float(breakdown.score),
                    "magnitude": float(breakdown.magnitude),
                    "structure_modifier": float(breakdown.structure_modifier),
                    "magnitude_eff": float(breakdown.magnitude_eff),
                    "coherence": float(breakdown.coherence),
                    "novelty": float(breakdown.novelty),
                    "metric_size": None,
                    "metric_z": None,
                    "baseline_median": None,
                    "baseline_mad": None,
                    "local_baseline_median": None,
                    "local_baseline_mad": None,
                    "local_z_score": None,
                    "baseline_scope": "composite",
                    "event_phase": comp_event_phase,
                    "phase_progress": comp_phase_progress,
                    "volatility_event": None,
                    "volatility_baseline": None,
                    "volatility_ratio": None,
                    "volatility_ratio_agg": (
                        float(breakdown.volatility_ratio_agg) if breakdown.volatility_ratio_agg is not None else None
                    ),
                    "metric_sizes_json": json.dumps(metric_sizes_all, sort_keys=True),
                    "metric_z_json": json.dumps({k: float(v) for k, v in breakdown.metric_z.items()}, sort_keys=True),
                    "baseline_median_json": json.dumps(baseline_median_map, sort_keys=True),
                    "baseline_mad_json": json.dumps(baseline_mad_map, sort_keys=True),
                    "volatility_event_json": json.dumps(volatility_event_all, sort_keys=True),
                    "volatility_baseline_json": json.dumps(vb_map, sort_keys=True),
                    "volatility_ratio_json": json.dumps(vr_map, sort_keys=True),
                }
            )
            event_id += 1

    df = pd.DataFrame(events)

    # Always write with the same columns, even if no events were found
    if df.empty:
        df = pd.DataFrame(columns=out_cols)
    else:
        # Ensure column set/order is stable
        for c in out_cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[out_cols]

        # Deterministic ordering (nice for diffs + reports)
        df = df.sort_values(["score", "layer", "metric", "step"], ascending=[False, True, True, True]).reset_index(drop=True)

        # Reassign event_id after sorting so IDs correspond to report ordering (optional)
        df["event_id"] = [f"e{i}" for i in range(len(df))]

    df.to_parquet(out, index=False)

    # Compute and return event entropy metrics
    n_layers = int(geom["layer"].nunique()) if "layer" in geom.columns else 1
    entropy_metrics = compute_event_entropy(df, total_steps, n_layers)

    return entropy_metrics
