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
    """Compute adaptive threshold as median + k * MAD."""
    if not deltas:
        return min_threshold
    med = _median(deltas)
    mad = _mad(deltas)
    threshold = med + k * mad
    return max(threshold, min_threshold)


def _compute_local_baseline(
    deltas: list[float],
    center_idx: int,
    window_size: int,
) -> tuple[float, float]:
    """Compute local baseline (median, MAD) around a center index."""
    n = len(deltas)
    start = max(0, center_idx - window_size)
    end = min(n, center_idx + window_size + 1)
    local_deltas = deltas[start:end]
    if not local_deltas:
        return 0.0, 1.0
    med = _median(local_deltas)
    mad = _mad(local_deltas)
    if mad < 1e-8:
        mad = 1e-8
    return med, mad


@dataclass
class EventEntropyMetrics:
    """Summary metrics for event distribution (entropy-based)."""

    n_events: int
    n_single_metric: int
    n_composite: int

    step_variance: float
    step_cv: float
    temporal_entropy: float

    layer_entropy: float
    layer_coverage: float

    metric_entropy: float
    n_unique_metrics: int

    n_shaping: int
    n_transition: int
    n_locking: int


def _shannon_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def compute_event_entropy(events_df: pd.DataFrame, total_steps: int, n_layers: int) -> EventEntropyMetrics:
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

    steps = events_df["step"].to_numpy()
    if len(steps) > 1:
        step_mean = float(steps.mean())
        step_std = float(steps.std())
        step_variance = float(steps.var())
        step_cv = step_std / step_mean if step_mean > 0 else 0.0
    else:
        step_variance = 0.0
        step_cv = 0.0

    n_bins = min(10, max(1, total_steps // 5))
    if n_bins > 0 and total_steps > 0:
        bin_counts = [0] * n_bins
        for s in steps:
            bin_idx = min(int(s * n_bins / total_steps), n_bins - 1)
            bin_counts[bin_idx] += 1
        temporal_entropy = _shannon_entropy(bin_counts)
    else:
        temporal_entropy = 0.0

    layer_counts: dict[int, int] = {}
    for layer in events_df["layer"]:
        layer_counts[int(layer)] = layer_counts.get(int(layer), 0) + 1
    layer_entropy = _shannon_entropy(list(layer_counts.values()))
    layer_coverage = len(layer_counts) / n_layers if n_layers > 0 else 0.0

    metrics_series = events_df[events_df["metric"] != "__composite__"]["metric"]
    metric_counts: dict[str, int] = {}
    for m in metrics_series:
        metric_counts[str(m)] = metric_counts.get(str(m), 0) + 1
    metric_entropy = _shannon_entropy(list(metric_counts.values()))
    n_unique_metrics = len(metric_counts)

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


@dataclass
class SelectionStats:
    """Stats about why candidates were kept or skipped during peak selection."""

    n_candidates: int = 0
    n_selected: int = 0
    n_skipped_suppression: int = 0
    n_skipped_topk: int = 0
    n_skipped_pre_warmup_cap: int = 0
    n_candidates_pre: int = 0
    n_candidates_post: int = 0
    n_selected_pre: int = 0
    n_selected_post: int = 0
    # Pre/post breakdown for suppression and topk
    n_skipped_suppression_pre: int = 0
    n_skipped_suppression_post: int = 0
    n_skipped_topk_pre: int = 0
    n_skipped_topk_post: int = 0


def _select_top_k_peaks_stepwise(
    hit_idxs: list[int],
    deltas_abs: list[float],
    steps_by_idx: list[int],
    max_peaks: int,
    suppression_radius_steps: int,
    return_stats: bool = False,
) -> list[int] | tuple[list[int], SelectionStats]:
    """
    Select top-K peaks by |delta| with non-maximum suppression measured in STEP units.

    Notes:
      - delta index i corresponds to change between steps[i] -> steps[i+1]
      - steps_by_idx must have length == len(deltas_abs)
    """
    stats = SelectionStats(n_candidates=len(hit_idxs))

    if not hit_idxs:
        if return_stats:
            return [], stats
        return []

    sorted_hits = sorted(hit_idxs, key=lambda i: abs(deltas_abs[i]), reverse=True)

    selected: list[int] = []
    for idx in sorted_hits:
        if len(selected) >= max_peaks:
            stats.n_skipped_topk += 1
            continue

        step = steps_by_idx[idx]
        too_close = any(abs(step - steps_by_idx[s]) <= suppression_radius_steps for s in selected)
        if too_close:
            stats.n_skipped_suppression += 1
            continue

        selected.append(idx)

    stats.n_selected = len(selected)

    result = sorted(selected, key=lambda i: steps_by_idx[i])
    if return_stats:
        return result, stats
    return result


def _select_top_k_peaks_with_warmup_stepwise(
    hit_idxs: list[int],
    deltas_abs: list[float],
    steps_by_idx: list[int],
    max_peaks: int,
    suppression_radius_steps: int,
    warmup_end_step: int,
    max_pre_warmup: int = 1,
    return_stats: bool = False,
) -> list[int] | tuple[list[int], SelectionStats]:
    """Select top-K peaks with SEPARATE budgets for pre/post warmup.

    Budgets:
      - max_peaks_post = max_peaks - max_pre_warmup (derived, for post-warmup)
      - max_pre_warmup (for pre-warmup)

    This prevents early transients from stealing post-warmup capacity.
    """

    post_hits = [i for i in hit_idxs if steps_by_idx[i] > warmup_end_step]
    pre_hits = [i for i in hit_idxs if steps_by_idx[i] <= warmup_end_step]

    stats = SelectionStats(
        n_candidates=len(hit_idxs),
        n_candidates_pre=len(pre_hits),
        n_candidates_post=len(post_hits),
    )

    if not hit_idxs:
        if return_stats:
            return [], stats
        return []

    sorted_post = sorted(post_hits, key=lambda i: abs(deltas_abs[i]), reverse=True)
    sorted_pre = sorted(pre_hits, key=lambda i: abs(deltas_abs[i]), reverse=True)

    selected: list[int] = []
    selected_pre: list[int] = []
    selected_post: list[int] = []

    # SEPARATE BUDGETS: pre and post don't compete for slots
    # - max_peaks_post = max_peaks - max_pre_warmup (post-warmup budget)
    # - max_pre_warmup = pre-warmup budget (already a parameter)
    # This prevents early transients from stealing post-warmup capacity
    max_peaks_post = max(1, max_peaks - max_pre_warmup)

    # 1) Select from post-warmup (up to max_peaks_post, with suppression)
    for idx in sorted_post:
        if len(selected_post) >= max_peaks_post:
            stats.n_skipped_topk += 1
            stats.n_skipped_topk_post += 1
            continue
        step = steps_by_idx[idx]
        if any(abs(step - steps_by_idx[s]) <= suppression_radius_steps for s in selected):
            stats.n_skipped_suppression += 1
            stats.n_skipped_suppression_post += 1
            continue
        selected.append(idx)
        selected_post.append(idx)

    # 2) Select from pre-warmup (up to max_pre_warmup, with suppression)
    # Pre-warmup has its OWN budget - doesn't compete with post
    for idx in sorted_pre:
        if len(selected_pre) >= max(0, int(max_pre_warmup)):
            # Use pre_warmup_cap for pre-budget overflow (distinct from post top-k)
            stats.n_skipped_pre_warmup_cap += 1
            continue
        step = steps_by_idx[idx]
        if any(abs(step - steps_by_idx[s]) <= suppression_radius_steps for s in selected):
            stats.n_skipped_suppression += 1
            stats.n_skipped_suppression_pre += 1
            continue
        selected.append(idx)
        selected_pre.append(idx)

    stats.n_selected = len(selected)
    stats.n_selected_pre = len(selected_pre)
    stats.n_selected_post = len(selected_post)

    result = sorted(selected, key=lambda i: steps_by_idx[i])
    if return_stats:
        return result, stats
    return result



def _get_warmup_end_step(run_id: str, step_min: int, fallback_fraction: float, step_max: int) -> int:
    """Get warmup end step from meta.json if possible, otherwise fallback."""
    meta_path = paths.run_dir(run_id) / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())

            scheduler = meta.get("scheduler", {})
            if isinstance(scheduler, dict):
                warmup_steps = scheduler.get("warmup_steps")
                if warmup_steps is not None:
                    # warmup_steps is a COUNT of optimizer steps; treat it as relative to step_min
                    return min(int(step_min + int(warmup_steps)), int(step_max))

            optimizer = meta.get("optimizer", {})
            if isinstance(optimizer, dict):
                warmup_steps = optimizer.get("warmup_steps")
                if warmup_steps is not None:
                    return min(int(step_min + int(warmup_steps)), int(step_max))
        except Exception:
            pass

    return int(step_min + (step_max - step_min) * float(fallback_fraction))


def _compute_window_delta_median(
    values: list[float],
    peak_idx: int,
    event_window_radius: int,
) -> tuple[float, int]:
    """
    Compute signed delta from pre/post window MEDIANS (more robust than adjacent points).

    peak_idx is a DELTA index: change between values[peak_idx] -> values[peak_idx+1]
    """
    n = len(values)
    if n == 0:
        return 0.0, 1

    # Use windows around the change boundary (between peak_idx and peak_idx+1)
    pre_end = peak_idx + 1  # include values[peak_idx]
    pre_start = max(0, pre_end - max(1, int(event_window_radius)))
    pre_window = values[pre_start:pre_end]

    post_start = min(n, peak_idx + 1)  # start at values[peak_idx+1]
    post_end = min(n, post_start + max(1, int(event_window_radius)))
    post_window = values[post_start:post_end]

    pre_med = _median([float(x) for x in pre_window]) if pre_window else float(values[max(0, peak_idx)])
    post_med = _median([float(x) for x in post_window]) if post_window else float(values[min(n - 1, peak_idx + 1)])

    delta = float(post_med - pre_med)
    polarity = 1 if delta >= 0 else -1
    return delta, polarity


def detect_events(
    run_id: str,
    *,
    analysis_id: str | None = None,
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
    adaptive_threshold: bool = True,
    adaptive_k: float = 2.5,
    adaptive_min_threshold: float = 0.01,
    local_baseline_window: int | None = 20,         # preferred if set
    local_baseline_fraction: float | None = 0.15,    # fallback if window is None
    composite_stabilization_steps: int = 10,
    _min_event_separation: int = 5,  # NOTE: currently unused; keep if you plan to enforce later
    max_events_per_series: int = 5,
    peak_suppression_radius: int = 15,  # Step units (≈3 capture intervals at cadence=5)
    event_window_radius: int = 1,      # NOW treated as CAPTURE-POINT COUNT (median window). Keep small.
    warmup_fraction: float = 0.1,
    use_warmup_handling: bool = True,
    max_pre_warmup: int = 1,  # Pre-warmup budget (post budget = max_events_per_series - max_pre_warmup)
) -> Optional[EventEntropyMetrics]:
    """
    Detect change point events in geometric state trajectories.

    Args:
        run_id: The run to analyze
        analysis_id: Optional analysis version identifier. If None, auto-generates from
                     detection parameters (e.g., "w10_p1_r15_e5_k25"). Used for versioning
                     output files so different parameter configurations don't overwrite each other.
        ... (other args as before)

    Returns:
        EventEntropyMetrics summarizing the detected events, or None if no geometry found
    """
    # Auto-generate analysis_id from detection parameters if not provided
    if analysis_id is None:
        analysis_id = paths.generate_analysis_id(
            warmup_fraction=warmup_fraction,
            max_pre_warmup=max_pre_warmup,
            peak_suppression_radius=peak_suppression_radius,
            max_events_per_series=max_events_per_series,
            adaptive_k=adaptive_k,
        )
        print(f"[detect_events] Auto-generated analysis_id: {analysis_id}")
    geom_path = paths.geometry_state_path(run_id)
    if not geom_path.exists():
        raise FileNotFoundError(
            f"Geometry state parquet not found for run_id='{run_id}'. Expected: {geom_path}\n"
            f"Run geometry computation first."
        )

    geom = pd.read_parquet(geom_path)

    required = {"run_id", "step", "layer", "metric", "value"}
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

    out = paths.events_candidates_path(run_id, analysis_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[detect_events] Output path: {out}")

    created_at_utc = datetime.now(timezone.utc)
    schema_version = "events_candidates@2.0"

    out_cols = [
        "run_id",
        "analysis_id",
        "schema_version",
        "created_at_utc",
        "event_id",
        "layer",
        "metric",
        "step",
        "start_step",
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
        "polarity",
        "delta",
        "normalized_score",
        "event_rank_in_series",
        "series_id",
        "local_baseline_median",
        "local_baseline_mad",
        "local_z_score",
        "baseline_scope",
        "event_phase",
        "phase_progress",
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

    step_min = int(geom["step"].min())
    step_max = int(geom["step"].max())
    total_steps = max(1, step_max - step_min)

    warmup_end_step = 0
    if use_warmup_handling:
        warmup_end_step = _get_warmup_end_step(run_id, step_min, warmup_fraction, step_max)
        print(
            f"[detect_events] warmup_end_step={warmup_end_step}, step_min={step_min}, "
            f"step_max={step_max}, warmup_fraction={warmup_fraction}"
        )

    # Local baseline window precedence: explicit window wins; fraction only used if window is None.
    if local_baseline_window is not None:
        local_window = int(local_baseline_window)
    elif local_baseline_fraction is not None:
        local_window = max(5, int(total_steps * float(local_baseline_fraction)))
    else:
        local_window = None

    shaping_boundary = 0.30
    locking_boundary = 0.70

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
        return float(rank_threshold)

    events: list[dict] = []
    event_id = 0

    # Collect per-series detection summary for retention analysis
    detection_summaries: list[dict] = []

    layer_data: dict[int, dict[str, dict[str, object]]] = {}

    for (layer, metric), g in geom.groupby(["layer", "metric"], sort=True):
        g = g.sort_values("step")
        values_np = g["value"].to_numpy()
        steps_np = g["step"].to_numpy()

        if len(values_np) < 2:
            continue

        deltas_abs = [float(abs(values_np[i] - values_np[i - 1])) for i in range(1, len(values_np))]
        if not deltas_abs:
            continue

        if adaptive_threshold:
            thr = _compute_adaptive_threshold(deltas_abs, k=adaptive_k, min_threshold=adaptive_min_threshold)
        else:
            thr = _threshold_for_metric(str(metric))

        hit_idxs = [i for i, d in enumerate(deltas_abs) if d > thr]
        if not hit_idxs:
            continue

        metric_key = str(metric)
        vb = volatility_baseline.get(metric_key)

        # steps_by_idx is the "event timestamp" for each delta index i: steps[i+1]
        steps_by_idx = [int(steps_np[min(i + 1, len(steps_np) - 1)]) for i in range(len(deltas_abs))]

        series_id = f"{run_id}:{int(layer)}:{metric_key}"

        # Select peaks with stats for retention analysis
        if use_warmup_handling and warmup_end_step > step_min:
            selected_peaks, select_stats = _select_top_k_peaks_with_warmup_stepwise(
                hit_idxs=hit_idxs,
                deltas_abs=deltas_abs,
                steps_by_idx=steps_by_idx,
                max_peaks=max_events_per_series,
                suppression_radius_steps=int(peak_suppression_radius),
                warmup_end_step=int(warmup_end_step),
                max_pre_warmup=int(max_pre_warmup),
                return_stats=True,
            )
        else:
            selected_peaks, select_stats = _select_top_k_peaks_stepwise(
                hit_idxs=hit_idxs,
                deltas_abs=deltas_abs,
                steps_by_idx=steps_by_idx,
                max_peaks=max_events_per_series,
                suppression_radius_steps=int(peak_suppression_radius),
                return_stats=True,
            )

        # Record detection summary for this series
        n_raw = select_stats.n_candidates
        retention = select_stats.n_selected / n_raw if n_raw > 0 else 1.0
        detection_summaries.append({
            "run_id": run_id,
            "layer": int(layer),
            "metric": metric_key,
            "series_id": series_id,
            # Config snapshot
            "adaptive_k": float(adaptive_k),
            "suppression_radius_steps": int(peak_suppression_radius),
            "max_peaks": int(max_events_per_series),
            "warmup_end_step": int(warmup_end_step) if use_warmup_handling else None,
            "max_pre_warmup": int(max_pre_warmup) if use_warmup_handling else None,
            # Counts
            "n_candidates_raw": select_stats.n_candidates,
            "n_candidates_pre": select_stats.n_candidates_pre,
            "n_candidates_post": select_stats.n_candidates_post,
            "n_selected_final": select_stats.n_selected,
            "n_selected_pre": select_stats.n_selected_pre,
            "n_selected_post": select_stats.n_selected_post,
            # Skip reasons (totals)
            "n_skipped_suppression": select_stats.n_skipped_suppression,
            "n_skipped_topk": select_stats.n_skipped_topk,
            "n_skipped_pre_warmup_cap": select_stats.n_skipped_pre_warmup_cap,
            # Skip reasons (pre/post breakdown)
            "n_skipped_suppression_pre": select_stats.n_skipped_suppression_pre,
            "n_skipped_suppression_post": select_stats.n_skipped_suppression_post,
            "n_skipped_topk_pre": select_stats.n_skipped_topk_pre,
            "n_skipped_topk_post": select_stats.n_skipped_topk_post,
            # Ratios
            "retention_ratio": float(retention),
            "suppression_skip_ratio": (
                float(select_stats.n_skipped_suppression / n_raw) if n_raw > 0 else 0.0
            ),
            "topk_skip_ratio": (
                float(select_stats.n_skipped_topk / n_raw) if n_raw > 0 else 0.0
            ),
        })

        # Build composite windows from SELECTED peaks only (not all hits)
        selected_windows: list[tuple[int, int]] = []
        for idx in selected_peaks:
            s = max(0, idx - int(window_radius_steps))
            e = min(len(deltas_abs) - 1, idx + int(window_radius_steps))
            selected_windows.append((s, e))
        selected_windows = _merge_overlapping_windows(selected_windows)

        ld = layer_data.setdefault(int(layer), {})
        ld[metric_key] = {
            "steps": steps_np,
            "values": values_np,
            "deltas_abs": deltas_abs,
            "windows": selected_windows,
            "steps_by_idx": steps_by_idx,
        }

        values_list = [float(v) for v in values_np]
        steps_list = [int(s) for s in steps_np]

        for peak_idx in selected_peaks:
            metric_size = float(deltas_abs[peak_idx])

            # Direction-aware delta/polarity using robust pre/post medians
            delta, polarity = _compute_window_delta_median(values_list, peak_idx, int(event_window_radius))

            ctx_start = max(0, peak_idx - int(window_radius_steps))
            ctx_end = min(len(deltas_abs) - 1, peak_idx + int(window_radius_steps))
            context_deltas = deltas_abs[ctx_start : ctx_end + 1]
            volatility_event = float(_median(context_deltas))

            # Canonical event step is "after" step: steps[peak_idx+1]
            ev_step = int(steps_list[min(peak_idx + 1, len(steps_list) - 1)])

            # Make (start_step, end_step) reflect context around the delta index in capture steps
            # start at steps[peak_idx] (before), and allow radius around that boundary in capture-grid indices
            start_idx = max(0, peak_idx - int(event_window_radius))
            end_idx = min(len(steps_list) - 1, peak_idx + 1 + int(event_window_radius))
            ev_start_step = int(steps_list[start_idx])
            ev_end_step = int(steps_list[end_idx])

            phase_progress = float((ev_step - step_min) / float(total_steps))
            if phase_progress < shaping_boundary:
                event_phase = "shaping"
            elif phase_progress > locking_boundary:
                event_phase = "locking"
            else:
                event_phase = "transition"

            local_med, local_mad, local_z = None, None, None
            baseline_scope = "global"
            if local_window is not None:
                local_med, local_mad = _compute_local_baseline(deltas_abs, peak_idx, int(local_window))
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

            if local_z is not None and local_window is not None:
                score = max(score, abs(local_z) / 3.0)

            events.append(
                {
                    "run_id": run_id,
                    "analysis_id": analysis_id,
                    "schema_version": schema_version,
                    "created_at_utc": created_at_utc,
                    "event_id": f"e{event_id}",
                    "layer": int(layer),
                    "metric": metric_key,
                    "step": ev_step,
                    "start_step": ev_start_step,
                    "end_step": ev_end_step,
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
                    "polarity": int(polarity),
                    "delta": float(delta),
                    "normalized_score": None,
                    "event_rank_in_series": None,
                    "series_id": series_id,
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

    single_metric_events_per_layer: dict[int, int] = {}
    for ev in events:
        layer_v = ev.get("layer")
        if layer_v is not None and ev.get("event_type") == "change_point":
            single_metric_events_per_layer[int(layer_v)] = single_metric_events_per_layer.get(int(layer_v), 0) + 1

    for layer, metrics in sorted(layer_data.items()):
        all_windows: list[tuple[int, int]] = []
        for info in metrics.values():
            ws = info.get("windows")
            if isinstance(ws, list):
                all_windows.extend(ws)
        if not all_windows:
            continue

        min_single_events = max(2, composite_quorum_k)
        if single_metric_events_per_layer.get(int(layer), 0) < min_single_events:
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

        steps_ref_list = [int(s) for s in steps_ref]

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

            step = int(steps_ref_list[min(peak_idx + 1, len(steps_ref_list) - 1)])
            start_step = int(steps_ref_list[max(0, s_idx)])
            end_step = int(steps_ref_list[min(e_idx + 1, len(steps_ref_list) - 1)])

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

            comp_phase_progress = float((step - step_min) / float(total_steps))
            if comp_phase_progress < shaping_boundary:
                comp_event_phase = "shaping"
            elif comp_phase_progress > locking_boundary:
                comp_event_phase = "locking"
            else:
                comp_event_phase = "transition"

            if step < step_min + composite_stabilization_steps:
                continue

            comp_series_id = f"{run_id}:{int(layer)}:__composite__"

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
                    "polarity": None,
                    "delta": None,
                    "normalized_score": None,
                    "event_rank_in_series": None,
                    "series_id": comp_series_id,
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

    if df.empty:
        df = pd.DataFrame(columns=out_cols)
    else:
        # Rank + normalized score per series_id by score descending
        if "series_id" in df.columns and "score" in df.columns:
            for sid in df["series_id"].dropna().unique():
                mask = df["series_id"] == sid
                # Convert to float; NaNs (unlikely) will rank last
                series_scores = pd.to_numeric(df.loc[mask, "score"], errors="coerce")
                n_ev = int(series_scores.shape[0])
                if n_ev <= 0:
                    continue
                ranks = series_scores.rank(ascending=False, method="min").astype(int)
                df.loc[mask, "event_rank_in_series"] = ranks.values
                df.loc[mask, "normalized_score"] = ((n_ev - ranks + 1) / n_ev).astype(float).values

        for c in out_cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[out_cols]

        df = df.sort_values(["score", "layer", "metric", "step"], ascending=[False, True, True, True]).reset_index(
            drop=True
        )
        df["event_id"] = [f"e{i}" for i in range(len(df))]

    df.to_parquet(out, index=False)

    # Write detection summary (per-series retention stats)
    if detection_summaries:
        summary_df = pd.DataFrame(detection_summaries)
        # Add analysis_id to summary for traceability
        summary_df["analysis_id"] = analysis_id
        summary_path = paths.detection_summary_path(run_id, analysis_id)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_parquet(summary_path, index=False)
        print(f"[detect_events] Wrote detection summary: {len(summary_df)} series to {summary_path}")

    n_layers = int(geom["layer"].nunique()) if "layer" in geom.columns else 1
    entropy_metrics = compute_event_entropy(df, total_steps, n_layers)
    return entropy_metrics