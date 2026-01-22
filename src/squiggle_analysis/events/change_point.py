from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd
from squiggle_core import paths
from squiggle_core.scoring.baselines import (
    build_metric_baselines_from_run,
    load_baseline,
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
) -> None:
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
        return

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

        thr = _threshold_for_metric(str(metric))

        deltas_abs = [float(abs(values[i] - values[i - 1])) for i in range(1, len(values))]
        if not deltas_abs:
            continue

        hit_idxs = [i for i, d in enumerate(deltas_abs) if d > thr]
        if not hit_idxs:
            continue

        raw_windows: list[tuple[int, int]] = []
        for idx in hit_idxs:
            s = max(0, idx - int(window_radius_steps))
            e = min(len(deltas_abs) - 1, idx + int(window_radius_steps))
            raw_windows.append((s, e))
        windows = _merge_overlapping_windows(raw_windows)

        metric_key = str(metric)
        vb = volatility_baseline.get(metric_key)

        ld = layer_data.setdefault(int(layer), {})
        ld[metric_key] = {
            "steps": steps,
            "deltas_abs": deltas_abs,
            "windows": windows,
        }

        for s_idx, e_idx in windows:
            metric_size = float(max(deltas_abs[s_idx : e_idx + 1]))
            volatility_event = float(_median(deltas_abs[s_idx : e_idx + 1]))

            start_step = int(steps[s_idx])
            end_step = int(steps[e_idx + 1])
            step = end_step

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

    for layer, metrics in sorted(layer_data.items()):
        all_windows: list[tuple[int, int]] = []
        for info in metrics.values():
            ws = info.get("windows")
            if isinstance(ws, list):
                all_windows.extend(ws)
        if not all_windows:
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

            start_step = int(steps_ref[s_idx])
            end_step = int(steps_ref[e_idx + 1])
            step = end_step

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
