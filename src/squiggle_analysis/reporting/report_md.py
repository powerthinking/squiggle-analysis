from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from squiggle_core import paths


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _try_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _count_captures_by_source(samples_dir: Path) -> dict:
    """
    Count step capture folders by sample_meta.json["source"].
    """
    if not samples_dir.exists():
        return {}

    counts: dict[str, int] = {}
    for step_dir in sorted(samples_dir.glob("step_*")):
        meta_path = step_dir / "sample_meta.json"
        source = "unknown"
        if meta_path.exists():
            try:
                source = json.loads(meta_path.read_text()).get("source", "unknown")
            except Exception:
                source = "unknown"
        counts[source] = counts.get(source, 0) + 1

    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _latest_nonnull_row(df: pd.DataFrame, cols: list[str]) -> Optional[pd.Series]:
    for c in cols:
        if c not in df.columns:
            return None
    sub = df.dropna(subset=cols)
    if sub.empty:
        return None
    return sub.iloc[-1]


def _probe_summary_lines(scalars: pd.DataFrame) -> list[str]:
    """
    Supports two formats:
      A) probe_acc / probe_loss
      B) probe_acc_A / probe_loss_A (+ optional _B)
    """
    lines: list[str] = []

    if scalars is None or scalars.empty:
        return ["_No probe metrics available (missing scalars)._"]

    # Format B: A/B
    if "probe_acc_A" in scalars.columns or "probe_loss_A" in scalars.columns:
        lastA = _latest_nonnull_row(scalars, ["probe_acc_A", "probe_loss_A"])
        if lastA is not None:
            peakA = float(scalars["probe_acc_A"].max())
            lines.append(
                f"- Probe A: latest acc **{float(lastA['probe_acc_A']):.4f}**, "
                f"loss **{float(lastA['probe_loss_A']):.6g}** | peak acc **{peakA:.4f}**"
            )
        else:
            lines.append("- Probe A: _no probe rows logged_")

        if "probe_acc_B" in scalars.columns or "probe_loss_B" in scalars.columns:
            lastB = _latest_nonnull_row(scalars, ["probe_acc_B", "probe_loss_B"])
            if lastB is not None:
                peakB = float(scalars["probe_acc_B"].max())
                lines.append(
                    f"- Probe B (holdout): latest acc **{float(lastB['probe_acc_B']):.4f}**, "
                    f"loss **{float(lastB['probe_loss_B']):.6g}** | peak acc **{peakB:.4f}**"
                )
            else:
                lines.append("- Probe B (holdout): _no probe rows logged_")

        return lines

    # Format A: single probe
    if "probe_acc" in scalars.columns or "probe_loss" in scalars.columns:
        last = _latest_nonnull_row(scalars, ["probe_acc", "probe_loss"])
        if last is None:
            return ["- Probe: _no probe rows logged_"]
        peak = float(scalars["probe_acc"].max()) if "probe_acc" in scalars.columns else float("nan")
        return [
            f"- Probe: latest acc **{float(last['probe_acc']):.4f}**, "
            f"loss **{float(last['probe_loss']):.6g}** | peak acc **{peak:.4f}**"
        ]

    return ["_No probe columns found in scalars._"]


def write_report(
    run_id: str,
    geom: Optional[pd.DataFrame] = None,
    events: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Write a Markdown report for a run.
    """
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_path = run_dir / "meta.json"
    scalars_long_path = paths.metrics_scalar_path(run_id)
    scalars_wide_path = paths.metrics_wide_path(run_id)
    captures_dir = paths.captures_dir(run_id)

    geometry_path = paths.geometry_state_long_path(run_id)
    events_path = paths.events_path(run_id)

    meta = _read_json(meta_path)
    scalars_wide = _try_read_parquet(scalars_wide_path)
    scalars_long = _try_read_parquet(scalars_long_path)

    # Prefer wide-form scalars for human summaries; if only long-form exists, pivot it to wide.
    scalars = None
    if scalars_wide is not None and not scalars_wide.empty:
        scalars = scalars_wide
    elif scalars_long is not None and not scalars_long.empty:
        try:
            tmp = scalars_long.copy()
            if "wall_time" in tmp.columns:
                tmp = tmp.drop(columns=["wall_time"])
            wide = (
                tmp.pivot_table(
                    index=["run_id", "step"],
                    columns="metric_name",
                    values="value",
                    aggfunc="last",
                )
                .reset_index()
            )
            wide.columns = [str(c) for c in wide.columns]
            if "step" in wide.columns:
                wide = wide.sort_values("step").reset_index(drop=True)
            scalars = wide
        except Exception:
            scalars = scalars_long

    if geom is None:
        geom = _try_read_parquet(geometry_path)
    if events is None:
        events = _try_read_parquet(events_path)

    # ---- Scalars summary ----
    scalar_lines: list[str] = []
    if scalars is None or scalars.empty:
        scalar_lines.append("_No scalars parquet found._")
    else:
        last = scalars.iloc[-1]
        best_loss = float(scalars["loss"].min()) if "loss" in scalars.columns else float("nan")
        best_step = int(scalars.loc[scalars["loss"].idxmin(), "step"]) if "loss" in scalars.columns else -1

        if "step" in scalars.columns:
            scalar_lines.append(f"- steps: **{int(last['step']) + 1:,}**")
        if "loss" in scalars.columns:
            scalar_lines.append(f"- final loss: **{float(last['loss']):.6g}**")
            scalar_lines.append(f"- best loss: **{best_loss:.6g}** @ step **{best_step:,}**")
        if "lr" in scalars.columns:
            scalar_lines.append(f"- lr (final): **{float(last['lr']):.6g}**")
        if "grad_norm" in scalars.columns and pd.notna(last.get("grad_norm", None)):
            scalar_lines.append(f"- grad_norm (final): **{float(last['grad_norm']):.6g}**")

    # ---- Probe summary ----
    probe_lines = _probe_summary_lines(scalars) if scalars is not None else ["_No probe metrics available._"]

    # ---- Capture inventory ----
    cap_counts = _count_captures_by_source(captures_dir)
    capture_lines: list[str] = []
    if not cap_counts:
        capture_lines.append("_No captures found._")
    else:
        total = sum(cap_counts.values())
        capture_lines.append(f"- total capture steps: **{total:,}**")
        for k, v in cap_counts.items():
            capture_lines.append(f"  - {k}: **{v:,}**")

    # ---- Geometry summary ----
    geom_lines: list[str] = []
    if geom is None or geom.empty:
        geom_lines.append("_No geometry_state parquet found._")
    else:
        geom_lines.append(f"- rows: **{len(geom):,}**")
        if "metric" in geom.columns:
            uniq = sorted(set(geom["metric"].astype(str).tolist()))
            geom_lines.append(f"- metrics: `{', '.join(uniq[:20])}`" + (" …" if len(uniq) > 20 else ""))

    # ---- Events table (interval-aware) ----
    events_section = ""
    if events is None or events.empty:
        events_section = "_No events detected (or events file missing)._"
    else:
        # Choose the most informative columns available.
        preferred_cols = [
            "run_id",
            "event_id",
            "event_type",
            "layer",
            "metric",
            "step",        # canonical event timestamp (always present  in new schema)
            "start_step",  # interval context (if present)
            "end_step",
            "score",
        ]
        cols = [c for c in preferred_cols if c in events.columns]

        show = events[cols].copy()

        # Stable, sensible ordering for readability
        if "score" in show.columns:
            show = show.sort_values(
                by=["score", "layer", "metric", "step"] if "step" in show.columns else ["score", "layer", "metric"],
                ascending=[False, True, True, True] if "step" in show.columns else [False, True, True],
            )
        else:
            # fallback ordering if score is missing for some reason
            if "step" in show.columns:
                show = show.sort_values(["layer", "metric", "step"], ascending=[True, True, True])
            else:
                show = show.sort_values(["layer", "metric"], ascending=[True, True])

        show = show.head(30)
        events_section = show.to_markdown(index=False)

    # ---- Build report ----
    lines: list[str] = []
    lines.append(f"# Squiggle Report: `{run_id}`")
    lines.append("")

    lines.append("## Run metadata")
    if meta:
        lines.append(f"- run_name: **{meta.get('run_name', 'n/a')}**")
        lines.append(f"- created_at: **{meta.get('created_at', 'n/a')}**")
        lines.append(f"- device: **{meta.get('device', 'n/a')}**")
        lines.append(f"- python: **{meta.get('python', 'n/a')}**")
        lines.append(f"- torch: **{meta.get('torch', 'n/a')}**")

        task = meta.get("task", {})
        if isinstance(task, dict):
            lines.append(f"- task: **{task.get('type', 'n/a')}** (p={task.get('p', 'n/a')})")

        model = meta.get("model", {})
        if isinstance(model, dict):
            lines.append(
                "- model: "
                f"d_model={model.get('d_model','?')}, "
                f"n_layers={model.get('n_layers','?')}, "
                f"n_heads={model.get('n_heads','?')}, "
                f"d_ff={model.get('d_ff','?')}, "
                f"dropout={model.get('dropout','?')}"
            )

        lines.append("")
        lines.append(f"- meta.json: `{meta_path}`")
    else:
        lines.append("_meta.json missing._")

    lines.append("")
    lines.append("## Scalars")
    lines.extend(scalar_lines)

    lines.append("")
    lines.append("## Probes")
    lines.extend(probe_lines)

    lines.append("")
    lines.append("## Captures")
    lines.extend(capture_lines)

    lines.append("")
    lines.append("## Geometry state")
    lines.extend(geom_lines)
    lines.append("")
    lines.append(f"- geometry_state: `{geometry_path}`")

    lines.append("")
    lines.append("## Events")
    lines.append(events_section)
    lines.append("")
    lines.append(f"- events: `{events_path}`")

    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- scalars (wide): `{scalars_wide_path}`")
    lines.append(f"- scalars (long): `{scalars_long_path}`")
    lines.append(f"- captures_dir: `{captures_dir}`")
    lines.append("")

    report_path = paths.report_md_path(run_id)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    return report_path

