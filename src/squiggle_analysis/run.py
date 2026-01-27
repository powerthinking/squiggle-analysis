from __future__ import annotations

import pandas as pd

from squiggle_core import paths
from squiggle_core.schemas import parquet_schemas

from .geometry.compute_state import compute_geometry_state
from .events.change_point import detect_events
from .reporting.report_md import write_report


def run_analysis(
    run_id: str,
    force: bool = False,
    *,
    analysis_id: str = "analysis@2.0",
    baseline_run_id: str | None = None,
    baseline_id: str | None = None,
):
    """
    End-to-end analysis for a single run.

    Args:
        run_id: The run to analyze
        force: Recompute all artifacts even if they exist
        analysis_id: Version identifier for this analysis
        baseline_run_id: Use another run's geometry as scoring baseline
        baseline_id: Load a persisted baseline file by ID
    """

    captures_dir = paths.captures_dir(run_id)
    geometry_path = paths.geometry_state_path(run_id)
    events_candidates_path = paths.events_candidates_path(run_id)
    report_path = paths.report_md_path(run_id)

    # 0) basic existence check for v0 sanity
    if not captures_dir.exists():
        raise FileNotFoundError(
            f"No captures directory for run_id='{run_id}'. Expected: {captures_dir}\n"
            "Run the scout training first so captures get written."
        )

    # 1) Geometry state (long-form)
    if force or not geometry_path.exists():
        compute_geometry_state(run_id, analysis_id=analysis_id)
    if not geometry_path.exists():
        raise RuntimeError(f"compute_geometry_state did not write: {geometry_path}")

    # Validate geometry after computation
    geom = pd.read_parquet(geometry_path)
    parquet_schemas.validate_geometry_state_df(geom)

    # 2) Events (with optional cross-run baseline)
    if force or not events_candidates_path.exists():
        detect_events(
            run_id,
            analysis_id=analysis_id,
            baseline_run_id=baseline_run_id,
            baseline_id=baseline_id,
            rank_threshold=0.2,
            mass_threshold=0.03,
        )
    if not events_candidates_path.exists():
        raise RuntimeError(f"detect_events did not write: {events_candidates_path}")

    # Validate events after computation
    events = pd.read_parquet(events_candidates_path)
    parquet_schemas.validate_events_df(events)

    # 3) Report (pass loaded dfs so report is consistent with what we validated)
    write_report(run_id, geom=geom, events=events)
    if not report_path.exists():
        raise RuntimeError(f"Report was not written: {report_path}")

    print(f"[✓] Analysis complete for run {run_id}")
    print(f"    Report: {report_path}")
