from squiggle_core import paths
from squiggle_core.schemas import parquet_schemas

from .geometry.compute_state import compute_geometry_state
from .events.change_point import detect_events
from .reporting.report_md import write_report


def run_analysis(run_id: str, force: bool = False):
    """
    End-to-end analysis for a single run.
    """

    # Resolve paths
    samples_dir = paths.samples_dir(run_id)
    geometry_path = paths.geometry_state_path(run_id)
    events_path = paths.events_path(run_id)
    report_path = paths.run_dir(run_id) / "report.md"

    # 1. Geometry state
    if force or not geometry_path.exists():
        compute_geometry_state(run_id)

    # 2. Events
    if force or not events_path.exists():
        detect_events(run_id, rank_threshold=0.2, mass_threshold=0.03)

    # 3. Report
    write_report(run_id)

    print(f"[✓] Analysis complete for run {run_id}")
    print(f"    Report: {report_path}")
