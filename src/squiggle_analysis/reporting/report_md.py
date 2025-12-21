import pandas as pd
from squiggle_core import paths


def write_report(run_id: str):
    run_dir = paths.run_dir(run_id)
    report_path = run_dir / "report.md"

    geom = pd.read_parquet(paths.geometry_state_path(run_id))
    events = pd.read_parquet(paths.events_path(run_id))

    lines = []
    lines.append(f"# Squiggle Analysis Report\n")
    lines.append(f"**Run ID:** `{run_id}`\n")

    lines.append("## Geometry State\n")
    lines.append(f"- Rows: {len(geom)}")
    lines.append(f"- Layers: {sorted(geom.layer.unique().tolist())}\n")

    lines.append("## Detected Events\n")
    if len(events) == 0:
        lines.append("_No events detected._\n")
    else:
        lines.append(events.to_markdown(index=False))

    run_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))

