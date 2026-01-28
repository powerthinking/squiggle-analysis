from __future__ import annotations

import pandas as pd
from squiggle_core import paths
from squiggle_core.schemas import parquet_schemas

from .events.change_point import detect_events
from .geometry.compute_state import compute_geometry_state
from .reporting.report_md import write_report


def run_analysis(
    run_id: str,
    force: bool = False,
    *,
    analysis_id: str = "analysis@2.0",
    baseline_run_id: str | None = None,
    baseline_id: str | None = None,
    event_detection_overrides: dict | None = None,
    llm_analysis: bool = False,
    llm_backend: str = "openai",
    llm_model: str = "gpt-4o",
    llm_question: str | None = None,
):
    """
    End-to-end analysis for a single run.

    Args:
        run_id: The run to analyze
        force: Recompute all artifacts even if they exist
        analysis_id: Version identifier for this analysis
        baseline_run_id: Use another run's geometry as scoring baseline
        baseline_id: Load a persisted baseline file by ID
        event_detection_overrides: Optional dict of parameter overrides for detect_events()
            Supported keys: peak_suppression_radius, max_events_per_series, warmup_fraction,
            max_pre_warmup
        llm_analysis: Whether to generate LLM qualitative analysis
        llm_backend: LLM backend ("openai" or "anthropic")
        llm_model: Model to use for analysis
        llm_question: Optional specific question to ask the LLM
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
        # Build event detection kwargs with optional overrides
        detect_kwargs = {
            "analysis_id": analysis_id,
            "baseline_run_id": baseline_run_id,
            "baseline_id": baseline_id,
            "rank_threshold": 0.2,
            "mass_threshold": 0.03,
        }
        # Apply CLI overrides if provided
        if event_detection_overrides:
            detect_kwargs.update(event_detection_overrides)

        detect_events(run_id, **detect_kwargs)
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

    # 4) Optional LLM analysis
    if llm_analysis:
        _run_llm_analysis(
            run_id=run_id,
            geom=geom,
            report_path=report_path,
            event_detection_overrides=event_detection_overrides,
            llm_backend=llm_backend,
            llm_model=llm_model,
            llm_question=llm_question,
        )


def _run_llm_analysis(
    run_id: str,
    geom: pd.DataFrame,
    report_path,
    event_detection_overrides: dict | None,
    llm_backend: str,
    llm_model: str,
    llm_question: str | None,
) -> None:
    """Run LLM analysis on the generated report."""
    import json

    from squiggle_analysis.llm_analysis.analyzer import (
        AnalysisRequest,
        analyze_report,
        write_analysis_result,
    )

    # Load metadata
    meta_path = paths.run_dir(run_id) / "meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    # Read the generated report
    report_content = report_path.read_text()

    # Build detection config from overrides (include all parameters)
    overrides = event_detection_overrides or {}
    detection_config = {
        "warmup_fraction": overrides.get("warmup_fraction", 0.1),
        "max_pre_warmup": overrides.get("max_pre_warmup", 1),
        "peak_suppression_radius": overrides.get("peak_suppression_radius", 15),
        "max_events_per_series": overrides.get("max_events_per_series", 5),
        "adaptive_k": overrides.get("adaptive_k", 2.5),
    }

    # Build run context
    run_context = {
        "analysis_mode": "single_run",
        "run_id": run_id,
        "seed": meta.get("seed"),
        "model_info": meta.get("model", {}),
        "step_range": [int(geom["step"].min()), int(geom["step"].max())],
        "detection_config": detection_config,
    }

    request = AnalysisRequest(
        run_context=run_context,
        primary_report=report_content,
        compare_report=None,
        artifacts=[],  # Could list plot paths here
        user_question=llm_question,
    )

    print(f"[...] Running LLM analysis with {llm_model}...")
    result = analyze_report(
        request,
        backend=llm_backend,
        model=llm_model,
    )

    # Write output
    analysis_path = report_path.parent / "llm_analysis.json"
    write_analysis_result(result, analysis_path)

    print(f"[✓] LLM analysis written to: {analysis_path}")

    if result.validation_errors:
        print(f"    WARNING: {len(result.validation_errors)} validation errors in response")
