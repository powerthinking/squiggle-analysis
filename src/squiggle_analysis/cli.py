import argparse
from pathlib import Path

from squiggle_core import paths

from .run import run_analysis


def main():
    parser = argparse.ArgumentParser(description="Run squiggle analysis on training runs")

    # Mutually exclusive: single-run analysis vs multi-run comparison vs list
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="Run ID to analyze (single-run mode)")
    group.add_argument(
        "--compare",
        nargs="+",
        metavar="RUN_ID",
        help="Compare multiple runs for seed invariance (multi-run mode)",
    )
    group.add_argument(
        "--list-analyses",
        metavar="RUN_ID",
        help="List available analysis versions for a run",
    )

    # Single-run options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute artifacts even if they already exist",
    )
    parser.add_argument(
        "--analysis-id",
        help="Analysis version identifier. If not provided, auto-generated from detection parameters "
             "(e.g., 'w10_p1_r15_e5_k25'). Different analysis IDs create separate output directories.",
    )
    parser.add_argument(
        "--baseline-run-id",
        help="Use another run's geometry as the scoring baseline (for cross-run comparison)",
    )
    parser.add_argument(
        "--baseline-id",
        help="Load a persisted baseline file by ID",
    )

    # Event detection parameter overrides (Part 4)
    parser.add_argument(
        "--suppression-radius",
        type=int,
        help="Minimum step separation between peaks (default: 15, ≈3 capture intervals)",
    )
    parser.add_argument(
        "--max-events-per-series",
        type=int,
        help="Maximum events per (layer, metric) series (default: 5)",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        help="Fraction of training to treat as warmup (default: 0.1)",
    )
    parser.add_argument(
        "--max-pre-warmup",
        type=int,
        help="Max early (pre-warmup) peaks allowed per series (default: 1, use 0 to block all)",
    )

    # Multi-run comparison options
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for comparison report (default: stdout)",
    )
    parser.add_argument(
        "--step-tolerance", "-t",
        type=int,
        default=5,
        help="Step tolerance for matching events in comparison (default: 5)",
    )
    parser.add_argument(
        "--compare-analysis-ids",
        nargs="+",
        help="Analysis IDs for each compared run (in same order as --compare args). "
             "Use 'auto' to auto-select the latest analysis for a run.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip trajectory plot generation in comparison mode",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        help="Directory for trajectory plots (default: plots/ next to output)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 12, 23],
        help="Layers for trajectory plots (default: 0 12 23)",
    )

    # Output options
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing report/analysis files instead of creating new versions",
    )

    # LLM analysis options
    parser.add_argument(
        "--llm-analysis",
        action="store_true",
        help="Generate LLM qualitative analysis of the report",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM backend to use (default: openai)",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o",
        help="Model to use for LLM analysis (default: gpt-4o)",
    )
    parser.add_argument(
        "--llm-question",
        type=str,
        help="Specific question to ask the LLM about the report",
    )

    args = parser.parse_args()

    if args.list_analyses:
        # List available analysis versions for a run
        run_id = args.list_analyses
        analysis_ids = paths.list_analysis_ids(run_id)

        if not analysis_ids:
            # Check for legacy (non-versioned) analysis
            legacy_path = paths.events_candidates_path(run_id, None)
            if legacy_path.exists():
                print(f"Analysis versions for {run_id}:")
                print(f"  (legacy) - {legacy_path}")
            else:
                print(f"No analysis found for {run_id}")
        else:
            print(f"Analysis versions for {run_id}:")
            for aid in analysis_ids:
                events_path = paths.events_candidates_path(run_id, aid)
                report_path = paths.report_md_path(run_id, aid)
                has_report = report_path.exists()
                print(f"  {aid}")
                print(f"    events: {events_path}")
                if has_report:
                    print(f"    report: {report_path}")

    elif args.compare:
        # Multi-run comparison mode
        from .compare_runs import generate_comparison_report

        if len(args.compare) < 2:
            parser.error("--compare requires at least 2 run IDs")

        # Parse analysis_ids for comparison
        compare_analysis_ids = None
        if args.compare_analysis_ids:
            compare_analysis_ids = [
                None if aid.lower() == "auto" else aid
                for aid in args.compare_analysis_ids
            ]

        report = generate_comparison_report(
            run_ids=args.compare,
            output_path=args.output,
            step_tolerance=args.step_tolerance,
            generate_plots=not args.no_plots,
            plots_dir=args.plots_dir,
            layers=args.layers,
            analysis_ids=compare_analysis_ids,
            llm_analysis=args.llm_analysis,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            llm_question=args.llm_question,
            overwrite=args.overwrite,
        )

        if args.output is None:
            print(report)
    else:
        # Single-run analysis mode
        # Build event detection overrides from CLI args
        event_detection_overrides = {}
        if args.suppression_radius is not None:
            event_detection_overrides["peak_suppression_radius"] = args.suppression_radius
        if args.max_events_per_series is not None:
            event_detection_overrides["max_events_per_series"] = args.max_events_per_series
        if args.warmup_fraction is not None:
            event_detection_overrides["warmup_fraction"] = args.warmup_fraction
        if args.max_pre_warmup is not None:
            event_detection_overrides["max_pre_warmup"] = args.max_pre_warmup

        run_analysis(
            run_id=args.run_id,
            force=args.force,
            analysis_id=args.analysis_id,
            baseline_run_id=args.baseline_run_id,
            baseline_id=args.baseline_id,
            event_detection_overrides=event_detection_overrides if event_detection_overrides else None,
            llm_analysis=args.llm_analysis,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            llm_question=args.llm_question,
            overwrite=args.overwrite,
        )
