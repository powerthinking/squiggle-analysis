import argparse
from pathlib import Path

from .run import run_analysis


def main():
    parser = argparse.ArgumentParser(description="Run squiggle analysis on training runs")

    # Mutually exclusive: single-run analysis vs multi-run comparison
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="Run ID to analyze (single-run mode)")
    group.add_argument(
        "--compare",
        nargs="+",
        metavar="RUN_ID",
        help="Compare multiple runs for seed invariance (multi-run mode)",
    )

    # Single-run options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute artifacts even if they already exist",
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

    if args.compare:
        # Multi-run comparison mode
        from .compare_runs import generate_comparison_report

        if len(args.compare) < 2:
            parser.error("--compare requires at least 2 run IDs")

        report = generate_comparison_report(
            run_ids=args.compare,
            output_path=args.output,
            step_tolerance=args.step_tolerance,
            generate_plots=not args.no_plots,
            plots_dir=args.plots_dir,
            layers=args.layers,
            llm_analysis=args.llm_analysis,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            llm_question=args.llm_question,
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
            baseline_run_id=args.baseline_run_id,
            baseline_id=args.baseline_id,
            event_detection_overrides=event_detection_overrides if event_detection_overrides else None,
            llm_analysis=args.llm_analysis,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            llm_question=args.llm_question,
        )
