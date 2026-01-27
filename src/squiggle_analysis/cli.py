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
        )

        if args.output is None:
            print(report)
    else:
        # Single-run analysis mode
        run_analysis(
            run_id=args.run_id,
            force=args.force,
            baseline_run_id=args.baseline_run_id,
            baseline_id=args.baseline_id,
        )
