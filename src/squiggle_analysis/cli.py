import argparse
from .run import run_analysis


def main():
    parser = argparse.ArgumentParser(description="Run squiggle analysis on a training run")
    parser.add_argument("--run-id", required=True, help="Run ID to analyze")
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
    args = parser.parse_args()

    run_analysis(
        run_id=args.run_id,
        force=args.force,
        baseline_run_id=args.baseline_run_id,
        baseline_id=args.baseline_id,
    )
