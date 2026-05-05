import argparse
from pathlib import Path

from .experiments import version_names
from .results import combine_summaries, finalize_single_method_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize a single-method ablation run.")
    parser.add_argument("--method", choices=version_names(), required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--plots_dir", default="outputs/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    plots_dir = Path(args.plots_dir)
    finalize_single_method_eval(run_dir, args.method)
    combine_summaries(
        [Path("outputs") / version / "results_summary.csv" for version in version_names()],
        plots_dir / "results_summary_all.csv",
    )


if __name__ == "__main__":
    main()
