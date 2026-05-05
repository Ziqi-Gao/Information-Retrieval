#!/bin/bash
#SBATCH --job-name=loopmat_plot
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00

# Pass site-specific partition/account options through sbatch or SBATCH_ARGS.

set -euo pipefail

module purge || true
source scripts/slurm_env.sh

SUMMARY_CSV=${SUMMARY_CSV:-outputs/preexp_eval/results_summary.csv}
PLOT_DIR=${PLOT_DIR:-outputs/preexp_eval/plots}

"${PYTHON_BIN}" -m src.plot_results \
  --summary_csv "${SUMMARY_CSV}" \
  --output_dir "${PLOT_DIR}"
