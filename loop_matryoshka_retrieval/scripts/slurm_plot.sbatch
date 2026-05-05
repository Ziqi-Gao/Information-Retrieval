#!/bin/bash
#SBATCH --job-name=loopmat_plot
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=short
#SBATCH --account=p32737

# Edit account, partition, time, and memory above for your HPC.

set -euo pipefail

module purge || true
set +u
source ~/.bashrc
set -u
CONDA_ENV=${CONDA_ENV:-/gpfs/projects/p32737/del6500_home/.conda/envs/ragen_ttt}
conda activate "${CONDA_ENV}"
PYTHON_BIN=${PYTHON_BIN:-python}

SUMMARY_CSV=${SUMMARY_CSV:-outputs/preexp_eval/results_summary.csv}
PLOT_DIR=${PLOT_DIR:-outputs/preexp_eval/plots}

"${PYTHON_BIN}" -m src.plot_results \
  --summary_csv "${SUMMARY_CSV}" \
  --output_dir "${PLOT_DIR}"
