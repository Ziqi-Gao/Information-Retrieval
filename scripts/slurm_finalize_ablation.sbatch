#!/bin/bash
#SBATCH --job-name=loopmat_finalize
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

# Pass site-specific partition/account options through sbatch or SBATCH_ARGS.

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"

module purge || true

source scripts/slurm_env.sh

METHODS=${METHODS:-"loop_final_last loop_final_none loop_matryoshka_last loop_matryoshka_none"}
PLOTS_DIR=${PLOTS_DIR:-outputs/plots}

mkdir -p "${PLOTS_DIR}" slurm_logs

for METHOD in ${METHODS}; do
  "${PYTHON_BIN}" -m src.finalize_ablation \
    --method "${METHOD}" \
    --run_dir "outputs/${METHOD}" \
    --plots_dir "${PLOTS_DIR}"
done

MPLCONFIGDIR=${MPLCONFIGDIR:-.mplconfig/${SLURM_JOB_ID:-slurm}}
export MPLCONFIGDIR
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON_BIN}" -m src.plot_results \
  --summary_csv "${PLOTS_DIR}/results_summary_all.csv" \
  --output_dir "${PLOTS_DIR}"

echo "End time: $(date)"
