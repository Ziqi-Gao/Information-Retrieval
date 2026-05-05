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
#SBATCH --partition=gengpu
#SBATCH --account=p32737

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"

module purge || true

set +u
source ~/.bashrc
set -u
CONDA_ENV=${CONDA_ENV:-/gpfs/projects/p32737/del6500_home/.conda/envs/ragen_ttt}
conda activate "${CONDA_ENV}"
PYTHON_BIN=${PYTHON_BIN:-python}

METHODS=${METHODS:-"loop_final_no_history loop_matryoshka_no_history"}
PLOTS_DIR=${PLOTS_DIR:-outputs/plots}

mkdir -p "${PLOTS_DIR}" slurm_logs

for METHOD in ${METHODS}; do
  "${PYTHON_BIN}" -m src.finalize_ablation \
    --method "${METHOD}" \
    --run_dir "outputs/${METHOD}" \
    --plots_dir "${PLOTS_DIR}"
done

MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-cache-${SLURM_JOB_ID}}
export MPLCONFIGDIR
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON_BIN}" -m src.plot_results \
  --summary_csv "${PLOTS_DIR}/results_summary_all.csv" \
  --output_dir "${PLOTS_DIR}"

echo "End time: $(date)"
