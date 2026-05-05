#!/bin/bash
#SBATCH --job-name=loopmat_eval
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# Pass site-specific partition/account options through sbatch or SBATCH_ARGS.

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"

module purge || true

source scripts/slurm_env.sh

echo "Conda env: ${CONDA_ENV:-<not activated by script>}"
echo "Python: $(which "${PYTHON_BIN}")"
"${PYTHON_BIN}" --version

VERSION=${VERSION:-standard}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-outputs/preexp/standard/final}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/preexp_eval}
EVAL_ALL_LOOPS=${EVAL_ALL_LOOPS:-true}

mkdir -p "${OUTPUT_DIR}" slurm_logs

"${PYTHON_BIN}" -m src.eval_mteb \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --version "${VERSION}" \
  --task_name SciFact \
  --eval_all_loops "${EVAL_ALL_LOOPS}" \
  --batch_size 32 \
  --output_dir "${OUTPUT_DIR}"

echo "End time: $(date)"
