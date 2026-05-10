#!/bin/bash
#SBATCH --job-name=loopmat_train
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

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
CONFIG=${CONFIG:-configs/preexp.yaml}
OUTPUT_BASE=${OUTPUT_BASE:-outputs/preexp}

OUTDIR="${OUTPUT_BASE}/${VERSION}"

mkdir -p "${OUTDIR}" slurm_logs

echo "Training version: ${VERSION}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTDIR}"

TRAIN_EXTRA_ARGS=()
if [ -n "${EPOCHS:-}" ]; then
  TRAIN_EXTRA_ARGS+=(--epochs "${EPOCHS}")
fi
if [ -n "${MAX_STEPS:-}" ]; then
  TRAIN_EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
fi
if [ -n "${SAVE_STEPS:-}" ]; then
  TRAIN_EXTRA_ARGS+=(--save_steps "${SAVE_STEPS}")
fi
if [ -n "${LOOP_MEMORY_MODE:-}" ]; then
  TRAIN_EXTRA_ARGS+=(--loop_memory_mode "${LOOP_MEMORY_MODE}")
fi
if [ "${#TRAIN_EXTRA_ARGS[@]}" -gt 0 ]; then
  echo "Extra train args: ${TRAIN_EXTRA_ARGS[*]}"
fi

"${PYTHON_BIN}" -m src.train \
  --config "${CONFIG}" \
  --version "${VERSION}" \
  --output_dir "${OUTDIR}" \
  "${TRAIN_EXTRA_ARGS[@]}"

echo "End time: $(date)"
