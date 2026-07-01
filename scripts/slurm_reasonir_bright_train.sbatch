#!/bin/bash
#SBATCH --job-name=reasonir_train
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

set -euo pipefail

if [ "${REASONIR_BRIGHT_SUBMIT_BATCH:-}" != "1" ]; then
  echo "Refusing direct sbatch. Use scripts/reasonir_bright_submit_batch.py." >&2
  exit 2
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"

module purge || true
source scripts/slurm_env.sh

VERSION=${VERSION:-standard}
CONFIG=${CONFIG:-configs/reasonir_bright_dev.yaml}
OUTPUT_BASE=${OUTPUT_BASE:-outputs/reasonir_bright/runs}
RUN_NAME=${RUN_NAME:-${VERSION}}
OUTDIR="${OUTPUT_BASE}/${RUN_NAME}"

mkdir -p "${OUTDIR}" slurm_logs

echo "Run name: ${RUN_NAME}"
echo "Training version: ${VERSION}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTDIR}"
echo "Python: $(which "${PYTHON_BIN}")"
"${PYTHON_BIN}" --version

TRAIN_EXTRA_ARGS=()
if [ -n "${MAX_STEPS:-}" ]; then
  TRAIN_EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
fi
if [ -n "${SAVE_STEPS:-}" ]; then
  TRAIN_EXTRA_ARGS+=(--save_steps "${SAVE_STEPS}")
fi

"${PYTHON_BIN}" -m src.train \
  --config "${CONFIG}" \
  --version "${VERSION}" \
  --output_dir "${OUTDIR}" \
  "${TRAIN_EXTRA_ARGS[@]}"

echo "End time: $(date)"
