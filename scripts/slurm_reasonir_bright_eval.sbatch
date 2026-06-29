#!/bin/bash
#SBATCH --job-name=bright_eval
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

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
CHECKPOINT_DIR=${CHECKPOINT_DIR:-outputs/reasonir_bright/runs/reasonir_bright_batch_001_dev/standard/final}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/reasonir_bright/eval/reasonir_bright_batch_001_dev/standard}
DOMAINS=${DOMAINS:-biology,economics,psychology,stackoverflow}
USE_LONG_DOCUMENTS=${USE_LONG_DOCUMENTS:-false}
EVAL_ALL_LOOPS=${EVAL_ALL_LOOPS:-false}
LOOP_IDX=${LOOP_IDX:-}
QUERY_BATCH_SIZE=${QUERY_BATCH_SIZE:-32}
CORPUS_BATCH_SIZE=${CORPUS_BATCH_SIZE:-64}
SCORE_CHUNK_SIZE=${SCORE_CHUNK_SIZE:-8192}

mkdir -p "${OUTPUT_DIR}" slurm_logs

EVAL_EXTRA_ARGS=()
if [ -n "${LOOP_IDX}" ]; then
  EVAL_EXTRA_ARGS+=(--loop_idx "${LOOP_IDX}")
fi

echo "BRIGHT domains: ${DOMAINS}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Python: $(which "${PYTHON_BIN}")"
"${PYTHON_BIN}" --version

"${PYTHON_BIN}" -m src.eval_bright \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --version "${VERSION}" \
  --domains "${DOMAINS}" \
  --use-long-documents "${USE_LONG_DOCUMENTS}" \
  --metric ndcg_at_10 \
  --eval_all_loops "${EVAL_ALL_LOOPS}" \
  --query_batch_size "${QUERY_BATCH_SIZE}" \
  --corpus_batch_size "${CORPUS_BATCH_SIZE}" \
  --score_chunk_size "${SCORE_CHUNK_SIZE}" \
  --output_dir "${OUTPUT_DIR}" \
  "${EVAL_EXTRA_ARGS[@]}"

echo "End time: $(date)"
