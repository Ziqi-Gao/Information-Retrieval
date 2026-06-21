#!/bin/bash
#SBATCH --job-name=loopmat_eval
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# Pass site-specific partition/account options through sbatch or SBATCH_ARGS.

set -euo pipefail

if [ "${GOAL_SUBMIT_BATCH:-}" != "1" ] && [ "${ALLOW_LEGACY_DIRECT_SBATCH:-}" != "1" ]; then
  echo "Refusing direct sbatch. Use scripts/goal_submit_batch.py, or set ALLOW_LEGACY_DIRECT_SBATCH=1 for an explicit manual override." >&2
  exit 2
fi

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
LOOP_IDX=${LOOP_IDX:-}
LOOP_DOCS=${LOOP_DOCS:-false}
DOC_LOOP_IDX=${DOC_LOOP_IDX:-}
FUSION_STANDARD_CHECKPOINT_DIR=${FUSION_STANDARD_CHECKPOINT_DIR:-}
FUSION_ALPHA=${FUSION_ALPHA:-}
FUSION_SCOPE=${FUSION_SCOPE:-}
DEFAULT_EVAL_TASKS=${DEFAULT_EVAL_TASKS:-SciFact,NFCorpus,SCIDOCS,FiQA2018,ArguAna,Touche2020,TRECCOVID}
if [ -n "${TASK_NAMES:-}" ]; then
  TASK_NAMES="${TASK_NAMES}"
elif [ -n "${TASK_NAME:-}" ]; then
  TASK_NAMES="${TASK_NAME}"
else
  TASK_NAMES="${DEFAULT_EVAL_TASKS}"
fi

mkdir -p "${OUTPUT_DIR}" slurm_logs

echo "Evaluation tasks: ${TASK_NAMES}"

EVAL_EXTRA_ARGS=()
if [ -n "${LOOP_IDX}" ]; then
  EVAL_EXTRA_ARGS+=(--loop_idx "${LOOP_IDX}")
fi
if [ -n "${LOOP_DOCS}" ]; then
  EVAL_EXTRA_ARGS+=(--loop_docs "${LOOP_DOCS}")
fi
if [ -n "${DOC_LOOP_IDX}" ]; then
  EVAL_EXTRA_ARGS+=(--doc_loop_idx "${DOC_LOOP_IDX}")
fi
if [ -n "${FUSION_STANDARD_CHECKPOINT_DIR}" ]; then
  EVAL_EXTRA_ARGS+=(--fusion_standard_checkpoint_dir "${FUSION_STANDARD_CHECKPOINT_DIR}")
fi
if [ -n "${FUSION_ALPHA}" ]; then
  EVAL_EXTRA_ARGS+=(--fusion_alpha "${FUSION_ALPHA}")
fi
if [ -n "${FUSION_SCOPE}" ]; then
  EVAL_EXTRA_ARGS+=(--fusion_scope "${FUSION_SCOPE}")
fi

"${PYTHON_BIN}" -m src.eval_mteb \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --version "${VERSION}" \
  --task_names "${TASK_NAMES}" \
  --eval_all_loops "${EVAL_ALL_LOOPS}" \
  --batch_size 32 \
  --output_dir "${OUTPUT_DIR}" \
  "${EVAL_EXTRA_ARGS[@]}"

echo "End time: $(date)"
