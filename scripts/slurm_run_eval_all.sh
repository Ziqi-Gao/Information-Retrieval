#!/bin/bash
set -euo pipefail

if [ "${ALLOW_LEGACY_DIRECT_SBATCH:-0}" != "1" ]; then
  echo "Refusing direct legacy sbatch submission." >&2
  echo "Use scripts/goal_submit_batch.py with a validated manifest for autonomous batches." >&2
  echo "For deliberate legacy operation, set ALLOW_LEGACY_DIRECT_SBATCH=1." >&2
  exit 1
fi

OUTPUT_BASE=${OUTPUT_BASE:-outputs/preexp}
EVAL_OUTPUT=${EVAL_OUTPUT:-outputs/preexp_eval}
SBATCH_ARGS=${SBATCH_ARGS:-}
DEFAULT_EVAL_TASKS=${DEFAULT_EVAL_TASKS:-SciFact,NFCorpus,SCIDOCS,FiQA2018,ArguAna,Touche2020,TRECCOVID}
if [ -n "${TASK_NAMES:-}" ]; then
  TASK_NAMES="${TASK_NAMES}"
elif [ -n "${TASK_NAME:-}" ]; then
  TASK_NAMES="${TASK_NAME}"
else
  TASK_NAMES="${DEFAULT_EVAL_TASKS}"
fi
export TASK_NAMES

mkdir -p slurm_logs "${EVAL_OUTPUT}"

submit_eval() {
  local version="$1"
  local run_name="$2"
  local eval_all_loops="$3"

  sbatch ${SBATCH_ARGS} \
    --export="NONE,VERSION=${version},CHECKPOINT_DIR=${OUTPUT_BASE}/${run_name}/final,OUTPUT_DIR=${EVAL_OUTPUT}/${run_name},EVAL_ALL_LOOPS=${eval_all_loops},TASK_NAMES=${TASK_NAMES}" \
    scripts/slurm_eval.sbatch
}

submit_eval standard standard false
submit_eval loop_final loop_final_mean_pool true
submit_eval loop_matryoshka loop_matryoshka_mean_pool true
submit_eval loop_final_recurrent_mean_pool loop_final_recurrent_mean_pool true
submit_eval loop_matryoshka_recurrent_mean_pool loop_matryoshka_recurrent_mean_pool true
submit_eval loop_final_recurrent_no_memory loop_final_recurrent_no_memory true
submit_eval loop_matryoshka_recurrent_no_memory loop_matryoshka_recurrent_no_memory true
