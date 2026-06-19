#!/bin/bash
set -euo pipefail

if [ "${ALLOW_LEGACY_DIRECT_SBATCH:-0}" != "1" ]; then
  echo "Refusing direct legacy sbatch submission." >&2
  echo "Use scripts/goal_submit_batch.py with a validated manifest for autonomous batches." >&2
  echo "For deliberate legacy operation, set ALLOW_LEGACY_DIRECT_SBATCH=1." >&2
  exit 1
fi

mkdir -p slurm_logs outputs/smoke outputs/smoke_eval

CONFIG=configs/smoke.yaml
OUTPUT_BASE=outputs/smoke
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

jid_standard=$(sbatch ${SBATCH_ARGS} --parsable --export=NONE,VERSION=standard,CONFIG=${CONFIG},OUTPUT_BASE=${OUTPUT_BASE} scripts/slurm_train.sbatch)
jid_loop_final=$(sbatch ${SBATCH_ARGS} --parsable --export=NONE,VERSION=loop_final,CONFIG=${CONFIG},OUTPUT_BASE=${OUTPUT_BASE} scripts/slurm_train.sbatch)
jid_loop_mat=$(sbatch ${SBATCH_ARGS} --parsable --export=NONE,VERSION=loop_matryoshka,CONFIG=${CONFIG},OUTPUT_BASE=${OUTPUT_BASE} scripts/slurm_train.sbatch)
jid_loop_final_recurrent=$(sbatch ${SBATCH_ARGS} --parsable --export=NONE,VERSION=loop_final_recurrent_mean_pool,CONFIG=${CONFIG},OUTPUT_BASE=${OUTPUT_BASE} scripts/slurm_train.sbatch)
jid_loop_mat_recurrent=$(sbatch ${SBATCH_ARGS} --parsable --export=NONE,VERSION=loop_matryoshka_recurrent_mean_pool,CONFIG=${CONFIG},OUTPUT_BASE=${OUTPUT_BASE} scripts/slurm_train.sbatch)
jid_loop_final_no_memory=$(sbatch ${SBATCH_ARGS} --parsable --export=NONE,VERSION=loop_final_recurrent_no_memory,CONFIG=${CONFIG},OUTPUT_BASE=${OUTPUT_BASE} scripts/slurm_train.sbatch)
jid_loop_mat_no_memory=$(sbatch ${SBATCH_ARGS} --parsable --export=NONE,VERSION=loop_matryoshka_recurrent_no_memory,CONFIG=${CONFIG},OUTPUT_BASE=${OUTPUT_BASE} scripts/slurm_train.sbatch)

echo "Submitted train jobs:"
echo "${jid_standard} standard"
echo "${jid_loop_final} loop_final"
echo "${jid_loop_mat} loop_matryoshka"
echo "${jid_loop_final_recurrent} loop_final_recurrent_mean_pool"
echo "${jid_loop_mat_recurrent} loop_matryoshka_recurrent_mean_pool"
echo "${jid_loop_final_no_memory} loop_final_recurrent_no_memory"
echo "${jid_loop_mat_no_memory} loop_matryoshka_recurrent_no_memory"

sbatch ${SBATCH_ARGS} --dependency=afterok:${jid_standard} --export=NONE,VERSION=standard,CHECKPOINT_DIR=${OUTPUT_BASE}/standard/final,OUTPUT_DIR=outputs/smoke_eval,EVAL_ALL_LOOPS=false,TASK_NAMES=${TASK_NAMES} scripts/slurm_eval.sbatch
sbatch ${SBATCH_ARGS} --dependency=afterok:${jid_loop_final} --export=NONE,VERSION=loop_final,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_final/final,OUTPUT_DIR=outputs/smoke_eval,EVAL_ALL_LOOPS=true,TASK_NAMES=${TASK_NAMES} scripts/slurm_eval.sbatch
sbatch ${SBATCH_ARGS} --dependency=afterok:${jid_loop_mat} --export=NONE,VERSION=loop_matryoshka,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_matryoshka/final,OUTPUT_DIR=outputs/smoke_eval,EVAL_ALL_LOOPS=true,TASK_NAMES=${TASK_NAMES} scripts/slurm_eval.sbatch
sbatch ${SBATCH_ARGS} --dependency=afterok:${jid_loop_final_recurrent} --export=NONE,VERSION=loop_final_recurrent_mean_pool,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_final_recurrent_mean_pool/final,OUTPUT_DIR=outputs/smoke_eval,EVAL_ALL_LOOPS=true,TASK_NAMES=${TASK_NAMES} scripts/slurm_eval.sbatch
sbatch ${SBATCH_ARGS} --dependency=afterok:${jid_loop_mat_recurrent} --export=NONE,VERSION=loop_matryoshka_recurrent_mean_pool,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_matryoshka_recurrent_mean_pool/final,OUTPUT_DIR=outputs/smoke_eval,EVAL_ALL_LOOPS=true,TASK_NAMES=${TASK_NAMES} scripts/slurm_eval.sbatch
sbatch ${SBATCH_ARGS} --dependency=afterok:${jid_loop_final_no_memory} --export=NONE,VERSION=loop_final_recurrent_no_memory,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_final_recurrent_no_memory/final,OUTPUT_DIR=outputs/smoke_eval,EVAL_ALL_LOOPS=true,TASK_NAMES=${TASK_NAMES} scripts/slurm_eval.sbatch
sbatch ${SBATCH_ARGS} --dependency=afterok:${jid_loop_mat_no_memory} --export=NONE,VERSION=loop_matryoshka_recurrent_no_memory,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_matryoshka_recurrent_no_memory/final,OUTPUT_DIR=outputs/smoke_eval,EVAL_ALL_LOOPS=true,TASK_NAMES=${TASK_NAMES} scripts/slurm_eval.sbatch
