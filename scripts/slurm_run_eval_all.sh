#!/bin/bash
set -euo pipefail

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

sbatch ${SBATCH_ARGS} --export=ALL,VERSION=standard,CHECKPOINT_DIR=${OUTPUT_BASE}/standard/final,OUTPUT_DIR=${EVAL_OUTPUT},EVAL_ALL_LOOPS=false scripts/slurm_eval.sbatch
sbatch ${SBATCH_ARGS} --export=ALL,VERSION=loop_final,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_final/final,OUTPUT_DIR=${EVAL_OUTPUT},EVAL_ALL_LOOPS=true scripts/slurm_eval.sbatch
sbatch ${SBATCH_ARGS} --export=ALL,VERSION=loop_matryoshka,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_matryoshka/final,OUTPUT_DIR=${EVAL_OUTPUT},EVAL_ALL_LOOPS=true scripts/slurm_eval.sbatch
sbatch ${SBATCH_ARGS} --export=ALL,VERSION=loop_final_recurrent_mean_pool,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_final_recurrent_mean_pool/final,OUTPUT_DIR=${EVAL_OUTPUT},EVAL_ALL_LOOPS=true scripts/slurm_eval.sbatch
sbatch ${SBATCH_ARGS} --export=ALL,VERSION=loop_matryoshka_recurrent_mean_pool,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_matryoshka_recurrent_mean_pool/final,OUTPUT_DIR=${EVAL_OUTPUT},EVAL_ALL_LOOPS=true scripts/slurm_eval.sbatch
sbatch ${SBATCH_ARGS} --export=ALL,VERSION=loop_final_recurrent_no_memory,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_final_recurrent_no_memory/final,OUTPUT_DIR=${EVAL_OUTPUT},EVAL_ALL_LOOPS=true scripts/slurm_eval.sbatch
sbatch ${SBATCH_ARGS} --export=ALL,VERSION=loop_matryoshka_recurrent_no_memory,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_matryoshka_recurrent_no_memory/final,OUTPUT_DIR=${EVAL_OUTPUT},EVAL_ALL_LOOPS=true scripts/slurm_eval.sbatch
