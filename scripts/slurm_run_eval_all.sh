#!/bin/bash
set -euo pipefail

OUTPUT_BASE=${OUTPUT_BASE:-outputs/preexp}
EVAL_OUTPUT=${EVAL_OUTPUT:-outputs/preexp_eval}
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p slurm_logs "${EVAL_OUTPUT}"

sbatch ${SBATCH_ARGS} --export=ALL,VERSION=standard,CHECKPOINT_DIR=${OUTPUT_BASE}/standard/final,OUTPUT_DIR=${EVAL_OUTPUT},EVAL_ALL_LOOPS=false scripts/slurm_eval.sbatch
sbatch ${SBATCH_ARGS} --export=ALL,VERSION=loop_final,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_final/final,OUTPUT_DIR=${EVAL_OUTPUT},EVAL_ALL_LOOPS=true scripts/slurm_eval.sbatch
sbatch ${SBATCH_ARGS} --export=ALL,VERSION=loop_matryoshka,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_matryoshka/final,OUTPUT_DIR=${EVAL_OUTPUT},EVAL_ALL_LOOPS=true scripts/slurm_eval.sbatch
