#!/bin/bash
set -euo pipefail

mkdir -p slurm_logs outputs/preexp outputs/preexp_eval

CONFIG=configs/preexp.yaml
OUTPUT_BASE=outputs/preexp
SBATCH_ARGS=${SBATCH_ARGS:-}

jid_standard=$(sbatch ${SBATCH_ARGS} --parsable --export=ALL,VERSION=standard,CONFIG=${CONFIG},OUTPUT_BASE=${OUTPUT_BASE} scripts/slurm_train.sbatch)
jid_loop_final=$(sbatch ${SBATCH_ARGS} --parsable --export=ALL,VERSION=loop_final,CONFIG=${CONFIG},OUTPUT_BASE=${OUTPUT_BASE} scripts/slurm_train.sbatch)
jid_loop_mat=$(sbatch ${SBATCH_ARGS} --parsable --export=ALL,VERSION=loop_matryoshka,CONFIG=${CONFIG},OUTPUT_BASE=${OUTPUT_BASE} scripts/slurm_train.sbatch)

echo "Submitted train jobs:"
echo "${jid_standard} standard"
echo "${jid_loop_final} loop_final"
echo "${jid_loop_mat} loop_matryoshka"

sbatch ${SBATCH_ARGS} --dependency=afterok:${jid_standard} --export=ALL,VERSION=standard,CHECKPOINT_DIR=${OUTPUT_BASE}/standard/final,OUTPUT_DIR=outputs/preexp_eval,EVAL_ALL_LOOPS=false scripts/slurm_eval.sbatch
sbatch ${SBATCH_ARGS} --dependency=afterok:${jid_loop_final} --export=ALL,VERSION=loop_final,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_final/final,OUTPUT_DIR=outputs/preexp_eval,EVAL_ALL_LOOPS=true scripts/slurm_eval.sbatch
sbatch ${SBATCH_ARGS} --dependency=afterok:${jid_loop_mat} --export=ALL,VERSION=loop_matryoshka,CHECKPOINT_DIR=${OUTPUT_BASE}/loop_matryoshka/final,OUTPUT_DIR=outputs/preexp_eval,EVAL_ALL_LOOPS=true scripts/slurm_eval.sbatch
