#!/bin/bash
set -euo pipefail

mkdir -p slurm_logs outputs

CONFIG=${CONFIG:-configs/preexp.yaml}
OUTPUT_BASE=${OUTPUT_BASE:-outputs}
SBATCH_ARGS=${SBATCH_ARGS:-}
METHODS=(loop_final_last loop_final_none loop_matryoshka_last loop_matryoshka_none)

for METHOD in "${METHODS[@]}"; do
  RUN_DIR="${OUTPUT_BASE}/${METHOD}"
  if [ -e "${RUN_DIR}" ]; then
    echo "Refusing to overwrite existing run directory: ${RUN_DIR}" >&2
    echo "Move or remove it before rerunning this ablation." >&2
    exit 1
  fi
done

declare -A TRAIN_JOBS
declare -A EVAL_JOBS

echo "Submitted memory-history range train jobs:"
for METHOD in "${METHODS[@]}"; do
  TRAIN_JOBS["${METHOD}"]=$(sbatch ${SBATCH_ARGS} --parsable --export=ALL,VERSION=${METHOD},CONFIG=${CONFIG},OUTPUT_BASE=${OUTPUT_BASE} scripts/slurm_train.sbatch)
  echo "${TRAIN_JOBS[${METHOD}]} ${METHOD}"
done

echo "Submitted memory-history range eval jobs:"
for METHOD in "${METHODS[@]}"; do
  EVAL_JOBS["${METHOD}"]=$(sbatch ${SBATCH_ARGS} --parsable --dependency=afterok:${TRAIN_JOBS[${METHOD}]} --export=ALL,VERSION=${METHOD},CHECKPOINT_DIR=${OUTPUT_BASE}/${METHOD}/final,OUTPUT_DIR=${OUTPUT_BASE}/${METHOD}/eval_tmp,EVAL_ALL_LOOPS=true scripts/slurm_eval.sbatch)
  echo "${EVAL_JOBS[${METHOD}]} ${METHOD}"
done

eval_dependencies=""
for METHOD in "${METHODS[@]}"; do
  if [ -z "${eval_dependencies}" ]; then
    eval_dependencies="${EVAL_JOBS[${METHOD}]}"
  else
    eval_dependencies="${eval_dependencies}:${EVAL_JOBS[${METHOD}]}"
  fi
done

jid_finalize=$(sbatch ${SBATCH_ARGS} --parsable --dependency=afterok:${eval_dependencies} --export=ALL,METHODS="${METHODS[*]}",PLOTS_DIR=${OUTPUT_BASE}/plots scripts/slurm_finalize_ablation.sbatch)

echo "Submitted finalize job:"
echo "${jid_finalize} finalize_memory_history_range"
