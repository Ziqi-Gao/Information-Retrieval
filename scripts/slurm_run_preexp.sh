#!/bin/bash
set -euo pipefail

CONFIG=${CONFIG:-configs/preexp.yaml}
OUTPUT_BASE=${OUTPUT_BASE:-outputs/preexp}
EVAL_OUTPUT_BASE=${EVAL_OUTPUT_BASE:-outputs/preexp_eval}
SBATCH_ARGS=${SBATCH_ARGS:-}

mkdir -p slurm_logs "${OUTPUT_BASE}" "${EVAL_OUTPUT_BASE}"

submit_train() {
  local version="$1"
  local run_name="$2"
  local loop_memory_mode="${3:-}"
  local loop_query_mode="${4:-}"
  local export_vars="ALL,VERSION=${version},CONFIG=${CONFIG},OUTPUT_BASE=${OUTPUT_BASE},RUN_NAME=${run_name}"

  if [ -n "${loop_memory_mode}" ]; then
    export_vars="${export_vars},LOOP_MEMORY_MODE=${loop_memory_mode}"
  fi
  if [ -n "${loop_query_mode}" ]; then
    export_vars="${export_vars},LOOP_QUERY_MODE=${loop_query_mode}"
  fi

  sbatch ${SBATCH_ARGS} --parsable --export="${export_vars}" scripts/slurm_train.sbatch
}

submit_eval() {
  local train_jid="$1"
  local version="$2"
  local run_name="$3"
  local eval_all_loops="$4"
  local checkpoint_dir="${OUTPUT_BASE}/${run_name}/final"
  local output_dir="${EVAL_OUTPUT_BASE}/${run_name}"

  sbatch ${SBATCH_ARGS} --parsable \
    --dependency=afterok:${train_jid} \
    --export=ALL,VERSION=${version},CHECKPOINT_DIR=${checkpoint_dir},OUTPUT_DIR=${output_dir},EVAL_ALL_LOOPS=${eval_all_loops} \
    scripts/slurm_eval.sbatch
}

train_jids=()
eval_jids=()
versions=()
run_names=()
eval_flags=()

jid=$(submit_train standard standard)
train_jids+=("${jid}")
versions+=(standard)
run_names+=(standard)
eval_flags+=(false)

for version in loop_final loop_matryoshka; do
  for mode in first_token mean_pool token_concat; do
    run_name="${version}_${mode}"
    jid=$(submit_train "${version}" "${run_name}" "${mode}")
    train_jids+=("${jid}")
    versions+=("${version}")
    run_names+=("${run_name}")
    eval_flags+=(true)
  done
done

for version in loop_final_recurrent_mean_pool loop_matryoshka_recurrent_mean_pool; do
  jid=$(submit_train "${version}" "${version}" mean_pool recurrent_hidden)
  train_jids+=("${jid}")
  versions+=("${version}")
  run_names+=("${version}")
  eval_flags+=(true)
done

for version in loop_final_recurrent_no_memory loop_matryoshka_recurrent_no_memory; do
  jid=$(submit_train "${version}" "${version}" none recurrent_hidden)
  train_jids+=("${jid}")
  versions+=("${version}")
  run_names+=("${version}")
  eval_flags+=(true)
done

echo "Submitted train jobs:"
for idx in "${!train_jids[@]}"; do
  echo "${train_jids[$idx]} ${run_names[$idx]}"
done

echo "Submitted eval jobs:"
for idx in "${!train_jids[@]}"; do
  jid=$(submit_eval "${train_jids[$idx]}" "${versions[$idx]}" "${run_names[$idx]}" "${eval_flags[$idx]}")
  eval_jids+=("${jid}")
  echo "${jid} afterok:${train_jids[$idx]} ${run_names[$idx]}"
done
