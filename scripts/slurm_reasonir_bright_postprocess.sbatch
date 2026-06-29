#!/bin/bash
#SBATCH --job-name=bright_postprocess
#SBATCH --output=slurm_logs/bright_postprocess_%j.out
#SBATCH --error=slurm_logs/bright_postprocess_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

if [ "${REASONIR_BRIGHT_SUBMIT_BATCH:-}" != "1" ]; then
  echo "Refusing direct sbatch. Use scripts/reasonir_bright_submit_batch.py." >&2
  exit 2
fi

if [ -z "${BATCH_ID:-}" ]; then
  echo "BATCH_ID is required." >&2
  exit 2
fi

RUN_ROOT=${RUN_ROOT:-outputs/reasonir_bright/runs}
EVAL_ROOT=${EVAL_ROOT:-outputs/reasonir_bright/eval}
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/${BATCH_ID}}
METRIC=${METRIC:-ndcg_at_10}
MANIFEST=${MANIFEST:-${OUTPUT_DIR}/batch_manifest.submitted.yaml}

mkdir -p "${OUTPUT_DIR}" slurm_logs

timestamp_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

write_failed_marker() {
  local exit_code="$1"
  local failed_line="$2"
  cat > "${OUTPUT_DIR}/postprocess_failed.json" <<EOF
{
  "batch_id": "${BATCH_ID}",
  "failed_at": "$(timestamp_utc)",
  "job_id": "${SLURM_JOB_ID:-}",
  "status": "failed",
  "exit_code": ${exit_code},
  "line": ${failed_line}
}
EOF
}

on_error() {
  local exit_code="$?"
  local failed_line="${BASH_LINENO[0]:-0}"
  trap - ERR
  write_failed_marker "${exit_code}" "${failed_line}" || true
  exit "${exit_code}"
}
trap on_error ERR

echo "ReasonIR-BRIGHT postprocess job ID: ${SLURM_JOB_ID:-<unknown>}"
echo "Batch ID: ${BATCH_ID}"
echo "Start time: $(date)"

module purge || true
source scripts/slurm_env.sh

"${PYTHON_BIN}" scripts/reasonir_bright_collect.py \
  --batch-id "${BATCH_ID}" \
  --run-root "${RUN_ROOT}" \
  --eval-root "${EVAL_ROOT}" \
  --output "${OUTPUT_DIR}/collected_results.csv"

"${PYTHON_BIN}" scripts/reasonir_bright_scoreboard.py \
  --results "${OUTPUT_DIR}/collected_results.csv" \
  --manifest "${MANIFEST}" \
  --metric "${METRIC}" \
  --output-csv "${OUTPUT_DIR}/scoreboard.csv" \
  --output-json "${OUTPUT_DIR}/scoreboard.json"

cat > "${OUTPUT_DIR}/postprocess_done.json" <<EOF
{
  "batch_id": "${BATCH_ID}",
  "completed_at": "$(timestamp_utc)",
  "job_id": "${SLURM_JOB_ID:-}",
  "status": "completed",
  "collected_results": "${OUTPUT_DIR}/collected_results.csv",
  "scoreboard_csv": "${OUTPUT_DIR}/scoreboard.csv",
  "scoreboard_json": "${OUTPUT_DIR}/scoreboard.json"
}
EOF

echo "End time: $(date)"
