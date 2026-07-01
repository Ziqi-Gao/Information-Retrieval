#!/bin/bash
#SBATCH --job-name=loopmat_postprocess
#SBATCH --output=slurm_logs/postprocess_%j.out
#SBATCH --error=slurm_logs/postprocess_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# Pass site-specific partition/account options through sbatch or SBATCH_ARGS.

set -euo pipefail

if [ "${GOAL_SUBMIT_BATCH:-}" != "1" ] && [ "${ALLOW_LEGACY_DIRECT_SBATCH:-}" != "1" ]; then
  echo "Refusing direct sbatch. Use scripts/goal_submit_batch.py, or set ALLOW_LEGACY_DIRECT_SBATCH=1 for an explicit manual override." >&2
  exit 2
fi

if [ -z "${BATCH_ID:-}" ]; then
  echo "BATCH_ID is required for Slurm postprocessing." >&2
  exit 2
fi

STATE_PATH=${STATE_PATH:-outputs/goal/state.json}
EVAL_ROOT=${EVAL_ROOT:-outputs/goal/eval}
BASELINE_CSV=${BASELINE_CSV:-outputs/baselines/standard_frozen/results_summary.csv}
METRIC=${METRIC:-ndcg_at_10}
MARGIN=${MARGIN:-0.001}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/goal/runs/${BATCH_ID}}
AUTO_CODEX=${AUTO_CODEX:-false}

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

echo "Postprocess job ID: ${SLURM_JOB_ID:-<unknown>}"
echo "Batch ID: ${BATCH_ID}"
echo "Node: ${SLURMD_NODENAME:-<unknown>}"
echo "Start time: $(date)"

module purge || true

source scripts/slurm_env.sh

echo "Conda env: ${CONDA_ENV:-<not activated by script>}"
echo "Python: $(which "${PYTHON_BIN}")"
"${PYTHON_BIN}" --version

STATUS_REFRESH="completed"
if ! "${PYTHON_BIN}" scripts/goal_status.py \
  --state "${STATE_PATH}" \
  --batch-id "${BATCH_ID}" \
  --update-state; then
  STATUS_REFRESH="failed"
  echo "WARNING: goal_status.py failed; continuing to deterministic collection and scoring because the Slurm dependency has fired." >&2
fi

"${PYTHON_BIN}" scripts/goal_collect.py \
  --batch-id "${BATCH_ID}" \
  --eval-root "${EVAL_ROOT}" \
  --output "${OUTPUT_DIR}/collected_results.csv"

"${PYTHON_BIN}" scripts/goal_scoreboard.py \
  --baseline "${BASELINE_CSV}" \
  --results "${OUTPUT_DIR}/collected_results.csv" \
  --metric "${METRIC}" \
  --margin "${MARGIN}" \
  --output-csv "${OUTPUT_DIR}/scoreboard.csv" \
  --output-json "${OUTPUT_DIR}/scoreboard.json"

AUTO_CODEX_NORMALIZED=$(printf "%s" "${AUTO_CODEX}" | tr '[:upper:]' '[:lower:]')
CODEX_STATUS="disabled"
if [ "${AUTO_CODEX_NORMALIZED}" = "true" ]; then
  SENTINEL="${OUTPUT_DIR}/.codex_resume_launched"
  if [ -e "${SENTINEL}" ]; then
    CODEX_STATUS="skipped_sentinel_exists"
  elif ! command -v codex >/dev/null 2>&1; then
    CODEX_STATUS="skipped_codex_not_found"
    cat > "${OUTPUT_DIR}/codex_resume_manual.md" <<EOF
# Manual Resume Needed

The Slurm postprocess job completed deterministic collection and scoring for ${BATCH_ID}, but \`AUTO_CODEX=true\` could not launch Codex because the \`codex\` command was not available on the compute node.

Resume manually by reading \`outputs/goal/state.json\`, \`${OUTPUT_DIR}/scoreboard.json\`, and \`docs/agent_lab_notebook.md\`.
EOF
  else
    cat > "${SENTINEL}" <<EOF
{"batch_id":"${BATCH_ID}","launched_at":"$(timestamp_utc)","job_id":"${SLURM_JOB_ID:-}"}
EOF
    set +e
    codex exec --sandbox workspace-write "Postprocess completed for ${BATCH_ID}. Read AGENTS.md, docs/goal_protocol.md, outputs/goal/state.json, and ${OUTPUT_DIR}/scoreboard.json. Summarize results and decide the next autonomous retrieval workflow step without submitting new jobs." \
      > "${OUTPUT_DIR}/codex_resume_after_postprocess.md" \
      2> "${OUTPUT_DIR}/codex_resume_after_postprocess.err"
    CODEX_RC="$?"
    set -e
    if [ "${CODEX_RC}" -eq 0 ]; then
      CODEX_STATUS="launched"
    else
      CODEX_STATUS="failed_rc_${CODEX_RC}"
      cat > "${OUTPUT_DIR}/codex_resume_manual.md" <<EOF
# Manual Resume Needed

The Slurm postprocess job completed deterministic collection and scoring for ${BATCH_ID}, but \`codex exec\` returned ${CODEX_RC}. Inspect:

- ${OUTPUT_DIR}/codex_resume_after_postprocess.md
- ${OUTPUT_DIR}/codex_resume_after_postprocess.err
- ${OUTPUT_DIR}/scoreboard.json
EOF
    fi
  fi
fi

cat > "${OUTPUT_DIR}/postprocess_done.json" <<EOF
{
  "batch_id": "${BATCH_ID}",
  "completed_at": "$(timestamp_utc)",
  "job_id": "${SLURM_JOB_ID:-}",
  "status": "completed",
  "collected_results": "${OUTPUT_DIR}/collected_results.csv",
  "scoreboard_csv": "${OUTPUT_DIR}/scoreboard.csv",
  "scoreboard_json": "${OUTPUT_DIR}/scoreboard.json",
  "status_refresh": "${STATUS_REFRESH}",
  "auto_codex": "${AUTO_CODEX_NORMALIZED}",
  "codex_status": "${CODEX_STATUS}"
}
EOF

echo "End time: $(date)"
