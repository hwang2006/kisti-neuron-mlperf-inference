#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

QUERY_COUNT="${QUERY_COUNT:-800}"
ITERATIONS="${ITERATIONS:-4}"

# Experimentally established initial bracket:
LOW="${LOW:-0.50}"
HIGH="${HIGH:-0.80}"

SEARCH_ROOT="${RUNS_ROOT}/server-qps-binary-search"
GENERATED_CONFIG_DIR="${CONFIG_DIR}/generated"

RESULT_CSV="${SEARCH_ROOT}/server-qps-binary-search-results.csv"

mkdir -p "${SEARCH_ROOT}"
mkdir -p "${GENERATED_CONFIG_DIR}"

check_file() {
    local path="$1"
    local label="$2"

    if [[ ! -f "${path}" ]]; then
        echo "ERROR: ${label} not found:"
        echo "  ${path}"
        exit 2
    fi
}

check_dir() {
    local path="$1"
    local label="$2"

    if [[ ! -d "${path}" ]]; then
        echo "ERROR: ${label} not found:"
        echo "  ${path}"
        exit 2
    fi
}

check_file "${CONDA_SH}" "Conda initialization script"
check_dir  "${ENV_DIR}" "Conda environment"
check_dir  "${BENCH}" "MLCommons benchmark directory"
check_dir  "${MODEL_PATH}" "Model directory"
check_file "${DATASET_PATH}" "Dataset"
check_file "${SERVER_MAIN}" "KISTI Server main"
check_file "${PATCH_DIR}/SUT_VLLM_serverfix.py" "KISTI Server SUT patch"

# Avoid collision with another Server benchmark.
if pgrep -u "${USER}" -f "main_serverfix.py.*--scenario Server" >/dev/null 2>&1; then
    echo "ERROR: another Server benchmark is already running."
    pgrep -a -u "${USER}" -f "main_serverfix.py.*--scenario Server" || true
    exit 2
fi

source "${CONDA_SH}"
conda activate "${ENV_DIR}"

export PYTHONPATH="${PATCH_DIR}:${BENCH}:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

echo "============================================================"
echo "MLPerf Inference - Server QPS Binary Search"
echo "============================================================"
echo "HOST          = $(hostname)"
echo "START         = $(date)"
echo "BENCH         = ${BENCH}"
echo "MODEL_PATH    = ${MODEL_PATH}"
echo "DATASET       = ${DATASET_PATH}"
echo "SERVER_MAIN   = ${SERVER_MAIN}"
echo "QUERY_COUNT   = ${QUERY_COUNT}"
echo "ITERATIONS    = ${ITERATIONS}"
echo "VALID LOW     = ${LOW}"
echo "INVALID HIGH  = ${HIGH}"
echo "SEARCH_ROOT   = ${SEARCH_ROOT}"
echo "============================================================"

echo \
"iteration,target_qps,result,completed_samples_per_sec,ttft_mean_ns,ttft_p99_ns,tpot_mean_ns,return_code" \
> "${RESULT_CSV}"

for ITER in $(seq 1 "${ITERATIONS}"); do

    QPS=$(awk -v lo="${LOW}" -v hi="${HIGH}" \
        "BEGIN { printf \"%.5f\", (lo + hi) / 2.0 }")

    TAG=$(echo "${QPS}" | sed "s/\\./p/g")

    CONF="${GENERATED_CONFIG_DIR}/server-bsearch-qps${TAG}-${QUERY_COUNT}q.conf"
    RUN_ROOT="${SEARCH_ROOT}/server-bsearch-qps${TAG}-${QUERY_COUNT}q"
    RUN_LOG="${SEARCH_ROOT}/server-bsearch-qps${TAG}-${QUERY_COUNT}q.log"

    cat > "${CONF}" <<CONFEOF
*.Server.target_qps = ${QPS}
*.Server.min_duration = 120000
*.Server.max_duration = 2400000
*.Server.min_query_count = ${QUERY_COUNT}
*.Server.max_query_count = ${QUERY_COUNT}
CONFEOF

    rm -rf "${RUN_ROOT}"
    mkdir -p "${RUN_ROOT}"

    echo
    echo "============================================================"
    echo "ITERATION ${ITER}/${ITERATIONS}"
    echo "LOW  (VALID)   = ${LOW}"
    echo "HIGH (INVALID) = ${HIGH}"
    echo "TEST QPS       = ${QPS}"
    echo "CONFIG         = ${CONF}"
    echo "RUN_ROOT       = ${RUN_ROOT}"
    echo "START          = $(date)"
    echo "============================================================"

    set +e

    (
        cd "${BENCH}"

        python -u "${SERVER_MAIN}" \
          --scenario Server \
          --model-path "${MODEL_PATH}" \
          --batch-size 4 \
          --dtype bfloat16 \
          --user-conf "${CONF}" \
          --total-sample-count 13368 \
          --dataset-path "${DATASET_PATH}" \
          --output-log-dir "${RUN_ROOT}" \
          --tensor-parallel-size 2 \
          --vllm
    ) > "${RUN_LOG}" 2>&1

    RC=$?

    set -e

    SUMMARY="${RUN_ROOT}/mlperf_log_summary.txt"

    if [[ -f "${SUMMARY}" ]] && grep -q "Result is : VALID" "${SUMMARY}"; then
        RESULT="VALID"
        LOW="${QPS}"
    else
        RESULT="INVALID"
        HIGH="${QPS}"
    fi

    COMPLETED=""
    TTFT_MEAN=""
    TTFT_P99=""
    TPOT_MEAN=""

    if [[ -f "${SUMMARY}" ]]; then

        COMPLETED=$(awk -F: \
          "/Completed samples per second/ {
             gsub(/[[:space:]]/, \"\", \$2);
             print \$2;
             exit
           }" "${SUMMARY}")

        TTFT_MEAN=$(awk -F: \
          "/Mean First Token latency/ {
             gsub(/[[:space:]]/, \"\", \$2);
             print \$2;
             exit
           }" "${SUMMARY}")

        TTFT_P99=$(awk -F: \
          "/99.00 percentile first token latency/ {
             gsub(/[[:space:]]/, \"\", \$2);
             print \$2;
             exit
           }" "${SUMMARY}")

        TPOT_MEAN=$(awk -F: \
          "/Mean Time per Output Token/ {
             gsub(/[[:space:]]/, \"\", \$2);
             print \$2;
             exit
           }" "${SUMMARY}")

    fi

    echo \
"${ITER},${QPS},${RESULT},${COMPLETED},${TTFT_MEAN},${TTFT_P99},${TPOT_MEAN},${RC}" \
      >> "${RESULT_CSV}"

    echo
    echo "RESULT"
    echo "  QPS                 = ${QPS}"
    echo "  MLPerf result       = ${RESULT}"
    echo "  Completed samples/s = ${COMPLETED:-N/A}"
    echo "  TTFT mean (ns)      = ${TTFT_MEAN:-N/A}"
    echo "  TTFT p99  (ns)      = ${TTFT_P99:-N/A}"
    echo "  TPOT mean (ns)      = ${TPOT_MEAN:-N/A}"
    echo "  RETURN CODE         = ${RC}"
    echo "  LOG                 = ${RUN_LOG}"

    echo
    echo "NEW SEARCH RANGE"
    echo "  VALID   <= ${LOW}"
    echo "  INVALID >= ${HIGH}"

done

echo
echo "============================================================"
echo "BINARY SEARCH COMPLETE"
echo "============================================================"
echo "END: $(date)"
echo
echo "Estimated sustainable QPS boundary:"
echo "  highest tested VALID bound : ${LOW}"
echo "  lowest tested INVALID bound: ${HIGH}"
echo
echo "RESULT TABLE:"
cat "${RESULT_CSV}"
