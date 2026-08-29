#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage:"
    echo "  $0 <config-file-name> <run-directory-name>"
    exit 2
fi

CONFIG_NAME="$1"
RUN_NAME="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

CONF="${CONFIG_DIR}/${CONFIG_NAME}"
RUN_ROOT="${RUNS_ROOT}/${RUN_NAME}"

check_path() {
    local type="$1"
    local path="$2"
    local label="$3"

    if [[ "${type}" == "file" && ! -f "${path}" ]]; then
        echo "ERROR: ${label} not found:"
        echo "  ${path}"
        exit 2
    fi

    if [[ "${type}" == "dir" && ! -d "${path}" ]]; then
        echo "ERROR: ${label} not found:"
        echo "  ${path}"
        exit 2
    fi
}

check_path file "${CONDA_SH}" "Conda initialization script"
check_path dir  "${ENV_DIR}" "Conda environment"
check_path dir  "${BENCH}" "MLCommons benchmark directory"
check_path dir  "${MODEL_PATH}" "Model directory"
check_path file "${DATASET_PATH}" "Dataset"
check_path file "${CONF}" "Server configuration"
check_path file "${SERVER_MAIN}" "KISTI Server main"
check_path file "${PATCH_DIR}/SUT_VLLM_serverfix.py" "KISTI Server SUT patch"

source "${CONDA_SH}"
conda activate "${ENV_DIR}"

export PYTHONPATH="${PATCH_DIR}:${BENCH}:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

rm -rf "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"

echo "============================================================"
echo "MLPerf Inference - Llama 3.1 8B Server"
echo "============================================================"
echo "HOST        = $(hostname)"
echo "START       = $(date)"
echo "BENCH       = ${BENCH}"
echo "MODEL_PATH  = ${MODEL_PATH}"
echo "DATASET     = ${DATASET_PATH}"
echo "CONFIG      = ${CONF}"
echo "RUN_ROOT    = ${RUN_ROOT}"
echo "SERVER_MAIN = ${SERVER_MAIN}"
echo "GPU         = 2 x NVIDIA H200"
echo "TP          = 2"
echo "BATCH       = 4"
echo "DTYPE       = bfloat16"
echo "VLLM MP     = ${VLLM_WORKER_MULTIPROC_METHOD}"
echo "============================================================"

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

RC=$?

echo
echo "============================================================"
echo "FINISHED"
echo "RETURN_CODE=${RC}"
echo "END=$(date)"
echo "============================================================"

exit "${RC}"
