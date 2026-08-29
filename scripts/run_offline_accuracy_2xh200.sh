#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

RUN_ROOT="${RUNS_ROOT}/offline-accuracy-2xh200"

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "ERROR: Conda initialization script not found:"
    echo "  ${CONDA_SH}"
    exit 2
fi

if [[ ! -d "${BENCH}" ]]; then
    echo "ERROR: MLCommons benchmark directory not found:"
    echo "  ${BENCH}"
    exit 2
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH does not exist:"
    echo "  ${MODEL_PATH}"
    echo "Set MODEL_PATH to the local Llama-3.1-8B-Instruct directory."
    exit 2
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
    echo "ERROR: DATASET_PATH does not exist:"
    echo "  ${DATASET_PATH}"
    exit 2
fi

source "${CONDA_SH}"
conda activate "${ENV_DIR}"

mkdir -p "${RUN_ROOT}"

echo "============================================================"
echo "MLPerf Inference - Llama 3.1 8B Offline Accuracy"
echo "============================================================"
echo "HOST        = $(hostname)"
echo "START       = $(date)"
echo "BENCH       = ${BENCH}"
echo "MODEL_PATH  = ${MODEL_PATH}"
echo "DATASET     = ${DATASET_PATH}"
echo "RUN_ROOT    = ${RUN_ROOT}"
echo "GPU         = 2 x NVIDIA H200"
echo "TP          = 2"
echo "BATCH       = 16"
echo "DTYPE       = bfloat16"
echo "VLLM MP     = ${VLLM_WORKER_MULTIPROC_METHOD}"
echo "============================================================"

cd "${BENCH}"

echo
echo "============================================================"
echo "1. ACCURACY GENERATION"
echo "============================================================"

python -u main.py \
  --scenario Offline \
  --model-path "${MODEL_PATH}" \
  --batch-size 16 \
  --dtype bfloat16 \
  --user-conf "${BENCH}/user.conf" \
  --total-sample-count 13368 \
  --dataset-path "${DATASET_PATH}" \
  --output-log-dir "${RUN_ROOT}" \
  --tensor-parallel-size 2 \
  --accuracy \
  --vllm

echo
echo "============================================================"
echo "2. ACCURACY EVALUATION"
echo "============================================================"

python -u evaluation.py \
  --mlperf-accuracy-file "${RUN_ROOT}/mlperf_log_accuracy.json" \
  --dataset-file "${DATASET_PATH}" \
  --model-name "${MODEL_PATH}" \
  --dtype int32 \
  2>&1 | tee "${RUN_ROOT}/accuracy-evaluation-int32.txt"

echo
echo "============================================================"
echo "3. OUTPUT FILES"
echo "============================================================"

find "${RUN_ROOT}" \
  -maxdepth 1 \
  -type f \
  -printf "%12s %f\n" \
  | sort

echo
echo "END=$(date)"
