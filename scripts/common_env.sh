#!/bin/bash

# ------------------------------------------------------------
# Common environment for KISTI Neuron MLPerf Inference scripts
# ------------------------------------------------------------

# Public/reproducibility workspace.
#
# Override this if your MLCommons reference tree, model, dataset,
# and environment are stored elsewhere.
export INFER_ROOT="${INFER_ROOT:-/scratch/${USER}/mlperf-inference-llama31}"

# MLCommons reference repository
export MLCOMMONS_ROOT="${MLCOMMONS_ROOT:-${INFER_ROOT}/mlcommons-inference}"
export BENCH="${BENCH:-${MLCOMMONS_ROOT}/language/llama3.1-8b}"

# Python / Conda environment
export ENV_DIR="${ENV_DIR:-${INFER_ROOT}/envs/mlperf-inference-v6}"

# Conda installation.
# KISTI Neuron users may override this if Miniconda is installed elsewhere.
export CONDA_SH="${CONDA_SH:-/scratch/${USER}/miniconda3/etc/profile.d/conda.sh}"

# Model and dataset paths.
#
# MODEL_PATH should point to the local Hugging Face model directory.
# DATASET_PATH should point to the MLCommons preprocessed CNN/DailyMail file.
export MODEL_PATH="${MODEL_PATH:-${INFER_ROOT}/models/Llama-3.1-8B-Instruct}"
export DATASET_PATH="${DATASET_PATH:-${INFER_ROOT}/data/cnn-dailymail/cnn_eval.json}"

# Result directory
export RUNS_ROOT="${RUNS_ROOT:-${INFER_ROOT}/runs}"

# vLLM CUDA multiprocessing requirement observed on KISTI Neuron.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# Public Git repository root
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Repository-local configs and KISTI Server patch
export CONFIG_DIR="${CONFIG_DIR:-${REPO_ROOT}/configs}"
export PATCH_DIR="${PATCH_DIR:-${REPO_ROOT}/patches}"
export SERVER_MAIN="${SERVER_MAIN:-${PATCH_DIR}/main_serverfix.py}"
