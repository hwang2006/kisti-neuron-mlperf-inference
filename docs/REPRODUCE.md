# Reproducing MLPerf Inference on KISTI Neuron with 2 x NVIDIA H200

This document describes how to reproduce the MLPerf Inference experiments
reported in this repository on the KISTI Neuron GPU cluster using two
NVIDIA H200 GPUs.

The target workload is:

- MLPerf Inference
- Llama 3.1 8B Instruct
- Datacenter category
- Offline scenario
- Server scenario
- 2 x NVIDIA H200
- Tensor Parallelism = 2
- vLLM reference backend

> **Important**
>
> The experiments documented here are reference/reproducibility measurements.
> They are not official MLCommons-published MLPerf results.

---

## 1. Target Directory Layout

The recommended workspace is:

```text
/scratch/$USER/mlperf-inference-llama31/
|
|-- mlcommons-inference/
|-- kisti-neuron-mlperf-inference/
|-- models/
|-- data/
|-- envs/
|-- runs/
`-- logs/
```

Set the root directory:

```bash
export INFER_ROOT=/scratch/$USER/mlperf-inference-llama31

mkdir -p \
  "${INFER_ROOT}/models" \
  "${INFER_ROOT}/data" \
  "${INFER_ROOT}/envs" \
  "${INFER_ROOT}/runs" \
  "${INFER_ROOT}/logs"

cd "${INFER_ROOT}"
```

---

## 2. Obtain a 2 x H200 Allocation

The experiments were performed on a KISTI Neuron H200 compute node.

The currently relevant H200 Slurm partition is:

```text
amd_h200nv_8
```

Request two H200 GPUs according to the current Neuron Slurm policy.

Because Neuron partition and allocation policies may change, verify the
current configuration before submitting a job:

```bash
sinfo
```

After entering the allocated compute node, verify:

```bash
hostname
nvidia-smi
```

The benchmark must be executed on the allocated GPU compute node, not on a
login node.

The expected GPU configuration is:

```text
2 x NVIDIA H200
```

---

## 3. Clone This Repository

```bash
cd "${INFER_ROOT}"

git clone \
  https://github.com/hwang2006/kisti-neuron-mlperf-inference.git

cd kisti-neuron-mlperf-inference
```

---

## 4. Clone the MLCommons Reference Repository

Clone the upstream MLCommons Inference repository separately.

```bash
cd "${INFER_ROOT}"

git clone https://github.com/mlcommons/inference.git mlcommons-inference

cd mlcommons-inference

git checkout v6.0.0pre
```

Verify the exact revision:

```bash
git rev-parse HEAD
```

The revision used in the original KISTI experiment was:

```text
7f42a83e543660fd4699f1f85a05ef06b4dc334a
```

For strict reproduction, verify that the checked-out revision matches.

---

## 5. Verify the Llama 3.1 8B Reference Benchmark

Verify that the Llama 3.1 8B benchmark directory exists:

```bash
ls -ld \
  "${INFER_ROOT}/mlcommons-inference/language/llama3.1-8b"
```

Move to the benchmark directory:

```bash
cd "${INFER_ROOT}/mlcommons-inference/language/llama3.1-8b"
```

List the files:

```bash
ls -al
```

The directory should contain the MLCommons Llama 3.1 8B reference
implementation. Important files include:

```text
main.py
SUT_VLLM.py
dataset.py
evaluation.py
user.conf
requirements.txt
```

Verify these files explicitly:

```bash
for f in \
  main.py \
  SUT_VLLM.py \
  dataset.py \
  evaluation.py \
  user.conf \
  requirements.txt
do
    if [ -f "$f" ]; then
        echo "OK: $f"
    else
        echo "MISSING: $f"
    fi
done
```

Expected output:

```text
OK: main.py
OK: SUT_VLLM.py
OK: dataset.py
OK: evaluation.py
OK: user.conf
OK: requirements.txt
```

If any file is reported as `MISSING`, verify that the MLCommons Inference
repository was cloned successfully and that the correct reference version is
checked out.

Verify the MLCommons tag and exact Git revision:

```bash
cd "${INFER_ROOT}/mlcommons-inference"

git describe --tags --exact-match 2>/dev/null || true
git rev-parse HEAD
```

For this reproduction, the expected version is:

```text
v6.0.0pre
7f42a83e543660fd4699f1f85a05ef06b4dc334a
```

Return to the Llama 3.1 8B benchmark directory before continuing:

```bash
cd "${INFER_ROOT}/mlcommons-inference/language/llama3.1-8b"
```

---

---

## 6. Create the Python Environment

The original experiment used:

```text
Python       3.10.20
PyTorch      2.4.0+cu121
Transformers 4.46.2
vLLM         0.6.3
LoadGen      6.0.2
```

Initialize Conda.

On the original Neuron environment, Miniconda was installed under:

```text
/scratch/$USER/miniconda3
```

If your Conda installation is elsewhere, adjust the path accordingly.

```bash
source /scratch/$USER/miniconda3/etc/profile.d/conda.sh
```

Create a new Conda environment for this reproduction:

```bash
conda create -y \
  -p "${INFER_ROOT}/envs/mlperf-inference-v6" \
  python=3.10
```

Activate it:

```bash
conda activate "${INFER_ROOT}/envs/mlperf-inference-v6"
```

Verify that the expected environment is active:

```bash
which python
python --version
```

The Python executable should be located under:

```text
${INFER_ROOT}/envs/mlperf-inference-v6/
```

Upgrade the basic Python packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install the Llama 3.1 8B reference requirements:

```bash
cd "${INFER_ROOT}/mlcommons-inference/language/llama3.1-8b"

python -m pip install -r requirements.txt
```

After installation, verify the main software versions:

```bash
python - << "PY"
import sys
import torch
import transformers
import vllm

print("Python       :", sys.version.split()[0])
print("PyTorch      :", torch.__version__)
print("CUDA build   :", torch.version.cuda)
print("Transformers :", transformers.__version__)
print("vLLM         :", getattr(vllm, "__version__", "unknown"))
PY
```

In the current reproduction on KISTI Neuron, the environment was:

```text
Python       3.10.21
PyTorch      2.4.0+cu121
CUDA build   12.1
Transformers 4.46.2
vLLM         dev
```

The Python version differs slightly from the original experiment
(`3.10.20` versus `3.10.21`), but the PyTorch and Transformers versions
match the original environment.

When importing vLLM, the following warning may appear:

```text
RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
```

In the current reproduction, vLLM therefore reports its version as:

```text
dev
```

This warning was also observed in the original benchmark environment and
did not prevent benchmark execution.

You can inspect the current pip configuration with:

```bash
python -m pip config list
```

On KISTI Neuron, the system-wide pip configuration may include the NVIDIA
Python package index:

```text
global.extra-index-url='https://pypi.ngc.nvidia.com'
global.trusted-host='pypi.ngc.nvidia.com'
```

If DNS resolution for `pypi.ngc.nvidia.com` is unavailable from a Neuron
login node, pip may repeatedly print warnings such as:

```text
NameResolutionError:
Failed to resolve 'pypi.ngc.nvidia.com'
```

For example:

```text
WARNING: Retrying ... after connection broken by
'NameResolutionError(...)': /transformers/
```

This does not necessarily indicate an installation failure.

If pip subsequently downloads the required packages from:

```text
https://pypi.org/simple
```

and the installation completes successfully, the NVIDIA index DNS warnings
can be ignored.

The important verification is that the required package versions can be
imported successfully after installation.

Before continuing, verify again that the Conda environment is active:

```bash
echo "${CONDA_PREFIX}"
python --version
```

Expected Conda prefix:

```text
${INFER_ROOT}/envs/mlperf-inference-v6
```

---

---

## 7. Install MLCommons LoadGen

Install MLCommons LoadGen from the same checked-out MLCommons Inference
repository used for this reproduction.

Move to the MLCommons repository root:

```bash
cd "${INFER_ROOT}/mlcommons-inference"
```

Install LoadGen in editable mode:

```bash
python -m pip install -e loadgen
```

A successful installation should end with output similar to:

```text
Successfully built mlcommons_loadgen
Successfully installed mlcommons_loadgen-6.0.2
```

Verify that LoadGen can be imported:

```bash
python - << "PY"
import mlperf_loadgen as lg

print("MLPerf LoadGen import: OK")
PY
```

Expected output:

```text
MLPerf LoadGen import: OK
```

You can also verify the installed package version:

```bash
python -m pip show mlcommons-loadgen
```

For this reproduction, the expected version is:

```text
Version: 6.0.2
```

On KISTI Neuron, pip may display DNS warnings for:

```text
https://pypi.ngc.nvidia.com
```

If the installation subsequently continues and ends with:

```text
Successfully installed mlcommons_loadgen-6.0.2
```

the NVIDIA PyPI DNS warnings can be ignored.

Before continuing, verify that the intended Conda environment is still active:

```bash
echo "${CONDA_PREFIX}"
python --version
```

Expected Conda prefix:

```text
${INFER_ROOT}/envs/mlperf-inference-v6
```

---

---

## 8. Verify the Software Stack

Verify the main software components in the active Conda environment:

```bash
python - << "PY"
import sys
import torch
import transformers
import vllm
import mlperf_loadgen

print("Python       :", sys.version.split()[0])
print("PyTorch      :", torch.__version__)
print("CUDA build   :", torch.version.cuda)
print("CUDA avail.  :", torch.cuda.is_available())
print("Transformers :", transformers.__version__)
print("vLLM         :", getattr(vllm, "__version__", "unknown"))
print("LoadGen      : import OK")
PY
```

In the current KISTI Neuron reproduction environment, the result was:

```text
Python       : 3.10.21
PyTorch      : 2.4.0+cu121
CUDA build   : 12.1
CUDA avail.  : True
Transformers : 4.46.2
vLLM         : dev
LoadGen      : import OK
```

The original experiment used:

```text
Python       3.10.20
PyTorch      2.4.0+cu121
Transformers 4.46.2
vLLM         0.6.3
LoadGen      6.0.2
```

The reproduced environment therefore matches the original PyTorch,
Transformers, and LoadGen software stack. The Python version differs only
at the patch level (`3.10.20` versus `3.10.21`).

Verify the installed LoadGen package version separately:

```bash
python -m pip show mlcommons-loadgen
```

Expected version:

```text
Version: 6.0.2
```

When importing vLLM, the following warning may appear:

```text
RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
```

In this case, vLLM may report:

```text
vLLM : dev
```

This warning was observed in the tested environment and did not prevent
benchmark execution.

The most important GPU check is:

```text
CUDA avail. : True
```

If this reports `False`, do not continue with the benchmark. Verify that the
command is being executed on a GPU compute node with an active GPU
allocation.

You can also verify the visible GPUs with:

```bash
nvidia-smi
```

For the target benchmark run, two NVIDIA H200 GPUs should be visible.

---

## 9. Download Llama 3.1 8B Instruct

The model used in the experiment was:

```text
meta-llama/Llama-3.1-8B-Instruct
```

Exact Hugging Face revision:

```text
0e9e39f249a16976918f6564b8830bc894c89659
```

Access to the Meta Llama model on Hugging Face is gated.

Authenticate with Hugging Face using your own account credentials.

Do not store Hugging Face tokens in this Git repository.

After authentication, download the exact revision:

```bash
export MODEL_PATH="${INFER_ROOT}/models/Llama-3.1-8B-Instruct"

python - << "PY"
import os
from huggingface_hub import snapshot_download

repo_id = "meta-llama/Llama-3.1-8B-Instruct"
revision = "0e9e39f249a16976918f6564b8830bc894c89659"
local_dir = os.environ["MODEL_PATH"]

snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=local_dir,
)

print("Model downloaded to:", local_dir)
PY
```

A successful download should finish with output similar to:

```text
Fetching 17 files: 100%
Model downloaded to: /scratch/$USER/mlperf-inference-llama31/models/Llama-3.1-8B-Instruct
```

Verify that the model directory exists and inspect its contents:

```bash
echo "=== MODEL DIRECTORY ==="
ls -lh "${MODEL_PATH}"

echo
echo "=== MODEL FILE TREE ==="
find "${MODEL_PATH}" -maxdepth 2 -type f | sort
```

The downloaded snapshot should include the Hugging Face Transformers model
configuration, tokenizer files, and sharded safetensors weights.

Important files include:

```text
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
model.safetensors.index.json
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
```

The snapshot may also contain the original Meta model files under:

```text
original/
```

For example:

```text
original/consolidated.00.pth
original/params.json
original/tokenizer.model
```

The MLPerf/vLLM reproduction in this repository uses the Hugging Face
Transformers-compatible files at the top level of `${MODEL_PATH}`.

Verify the required Transformers/vLLM files explicitly:

```bash
echo "=== TRANSFORMERS FILE CHECK ==="

for f in \
  config.json \
  tokenizer.json \
  tokenizer_config.json \
  model.safetensors.index.json
do
    if [ -f "${MODEL_PATH}/$f" ]; then
        echo "OK: $f"
    else
        echo "MISSING: $f"
    fi
done
```

Expected output:

```text
OK: config.json
OK: tokenizer.json
OK: tokenizer_config.json
OK: model.safetensors.index.json
```

Verify that all four safetensors weight shards are present:

```bash
echo "=== MODEL WEIGHT SHARDS ==="

ls -lh "${MODEL_PATH}"/model-*.safetensors
```

Expected files:

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
```

You can also verify that Transformers can read the model configuration
without loading the full model into GPU memory:

```bash
python - << "PY"
import os
from transformers import AutoConfig, AutoTokenizer

model_path = os.environ["MODEL_PATH"]

config = AutoConfig.from_pretrained(
    model_path,
    local_files_only=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)

print("Model type       :", config.model_type)
print("Architecture     :", config.architectures)
print("Hidden size      :", config.hidden_size)
print("Number of layers :", config.num_hidden_layers)
print("Tokenizer class  :", tokenizer.__class__.__name__)
print("Vocabulary size  :", len(tokenizer))
print("Model files      : OK")
PY
```

This validation intentionally loads only the model configuration and tokenizer.
It does not load the 8B model weights into GPU memory.

If the configuration and tokenizer load successfully and all required
safetensors files are present, the model preparation step is complete.

---

## 10. Download the MLCommons CNN/DailyMail Dataset

Create the dataset directory and define the dataset path:

```bash
export INFER_ROOT=/scratch/$USER/mlperf-inference-llama31

mkdir -p "${INFER_ROOT}/data/cnn-dailymail"

export DATASET_PATH="${INFER_ROOT}/data/cnn-dailymail/cnn_eval.json"
```

The MLCommons metadata URL for the Llama 3.1 8B CNN/DailyMail evaluation
dataset is:

```text
https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri
```

This `.uri` file is not the JSON dataset itself. It contains a pointer to
the actual dataset location.

For example:

```bash
curl -fL \
  https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri
```

should return:

```text
https://inference.mlcommons-storage.org/llama3.1_8b/datasets
```

Do not download this `.uri` file directly as `cnn_eval.json`.

For example, the following command is incorrect:

```bash
curl -L \
  https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri \
  -o "${DATASET_PATH}"
```

This only creates a very small text file containing the dataset URL rather
than the actual JSON dataset.

### Download the MLCommons R2 Downloader

On KISTI Neuron, use a user-specific temporary directory:

```bash
export TMPDIR=/tmp/$USER

mkdir -p "${TMPDIR}"
```

Download the MLCommons R2 downloader script:

```bash
curl -fL \
  https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh \
  -o "${TMPDIR}/mlc-r2-downloader.sh"
```

Verify the downloader script:

```bash
echo "=== DOWNLOADER FILE ==="
ls -lh "${TMPDIR}/mlc-r2-downloader.sh"

echo
echo "=== DOWNLOADER LINE COUNT ==="
wc -l "${TMPDIR}/mlc-r2-downloader.sh"

echo
echo "=== DOWNLOADER HEAD ==="
head -20 "${TMPDIR}/mlc-r2-downloader.sh"
```

In the tested KISTI Neuron environment, the downloader script was
approximately:

```text
34 KB
956 lines
```

The first line should be:

```text
#!/bin/bash
```

### Download the Actual Dataset

Create the destination directory if it does not already exist:

```bash
mkdir -p "${INFER_ROOT}/data/cnn-dailymail"
```

Run the MLCommons downloader:

```bash
/bin/bash "${TMPDIR}/mlc-r2-downloader.sh" \
  -d "${INFER_ROOT}/data/cnn-dailymail" \
  https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri
```

After the download completes, set the dataset path again:

```bash
export DATASET_PATH="${INFER_ROOT}/data/cnn-dailymail/cnn_eval.json"
```

### Verify the Downloaded Dataset

Inspect the dataset directory:

```bash
echo "=== DATASET DIRECTORY ==="

ls -alh "${INFER_ROOT}/data/cnn-dailymail"
```

Verify that `cnn_eval.json` exists:

```bash
echo
echo "=== DATASET FILE ==="

ls -lh "${DATASET_PATH}"
```

Do not continue if the file is only a few bytes in size.

For example, a file of approximately:

```text
61 bytes
```

indicates that the `.uri` metadata file was saved instead of the actual
dataset.

Verify that the file contains valid JSON and check the number of samples:

```bash
python - << "PY"
import json
import os

path = os.environ["DATASET_PATH"]

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Dataset path :", path)
print("Samples      :", len(data))
print("First keys   :", list(data[0].keys()))
PY
```

Expected sample count:

```text
Samples      : 13368
```

Typical sample keys include:

```text
input
instruction
output
tok_input
```

You can also inspect the beginning of the JSON file without printing the
entire dataset:

```bash
head -c 300 "${DATASET_PATH}"
echo
```

The output should contain JSON content rather than a single URL.

If JSON parsing succeeds and the dataset contains 13,368 samples, the
dataset preparation step is complete.

---

## 11. Configure the Reproduction Environment

Return to the KISTI Neuron MLPerf Inference repository:

```bash
export INFER_ROOT=/scratch/$USER/mlperf-inference-llama31

cd "${INFER_ROOT}/kisti-neuron-mlperf-inference"
```

Set the main paths:

```bash
export MODEL_PATH="${INFER_ROOT}/models/Llama-3.1-8B-Instruct"
export DATASET_PATH="${INFER_ROOT}/data/cnn-dailymail/cnn_eval.json"
export CONDA_SH=/scratch/$USER/miniconda3/etc/profile.d/conda.sh
```

Load the repository environment:

```bash
source scripts/common_env.sh
```

Verify the effective configuration:

```bash
echo "INFER_ROOT   = ${INFER_ROOT}"
echo "BENCH        = ${BENCH}"
echo "ENV_DIR      = ${ENV_DIR}"
echo "MODEL_PATH   = ${MODEL_PATH}"
echo "DATASET_PATH = ${DATASET_PATH}"
echo "RUNS_ROOT    = ${RUNS_ROOT}"
```

Verify that the important paths exist:

```bash
ls -ld "${BENCH}"
ls -ld "${MODEL_PATH}"
ls -lh "${DATASET_PATH}"
ls -ld "${ENV_DIR}"
```

If the Conda environment is not active:

```bash
source "${CONDA_SH}"
conda activate "${ENV_DIR}"
```

Verify:

```bash
which python
python --version
```

> When entering a new compute node, set `INFER_ROOT` again and run
> `source scripts/common_env.sh` before starting the benchmark.

---

## 12. Set the vLLM Multiprocessing Method

For the 2-GPU vLLM run on KISTI Neuron, use the `spawn` multiprocessing
method:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

The repository already sets this value in:

```text
scripts/common_env.sh
```

Verify it after loading the common environment:

```bash
source scripts/common_env.sh

echo "${VLLM_WORKER_MULTIPROC_METHOD}"
```

Expected output:

```text
spawn
```

This setting is required because the default fork-based multiprocessing path
can fail when CUDA is re-initialized in a child process.

A typical error is:

```text
RuntimeError: Cannot re-initialize CUDA in forked subprocess
```

If `spawn` is set correctly, continue to the Offline performance benchmark.

---

## 13. Run Offline Performance

Make sure you are on the allocated H200 compute node:

```bash
hostname
nvidia-smi
```

Verify that two NVIDIA H200 GPUs are visible.

Load the reproduction environment if needed:

```bash
export INFER_ROOT=/scratch/$USER/mlperf-inference-llama31

cd "${INFER_ROOT}/kisti-neuron-mlperf-inference"

source scripts/common_env.sh
```

If the Conda environment is not already active:

```bash
source "${CONDA_SH}"
conda activate "${ENV_DIR}"
```

Verify the main paths before starting the benchmark:

```bash
echo "MODEL_PATH   = ${MODEL_PATH}"
echo "DATASET_PATH = ${DATASET_PATH}"
echo "BENCH        = ${BENCH}"
```

Run the Offline performance benchmark:

```bash
./scripts/run_offline_performance_2xh200.sh
```

The benchmark configuration is:

```text
Scenario         Offline
GPUs             2 x NVIDIA H200
Tensor Parallel  2
Batch Size       16
Precision        BF16
Samples          13,368
```

The original measured result was:

```text
Samples per second : 15.4041
Tokens per second  : 1971.72
Result              : VALID
```

The reproduced result may vary slightly because of GPU state, system load,
filesystem activity, and runtime variation.

The important validation point is that the benchmark completes successfully
and MLPerf LoadGen reports:

```text
Result is : VALID
```

---

## 14. Check Offline Performance Results

The Offline performance results are written under:

```text
${INFER_ROOT}/runs/offline-performance-2xh200
```

Check the result directory:

```bash
ls -alh "${INFER_ROOT}/runs/offline-performance-2xh200"
```

Inspect the MLPerf LoadGen summary:

```bash
cat \
  "${INFER_ROOT}/runs/offline-performance-2xh200/mlperf_log_summary.txt"
```

For a shorter check:

```bash
grep -E \
  "Samples per second|Tokens per second|Result is" \
  "${INFER_ROOT}/runs/offline-performance-2xh200/mlperf_log_summary.txt"
```

The original KISTI 2 x H200 result was:

```text
Samples per second : 15.4041
Tokens per second  : 1971.72
Result is : VALID
```

The exact throughput may vary slightly between runs.

The most important check is:

```text
Result is : VALID
```

If LoadGen reports `VALID`, continue to the Offline accuracy test.

---

## 15. Run Offline Accuracy

Make sure the reproduction environment is loaded:

```bash
export INFER_ROOT=/scratch/$USER/mlperf-inference-llama31

cd "${INFER_ROOT}/kisti-neuron-mlperf-inference"

source scripts/common_env.sh
```

If necessary, activate the Conda environment:

```bash
source "${CONDA_SH}"
conda activate "${ENV_DIR}"
```

Run the Offline accuracy benchmark:

```bash
./scripts/run_offline_accuracy_2xh200.sh
```

The script performs the MLPerf accuracy run and then evaluates the generated
outputs using the MLCommons `evaluation.py` script.

The evaluation command uses:

```text
--dtype int32
```

This is intentional. In the tested environment, using the default int64
decoding path caused an integer overflow during tokenizer decoding.

---

## 16. Check Offline Accuracy Results

The accuracy results are written under:

```text
${INFER_ROOT}/runs/offline-accuracy-2xh200
```

Inspect the result directory:

```bash
ls -alh "${INFER_ROOT}/runs/offline-accuracy-2xh200"
```

Check the evaluation output:

```bash
cat \
  "${INFER_ROOT}/runs/offline-accuracy-2xh200/accuracy-evaluation-int32.txt"
```

The original KISTI result was:

```text
ROUGE-1    : 38.8414
ROUGE-2    : 15.9717
ROUGE-L    : 24.5673
ROUGE-Lsum : 35.8933
gen_len    : 8162948
gen_num    : 13368
```

The important checks are that all 13,368 samples are evaluated and that the
ROUGE values are close to the original result.

---

## 17. Understand the Server Patch

The upstream MLCommons Server implementation did not operate reliably with
the tested vLLM asynchronous execution path on KISTI Neuron.

The original MLCommons files remain unchanged.

KISTI-specific Server copies are provided under:

```text
patches/
├── SUT_VLLM_serverfix.py
├── main_serverfix.py
└── README.md
```

The main change is the use of a persistent asyncio event loop for the Server
worker instead of repeatedly creating and destroying an event loop for each
request.

The repository scripts load the patched implementation through `PYTHONPATH`.

For details:

```bash
cat patches/README.md
```

---

## 18. Run a Known-Valid Server Point

Start with the known-valid Server configuration:

```text
Target QPS : 0.50
Queries    : 800
```

Load the reproduction environment:

```bash
export INFER_ROOT=/scratch/$USER/mlperf-inference-llama31

cd "${INFER_ROOT}/kisti-neuron-mlperf-inference"

source scripts/common_env.sh
```

If necessary:

```bash
source "${CONDA_SH}"
conda activate "${ENV_DIR}"
```

Run:

```bash
./scripts/run_server_qps05_800q.sh
```

The original 2 x H200 experiment produced:

```text
Result is : VALID
```

at target QPS 0.50.

---

## 19. Inspect the Server Result

After the run completes, locate the generated result directory:

```bash
find "${RUNS_ROOT}" \
  -name mlperf_log_summary.txt \
  -type f \
  -printf "%T@ %p\n" \
  | sort -n \
  | tail
```

Inspect the relevant summary:

```bash
cat /path/to/mlperf_log_summary.txt
```

Useful metrics include:

```text
Result is
Completed samples per second
Mean First Token latency
99.00 percentile first token latency
Mean Time per Output Token
```

A successful QPS 0.50 reproduction should report:

```text
Result is : VALID
```

---

## 20. Run QPS = 0.546875

The original experiment found that QPS 0.546875 was close to the
VALID/INVALID transition.

Run:

```bash
cd "${INFER_ROOT}/kisti-neuron-mlperf-inference"

./scripts/run_server_qps0546875_800q.sh
```

The original result was:

```text
INVALID
```

but it was close to satisfying the required latency behavior.

Because Server results near the boundary are sensitive to latency outliers,
repeated runs may differ slightly.

---

## 21. Run a Higher-QPS Test

To observe behavior above the sustainable region, run:

```bash
./scripts/run_server_qps08_500q.sh
```

The original result at target QPS 0.8 was:

```text
INVALID
```

At higher request rates, TTFT increased significantly while TPOT remained
relatively stable.

---

## 22. Run an Extreme Overload Test

Run:

```bash
./scripts/run_server_extreme_qps15_500q.sh
```

The original experiment at target QPS 1.5 showed approximately:

```text
Completed samples/s : 1.45
TTFT mean           : 7.23 s
TTFT p99            : 17.88 s
TTFT maximum        : 18.90 s
TPOT mean           : 5.10 ms
TPOT p99            : 5.34 ms
Result              : INVALID
```

The large increase in TTFT indicates that queueing and scheduling delay
dominate under overload.

---

## 23. Run Automated Server QPS Binary Search

The repository includes:

```text
scripts/run_server_qps_binary_search.sh
```

The default search parameters are:

```text
LOW         = 0.50
HIGH        = 0.80
QUERY_COUNT = 800
ITERATIONS  = 4
```

Run:

```bash
cd "${INFER_ROOT}/kisti-neuron-mlperf-inference"

./scripts/run_server_qps_binary_search.sh
```

The parameters can also be overridden:

```bash
LOW=0.50 \
HIGH=0.80 \
QUERY_COUNT=800 \
ITERATIONS=4 \
./scripts/run_server_qps_binary_search.sh
```

Generated configurations are written under:

```text
configs/generated/
```

Binary-search results are written under:

```text
${RUNS_ROOT}/server-qps-binary-search/
```

---

## 24. Original Server QPS Characterization

The original KISTI 2 x H200 measurements were:

```text
Target QPS   Result
----------   -------
0.45         VALID
0.50         VALID
0.53750      VALID
0.546875     INVALID, borderline
0.55625      INVALID
0.57500      INVALID
0.65000      INVALID
0.80000      INVALID
1.50000      INVALID
```

The highest tested VALID point was:

```text
0.53750
```

and the closest tested INVALID point was:

```text
0.546875
```

The observed transition region was therefore approximately:

```text
0.5375 < sustainable QPS < 0.546875
```

This should be interpreted as an experimentally observed region rather than
an exact architectural limit.

---

## 25. Interpret Server Latency

Three metrics are particularly useful when analyzing the Server scenario:

```text
QPS   Queries Per Second
TTFT  Time To First Token
TPOT  Time Per Output Token
```

In the original experiment, TPOT remained close to approximately 5 ms over
a relatively wide QPS range.

In contrast, TTFT increased sharply when the request rate exceeded the
sustainable Server region.

This indicates that request queueing and scheduling delay were the dominant
overload effects rather than a major slowdown in token generation itself.

---

## 26. Expected Run-to-Run Variation

Exact numerical reproduction is not expected.

Results can vary because of GPU clock state, temperature, CPU scheduling,
filesystem activity, background load, request timing, and vLLM runtime
behavior.

Server measurements near the VALID/INVALID boundary are particularly
sensitive to latency outliers.

Therefore, results such as:

```text
0.53750  VALID
0.546875 INVALID
```

should be interpreted as a measured transition region rather than a
deterministic threshold.

---

## 27. Check for Leftover GPU Processes

Before starting another benchmark, check for existing GPU processes:

```bash
nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv
```

An orphaned vLLM worker may retain a large amount of GPU memory and cause
the next run to fail with an out-of-memory error.

Verify process ownership before terminating any process.

---

## 28. Troubleshooting

### Conda environment is not active

If Python reports errors such as:

```text
ModuleNotFoundError: No module named 'torch'
```

reload the environment:

```bash
export INFER_ROOT=/scratch/$USER/mlperf-inference-llama31

cd "${INFER_ROOT}/kisti-neuron-mlperf-inference"

source scripts/common_env.sh
source "${CONDA_SH}"
conda activate "${ENV_DIR}"
```

Verify:

```bash
which python
python --version
```

### Environment variables disappear on a compute node

Environment variables set in a previous shell may not be available after
entering another node.

Set:

```bash
export INFER_ROOT=/scratch/$USER/mlperf-inference-llama31

cd "${INFER_ROOT}/kisti-neuron-mlperf-inference"

source scripts/common_env.sh
```

before running the benchmark.

### CUDA fork error

If you see:

```text
Cannot re-initialize CUDA in forked subprocess
```

verify:

```bash
echo "${VLLM_WORKER_MULTIPROC_METHOD}"
```

Expected:

```text
spawn
```

### Accuracy tokenizer overflow

If accuracy evaluation reports an integer conversion overflow, use:

```text
--dtype int32
```

The repository accuracy script already uses this setting.

### Server stalls

Use the KISTI Server implementation under:

```text
patches/
```

through the provided Server scripts rather than replacing the upstream
MLCommons files.

### Server shutdown warnings

Some Server runs may finish with messages such as:

```text
Task was destroyed but it is pending!
```

or multiprocessing shared-memory cleanup warnings.

These warnings were observed after LoadGen completed, but the asynchronous
engine cleanup should be improved before treating the implementation as
formal submission-quality code.

---

## 29. Reproduction Checklist

Before considering the benchmark reproduced, verify:

```text
[ ] Running on an allocated H200 compute node
[ ] Two NVIDIA H200 GPUs visible
[ ] MLCommons inference v6.0.0pre checked out
[ ] Reference commit 7f42a83e... verified
[ ] Conda environment active
[ ] PyTorch 2.4.0+cu121
[ ] Transformers 4.46.2
[ ] LoadGen 6.0.2
[ ] Llama 3.1 8B Instruct downloaded
[ ] Model revision 0e9e39f... used
[ ] Model config/tokenizer/safetensors verified
[ ] cnn_eval.json downloaded using MLCommons R2 downloader
[ ] Dataset MD5 verification passed
[ ] Dataset contains 13,368 samples
[ ] VLLM_WORKER_MULTIPROC_METHOD=spawn
[ ] Offline Performance completed
[ ] Offline LoadGen result checked
[ ] Offline Accuracy completed
[ ] ROUGE values checked
[ ] Server QPS 0.50 tested
[ ] Server patch used
[ ] Server summary inspected
```

---

## 30. Original KISTI Reference Results

### Offline Performance

```text
Samples/s : 15.4041
Tokens/s  : 1971.72
Result     : VALID
```

### Offline Accuracy

```text
ROUGE-1    : 38.8414
ROUGE-2    : 15.9717
ROUGE-L    : 24.5673
ROUGE-Lsum : 35.8933
Samples    : 13368
```

### Server

```text
QPS 0.50      : VALID
QPS 0.53750   : VALID
QPS 0.546875  : borderline INVALID
QPS 0.55625   : INVALID
QPS 0.80      : INVALID
QPS 1.50      : INVALID
```

---

## 31. Official MLPerf Submission Note

The experiments in this repository are intended for performance
characterization and reproducibility.

They are not official MLCommons-published MLPerf results.

A formal MLPerf submission requires additional submission metadata,
compliance tests, result organization, submission checking, and adherence
to the rules of the applicable MLPerf division.

The KISTI Server patch should also be reviewed against the requirements of
the intended submission division before being used for a formal submission.

---

## 32. References

MLCommons Inference:

```text
https://github.com/mlcommons/inference
```

KISTI Neuron MLPerf Inference:

```text
https://github.com/hwang2006/kisti-neuron-mlperf-inference
```

KISTI Neuron User Guide:

```text
https://docs-ksc.gitbook.io/neuron-user-guide/
```

Meta Llama 3.1 8B Instruct:

```text
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

---
