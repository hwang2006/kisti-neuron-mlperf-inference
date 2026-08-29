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

Verify:

```bash
ls -lh "${MODEL_PATH}"
```

The directory should contain files such as:

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors.index.json
model-00001-of-00004.safetensors
...
```

---

## 10. Download the MLCommons CNN/DailyMail Dataset

Create the dataset directory:

```bash
mkdir -p "${INFER_ROOT}/data/cnn-dailymail"

export DATASET_PATH="${INFER_ROOT}/data/cnn-dailymail/cnn_eval.json"
```

Download the preprocessed MLCommons evaluation dataset:

```bash
curl -L \
  https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri \
  -o "${DATASET_PATH}"
```

Verify that the file exists:

```bash
ls -lh "${DATASET_PATH}"
```

Verify the sample count:

```bash
python - << "PY"
import json
import os

path = os.environ["DATASET_PATH"]

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Samples:", len(data))
print("First sample keys:", list(data[0].keys()))
PY
```

Expected:

```text
Samples: 13368
```

Typical sample keys include:

```text
input
instruction
output
tok_input
```

---

## 11. Configure the Public Reproduction Scripts

Return to the KISTI repository:

```bash
cd "${INFER_ROOT}/kisti-neuron-mlperf-inference"
```

The scripts use:

```text
scripts/common_env.sh
```

By default, the scripts assume:

```text
INFER_ROOT=/scratch/$USER/mlperf-inference-llama31
```

Set explicit paths before running:

```bash
export INFER_ROOT=/scratch/$USER/mlperf-inference-llama31

export MODEL_PATH="${INFER_ROOT}/models/Llama-3.1-8B-Instruct"

export DATASET_PATH="${INFER_ROOT}/data/cnn-dailymail/cnn_eval.json"
```

If Conda is installed somewhere other than:

```text
/scratch/$USER/miniconda3
```

set:

```bash
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
```

---

## 12. Required vLLM Multiprocessing Setting

The original two-GPU run failed when CUDA was re-initialized from a
fork-based subprocess.

The following environment variable is therefore required:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

The public scripts set this automatically through `common_env.sh`.

The original failure was:

```text
RuntimeError: Cannot re-initialize CUDA in forked subprocess.
To use CUDA with multiprocessing, you must use the spawn start method.
```

---

## 13. Run Offline Performance

Make sure you are on the allocated H200 compute node:

```bash
hostname
nvidia-smi
```

Then run:

```bash
cd "${INFER_ROOT}/kisti-neuron-mlperf-inference"

./scripts/run_offline_performance_2xh200.sh
```

The benchmark configuration is:

```text
Scenario         Offline
GPUs             2 x H200
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

The reproduced value does not have to be bit-for-bit identical because of
runtime variation, GPU state, system sharing, filesystem activity, and other
environmental effects.

The important validation point is that the benchmark completes normally and
LoadGen reports a valid result under the intended configuration.

---

## 14. Check Offline Performance Results

The default result directory is:

```text
${INFER_ROOT}/runs/offline-performance-2xh200
```

Inspect:

```bash
cat \
  "${INFER_ROOT}/runs/offline-performance-2xh200/mlperf_log_summary.txt"
```

Useful checks:

```bash
grep -E \
"Samples per second|Tokens per second|Result is" \
"${INFER_ROOT}/runs/offline-performance-2xh200/mlperf_log_summary.txt"
```

Expected original result:

```text
Samples per second: 15.4041
Tokens per second: 1971.72
Result is : VALID
```

---

## 15. Run Offline Accuracy

Run:

```bash
cd "${INFER_ROOT}/kisti-neuron-mlperf-inference"

./scripts/run_offline_accuracy_2xh200.sh
```

This performs:

1. MLPerf accuracy inference
2. creation of `mlperf_log_accuracy.json`
3. ROUGE evaluation

The public script explicitly uses:

```text
--dtype int32
```

during evaluation.

This is required because the tested reference evaluation path using int64
caused:

```text
OverflowError: out of range integral type conversion attempted
```

---

## 16. Check Offline Accuracy

The result directory is:

```text
${INFER_ROOT}/runs/offline-accuracy-2xh200
```

Inspect:

```bash
cat \
  "${INFER_ROOT}/runs/offline-accuracy-2xh200/accuracy-evaluation-int32.txt"
```

Original KISTI result:

```text
ROUGE-1    38.8414
ROUGE-2    15.9717
ROUGE-L    24.5673
ROUGE-Lsum 35.8933
gen_len    8162948
gen_num    13368
```

---

## 17. Understand the Server Patch

The upstream Server reference implementation did not operate reliably with
the tested vLLM asynchronous execution path on the KISTI H200 environment.

The original upstream files remain unchanged.

KISTI-specific copies are provided under:

```text
patches/
|-- SUT_VLLM_serverfix.py
|-- main_serverfix.py
`-- README.md
```

The principal change is that the Server worker uses a persistent asyncio
event loop instead of creating and destroying an event loop for each request.

The public Server scripts use the patch through `PYTHONPATH` and do not
require overwriting the MLCommons reference files.

See:

```text
patches/README.md
```

for details.

---

## 18. Run a Known-Valid Server Point

The most straightforward Server reproduction point is:

```text
Target QPS = 0.50
Queries    = 800
```

Run:

```bash
cd "${INFER_ROOT}/kisti-neuron-mlperf-inference"

./scripts/run_server_qps05_800q.sh
```

The original experiment produced a VALID result at this point.

---

## 19. Run the Highest Tested VALID QPS Region

The original binary search found:

```text
QPS 0.53750  -> VALID
QPS 0.546875 -> INVALID, borderline
```

The observed transition region was therefore approximately:

```text
0.5375 < sustainable QPS < 0.546875
```

This boundary is statistical and should not be interpreted as an exact
hardware limit.

---

## 20. Run QPS = 0.546875

Run:

```bash
./scripts/run_server_qps0546875_800q.sh
```

The original result was:

```text
Result: INVALID
```

but it was very close to satisfying the statistical early-stopping
requirement.

The run was estimated to require only 38 additional clean queries.

---

## 21. Run Near-Saturation Test

Run:

```bash
./scripts/run_server_qps08_500q.sh
```

The original result was INVALID.

Typical behavior:

```text
TTFT p99 approached approximately 2 seconds
TPOT remained near approximately 5 ms
```

This indicates increasing request-queueing pressure.

---

## 22. Run Extreme Overload Test

Run:

```bash
./scripts/run_server_extreme_qps15_500q.sh
```

Original result:

```text
Target QPS          1.5
Result              INVALID
Completed samples/s approximately 1.45
TTFT mean           approximately 7.23 s
TTFT p99            approximately 17.88 s
TPOT mean           approximately 5.10 ms
```

This demonstrates that under excessive request arrival rates the dominant
problem is TTFT queueing rather than token-generation TPOT degradation.

---

## 23. Run Automated Server QPS Binary Search

The repository provides:

```text
scripts/run_server_qps_binary_search.sh
```

Default parameters:

```text
LOW         = 0.50
HIGH        = 0.80
QUERY_COUNT = 800
ITERATIONS  = 4
```

Run:

```bash
./scripts/run_server_qps_binary_search.sh
```

Parameters can be overridden:

```bash
LOW=0.50 \
HIGH=0.80 \
QUERY_COUNT=800 \
ITERATIONS=4 \
./scripts/run_server_qps_binary_search.sh
```

Generated configurations are placed under:

```text
configs/generated/
```

Run outputs are placed under:

```text
${INFER_ROOT}/runs/server-qps-binary-search/
```

---

## 24. Original Binary Search Result

The original H200 experiment produced approximately:

| Target QPS | Result |
|---:|---|
| 0.65000 | INVALID |
| 0.57500 | INVALID |
| 0.53750 | VALID |
| 0.55625 | INVALID |

Additional testing produced:

| Target QPS | Result |
|---:|---|
| 0.546875 | INVALID, borderline |

Therefore:

```text
highest tested VALID  = 0.53750
lowest nearby INVALID = 0.546875
```

---

## 25. Server Latency Interpretation

The Server scenario evaluates more than raw throughput.

Important metrics include:

### TTFT

```text
Time To First Token
```

Time from request arrival until the first output token is returned.

### TPOT

```text
Time Per Output Token
```

Token-generation interval after the first token.

### QPS

```text
Queries Per Second
```

Request arrival rate.

The original experiment showed that TPOT remained close to 5 ms across a
wide range of QPS values while TTFT increased dramatically under overload.

This is consistent with queueing and scheduling becoming the dominant
bottleneck.

---

## 26. Expected Differences Between Repeated Runs

Exact numerical reproduction is not guaranteed.

Small differences can result from:

- GPU clock state
- GPU temperature
- other activity on a shared node
- filesystem I/O
- CPU scheduling
- vLLM runtime behavior
- request arrival randomness
- LoadGen statistical early stopping

Server VALID/INVALID behavior near the boundary is especially sensitive to
the number and distribution of latency outliers.

For this reason:

```text
0.5375 VALID
0.546875 borderline INVALID
```

should be treated as an experimentally observed transition region rather
than an exact universal threshold.

---

## 27. Verify No Leftover GPU Processes

Before starting another vLLM benchmark, check:

```bash
nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

An orphaned vLLM worker can retain a large amount of H200 memory and cause
the next run to fail with CUDA out-of-memory errors.

Do not use a broad `pkill` unless the process ownership and PID have first
been verified.

---

## 28. Troubleshooting

### CUDA fork error

Symptom:

```text
Cannot re-initialize CUDA in forked subprocess
```

Solution:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

### Accuracy tokenizer overflow

Symptom:

```text
OverflowError: out of range integral type conversion attempted
```

Solution:

```text
--dtype int32
```

The public accuracy script already contains this setting.

### Server stalls after first request

Use the KISTI Server patch included under:

```text
patches/
```

and execute Server through the repository-provided scripts.

### Server shutdown warnings

Some Server runs may finish with warnings such as:

```text
Task was destroyed but it is pending!
```

or multiprocessing shared-memory cleanup warnings.

These were observed after successful LoadGen completion.

They indicate an incomplete asynchronous-engine cleanup path and should be
resolved before treating the implementation as formal submission-quality
code.

---

## 29. Reproduction Checklist

Before considering the experiment reproduced, verify:

```text
[ ] Running on allocated H200 compute node
[ ] 2 x NVIDIA H200 visible
[ ] MLCommons inference v6.0.0pre checked out
[ ] Reference commit verified
[ ] Python environment created
[ ] LoadGen installed
[ ] Llama 3.1 8B Instruct downloaded
[ ] Exact model revision recorded
[ ] CNN/DailyMail dataset contains 13,368 samples
[ ] VLLM_WORKER_MULTIPROC_METHOD=spawn
[ ] Offline Performance completed
[ ] Offline LoadGen result checked
[ ] Offline Accuracy completed
[ ] ROUGE values checked
[ ] Server QPS=0.50 completed
[ ] Server patch used
[ ] Server summary inspected
```

---

## 30. Original KISTI Reference Results

### Offline

```text
Samples/s : 15.4041
Tokens/s  : 1971.72
Result     : VALID
```

### Accuracy

```text
ROUGE-1    : 38.8414
ROUGE-2    : 15.9717
ROUGE-L    : 24.5673
ROUGE-Lsum : 35.8933
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

Successful reproduction of these experiments does not by itself constitute
an official MLPerf submission.

Formal submission additionally requires the applicable MLCommons:

- model requirements
- dataset requirements
- system metadata
- submitter metadata
- performance run structure
- accuracy run structure
- compliance tests
- submission checker
- reproducibility requirements
- division-specific rules

The Server patch in this repository must also be reviewed against the
requirements of the intended MLPerf submission division before it is used
for a formal submission.

---

## 32. References

MLCommons Inference:

https://github.com/mlcommons/inference

KISTI Neuron MLPerf Inference repository:

https://github.com/hwang2006/kisti-neuron-mlperf-inference

KISTI Neuron User Guide:

https://docs-ksc.gitbook.io/neuron-user-guide/

Meta Llama 3.1 8B Instruct:

https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

