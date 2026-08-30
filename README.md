# MLPerf Inference on KISTI Neuron GPU Cluster

This repository documents MLPerf Inference reference-benchmark experiments
performed on the KISTI Neuron GPU cluster using NVIDIA H200 GPUs.

The experiments focus on Llama 3.1 8B Instruct and cover:

- Offline performance
- Offline accuracy
- Server performance
- Server QPS tuning
- Reference implementation portability issues on the KISTI Neuron system

## Reproduction Guide

For a step-by-step guide to reproducing the complete benchmark on
2 x NVIDIA H200 GPUs on the KISTI Neuron system, see:

**[docs/REPRODUCE.md](docs/REPRODUCE.md)**

The reproduction guide covers environment setup, MLCommons reference
checkout, model and dataset preparation, Offline performance, Accuracy,
Server execution, QPS tuning, and troubleshooting.

> **Note**
>
> The results in this repository are experimental/internal benchmark results.
> They are not official MLCommons-published MLPerf results.

---

## 1. Benchmark Overview

| Item | Configuration |
|---|---|
| Benchmark | MLPerf Inference |
| Model | Llama 3.1 8B Instruct |
| Scenario | Offline, Server |
| System | KISTI Neuron GPU Cluster |
| GPU | 2 x NVIDIA H200 |
| Tensor Parallelism | TP=2 |
| Precision | BF16 |
| Backend | vLLM |
| Dataset | CNN/DailyMail |
| Dataset Samples | 13,368 |

---

## 2. MLCommons Reference Implementation

The experiments are based on the MLCommons MLPerf Inference reference
implementation.

Reference repository:

https://github.com/mlcommons/inference

Version used:

```text
Tag    : v6.0.0pre
Commit : 7f42a83e543660fd4699f1f85a05ef06b4dc334a
```

The benchmark implementation used in this study is:

```text
language/llama3.1-8b
```

---

## 3. KISTI Neuron Environment

The benchmark was executed on a KISTI Neuron compute node equipped with
NVIDIA H200 GPUs.

Typical configuration used in this study:

| Component | Configuration |
|---|---|
| GPU | 2 x NVIDIA H200 |
| GPU Memory | approximately 141 GB per GPU |
| NVIDIA Driver | 580.105.08 |
| Host CUDA | 13.0 |
| Filesystem | Lustre `/scratch` |
| Scheduler | Slurm |
| Tensor Parallelism | TP=2 |

The original benchmark working environment was placed under:

```text
/scratch/$USER/mlperf/inference
```

The public reproducibility repository is maintained separately under:

```text
/scratch/$USER/mlperf-inference-llama31/kisti-neuron-mlperf-inference
```

---

## 4. Model

The model used for the benchmark was:

```text
meta-llama/Llama-3.1-8B-Instruct
```

Hugging Face revision:

```text
0e9e39f249a16976918f6564b8830bc894c89659
```

The revision specified in older MLCommons reference instructions was no
longer available at the time of this experiment.

Therefore, the exact model artifact used in this repository may differ from
the canonical MLCommons reference artifact.

This difference is recorded explicitly for reproducibility.

---

## 5. Dataset

The official preprocessed CNN/DailyMail evaluation dataset distributed by
MLCommons was used.

```text
Number of samples : 13,368
Dataset file      : cnn_eval.json
```

The dataset was downloaded from the public MLCommons storage location and
verified before benchmark execution.

---

## 6. Software Environment

The reference implementation was executed using the following main software
components:

| Component | Version / Configuration |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.4.0+cu121 |
| Transformers | 4.46.2 |
| vLLM | 0.6.3 |
| MLCommons LoadGen | 6.0.2 |
| MLPerf Inference | v6.0.0pre |

The Python environment used during the experiment was located under:

```text
/scratch/$USER/mlperf/inference/envs/mlperf-inference-v6
```

---

## 7. Offline Performance

The final Offline performance run used:

```text
GPU Count        : 2
Tensor Parallel  : 2
Batch Size       : 16
Precision        : BF16
Samples          : 13,368
Scenario         : Offline
Mode             : PerformanceOnly
```

The original MLCommons reference `user.conf` was used for the final
performance measurement.

### Result

| Metric | Result |
|---|---:|
| Samples per second | 15.4041 |
| Tokens per second | 1,971.72 |
| LoadGen result | VALID |

The LoadGen minimum-duration and minimum-query requirements were satisfied.

---

## 8. Offline Accuracy

Accuracy inference was completed for all 13,368 evaluation samples.

The generated MLPerf accuracy log contained 13,368 entries.

### Accuracy Result

| Metric | Result |
|---|---:|
| ROUGE-1 | 38.8414 |
| ROUGE-2 | 15.9717 |
| ROUGE-L | 24.5673 |
| ROUGE-Lsum | 35.8933 |
| Generated tokens | 8,162,948 |
| Samples | 13,368 |

The MLCommons evaluation script was executed explicitly with:

```text
--dtype int32
```

because the default int64 decoding path caused an overflow error while
decoding token IDs in this environment.

The generated MLPerf accuracy log itself was not modified.

---

## 9. Server Performance

The Server scenario was evaluated on the same 2 x NVIDIA H200 configuration.

The Server workload was executed using:

```text
GPU Count        : 2
Tensor Parallel  : 2
Batch Size       : 4
Precision        : BF16
Scenario         : Server
Backend          : vLLM AsyncLLMEngine
```

A key finding during reproduction was that **CPU/NUMA placement materially affected
Server tail latency**.

On `gpu54`, the two allocated H200 GPUs are local to NUMA node 1. The Slurm job
provided 16 CPU cores split across the two NUMA nodes:

```text
NUMA 0 : 16-21,26-27
NUMA 1 : 48-52,57-59
```

The GPU-local CPU set was:

```text
48-52,57-59
```

Without explicit NUMA binding, repeated QPS 0.50 runs showed unstable TTFT tail
latency and could fail MLPerf statistical early stopping even when the observed
p99 TTFT remained below the configured 2.0 s limit.

Binding the Server benchmark to the GPU-local NUMA node restored stable behavior:

```bash
numactl \
  --physcpubind=48-52,57-59 \
  --membind=1 \
  /bin/bash ./scripts/run_server_qps05_800q.sh
```

### Reproduced Server Results on gpu54

| Target QPS | Queries | CPU/NUMA placement | Result | Mean TTFT | p99 TTFT | Mean TPOT |
|---:|---:|---|---|---:|---:|---:|
| 0.5000 | 800 | default | INVALID | 198.0 ms | 1.597 s | 5.150 ms |
| 0.5000 | 800 | NUMA1 pinned | **VALID** | 194.3 ms | 1.550 s | 5.117 ms |
| 0.5375 | 800 | NUMA1 pinned | **VALID** | 212.6 ms | 1.566 s | 5.112 ms |

For the reproduced QPS 0.5375 run:

```text
Completed samples per second : 0.53
Completed tokens per second  : 67.56
Result is                    : VALID

Performance constraints      : Yes
Min duration                 : Yes
Min queries                  : Yes
Early stopping               : Yes

Mean TTFT                    : 212.63 ms
p99 TTFT                     : 1.566 s
Max TTFT                     : 2.168 s

Mean TPOT                    : 5.112 ms
p99 TPOT                     : 5.414 ms
```

The configured Server latency limits were:

```text
TTFT : 2.0 s
TPOT : 100 ms
```

These observations indicate that reproducible Server measurements depend not only
on GPU throughput but also on CPU scheduling and NUMA locality. In this test
environment, GPU-local CPU and memory binding significantly improved TTFT tail
stability.

---

## 10. Server QPS Tuning

Several bounded Server runs were performed to characterize the sustainable QPS
region.

### Original characterization

The earlier experiments, performed primarily on `gpu55`, produced:

| Target QPS | Result |
|---:|---|
| 0.45 | VALID |
| 0.50 | VALID |
| 0.53750 | VALID |
| 0.546875 | INVALID, borderline |
| 0.55625 | INVALID |
| 0.575 | INVALID |
| 0.650 | INVALID |
| 0.800 | INVALID |
| 1.500 | INVALID |

The highest tested VALID point was:

```text
QPS = 0.5375
```

The observed transition region was approximately:

```text
0.5375 < sustainable QPS < 0.546875
```

### Reproduction on gpu54

Initial runs on `gpu54` without explicit NUMA binding showed substantial
run-to-run TTFT tail variability. For example, a QPS 0.50 / 1200-query rerun
had:

```text
Mean TTFT : approximately 0.222 s
p99 TTFT  : approximately 1.776 s
Max TTFT  : approximately 6.784 s
```

while TPOT remained near 5.16 ms.

After binding the process and memory to GPU-local NUMA node 1, QPS 0.50 and
QPS 0.5375 both reproduced as VALID with 800 queries.

The next characterization point is:

```text
QPS 0.546875 / 800 queries / NUMA1 pinned
```

The QPS boundary should be interpreted as an approximate operating region rather
than a deterministic single-point limit.

---

## 11. Server Latency Behavior

The Server scenario showed a clear distinction between first-token latency and
token-generation latency.

- **TTFT (Time To First Token)** measures the time from request arrival until the
  first output token is returned.
- **TPOT (Time Per Output Token)** measures the average time required for
  subsequent generated tokens.

At moderate QPS values, TPOT stayed close to approximately 5 ms while TTFT was
more sensitive to request scheduling, CPU placement, and queueing.

At excessive request rates, request queueing increased sharply.

For example, at target QPS 1.5:

```text
Completed throughput : approximately 1.45 samples/s
TTFT mean            : approximately 7.23 s
TTFT p99             : approximately 17.88 s
TTFT maximum         : approximately 18.90 s
TPOT mean            : approximately 5.10 ms
TPOT p99             : approximately 5.34 ms
```

The token-generation time remained nearly unchanged while TTFT increased
dramatically.

This indicates that the dominant saturation behavior was request queueing and
scheduling delay rather than degradation of the token-generation kernel.

The NUMA-pinning experiments further showed that CPU locality can affect TTFT
tail stability even when GPU-side TPOT remains nearly unchanged.

---

## 12. Differences from the MLCommons Reference Benchmark

Several KISTI-specific adaptations were required to execute the MLCommons
reference implementation reliably on the Neuron H200 system.

These changes are documented explicitly so that the upstream reference code
and the KISTI-specific portability changes can be distinguished.

### 12.1 Execution Environment

The MLCommons reference implementation provides a generic benchmark
implementation.

The KISTI experiment was executed in the following environment:

```text
KISTI Neuron GPU Cluster
Slurm scheduler
Lustre filesystem
2 x NVIDIA H200
Tensor Parallelism = 2
```

Models, datasets, environments, and benchmark logs were stored under the
KISTI `/scratch` filesystem.

### 12.2 Model Artifact Difference

The benchmark used:

```text
meta-llama/Llama-3.1-8B-Instruct
```

with Hugging Face revision:

```text
0e9e39f249a16976918f6564b8830bc894c89659
```

An older model revision referenced by the MLCommons instructions was no
longer available.

Therefore, this experiment does not use exactly the same model artifact as
that older reference documentation.

### 12.3 Multiprocessing Method

The initial 2-GPU reference run failed with:

```text
RuntimeError: Cannot re-initialize CUDA in forked subprocess.
To use CUDA with multiprocessing, you must use the spawn start method.
```

To avoid CUDA initialization through the default fork-based process model,
the following environment variable was added:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

No MLCommons source-code modification was required for the Offline scenario.

### 12.4 Offline Scenario

The final Offline benchmark used the original MLCommons Llama 3.1 8B
reference implementation.

KISTI-specific runtime parameters were:

```text
2 x H200
TP = 2
Batch size = 16
BF16
13,368 samples
```

The final LoadGen result was VALID.

### 12.5 Accuracy Evaluation dtype

The generated MLPerf accuracy log itself was valid.

However, the MLCommons evaluation script defaults to an int64 decoding path
in the tested version.

This caused:

```text
OverflowError: out of range integral type conversion attempted
```

during tokenizer batch decoding.

The accuracy evaluation was therefore rerun with:

```text
--dtype int32
```

The generated inference results were not altered.

### 12.6 Server AsyncIO Behavior

The largest difference from the upstream reference implementation occurred
in the Server scenario.

The original `ServerSUT` creates a vLLM `AsyncLLMEngine`, while individual
queries are executed through repeated calls to:

```python
asyncio.run(...)
```

In the tested KISTI environment, this repeatedly created and destroyed the
asyncio event loop associated with the asynchronous vLLM engine.

The benchmark processed the first request but then stalled.

The upstream MLCommons source file was preserved unchanged.

Separate KISTI-specific copies were created:

```text
SUT_VLLM_serverfix.py
main_serverfix.py
```

The modified implementation creates one persistent asyncio event loop for
the worker thread and reuses the same event loop for subsequent requests.

Reference behavior:

```text
request
  -> asyncio.run()
  -> create event loop
  -> execute request
  -> destroy event loop

next request
  -> create another event loop
```

KISTI modification:

```text
Server worker
  -> create persistent event loop
       -> request 1
       -> request 2
       -> request 3
       -> ...
  -> close event loop at shutdown
```

This modification allowed Server requests to be processed continuously.

### 12.7 Server Shutdown Handling

The tested reference Server shutdown path also attempted to join:

```text
ft_response_thread
```

although that attribute was not instantiated in the tested execution path.

The KISTI-specific copy therefore checks whether the attribute exists before
attempting to access or join it.

Conceptually:

```python
if hasattr(self, "ft_response_thread"):
    ...
```

The original MLCommons source remains unchanged.

### 12.8 Remaining Server Shutdown Warning

After successful Server runs, some asynchronous cleanup warnings were still
observed, including messages such as:

```text
Task was destroyed but it is pending!
```

and multiprocessing shared-memory cleanup warnings.

These messages occurred after LoadGen completed and did not prevent VALID
benchmark results.

However, the asynchronous engine shutdown lifecycle should be cleaned up
before treating the modified Server implementation as submission-quality
code.

### 12.9 Server QPS Tuning and NUMA Placement

The original MLCommons `user.conf` defines a target QPS and minimum duration.

For performance characterization, temporary tuning configuration files were
created with bounded:

```text
minimum query count
maximum query count
maximum duration
target QPS
```

These tuning runs were used to identify the sustainable QPS region.

During reproduction on `gpu54`, the Slurm allocation exposed 16 CPU cores across
both NUMA nodes, while the allocated H200 GPUs were local to NUMA node 1.
Explicitly binding the Server benchmark to the GPU-local CPU set:

```text
48-52,57-59
```

and memory node:

```text
NUMA node 1
```

significantly improved TTFT tail stability and reproduced VALID results at
QPS 0.50 and QPS 0.5375 with 800 queries.

These tuning runs are performance-engineering measurements and should not be
interpreted as official MLPerf submission runs.

---


## 13. Reference Code Preservation

A key principle of this repository is to preserve the original MLCommons
implementation wherever possible.

The upstream repository and the KISTI reproducibility repository are kept
separate.

Recommended layout:

```text
/scratch/$USER/mlperf-inference-llama31/
|
|-- mlcommons-inference/
|   `-- upstream MLCommons reference implementation
|
`-- kisti-neuron-mlperf-inference/
    |-- README.md
    |-- configs/
    |-- docs/
    |-- patches/
    |-- results/
    `-- scripts/
```

The original MLCommons source should not be silently modified.

KISTI-specific modifications are stored under `patches/` or documented
explicitly.

---

## 14. Repository Structure

The public repository is organized as follows:

```text
kisti-neuron-mlperf-inference/
|
|-- README.md
|-- .gitignore
|-- configs/
|-- docs/
|-- patches/
|-- results/
|   |-- offline/
|   `-- server/
`-- scripts/
```

### Directory Purpose

`configs/`

- Offline LoadGen configurations
- Server tuning configurations

`scripts/`

- Offline performance execution scripts
- Offline accuracy execution scripts
- Server execution scripts
- QPS tuning scripts

`patches/`

- KISTI-specific Server SUT implementation
- notes explaining differences from upstream

`results/`

- summarized benchmark results
- small text and CSV result files suitable for Git

`docs/`

- benchmark reports
- technical notes
- reproducibility documentation

Large models, datasets, runtime environments, raw accuracy logs, and large
benchmark logs are intentionally excluded from the Git repository.

---

## 15. Reproducing the Benchmark

A typical working layout is:

```text
/scratch/$USER/mlperf-inference-llama31/
|
|-- mlcommons-inference/
|-- kisti-neuron-mlperf-inference/
|-- data/
|-- models/
|-- envs/
|-- logs/
|-- runs/
|-- run-configs/
`-- run-scripts/
```

The MLCommons reference repository can be cloned separately with:

```bash
git clone https://github.com/mlcommons/inference.git mlcommons-inference
cd mlcommons-inference
git checkout v6.0.0pre
```

The KISTI repository can be cloned with:

```bash
git clone https://github.com/hwang2006/kisti-neuron-mlperf-inference.git
```

Before executing the 2-GPU vLLM benchmark, the multiprocessing method should
be configured as:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Detailed scripts and configurations are maintained under `scripts/` and
`configs/`.

---

## 16. Limitations

The current study has several limitations.

- Only one KISTI Neuron H200 node configuration was evaluated.
- The primary configuration used 2 x NVIDIA H200.
- The model was Llama 3.1 8B Instruct.
- The Offline scenario used the upstream reference SUT.
- The Server scenario required a KISTI-specific portability patch.
- Server QPS tuning runs used bounded query counts for characterization.
- Server TTFT tail latency was sensitive to CPU/NUMA placement; GPU-local NUMA binding improved reproducibility on `gpu54`.
- The model artifact differs from the older revision listed in the
  MLCommons reference documentation.
- Some asynchronous vLLM shutdown warnings remain.
- No formal MLPerf compliance run was completed.

---

## 17. Official MLPerf Submission Status

The results in this repository are not official published MLPerf results.

A formal MLPerf submission would additionally require, among other items:

- exact submission-approved model artifact
- exact approved dataset artifact
- official system metadata
- submitter metadata
- required performance runs
- required accuracy runs
- applicable compliance tests
- official MLPerf directory structure
- submission checker validation
- reproducibility verification
- confirmation that the Server implementation modifications comply with
  the intended submission division rules

Therefore, this repository should be interpreted as a:

```text
reference benchmark reproduction
+
performance engineering study
+
pre-submission investigation
```

on the KISTI Neuron GPU cluster.

---

## 18. Related Repository

MLPerf Training experiments performed on the KISTI Neuron system are
documented separately:

https://github.com/hwang2006/kisti-neuron-mlperf-training

The Training and Inference repositories are intended to provide complementary
records of MLPerf workload execution on the KISTI Neuron GPU infrastructure.

---

## 19. References

### MLCommons

MLCommons MLPerf Inference:

https://github.com/mlcommons/inference

MLPerf Inference benchmark information:

https://mlcommons.org/benchmarks/inference/

### KISTI

KISTI Neuron User Guide:

https://docs-ksc.gitbook.io/neuron-user-guide/

### Model

Meta Llama 3.1 8B Instruct:

https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

---

## Acknowledgement

The benchmark experiments documented in this repository were performed on
the KISTI Neuron GPU cluster.

The MLPerf benchmark suite and reference implementations are provided by
