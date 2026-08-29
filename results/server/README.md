# Server Results

## System

- KISTI Neuron GPU Cluster
- 2 x NVIDIA H200
- Tensor Parallelism: TP=2
- Precision: BF16
- Model: Llama 3.1 8B Instruct

## QPS Characterization

| Target QPS | Result | Notes |
|---:|---|---|
| 0.45 | VALID | 700-query validation run |
| 0.50 | VALID | 800-query run |
| 0.53750 | VALID | Highest tested VALID binary-search point |
| 0.546875 | INVALID | Borderline; estimated 38 more clean queries required |
| 0.55625 | INVALID | Binary-search run |
| 0.57500 | INVALID | Binary-search run |
| 0.65000 | INVALID | Binary-search run |
| 0.80000 | INVALID | Tail-latency instability |
| 1.50000 | INVALID | Clear overload |

Observed transition region:

```text
0.5375 < sustainable QPS < 0.546875
```

The value should be interpreted as an experimentally observed transition
region rather than a deterministic architectural limit.

## High-load Latency Behavior

At target QPS 1.5:

```text
Completed samples/s : approximately 1.45
TTFT mean           : approximately 7.23 s
TTFT p99            : approximately 17.88 s
TTFT maximum        : approximately 18.90 s
TPOT mean           : approximately 5.10 ms
TPOT p99            : approximately 5.34 ms
```

The key observation is that TPOT remained nearly constant while TTFT rose
sharply. This suggests that request queueing and scheduling delay dominated
the overload behavior rather than a large slowdown in token generation.
