# Offline Results

## System

- KISTI Neuron GPU Cluster
- 2 x NVIDIA H200
- Tensor Parallelism: TP=2
- Precision: BF16
- Model: Llama 3.1 8B Instruct
- Dataset: CNN/DailyMail, 13,368 samples

## Performance

| Metric | Result |
|---|---:|
| Samples per second | 15.4041 |
| Tokens per second | 1,971.72 |
| LoadGen result | VALID |

## Accuracy

| Metric | Result |
|---|---:|
| ROUGE-1 | 38.8414 |
| ROUGE-2 | 15.9717 |
| ROUGE-L | 24.5673 |
| ROUGE-Lsum | 35.8933 |
| Generated tokens | 8,162,948 |
| Samples | 13,368 |

The accuracy evaluation was performed with:

```text
--dtype int32
```

because the default int64 decoding path caused an overflow error in the
tested environment.
