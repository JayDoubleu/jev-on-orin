# Results

Machine: NVIDIA Jetson AGX Orin 64 GB developer kit, JetPack 6 (L4T r36, kernel 5.15.148-tegra, CUDA 12.6), 12-core Cortex-A78AE, integrated Ampere GPU (sm_87), 61 GB unified memory. Software: Python 3.10, torch 2.8.0 built for sm_87, transformers 5.17.0, peft 0.21.0. Measured 2026-09-20.

Method: four fixed calls (`bench/bench_common.py`): the Nimble README ticket with two questions, a six-question support-ticket schema, and a two-sentence refund-authorisation claim in short (~200 token) and long (~1000 token) forms. One warmup, then 5 repeats (big models) or 20 (encoders), medians. One request at a time, nothing else on the GPU. "Sustained" is back-to-back calls of the two-question case for 20 to 30 s.

## All models, stock code

| Model | load s | 2 fields short | 6 fields ~100 tok | 1 field ~1000 tok | 1 field short | sustained dec/s | GPU GiB |
|---|---|---|---|---|---|---|---|
| Nimble 9B (torch fallback kernels) | 20 | 625 ms | 4660 ms | 1549 ms | 294 ms | 1.6 | 17.5 |
| Nimble 9B (+fla/triton) | 35 | 467 ms | 3464 ms | 1139 ms | 214 ms | 2.1 | 17.5 |
| Nemotron-Diffusion 14B | 36 | 1007 ms | 3566 ms | 2052 ms | 534 ms | 1.0 | 25.2 |
| Kotoba DeBERTa-v3-large 437M | 8 | 82 ms | 99 ms | 123 ms | 80 ms | 12.2 | 1.7 |
| von-1.0 ModernBERT-large 395M | 10 | 108 ms | 353 ms | 81 ms | 57 ms | 9.3 | 0.8 |
| Verdict 2.0 ModernBERT-base 151M | 5 | 46 ms | 120 ms | 128 ms | 41 ms | 23.0 | 0.6 |
| Kotoba (CPU) | 6 | 1579 ms | 2616 ms | 3478 ms | 1485 ms | 0.6 | cpu |
| von (CPU) | 5 | 987 ms | 11.9 s | 61.0 s | 1111 ms | 0.9 | cpu |
| Verdict (CPU) | 5 | 1399 ms | 29.3 s | 22.6 s | 432 ms | 1.3 | cpu |

The Nimble rows use its stock `CudaCandidateScorer` (one full prompt per field). The first row has transformers' reference PyTorch kernels for the linear-attention layers; the second has `flash-linear-attention` and `triton` installed, `causal-conv1d` still on the fallback.

## Nimble, tuned (`results/bench-nimble-prefix-conv.json`)

All fast kernels installed (fla, triton, causal-conv1d built from source). Three scoring modes from `nimble_orin/cuda_prefix_scorer.py`; `vs independent` is the largest logit and probability difference against the stock path over every candidate of every field.

| Call | independent (stock) | cached serial | parallel | auto picks | max logit diff | max prob diff | same answers |
|---|---|---|---|---|---|---|---|
| 2 questions, short | 428 ms | 544 ms | **381 ms** | parallel | 0.09 | 0.001 | yes |
| 6 questions, ticket | 3481 ms | 1572 ms | **779 ms** | parallel | 0.07 | 0.003 | yes |
| 1 question, ~1000 tokens | **1033 ms** | 1232 ms | 1236 ms | independent | 0.06 | 0.00002 | yes |
| 1 question, short | **200 ms** | 363 ms | 367 ms | independent | 0.10 | 0.000001 | yes |

Sustained two-question calls in parallel mode: 2.6 per second. Load 18 s, 17.5 GiB.

Why single questions are slower with the cache: a profile (`bench/profile_prefix.py`) shows a 12-token suffix pass costs 190 ms and even a 1-token pass 136 ms, against 210 ms for the whole 185-token prefix. One forward pass has a ~130 ms floor on the Orin, dominated by CPU-side kernel dispatch through 32 hybrid layers; the marginal cost above that is about 0.9 ms per token (~1,100 tokens per second). So the cache saves prefills but adds one pass. CUDA graph capture is the obvious next lever for the floor and was not attempted.

## Answers each model gave

Six-question ticket (a furious three-year Pro customer, export button returns 500, wants a refund and a human):

| Model | route | sentiment | severity | mentions refund | is customer | needs human |
|---|---|---|---|---|---|---|
| Nimble | TECHNICAL | NEGATIVE | S3 | true | true | true |
| Nemotron | TECHNICAL (0.98) | NEGATIVE (0.99) | score 1.3 (between S2 and S3) | 0.99 | 1.00 | 0.99 |
| Kotoba | TECHNICAL (0.80) | NEGATIVE (0.74) | score 0.84 | 0.90 | 0.92 | 0.91 |
| von | BILLING (0.34) | NEGATIVE (1.00) | score 1.0 (0.99) | 1.00 | 1.00 | 1.00 |
| Verdict | TECHNICAL (0.44) | NEGATIVE (0.83) | abstained | (noul) | (noul) | (noul) |

Claim pair. Policy: only Mira may authorise refunds on account 42. Long text (28 filler paragraphs first): the sole authorisation was signed by Noah, so NOT_SUPPORTED is right. Short text: signed by Mira, so SUPPORTED is right.

| Model | long (Noah) | short (Mira) |
|---|---|---|
| Nimble | NOT_SUPPORTED | SUPPORTED |
| Nemotron | NOT_SUPPORTED (0.84) | SUPPORTED (0.92) |
| Kotoba | NOT_SUPPORTED (0.53) | NOT_SUPPORTED (0.52) |
| von | SUPPORTED (0.99) | SUPPORTED (0.99) |
| Verdict | abstained (0.41) | SUPPORTED (0.96) |

## Caveats

- One machine, one day, one hand-written prompt set. The claim pair discriminates reasoning from pattern matching; it says nothing about which of Nimble and Nemotron is more accurate in general. Nimble's own 324-example held-out set puts it at 90.1 percent against Jev 1.13.0's 93.2.
- Timings exclude tokenisation for Nimble and Nemotron (their scoring functions report model time) and include it for the three encoders (their APIs do not separate it). Tokenisation is a few milliseconds either way.
- The CPU rows are torch fp32 with 12 threads; Verdict and Kotoba ship ONNX exports that would do better on CPU, not measured.
- Real Jev numbers are not measured here. The 100 to 300 ms figure is what the upstream READMEs report for the cloud API, network included.
