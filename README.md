# jev-on-orin

Open-source Jev-style typed decisions on an NVIDIA Jetson AGX Orin 64 GB: which projects run, how fast, how well, and a tuned Nimble scorer that makes the best one 4.5x faster.

TypeSafe's Jev is a "System One" model: you give it text plus a schema of questions (choice, boolean, ordered score) and it answers every question from the model's next-token probabilities, with no text generation. Several open projects now do the same. This repo is the result of putting five of them on one Orin, with the same prompts, and measuring.

## Results at a glance

All on the Orin's integrated GPU, bf16, transformers 5.17, torch 2.8.0 built for sm_87. Medians after warmup, one request at a time.

| Model | Load | 2 questions, short text | 6 questions, ticket | 1 question, ~1000 tokens | Sustained | GPU memory | Reasons over evidence? |
|---|---|---|---|---|---|---|---|
| **Nimble 9B, tuned (this repo)** | 18 s | **380 ms** | **780 ms** | 1.0 s | 2.6/s | 17.5 GB | yes |
| Nimble 9B, stock CUDA scorer | 20 s | 625 ms | 4.7 s | 1.5 s | 1.6/s | 17.5 GB | yes |
| Nemotron-Labs-Diffusion 14B | 36 s | 1.0 s | 3.6 s | 2.1 s | 1.0/s | 25.2 GB | yes |
| Kotoba open-jev DeBERTa-v3-large | 8 s | 82 ms | 99 ms | 123 ms | 12/s | 1.7 GB | no (coin flip) |
| von-1.0 ModernBERT-large | 10 s | 108 ms | 353 ms | 81 ms | 9/s | 0.8 GB | no (confidently wrong) |
| openJev-verdict-2.0 ModernBERT-base | 5 s | 46 ms | 120 ms | 128 ms | 23/s | 0.6 GB | no (abstains) |

For scale, the projects above quote the real Jev API at roughly 100 to 300 ms per call. See [results/RESULTS.md](results/RESULTS.md) for every number, the answers each model gave, and the CPU runs (all useless on the Orin: 1 to 60 s).

"Reasons over evidence" is one hand-written pair: a policy sentence, an authorisation sentence, and a claim, where changing one name flips the right answer. Only Nimble and Nemotron got both halves right. It discriminates; it does not rank the winners.

## What was tuned

Nimble's stock CUDA scorer re-reads the whole prompt once per question. `nimble_orin/cuda_prefix_scorer.py` subclasses it and:

1. runs the shared prefix (system prompt, context, schema) once into a transformers `DynamicCache`,
2. expands that cache to one row per question and scores all question suffixes in a single batched forward pass (right-padded; the hidden state is read at each row's last real token, so padding never feeds a result),
3. falls back to the stock single pass when there is only one question, because on the Orin one forward pass costs about 0.13 s regardless of length and the cache only pays off from two questions up (`mode="auto"`).

Plus the kernels Nimble expects but the Jetson wheels lack: `flash-linear-attention` and `triton` install from PyPI and work on sm_87; `causal-conv1d` has to be built from source (`setup/03-build-causal-conv1d.sh`, ten minutes).

Across all four test calls and all three modes the chosen answers were identical to the stock scorer, with logits within 0.1 and probabilities within 0.3 percent (`results/bench-nimble-prefix-conv.json`, `vs_independent`).

```python
import json
from nimble_orin import PrefixCachedScorer   # needs third_party/nimble on the path, see setup/
scorer = PrefixCachedScorer(**json.load(open("nimble-model.json")))
result = scorer.score("The payment service is down for all customers.", {
    "priority": {"type": "enum", "choices": ["HIGH", "LOW"], "description": "Urgency based on current business impact."},
    "requires_review": {"type": "boolean", "description": "Whether customers are unable to complete a purchase."},
})
print(result["output"], result["fields"]["priority"]["scores"])
```

## Reproduce

Needs an AGX Orin (64 GB tested; Nimble alone needs about 20 GB free), JetPack 6 with CUDA 12.6, [uv](https://docs.astral.sh/uv/), and roughly 50 GB of disk for weights. The scripts run in order from the repo root:

```sh
setup/01-clone-upstream.sh        # the four upstream repos into third_party/, pinned
UV_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/ setup/02-make-venvs.sh
setup/03-build-causal-conv1d.sh   # optional, +10 percent on Nimble
setup/04-download-weights.sh      # WITH_NEMOTRON=1 for the extra 27 GB
setup/05-merge-nimble.sh          # LoRA onto its pinned Qwen3.5-9B base, CPU, 2 minutes
setup/06-run-all.sh               # sequential, prints the table
```

Read [docs/ORIN-NOTES.md](docs/ORIN-NOTES.md) first: it lists the Tegra-specific traps (torch wheel source, the unified-memory double-buffer crash, why CPU runs are hopeless).

## What was not run, and why

- **DiffusionGemma 26B-A4B** (razorback16/openjev, openjev-sglang): the open servers need a patched vLLM or SGLang plus NVFP4 weights on Blackwell; the bf16 weights alone are 52 GB.
- **reflex, open-alternative-jev, system-one-open**: the same next-token trick on Qwen3.5-4B or Gemma; Nimble covers that approach with a fine-tune on top.

## Safety notes on the upstream code

Each repo was read before anything ran. Nimble's scoring path is local-only (its network calls live in data curation and Modal deploy code). Verdict's weights are safetensors from `heman10x/rlcd-modernbert-151m`, sha256-pinned in its `ARTIFACTS.json`, not the repo its README links; its README's training claims do not match its own manifest. von and Kotoba are plain transformers loads of safetensors. Nemotron_Jev loads NVIDIA's model with `trust_remote_code`; the remote modelling files were read and contain only model code.

## Layout

```
nimble_orin/   the prefix-cached scorer (subclasses Nimble's CudaCandidateScorer, does not copy it)
bench/         one benchmark per project, shared prompts in bench_common.py, summarise.py prints the table
results/       every run's JSON plus RESULTS.md
setup/         the six scripts above and the upstream commit pins
docs/          Orin notes
third_party/   upstream clones (gitignored)   hf/  weights (gitignored)
```

Licence: Apache 2.0 for this repo. Upstream projects and model weights keep their own licences (Nimble's model is on Hugging Face under its own terms; Nemotron under the NVIDIA Nemotron Open Model License).
