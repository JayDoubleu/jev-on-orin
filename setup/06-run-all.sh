#!/usr/bin/env bash
# Run every benchmark sequentially (never in parallel: they share one GPU) and print the table.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 HF_HUB_CACHE="$PWD/hf/hub"
venv/bin/python bench/bench_nimble.py nimble-model.json 5 results/bench-nimble-cuda-fla.json
venv/bin/python bench/bench_prefix.py 5 results/bench-nimble-prefix-conv.json
for dev in cuda cpu; do
  venv-verdict/bin/python bench/bench_verdict.py $dev 20 results/bench-verdict-$dev.json
  venv-von/bin/python bench/bench_von.py $dev 20
  venv-verdict/bin/python bench/bench_kotoba.py $dev 20
done
[ "${WITH_NEMOTRON:-0}" = 1 ] && venv/bin/python bench/bench_nemotron.py 5
python3 bench/summarise.py
