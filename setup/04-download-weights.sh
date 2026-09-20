#!/usr/bin/env bash
# All weights go to hf/hub (about 50 GB). Nimble's adapter is merged onto its pinned base by 05.
set -euo pipefail
cd "$(dirname "$0")/.."
dl() { uvx --from huggingface-hub hf download "$@" --cache-dir hf/hub; }
dl bespokelabs/Bespoke-Nimble-9B
dl Qwen/Qwen3.5-9B --revision "$(python3 -c 'import json,glob;print(json.load(open(glob.glob("hf/hub/models--bespokelabs--Bespoke-Nimble-9B/snapshots/*/schema_config.json")[0]))["revision"])')"
dl com-kotobalabs/open-jev-deberta-v3-large
dl wfzyx/von-1.0
mkdir -p checkpoints && ln -sfn "$(ls -d "$PWD"/hf/hub/models--wfzyx--von-1.0/snapshots/*)" checkpoints/von-modernbert-rlcd   # von reads calibration.json from a local dir
python3 third_party/verdict/scripts/download_artifacts.py --output_dir third_party/verdict/artifacts/v2   # sha256-pinned
if [ "${WITH_NEMOTRON:-0}" = 1 ]; then dl nvidia/Nemotron-Labs-Diffusion-14B --exclude "assets/*"; fi   # 27 GB, optional
