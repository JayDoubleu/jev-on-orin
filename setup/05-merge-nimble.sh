#!/usr/bin/env bash
# Merge the Nimble LoRA onto its pinned base on the CPU (about 2 minutes, 18 GB written), write nimble-model.json.
set -euo pipefail
cd "$(dirname "$0")/.."
ADAPTER=$(ls -d "$PWD"/hf/hub/models--bespokelabs--Bespoke-Nimble-9B/snapshots/*)
REV=$(python3 -c "import json;print(json.load(open('$ADAPTER/schema_config.json'))['revision'])")
BASE="$PWD/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/$REV"
( cd third_party/nimble && HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" ../../venv/bin/python -m nimble.scoring.merge_local_adapter --adapter "$ADAPTER" --base "$BASE" --output ../../merged )
printf '{"model_path": "%s/merged", "model_id": "bespokelabs/Bespoke-Nimble-9B", "revision": "%s", "max_input_tokens": 2048}\n' "$PWD" "$(basename "$ADAPTER")" > nimble-model.json
cat nimble-model.json
