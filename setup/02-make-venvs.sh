#!/usr/bin/env bash
# Three venvs, Python 3.10, torch 2.8.0 from a Jetson wheel index (see docs/ORIN-NOTES.md).
# UV_INDEX must point at an index that serves an sm_87 torch 2.8.0 wheel before running this.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${UV_INDEX:?set UV_INDEX to a Jetson torch wheel index, e.g. https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/}"
uv venv --python 3.10 venv
uv pip install --python venv/bin/python "torch==2.8.0" "transformers==5.17.0" "peft==0.21.0" "accelerate==1.15.0" \
  "sentencepiece==0.2.2" pillow huggingface-hub "flash-linear-attention" "triton" setuptools ninja packaging wheel
uv venv --python 3.10 venv-verdict
uv pip install --python venv-verdict/bin/python "torch==2.8.0" gliclass safetensors onnxruntime "pydantic>=2" numpy accelerate sentencepiece
uv venv --python 3.10 venv-von
uv pip install --python venv-von/bin/python "torch==2.8.0" transformers accelerate "pydantic>=2" click httpx sentencepiece protobuf safetensors
venv/bin/python -c 'import torch, transformers, fla; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print("nimble venv ok", torch.__version__, transformers.__version__)'
