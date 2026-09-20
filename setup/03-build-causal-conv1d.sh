#!/usr/bin/env bash
# causal-conv1d has no aarch64 wheel; build it for sm_87. About 10 minutes on an AGX Orin.
set -euo pipefail
cd "$(dirname "$0")/.."
CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="8.7" MAX_JOBS="${MAX_JOBS:-8}" \
  uv pip install --python venv/bin/python --no-build-isolation "causal-conv1d>=1.4"
venv/bin/python -c 'import causal_conv1d; print("causal_conv1d", causal_conv1d.__version__)'
