# Running these on a Jetson AGX Orin

Things that bit, in the order they bite.

**torch must be a Jetson build.** Stock PyPI aarch64 torch has no sm_87 kernels and pulls CUDA 13 libraries the r36 driver rejects. Use `pypi.jetson-ai-lab.io/jp6/cu126` (torch 2.8.0 there works; 2.11.0 from that index did not link on this image) or build from source. `setup/02-make-venvs.sh` reads `UV_INDEX` so you can point it at whichever.

**Python 3.10.** The Jetson torch wheels are cp310. Nimble asks for 3.12 and von for 3.12+, but both run on 3.10 (von is imported from `third_party/von/src` via `sys.path` rather than installed, because its `requires-python` refuses 3.10). `flash-linear-attention` prints a warning that `torch.compile` needs 3.11 and uses its kernels uncompiled; they still work.

**Fast kernels.** `flash-linear-attention` 0.5.2 and `triton` 3.8.0 install from PyPI and run on sm_87 (Nimble +25 to 35 percent). `causal-conv1d` has no aarch64 wheel: build with `CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=8.7 uv pip install --no-build-isolation causal-conv1d`, with setuptools, ninja, packaging and wheel already in the venv. Ten minutes, another ~10 percent.

**Unified memory means no double buffering.** `from_pretrained(...).to("cuda")` builds the whole model in CPU RAM and then copies it to "GPU" memory that is the same RAM. For the 27 GB Nemotron checkpoint that peak killed the process with `NVML_SUCCESS == r INTERNAL ASSERT FAILED` in `CUDACachingAllocator.cpp` (the allocator's OOM path queries NVML, which Tegra does not support, so you get an assert instead of an OOM). Load with `device_map="cuda"` so weights stream straight to their final place. Nimble's 18 GB happened to survive the double buffer; do not rely on it.

**Set `PYTORCH_CUDA_ALLOC_CONF=backend:native`.** torch's expandable-segments allocator needs CUDA virtual memory management that Tegra lacks.

**`NvMapMemHandleAlloc: error 0` lines during load are noise** as long as the process continues.

**The CPU is not an option.** 12 Cortex-A78AE cores, no MKL, fp32 only: ModernBERT-large took 61 s on a 1000-token input, 437M DeBERTa 3.5 s. Every model here was 10 to 100 times slower on CPU than on the iGPU.

**One forward pass costs ~130 ms whatever its length** (for the 9B hybrid model). That is CPU-side dispatch through 32 layers on slow cores. Consequences: batch questions into one pass wherever possible (what `PrefixCachedScorer` does), and do not add a cached pass to a single-question call. CUDA graphs would remove most of it; not attempted here.

**Run benchmarks one at a time.** One GPU, shared memory bandwidth; concurrent runs distort both.

**nvidia-smi shows nothing useful** for the iGPU (no utilisation or memory columns). Use `torch.cuda.memory_allocated()` and `tegrastats`.
