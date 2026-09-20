# jev-on-orin

Benchmarks of open-source Jev-style typed-decision models on a Jetson AGX Orin, plus `nimble_orin/`, a prefix-cached scorer for Nimble. Read `README.md` for the story and `docs/ORIN-NOTES.md` before running anything on the GPU: it lists the Tegra traps that cost real time.

## Layout

- `nimble_orin/cuda_prefix_scorer.py`: subclasses Nimble's `CudaCandidateScorer` (imported from `third_party/nimble`, never copied). Three modes plus `auto`. Any change must keep `same_choices: true` and logit differences under about 0.1 against `independent` in `bench/bench_prefix.py`; that comparison is the correctness test.
- `bench/`: one script per project; the prompts and schemas all models share live in `bench_common.py` and must stay identical across scripts, or the table stops being a comparison. `summarise.py` renders the table from `results/`.
- `results/`: run JSONs are evidence. Never edit them by hand; re-run the benchmark and replace the file. When numbers change, update `results/RESULTS.md` and the table in `README.md` together.
- `setup/`: numbered scripts, run from the repo root in order. `setup/pins/` holds the upstream commits the results were measured against; bump a pin only alongside fresh results.
- `third_party/`, `hf/`, `merged/`, `venv*/` are gitignored and rebuilt by the setup scripts. Nothing heavier than a script or a JSON is ever committed.

## Working rules

- Benchmarks run one at a time. The iGPU and its memory bandwidth are shared, and concurrent runs distort every number.
- Load big checkpoints with `device_map="cuda"`. A CPU load followed by `.to("cuda")` double-buffers in unified memory and dies in an NVML assert (see the notes).
- Before adding a new upstream project: read its code for network calls, `pickle`/`torch.load` of untrusted files and `trust_remote_code`, prefer safetensors weights, pin the commit, and record what was found in the README's safety section. Unverifiable claims in an upstream README get stated against the upstream's own files, not editorialised.
- New models get the same four calls as everyone else; add a case only if every script gets it.
- Real Jev numbers here are what upstream READMEs quote for the cloud API. Do not present them as measured.
- British English, no em-dashes, no emojis, in code comments and docs alike.
- Commits are authored as Jay W. (`git.jaydoubleu@gmail.com`); the repo's local git config already sets this.

## Environment

Python 3.10 with a Jetson build of torch 2.8.0 (sm_87); stock PyPI torch does not work on the Orin. `setup/02-make-venvs.sh` takes the wheel index from `UV_INDEX`. Python runs through `uv`; never pip into the system interpreter.
