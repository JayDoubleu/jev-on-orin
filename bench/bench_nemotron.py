"""Benchmark Nemotron-Labs-Diffusion-14B typed decisions (pst2154/Nemotron_Jev scoring path) on the Orin."""
import glob, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.bench_common import *
sys.path.insert(0, str(TP / "nemotron-jev" / "app"))
from scoring import evaluate
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SNAP = glob.glob(str(HF / "models--nvidia--Nemotron-Labs-Diffusion-14B/snapshots/*"))[0]
from transformers import AutoModel, AutoTokenizer
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True, local_files_only=True)
model = AutoModel.from_pretrained(SNAP, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, device_map="cuda").eval()
torch.cuda.synchronize(); load_s = time.perf_counter() - t0
print(f"load {load_s:.1f}s cuda {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
readme_q = {"priority": {"type": "choice", "instructions": "Urgency based on current business impact.",
                         "criteria": {"HIGH": "A critical business operation is currently blocked.", "LOW": "An optional enhancement with no current business impact."}},
            "requires_review": {"type": "noul", "instructions": "Customers are unable to complete a purchase."}}
six_q = {"route": {"type": "choice", "instructions": "Which team should handle this message?", "criteria": ROUTE},
         "sentiment": {"type": "choice", "instructions": "Overall tone of the writer toward the company.", "criteria": {"POSITIVE": "Positive", "NEUTRAL": "Neutral", "NEGATIVE": "Negative"}},
         "severity": {"type": "score", "instructions": "Rate the severity of the reported problem.", "criteria": SEV},
         "mentions_refund": {"type": "noul", "instructions": "The writer asks for money back."},
         "is_customer": {"type": "noul", "instructions": "The writer states they are a paying customer."},
         "needs_human": {"type": "noul", "instructions": "A human must reply rather than an automated answer."}}
claim_q = {"supported": {"type": "choice", "instructions": "Is the claim in the final sentence supported by the preceding context?",
                         "criteria": {"SUPPORTED": "The claim is supported by the context.", "NOT_SUPPORTED": "The claim is not supported by the context."}}}
CASES = [("readme_2fields_short", SHORT, readme_q), ("support_6fields", SUPPORT, six_q), ("claim_1field_long", LONG, claim_q), ("claim_1field_short", SHORT_CLAIM, claim_q)]
res = {"machine": "NVIDIA Jetson AGX Orin 64 GB (Tegra, sm_87)", "device": "cuda", "torch": torch.__version__, "transformers": __import__("transformers").__version__,
       "model": "nvidia/Nemotron-Labs-Diffusion-14B", "params": sum(p.numel() for p in model.parameters()), "load_seconds": load_s,
       "cuda_allocated_gib_after_load": torch.cuda.memory_allocated() / 2**30, "repeats": REPEATS,
       "timing_scope": "scoring.evaluate() metrics.seconds: tokenisation excluded, causal prefill + one masked slot per question, questions sequential; 1 warmup then REPEATS measured", "cases": []}
def fmt(ans): return {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in a.items() if kk in ("choice", "noul", "score", "confidence")} for k, a in ans.items()}
for name, ctx, qs in CASES:
    payload = {"model": "jev-latest", "state": ctx, "questions": qs}
    r = evaluate(model, tok, payload); times = []
    for _ in range(REPEATS):
        r = evaluate(model, tok, payload); times.append(r["metrics"]["seconds"])
    st = stats(times); out = fmt(r["answers"]); toks = r["usage"]["input_tokens"]
    res["cases"].append({"case": name, "fields": len(qs), "total_prompt_tokens": toks, "seconds": st, "seconds_per_field_median": st["median"] / len(qs),
                         "prefill_tokens_per_second_median": toks / st["median"], "cuda_peak_gib": torch.cuda.max_memory_allocated() / 2**30, "output": out})
    print(f"{name}: fields={len(qs)} tokens={toks} median={st['median']:.3f}s p95={st['p95']:.3f}s ({toks/st['median']:.0f} tok/s) -> {out}", flush=True)
n, t0 = 0, time.perf_counter()
while time.perf_counter() - t0 < 30: evaluate(model, tok, {"model": "jev-latest", "state": SHORT, "questions": readme_q}); n += 1
el = time.perf_counter() - t0
res["sustained_readme_case"] = {"seconds": el, "decisions": n, "decisions_per_second": n / el}
print(f"sustained: {n} in {el:.1f}s = {n/el:.2f}/s", flush=True)
(ROOT / "results" / "bench-nemotron-cuda.json").write_text(json.dumps(res, indent=2) + "\n")
