"""Benchmark von-1.0 (ModernBERT-large NLI cross-encoder) on the Orin."""
import json, os, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "von" / "src"))
from bench.bench_common import *
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"; REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
os.environ["VON_DEVICE"] = DEVICE
import von
readme_q = {"priority": von.choice(instructions="Urgency based on current business impact.",
                                   criteria={"HIGH": "A critical business operation is currently blocked.", "LOW": "An optional enhancement with no current business impact."}),
            "requires_review": von.noul(instructions="Customers are unable to complete a purchase.")}
six_q = {"route": von.choice(instructions="Which team should handle this message?", criteria=ROUTE),
         "sentiment": von.choice(instructions="Overall tone of the writer toward the company.", criteria={"POSITIVE": "Positive", "NEUTRAL": "Neutral", "NEGATIVE": "Negative"}),
         "severity": von.score(instructions="Rate the severity of the reported problem.", criteria=SEV),
         "mentions_refund": von.noul(instructions="The writer asks for money back."),
         "is_customer": von.noul(instructions="The writer states they are a paying customer."),
         "needs_human": von.noul(instructions="A human must reply rather than an automated answer.")}
claim_q = {"supported": von.choice(instructions="Is the claim in the final sentence supported by the preceding context?",
                                   criteria={"SUPPORTED": "The claim is supported by the context.", "NOT_SUPPORTED": "The claim is not supported by the context."})}
CASES = [("readme_2fields_short", SHORT, readme_q), ("support_6fields", SUPPORT, six_q), ("claim_1field_long", LONG, claim_q), ("claim_1field_short", SHORT_CLAIM, claim_q)]
def fmt(ans):
    o = {}
    for k, a in ans.items():
        o[k] = {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in a.__dict__.items() if kk in ("choice", "noul", "score", "confidence")} if hasattr(a, "__dict__") else str(a)
    return o
t0 = time.perf_counter(); von.system_one(state=SHORT, questions=readme_q)
if DEVICE == "cuda": torch.cuda.synchronize()
load_s = time.perf_counter() - t0
print(f"load+first call {load_s:.1f}s", flush=True)
res = {"machine": "NVIDIA Jetson AGX Orin 64 GB (Tegra, sm_87)", "device": DEVICE, "torch": torch.__version__, "model": "wfzyx/von-1.0",
       "load_plus_first_call_seconds": load_s, "repeats": REPEATS, "timing_scope": "von.system_one() wall time incl. tokenisation; 3 warmups then REPEATS measured", "cases": []}
for name, ctx, qs in CASES:
    for _ in range(3): von.system_one(state=ctx, questions=qs)
    times = []
    for _ in range(REPEATS):
        if DEVICE == "cuda": torch.cuda.synchronize()
        t = time.perf_counter(); r = von.system_one(state=ctx, questions=qs)
        if DEVICE == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    st = stats(times); out = fmt(r.answers)
    res["cases"].append({"case": name, "fields": len(qs), "seconds": st, "seconds_per_field_median": st["median"] / len(qs), "output": out})
    print(f"{name}: fields={len(qs)} median={st['median']*1000:.1f}ms p95={st['p95']*1000:.1f}ms -> {out}", flush=True)
if DEVICE == "cuda": res["cuda_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
n, t0 = 0, time.perf_counter()
while time.perf_counter() - t0 < 20: von.system_one(state=SHORT, questions=readme_q); n += 1
el = time.perf_counter() - t0
res["sustained_readme_case"] = {"seconds": el, "decisions": n, "decisions_per_second": n / el}
print(f"sustained: {n} in {el:.1f}s = {n/el:.1f}/s", flush=True)
(ROOT / "results" / f"bench-von-{DEVICE}.json").write_text(json.dumps(res, indent=2) + "\n")
