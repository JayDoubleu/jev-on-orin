"""Benchmark com-kotobalabs/open-jev-deberta-v3-large on the Orin."""
import glob, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.bench_common import *
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"; REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SNAP = glob.glob(str(HF / "models--com-kotobalabs--open-jev-deberta-v3-large/snapshots/*"))[0]
sys.path.insert(0, SNAP)
from typed_decisions.open_jev import OpenJev
readme_q = [{"type": "choice", "instructions": "Urgency based on current business impact. HIGH: a critical business operation is currently blocked. LOW: an optional enhancement with no current business impact.", "options": ["HIGH", "LOW"]},
            {"type": "noul", "instructions": "Customers are unable to complete a purchase."}]
six_q = [{"type": "choice", "instructions": "Which team should handle this message?", "options": list(ROUTE)},
         {"type": "choice", "instructions": "Overall tone of the writer toward the company.", "options": ["POSITIVE", "NEUTRAL", "NEGATIVE"]},
         {"type": "score", "instructions": "Rate the severity of the reported problem.", "options": SEV},
         {"type": "noul", "instructions": "The writer asks for money back."},
         {"type": "noul", "instructions": "The writer states they are a paying customer."},
         {"type": "noul", "instructions": "A human must reply rather than an automated answer."}]
claim_q = [{"type": "choice", "instructions": "Is the claim in the final sentence supported by the preceding context?", "options": ["SUPPORTED", "NOT_SUPPORTED"]}]
CASES = [("readme_2fields_short", SHORT, readme_q), ("support_6fields", SUPPORT, six_q), ("claim_1field_long", LONG, claim_q), ("claim_1field_short", SHORT_CLAIM, claim_q)]
t0 = time.perf_counter(); m = OpenJev.from_pretrained(SNAP, device=DEVICE)
if DEVICE == "cuda": torch.cuda.synchronize()
load_s = time.perf_counter() - t0
nparams = sum(p.numel() for p in m.model.parameters()) if hasattr(m, "model") else None
print(f"load {load_s:.1f}s params {nparams}", flush=True)
res = {"machine": "NVIDIA Jetson AGX Orin 64 GB (Tegra, sm_87)", "device": DEVICE, "torch": torch.__version__, "model": "com-kotobalabs/open-jev-deberta-v3-large",
       "params": nparams, "load_seconds": load_s, "repeats": REPEATS, "timing_scope": "OpenJev.decide() wall time incl. tokenisation; 3 warmups then REPEATS measured", "cases": []}
def fmt(r):
    return [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in a.items() if k != "probabilities"} for a in r]
for name, ctx, qs in CASES:
    for _ in range(3): m.decide(ctx, qs)
    times = []
    for _ in range(REPEATS):
        if DEVICE == "cuda": torch.cuda.synchronize()
        t = time.perf_counter(); r = m.decide(ctx, qs)
        if DEVICE == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    st = stats(times); out = fmt(r)
    res["cases"].append({"case": name, "fields": len(qs), "seconds": st, "seconds_per_field_median": st["median"] / len(qs), "output": out})
    print(f"{name}: fields={len(qs)} median={st['median']*1000:.1f}ms p95={st['p95']*1000:.1f}ms -> {out}", flush=True)
if DEVICE == "cuda": res["cuda_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
n, t0 = 0, time.perf_counter()
while time.perf_counter() - t0 < 20: m.decide(SHORT, readme_q); n += 1
el = time.perf_counter() - t0
res["sustained_readme_case"] = {"seconds": el, "decisions": n, "decisions_per_second": n / el}
print(f"sustained: {n} in {el:.1f}s = {n/el:.1f}/s", flush=True)
(ROOT / "results" / f"bench-kotoba-{DEVICE}.json").write_text(json.dumps(res, indent=2) + "\n")
