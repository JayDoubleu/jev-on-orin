"""Benchmark Bespoke-Nimble-9B on the Jetson AGX Orin through Nimble's stock CUDA scorer (independent full prompt per field).

Usage: python bench/bench_nimble.py nimble-model.json [repeats] [out.json]"""
import json, os, platform, statistics, subprocess, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "nimble"))
from nimble.scoring.cuda_scorer import CudaCandidateScorer  # noqa: E402

CONFIG = json.loads(Path(sys.argv[1]).read_text())
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("bench-results.json")

def stats(values):
    o = sorted(values)
    i = (len(o) - 1) * 0.95
    lo = int(i); hi = min(lo + 1, len(o) - 1)
    return {"min": min(o), "median": statistics.median(o), "mean": statistics.mean(o),
            "p95": o[lo] + (o[hi] - o[lo]) * (i - lo), "max": max(o)}

def rss_gib():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 2**20
    return None

PRIORITY_SCHEMA = {
    "priority": {"type": "enum", "choices": ["HIGH", "LOW"],
                 "description": "Urgency based on current business impact.",
                 "choice_descriptions": {"HIGH": "A critical business operation is currently blocked.",
                                          "LOW": "An optional enhancement with no current business impact."}},
    "requires_review": {"type": "boolean", "description": "Whether customers are unable to complete a purchase."},
}
SIX_SCHEMA = {
    "route": {"type": "enum", "choices": ["BILLING", "TECHNICAL", "SALES", "ABUSE", "OTHER"],
              "description": "Which team should handle this message."},
    "sentiment": {"type": "enum", "choices": ["POSITIVE", "NEUTRAL", "NEGATIVE"],
                  "description": "Overall tone of the writer toward the company."},
    "severity": {"type": "enum", "choices": ["S1", "S2", "S3", "S4"],
                 "description": "Ordered severity; S1 is a full outage, S4 is cosmetic.",
                 "choice_descriptions": {"S1": "Service unusable for all customers.", "S2": "Major feature broken for many customers.",
                                          "S3": "Minor feature broken or workaround exists.", "S4": "Cosmetic or documentation issue."}},
    "mentions_refund": {"type": "boolean", "description": "Whether the writer asks for money back."},
    "is_customer": {"type": "boolean", "description": "Whether the writer states they are a paying customer."},
    "needs_human": {"type": "boolean", "description": "Whether a human must reply rather than an automated answer."},
}
CLAIM_SCHEMA = {
    "supported": {"type": "enum", "choices": ["SUPPORTED", "NOT_SUPPORTED"],
                  "description": "Whether the claim in the final sentence is supported by the preceding context."},
}

short_ctx = "The payment service is down for all customers."
support_msg = ("Hi, I have been a paying customer on the Pro plan for three years. Since this morning the export "
               "button on the reports page returns a 500 error every time, so my whole finance team cannot close "
               "the month. I have tried three browsers. Honestly I am furious; this is the second time this quarter. "
               "If this is not fixed today I want a refund for this month and I want to talk to an actual person, "
               "not a bot.")
filler = ("Section {n}. The regional office reported that quarterly maintenance was completed on schedule, "
          "with no outstanding defects noted by the inspection team, and the log was countersigned by the duty manager. ")
long_ctx = "".join(filler.format(n=i) for i in range(28)) + \
    "Only Mira may authorize refunds for account 42. The sole authorization for this refund on account 42 was signed by Noah. " + \
    "Claim: the refund on account 42 is authorized."

CASES = [
    ("readme_2fields_short", short_ctx, PRIORITY_SCHEMA),
    ("support_6fields", support_msg, SIX_SCHEMA),
    ("claim_1field_long", long_ctx, CLAIM_SCHEMA),
    ("claim_1field_short", "Only Mira may authorize refunds for account 42. The sole authorization for this refund on account 42 was signed by Mira. Claim: the refund on account 42 is authorized.", CLAIM_SCHEMA),
]

t0 = time.perf_counter()
scorer = CudaCandidateScorer(**CONFIG)
torch.cuda.synchronize()
load_s = time.perf_counter() - t0
print(f"load: {load_s:.1f} s, rss {rss_gib():.1f} GiB, cuda allocated {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

results = {"machine": "NVIDIA Jetson AGX Orin 64 GB (Tegra, sm_87, iGPU, unified memory)",
           "kernel": platform.release(), "python": platform.python_version(),
           "runtime": scorer.runtime, "transformers": __import__("transformers").__version__,
           "peft": __import__("peft").__version__,
           "load_seconds": load_s, "rss_gib_after_load": rss_gib(),
           "cuda_allocated_gib_after_load": torch.cuda.memory_allocated() / 2**30,
           "repeats": REPEATS, "timing_scope": "scorer.score() wall time: tokenisation excluded, full-prompt prefill per field included; 1 warmup then REPEATS measured",
           "cases": []}

for name, ctx, schema in CASES:
    r = scorer.score(ctx, schema)  # warmup
    times, peaks = [], []
    for _ in range(REPEATS):
        r = scorer.score(ctx, schema)
        times.append(r["metrics"]["total_seconds"])
        peaks.append(r["metrics"]["cuda_peak_active_gib"])
    toks = [f["prompt_token_count"] for f in r["fields"].values()]
    st = stats(times)
    case = {"case": name, "fields": len(schema), "prompt_tokens_per_field": toks,
            "total_prompt_tokens": sum(toks), "seconds": st,
            "seconds_per_field_median": st["median"] / len(schema),
            "prefill_tokens_per_second_median": sum(toks) / st["median"],
            "cuda_peak_active_gib": max(peaks), "output": r["output"],
            "scores": {k: v["scores"] for k, v in r["fields"].items()}}
    results["cases"].append(case)
    print(f"{name}: fields={len(schema)} tokens={toks} median={st['median']:.3f}s p95={st['p95']:.3f}s "
          f"({case['prefill_tokens_per_second_median']:.0f} tok/s) peak={max(peaks):.1f} GiB -> {r['output']}", flush=True)

# Throughput: sustained back-to-back short decisions for ~30 s
n, t0 = 0, time.perf_counter()
while time.perf_counter() - t0 < 30:
    scorer.score(short_ctx, PRIORITY_SCHEMA); n += 1
el = time.perf_counter() - t0
results["sustained_readme_case"] = {"seconds": el, "decisions": n, "decisions_per_second": n / el,
                                    "fields_per_second": 2 * n / el}
print(f"sustained: {n} two-field decisions in {el:.1f} s = {n/el:.2f} decisions/s", flush=True)
OUT.write_text(json.dumps(results, indent=2) + "\n")
print("wrote", OUT)
