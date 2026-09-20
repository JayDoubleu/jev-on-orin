import json, sys, time, statistics
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.bench_common import *
from nimble_orin.cuda_prefix_scorer import PrefixCachedScorer
CONFIG = json.loads(Path("nimble-model.json").read_text()); REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("bench-nimble-prefix.json")
PRIORITY = {"priority": {"type": "enum", "choices": ["HIGH", "LOW"], "description": "Urgency based on current business impact.",
                         "choice_descriptions": {"HIGH": "A critical business operation is currently blocked.", "LOW": "An optional enhancement with no current business impact."}},
            "requires_review": {"type": "boolean", "description": "Whether customers are unable to complete a purchase."}}
SIX = {"route": {"type": "enum", "choices": list(ROUTE), "description": "Which team should handle this message.", "choice_descriptions": ROUTE},
       "sentiment": {"type": "enum", "choices": ["POSITIVE", "NEUTRAL", "NEGATIVE"], "description": "Overall tone of the writer toward the company."},
       "severity": {"type": "enum", "choices": ["S1", "S2", "S3", "S4"], "description": "Ordered severity; S1 is a full outage, S4 is cosmetic.",
                    "choice_descriptions": dict(zip(["S1", "S2", "S3", "S4"], [s[3:] for s in SEV]))},
       "mentions_refund": {"type": "boolean", "description": "Whether the writer asks for money back."},
       "is_customer": {"type": "boolean", "description": "Whether the writer states they are a paying customer."},
       "needs_human": {"type": "boolean", "description": "Whether a human must reply rather than an automated answer."}}
CLAIM = {"supported": {"type": "enum", "choices": ["SUPPORTED", "NOT_SUPPORTED"], "description": "Whether the claim in the final sentence is supported by the preceding context."}}
CASES = [("readme_2fields_short", SHORT, PRIORITY), ("support_6fields", SUPPORT, SIX), ("claim_1field_long", LONG, CLAIM), ("claim_1field_short", SHORT_CLAIM, CLAIM)]
t0 = time.perf_counter(); s = PrefixCachedScorer(**CONFIG); torch.cuda.synchronize(); load = time.perf_counter() - t0
print(f"load {load:.1f}s", flush=True)
res = {"load_seconds": load, "runtime": s.runtime, "repeats": REPEATS, "cases": []}
for name, ctx, schema in CASES:
    row = {"case": name, "fields": len(schema), "modes": {}}
    ref = None
    for mode in ("independent", "cached_serial", "parallel"):
        r = s.score(ctx, schema, mode=mode); times = []
        for _ in range(REPEATS):
            r = s.score(ctx, schema, mode=mode); times.append(r["metrics"]["total_seconds"])
        st = stats(times)
        entry = {"seconds": st, "output": r["output"], "peak_gib": r["metrics"]["cuda_peak_active_gib"],
                 "prefix_tokens": r["metrics"].get("prefix_tokens"), "suffix_tokens": r["metrics"].get("suffix_tokens")}
        if ref is None: ref = r
        else:
            dl = max(abs(r["fields"][f]["logits"][k] - ref["fields"][f]["logits"][k]) for f in r["fields"] for k in r["fields"][f]["logits"])
            dp = max(abs(r["fields"][f]["scores"][k] - ref["fields"][f]["scores"][k]) for f in r["fields"] for k in r["fields"][f]["scores"])
            entry["vs_independent"] = {"max_logit_diff": dl, "max_prob_diff": dp, "same_choices": r["output"] == ref["output"]}
        row["modes"][mode] = entry
        print(f"{name} {mode:>13}: median={st['median']:.3f}s p95={st['p95']:.3f}s {entry.get('vs_independent','')} -> {r['output']}", flush=True)
    res["cases"].append(row)
n, t0 = 0, time.perf_counter()
while time.perf_counter() - t0 < 30: s.score(SHORT, PRIORITY, mode="parallel"); n += 1
el = time.perf_counter() - t0; res["sustained_parallel_readme"] = {"decisions": n, "seconds": el, "decisions_per_second": n / el}
print(f"sustained parallel: {n/el:.2f} dec/s", flush=True)
OUT.write_text(json.dumps(res, indent=2) + "\n")
