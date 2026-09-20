"""Benchmark openJev-verdict-2.0 (ModernBERT-base + GLiClass) on the Orin: torch CUDA and torch CPU."""
import json, platform, statistics, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "verdict"))
from core.engine_encoder import DecisionEngine
from core.primitives import Choice, Option, Noul, Score, Level
from core.calibration import TemperatureCalibrator

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(f"bench-verdict-{DEVICE}.json")
CKPT = Path(__file__).resolve().parents[1] / "third_party" / "verdict" / "artifacts" / "v2"

def stats(v):
    o = sorted(v); i = (len(o) - 1) * 0.95; lo = int(i); hi = min(lo + 1, len(o) - 1)
    return {"min": min(o), "median": statistics.median(o), "mean": statistics.mean(o),
            "p95": o[lo] + (o[hi] - o[lo]) * (i - lo), "max": max(o)}

def noul(i, p): return Noul(id=i, proposition=p, semantics="conditional_on_sufficient_evidence_v2")

readme_q = [Choice(id="priority", question="Urgency based on current business impact.",
                   options=[Option(id="HIGH", description="A critical business operation is currently blocked."),
                            Option(id="LOW", description="An optional enhancement with no current business impact.")]),
            noul("requires_review", "Customers are unable to complete a purchase.")]
six_q = [Choice(id="route", question="Which team should handle this message?",
                options=[Option(id=k, description=d) for k, d in [("BILLING", "Billing and payments"), ("TECHNICAL", "Technical bugs and outages"),
                         ("SALES", "Sales and upgrades"), ("ABUSE", "Abuse or policy violation"), ("OTHER", "Anything else")]]),
         Choice(id="sentiment", question="Overall tone of the writer toward the company.",
                options=[Option(id="POSITIVE", description="Positive"), Option(id="NEUTRAL", description="Neutral"), Option(id="NEGATIVE", description="Negative")]),
         Score(id="severity", question="Ordered severity of the reported problem.",
               levels=[Level(id="S1", description="Service unusable for all customers.", value=1), Level(id="S2", description="Major feature broken for many customers.", value=2),
                       Level(id="S3", description="Minor feature broken or workaround exists.", value=3), Level(id="S4", description="Cosmetic or documentation issue.", value=4)]),
         noul("mentions_refund", "The writer asks for money back."),
         noul("is_customer", "The writer states they are a paying customer."),
         noul("needs_human", "A human must reply rather than an automated answer.")]
claim_q = [Choice(id="supported", question="Is the claim in the final sentence supported by the preceding context?",
                  options=[Option(id="SUPPORTED", description="The claim is supported by the context."),
                           Option(id="NOT_SUPPORTED", description="The claim is not supported by the context.")])]
short_ctx = "The payment service is down for all customers."
support_msg = ("Hi, I have been a paying customer on the Pro plan for three years. Since this morning the export "
               "button on the reports page returns a 500 error every time, so my whole finance team cannot close "
               "the month. I have tried three browsers. Honestly I am furious; this is the second time this quarter. "
               "If this is not fixed today I want a refund for this month and I want to talk to an actual person, not a bot.")
filler = ("Section {n}. The regional office reported that quarterly maintenance was completed on schedule, "
          "with no outstanding defects noted by the inspection team, and the log was countersigned by the duty manager. ")
long_ctx = "".join(filler.format(n=i) for i in range(28)) + \
    "Only Mira may authorize refunds for account 42. The sole authorization for this refund on account 42 was signed by Noah. Claim: the refund on account 42 is authorized."
short_claim = "Only Mira may authorize refunds for account 42. The sole authorization for this refund on account 42 was signed by Mira. Claim: the refund on account 42 is authorized."
CASES = [("readme_2fields_short", short_ctx, readme_q), ("support_6fields", support_msg, six_q),
         ("claim_1field_long", long_ctx, claim_q), ("claim_1field_short", short_claim, claim_q)]

t0 = time.perf_counter()
cal = TemperatureCalibrator.load(CKPT / "calibrator.json")
engine = DecisionEngine(model_name_or_path=str(CKPT), calibrator=cal, device=DEVICE)
if DEVICE == "cuda": torch.cuda.synchronize()
load_s = time.perf_counter() - t0
nparams = sum(p.numel() for p in engine.model.parameters())
print(f"load {load_s:.1f}s params {nparams/1e6:.1f}M device {DEVICE}", flush=True)
res = {"machine": "NVIDIA Jetson AGX Orin 64 GB (Tegra, sm_87)", "device": DEVICE, "torch": torch.__version__,
       "transformers": __import__("transformers").__version__, "params_million": nparams / 1e6,
       "load_seconds": load_s, "repeats": REPEATS,
       "timing_scope": "engine.evaluate() wall time incl. tokenisation, all fields in one batched forward; 3 warmups then REPEATS measured", "cases": []}
for name, ctx, qs in CASES:
    for _ in range(3): engine.evaluate(ctx, qs)
    times = []
    for _ in range(REPEATS):
        if DEVICE == "cuda": torch.cuda.synchronize()
        t = time.perf_counter(); r = engine.evaluate(ctx, qs)
        if DEVICE == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    ntok = len(engine.tokenizer(ctx)["input_ids"])
    st = stats(times)
    out = {}
    for d in r.results:
        if d.kind == "choice":
            out[d.id] = {"selected": d.selected_id, "p": round(d.selected_probability, 3), "abstain": d.is_abstention}
        elif d.kind == "score":
            out[d.id] = {"selected": d.selected_level_id, "expected": d.expected_score, "abstain": d.is_abstention}
        else:
            out[d.id] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.model_dump().items()
                         if k in ("probability", "true_probability", "p_true", "is_abstention", "selected_id", "value")}
    case = {"case": name, "fields": len(qs), "context_tokens": ntok, "seconds": st,
            "seconds_per_field_median": st["median"] / len(qs), "output": out}
    res["cases"].append(case)
    print(f"{name}: fields={len(qs)} ctx_tokens={ntok} median={st['median']*1000:.1f}ms p95={st['p95']*1000:.1f}ms -> {out}", flush=True)
if DEVICE == "cuda":
    res["cuda_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
n, t0 = 0, time.perf_counter()
while time.perf_counter() - t0 < 20:
    engine.evaluate(short_ctx, readme_q); n += 1
el = time.perf_counter() - t0
res["sustained_readme_case"] = {"seconds": el, "decisions": n, "decisions_per_second": n / el}
print(f"sustained: {n} two-field decisions in {el:.1f}s = {n/el:.1f}/s", flush=True)
OUT.write_text(json.dumps(res, indent=2) + "\n"); print("wrote", OUT)
