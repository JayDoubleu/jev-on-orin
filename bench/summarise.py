import json, glob, sys
from pathlib import Path
RES = Path(__file__).resolve().parents[1] / "results"
rows = {"nimble-cuda": "Nimble 9B (torch fallback kernels)", "nimble-cuda-fla": "Nimble 9B (+fla/triton)", "nemotron-cuda": "Nemotron-Diffusion 14B",
        "kotoba-cuda": "Kotoba DeBERTa-v3-large 437M", "von-cuda": "von-1.0 ModernBERT-large 395M", "verdict-cuda": "Verdict 2.0 ModernBERT-base 151M",
        "kotoba-cpu": "Kotoba (CPU)", "von-cpu": "von (CPU)", "verdict-cpu": "Verdict (CPU)"}
print("| Model | load s | 2 fields short | 6 fields ~100 tok | 1 field ~1000 tok | 1 field short | sustained dec/s | GPU GiB |")
print("|---|---|---|---|---|---|---|---|")
for key, label in rows.items():
    try: d = json.load(open(RES / f"bench-{key}.json"))
    except FileNotFoundError: continue
    c = {x["case"]: x["seconds"]["median"] for x in d["cases"]}
    ms = lambda k: f"{c[k]*1000:.0f} ms" if c[k] < 10 else f"{c[k]:.1f} s"
    load = d.get("load_seconds") or d.get("load_plus_first_call_seconds")
    gib = d.get("cuda_peak_gib") or d.get("cuda_allocated_gib_after_load") or (d["cases"][0].get("cuda_peak_active_gib") if "cuda" in key else None)
    print(f"| {label} | {load:.0f} | {ms('readme_2fields_short')} | {ms('support_6fields')} | {ms('claim_1field_long')} | {ms('claim_1field_short')} | {d['sustained_readme_case']['decisions_per_second']:.1f} | {gib:.1f} |" if gib else
          f"| {label} | {load:.0f} | {ms('readme_2fields_short')} | {ms('support_6fields')} | {ms('claim_1field_long')} | {ms('claim_1field_short')} | {d['sustained_readme_case']['decisions_per_second']:.1f} | cpu |")
print()
print("Answers on the two claim cases (long: signed by Noah -> NOT_SUPPORTED is right; short: signed by Mira -> SUPPORTED is right):")
for key in ["nimble-cuda-fla", "nemotron-cuda", "kotoba-cuda", "von-cuda", "verdict-cuda"]:
    try: d = json.load(open(RES / f"bench-{key}.json"))
    except FileNotFoundError: continue
    o = {x["case"]: x["output"] for x in d["cases"]}
    print(f"  {rows[key]}: long={json.dumps(o['claim_1field_long'])[:120]} short={json.dumps(o['claim_1field_short'])[:120]}")
