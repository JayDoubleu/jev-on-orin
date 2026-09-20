from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TP = ROOT / "third_party"
HF = ROOT / "hf" / "hub"
import statistics
def stats(v):
    o = sorted(v); i = (len(o) - 1) * 0.95; lo = int(i); hi = min(lo + 1, len(o) - 1)
    return {"min": min(o), "median": statistics.median(o), "mean": statistics.mean(o),
            "p95": o[lo] + (o[hi] - o[lo]) * (i - lo), "max": max(o)}
SHORT = "The payment service is down for all customers."
SUPPORT = ("Hi, I have been a paying customer on the Pro plan for three years. Since this morning the export "
           "button on the reports page returns a 500 error every time, so my whole finance team cannot close "
           "the month. I have tried three browsers. Honestly I am furious; this is the second time this quarter. "
           "If this is not fixed today I want a refund for this month and I want to talk to an actual person, not a bot.")
_F = ("Section {n}. The regional office reported that quarterly maintenance was completed on schedule, "
      "with no outstanding defects noted by the inspection team, and the log was countersigned by the duty manager. ")
LONG = "".join(_F.format(n=i) for i in range(28)) + "Only Mira may authorize refunds for account 42. The sole authorization for this refund on account 42 was signed by Noah. Claim: the refund on account 42 is authorized."
SHORT_CLAIM = "Only Mira may authorize refunds for account 42. The sole authorization for this refund on account 42 was signed by Mira. Claim: the refund on account 42 is authorized."
ROUTE = {"BILLING": "Billing and payments", "TECHNICAL": "Technical bugs and outages", "SALES": "Sales and upgrades", "ABUSE": "Abuse or policy violation", "OTHER": "Anything else"}
SEV = ["S1 service unusable for all customers", "S2 major feature broken for many customers", "S3 minor feature broken or workaround exists", "S4 cosmetic or documentation issue"]
