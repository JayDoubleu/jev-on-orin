import copy, json, time, torch
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.bench_common import *
from nimble_orin.cuda_prefix_scorer import PrefixCachedScorer
from transformers import DynamicCache
s = PrefixCachedScorer(**json.loads(Path("nimble-model.json").read_text()))
CLAIM = {"supported": {"type": "enum", "choices": ["SUPPORTED", "NOT_SUPPORTED"], "description": "Whether the claim in the final sentence is supported by the preceding context."}}
p = s.prepare(SHORT_CLAIM, CLAIM)
prefix = torch.tensor([p.prefix_ids], device=s.device); full = torch.tensor([p.full_ids[0]], device=s.device); suf = torch.tensor([p.suffix_ids[0]], device=s.device)
def t(fn, n=5):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(n):
        torch.cuda.synchronize(); a = time.perf_counter(); r = fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - a)
    return sorted(ts)[len(ts)//2], r
with torch.inference_mode():
    print("prefix", len(p.prefix_ids), "suffix", len(p.suffix_ids[0]), "full", len(p.full_ids[0]))
    print("full no-cache        %.3f" % t(lambda: s.backbone(input_ids=full, use_cache=False))[0])
    print("prefix no-cache      %.3f" % t(lambda: s.backbone(input_ids=prefix, use_cache=False))[0])
    print("prefix with cache    %.3f" % t(lambda: s.backbone(input_ids=prefix, past_key_values=DynamicCache(config=s.model.config), use_cache=True))[0])
    cache = DynamicCache(config=s.model.config); s.backbone(input_ids=prefix, past_key_values=cache, use_cache=True)
    print("deepcopy cache       %.3f" % t(lambda: copy.deepcopy(cache))[0])
    print("suffix with cache    %.3f" % t(lambda: s.backbone(input_ids=suf, past_key_values=copy.deepcopy(cache), use_cache=True))[0])
    one = suf[:, :1]
    print("1 token with cache   %.3f" % t(lambda: s.backbone(input_ids=one, past_key_values=copy.deepcopy(cache), use_cache=True))[0])
    # where does suffix time go: profile top ops
    from torch.profiler import profile, ProfilerActivity
    b = copy.deepcopy(cache)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        s.backbone(input_ids=suf, past_key_values=b, use_cache=True); torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=60))
