"""Nimble CUDA scorer with shared-prefix caching (the trick the Mac ParallelScorer uses).

modes: independent (upstream: full prompt per field), cached_serial (one prefix prefill, then each
field's suffix on a copy of the cache), parallel (one prefill, cache expanded to N rows, all suffixes
right-padded in one forward; hidden read at each row's last real token, so pads never feed a result).
"""
import copy, hashlib, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "nimble"))
from nimble.scoring.cuda_scorer import CudaCandidateScorer, candidate_projection
from nimble.scoring.parallel_schema import choice_key
from transformers import DynamicCache


class PrefixCachedScorer(CudaCandidateScorer):
    @torch.inference_mode()
    def score(self, context, schema, mode="auto"):
        if mode == "auto":
            # One forward pass costs ~0.13 s on the Orin regardless of length, so a single field is
            # cheapest as one full-prompt pass; two or more fields share one prefill plus one batched pass.
            mode = "independent" if len(schema) == 1 else "parallel"
        if mode == "independent":
            return super().score(context, schema, mode="independent")
        prepared = self.prepare(context, schema)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        prefix = torch.tensor([prepared.prefix_ids], device=self.device)
        cache = DynamicCache(config=self.model.config)
        self.backbone(input_ids=prefix, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize(); prefill_s = time.perf_counter() - started
        hiddens = []
        if mode == "cached_serial":
            for ids in prepared.suffix_ids:
                branch = copy.deepcopy(cache)
                tokens = torch.tensor([ids], device=self.device)
                out = self.backbone(input_ids=tokens, past_key_values=branch, use_cache=True)
                hiddens.append(out.last_hidden_state[0, -1, :])
                del branch, out
        elif mode == "parallel":
            n = len(prepared.suffix_ids); width = max(map(len, prepared.suffix_ids))
            pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            rows = [ids + [pad] * (width - len(ids)) for ids in prepared.suffix_ids]
            last = torch.tensor([len(ids) - 1 for ids in prepared.suffix_ids], device=self.device)
            branch = copy.deepcopy(cache)
            branch.reorder_cache(torch.zeros(n, dtype=torch.long, device=self.device))  # expand batch 1 -> n
            tokens = torch.tensor(rows, device=self.device)
            out = self.backbone(input_ids=tokens, past_key_values=branch, use_cache=True)
            h = out.last_hidden_state
            hiddens = [h[i, last[i], :] for i in range(n)]
            del branch, out
        else:
            raise ValueError(mode)
        fields, output = {}, {}
        for name, choices, ids, candidates, hidden in zip(prepared.names, prepared.choices, prepared.full_ids,
                                                          prepared.candidate_ids, hiddens):
            logits = candidate_projection(hidden[None, :], self.head_weight, candidates)[0]
            if not torch.isfinite(logits).all():
                raise ValueError("non-finite logits")
            probs = torch.softmax(logits / self.temperature, dim=-1)
            best = logits.argmax().item(); keys = [choice_key(v) for v in choices]
            output[name] = choices[best]
            fields[name] = {"value": choices[best], "scores": dict(zip(keys, probs.tolist())),
                            "logits": dict(zip(keys, logits.tolist())), "prompt_token_count": len(ids)}
        torch.cuda.synchronize()
        return {"model": self.model_id, "backend": "cuda", "output": output, "fields": fields,
                "metrics": {"mode": mode, "fields": len(fields), "prefix_tokens": len(prepared.prefix_ids),
                            "suffix_tokens": list(map(len, prepared.suffix_ids)),
                            "prefill_seconds": prefill_s, "total_seconds": time.perf_counter() - started,
                            "cuda_peak_active_gib": torch.cuda.max_memory_allocated() / 2**30}}
