#!/usr/bin/env python3
"""A fairer comparison of the Gemma 3 1B plan with RFC #37226 (Grok's review): the RFC's 1/64 pushes small values
toward float16's less precise range, so it is also run with the plan's own factor (1/8) and with ~1/sqrt(hidden) =
1/34 (the size the RFC's author suggests), and everything is scored on a longer text too.

- held out: the 12 prompts of make_plan.py (teacher-forced top-1 / logits error, free-running 48 greedy tokens);
- perplexity and KL divergence vs float32 on WikiText-2 test, CHUNKS non-overlapping windows of CTX tokens;
- speed: the 12 teacher-forced passes and a 48-token greedy decode, each repeated REPS times (median and min).

    fair_check.py --model GEMMA_1B_DIR --wiki wiki.test.raw [--out fair.json]
"""
import argparse, copy, json, math, os, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from fp16fix import Applied, load_fixes

NEW, CTX, CHUNKS, REPS = 48, 1024, 40, 5
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--wiki", required=True)
ap.add_argument("--out", default=os.path.join(HERE, "fair.json"))
a = ap.parse_args()
P = json.load(open(os.path.join(HERE, "prompts.json")))
tok = AutoTokenizer.from_pretrained(a.model)
f32 = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager", device_map={"": 0}).eval()
f32.generation_config.max_length = None


def rfc(s):
    """RFC #37226 as a weight edit on a float32 copy, then float16: embedding x s, post-norm outputs x s."""
    m = copy.deepcopy(f32)
    m.model.embed_tokens.embed_scale = m.model.embed_tokens.embed_scale * s
    for layer in m.model.layers:
        for norm in (layer.post_attention_layernorm, layer.post_feedforward_layernorm):
            norm.weight.data = (1.0 + norm.weight.data) * s - 1.0
    return m.half()


plan_model = copy.deepcopy(f32).half()
fixes = load_fixes(os.path.join(HERE, "plan.json"), model=plan_model)
variants = {"rfc_1/8": rfc(1 / 8), "rfc_1/34": rfc(1 / 34), "rfc_1/64": rfc(1 / 64), "plan": plan_model}


def with_plan(name):
    return Applied(plan_model, fixes) if name == "plan" else _null()


class _null:
    def __enter__(self): return self
    def __exit__(self, *e): return False


def greedy(m, x):
    return m.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=NEW, min_new_tokens=NEW,
                      do_sample=False, pad_token_id=tok.pad_token_id)


def reps(f):
    f(); ts = []
    for _ in range(REPS):
        torch.cuda.synchronize(); t = time.time(); f(); torch.cuda.synchronize(); ts.append(time.time() - t)
    return {"median": statistics.median(ts), "min": min(ts)}


text = open(a.wiki).read()
wiki = tok(text, return_tensors="pt").input_ids[0]
chunks = [wiki[i * CTX:(i + 1) * CTX][None].cuda() for i in range(CHUNKS)]
ids = [tok(p, return_tensors="pt").input_ids.cuda() for p in P["held_out"]]
res = {"model": "unsloth/gemma-3-1b-pt", "transformers": transformers.__version__, "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(), "wiki_tokens": CHUNKS * CTX, "reps": REPS, "results": {}}
with torch.no_grad():
    seqs = [greedy(f32, x) for x in ids]
    ref = [f32(input_ids=s).logits.float() for s in seqs]
    nll32, ref_lp = 0.0, []
    for c in chunks:                                     # float32 log-probs, kept on the CPU in float16 to fit
        lp = F.log_softmax(f32(input_ids=c).logits[0, :-1].float(), -1)
        nll32 += -lp.gather(-1, c[0, 1:, None]).sum().item()
        ref_lp.append(lp.half().cpu())
    n_tok = CHUNKS * (CTX - 1)
    res["results"]["fp32"] = {"ppl": math.exp(nll32 / n_tok),
                              "time_teacher_forced": reps(lambda: [f32(input_ids=s) for s in seqs]),
                              "time_decode_48": reps(lambda: greedy(f32, ids[0]))}
    for name, m in variants.items():
        with with_plan(name):
            top1, err, same = [], [], []
            for x, s, r in zip(ids, seqs, ref):
                o = m(input_ids=s).logits.float()
                n0 = s.shape[1] - NEW
                sl = slice(n0 - 1, s.shape[1] - 1)
                top1.append((o[0, sl].argmax(-1) == r[0, sl].argmax(-1)).float().mean().item())
                err.append(((o[0, sl] - r[0, sl]).norm() / r[0, sl].norm()).item())
                g = greedy(m, x)
                same.append(int((g[0, n0:] == s[0, n0:]).cumprod(0).sum()))
            nll, kl = 0.0, 0.0
            for c, rlp in zip(chunks, ref_lp):
                lp = F.log_softmax(m(input_ids=c).logits[0, :-1].float(), -1)
                nll += -lp.gather(-1, c[0, 1:, None]).sum().item()
                r = rlp.cuda().float()
                kl += (r.exp() * (r - lp)).sum().item()
            e = sorted(err)
            res["results"][name] = {
                "held_out_top1": sum(top1) / len(top1), "held_out_rel_err_median": e[len(e) // 2], "held_out_rel_err_max": e[-1],
                "free_run_all_48_same": sum(x == NEW for x in same), "free_run_same_tokens_mean": sum(same) / len(same),
                "ppl": math.exp(nll / n_tok), "kl_mean": kl / n_tok,
                "time_teacher_forced": reps(lambda: [m(input_ids=s) for s in seqs]),
                "time_decode_48": reps(lambda: greedy(m, ids[0]))}
        print(name, json.dumps(res["results"][name]), flush=True)
print("fp32", json.dumps(res["results"]["fp32"]))
json.dump(res, open(a.out, "w"), indent=1)
