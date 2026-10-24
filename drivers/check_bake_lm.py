#!/usr/bin/env python3
"""Issue #16 on a causal LM: the plan applied with hooks (Applied) vs baked into the weights (bake), both in float16,
against float32 -- accuracy on held-out prompts and speed (teacher-forced passes and 48-token greedy decoding, each
the median of 5 runs after a warm-up). The plan is loaded (--plan) or made from the calibration prompts.

    check_bake_lm.py --model DIR --prompts prompts.json [--plan plan.json] --out OUT.json
"""
import argparse, copy, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "..")]
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from fp16fix import Recorder, Plan, Applied, load_fixes
from fp16fix.bake import bake

NEW, REPS = 48, 5
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--prompts", required=True)
ap.add_argument("--plan"); ap.add_argument("--out", required=True)
a = ap.parse_args()
P = json.load(open(a.prompts))
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager",
                                             device_map={"": 0}).eval()
model.generation_config.max_length = None


def greedy(m, x):
    return m.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=NEW, min_new_tokens=NEW,
                      do_sample=False, pad_token_id=tok.pad_token_id)


def reps(f):
    f(); ts = []
    for _ in range(REPS):
        torch.cuda.synchronize(); t = time.time(); f(); torch.cuda.synchronize(); ts.append(time.time() - t)
    return sorted(ts)[REPS // 2]


def scores(m, seqs):
    return [m(input_ids=s).logits.float() for s in seqs]


def compare(outs, gens, ref, seqs):
    top1, err, same = [], [], 0
    for o, g, r, s in zip(outs, gens, ref, seqs):
        n0 = s.shape[1] - NEW
        sl = slice(n0 - 1, s.shape[1] - 1)
        if not torch.isfinite(o[0, sl]).all():
            return {"finite": False}
        top1.append((o[0, sl].argmax(-1) == r[0, sl].argmax(-1)).float().mean().item())
        err.append(((o[0, sl] - r[0, sl]).norm() / r[0, sl].norm()).item())
        same += bool((g[0, n0:] == s[0, n0:]).all())
    e = sorted(err)
    return {"finite": True, "top1": sum(top1) / len(top1), "rel_err_median": e[len(e) // 2], "rel_err_max": e[-1],
            "greedy_48_identical": same}


ids = {p: tok(p, return_tensors="pt").input_ids.cuda() for p in P["calibration"] + P["held_out"]}
held_ids = [ids[p] for p in P["held_out"]]
res = {"model": os.path.basename(a.model.rstrip("/")), "transformers": transformers.__version__, "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(), "reps": REPS, "time_s": {}}
with torch.no_grad():
    seqs = [greedy(model, x) for x in held_ids]
    ref = scores(model, seqs)
    res["time_s"]["fp32"] = {"teacher_forced": reps(lambda: scores(model, seqs)), "decode_48": reps(lambda: greedy(model, held_ids[0]))}
    cal = [greedy(model, ids[p]) for p in P["calibration"]] if not a.plan else None
    if cal is not None:
        with Recorder(model) as r32:
            scores(model, cal)
    model.half()                                          # in place: a 4B model in float32 and float16 copies won't fit
    if a.plan:
        fixes = load_fixes(a.plan, model=model)
    else:
        with Recorder(model) as r16:
            scores(model, cal)
        fixes = Plan(r32, r16).fixes
    res["time_s"]["plain_fp16"] = {"teacher_forced": reps(lambda: scores(model, seqs)), "decode_48": reps(lambda: greedy(model, held_ids[0]))}
    with Applied(model, fixes):
        out_a, gen_a = scores(model, seqs), [greedy(model, x) for x in held_ids]
        res["time_s"]["fp16_applied"] = {"teacher_forced": reps(lambda: scores(model, seqs)), "decode_48": reps(lambda: greedy(model, held_ids[0]))}
    baked = copy.deepcopy(model)
    del model
    torch.cuda.empty_cache()
    t0 = time.time()
    rep = bake(baked, fixes, lambda m: m(input_ids=seqs[0]))
    res["bake_s"] = time.time() - t0
    print(rep.describe())
    left = Applied(baked, rep.leftover) if rep.leftover else None
    if left:
        left.__enter__()
    out_b, gen_b = scores(baked, seqs), [greedy(baked, x) for x in held_ids]
    res["time_s"]["fp16_baked"] = {"teacher_forced": reps(lambda: scores(baked, seqs)), "decode_48": reps(lambda: greedy(baked, held_ids[0]))}
    if left:
        left.__exit__(None, None, None)
res["fp16_applied"] = compare(out_a, gen_a, ref, seqs)
res["fp16_baked"] = compare(out_b, gen_b, ref, seqs)
res["baked_vs_applied_rel_diff_max"] = max(((b - x).norm() / x.norm()).item() for b, x in zip(out_b, out_a))
res["bake"] = {"baked": [list(b) for b in rep.baked], "leftover_rules": [f["rule"] for f in rep.leftover],
               "island_hooks": len(rep.hooks) // 2, "describe": rep.describe()}
json.dump(res, open(a.out, "w"), indent=1)
for k in ("fp16_applied", "fp16_baked"):
    print(k, json.dumps(res[k]))
print("baked vs applied, max rel. diff:", res["baked_vs_applied_rel_diff_max"])
print("time:", json.dumps(res["time_s"]))
