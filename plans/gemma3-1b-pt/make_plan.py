#!/usr/bin/env python3
"""The fp16fix plan for Gemma 3 1B (base model, unsloth/gemma-3-1b-pt = google/gemma-3-1b-pt), scored on held-out
prompts against float32 and against the published fix, transformers RFC #37226 (closed unmerged): scale the
embedding and every post_*_layernorm weight by 1/64, so the residual stream shrinks and the pre-norms cancel it.

Calibrates on prompts.json's "calibration" prompts (each with its 48-token float32 greedy continuation), writes
plan.json, then on the "held_out" prompts compares plain float16, float16 + plan and float16 + the RFC fix with
float32: teacher-forced (the float32 continuation fed in: top-1 agreement and logits relative error) and free-running
(48 greedy tokens: how many match float32's before the first difference).

    make_plan.py --model GEMMA_1B_DIR [--out-dir .]
Run it in the environment that will use the plan: a plan names exact operations, and transformers versions differ.
"""
import argparse, copy, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from fp16fix import Recorder, Plan, Applied, save_fixes

RFC_SCALE = 1 / 64
NEW = 48

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out-dir", default=HERE)
a = ap.parse_args()
P = json.load(open(os.path.join(HERE, "prompts.json")))
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager",
                                             device_map={"": 0}).eval()
model.generation_config.max_length = None                   # max_new_tokens decides (silences a warning)


def rfc_fix(m, s=RFC_SCALE):
    """RFC #37226 on a float32 copy, before casting: embedding x s, post-norm outputs x s ((1 + w) -> (1 + w) s)."""
    emb = m.model.embed_tokens
    emb.embed_scale = emb.embed_scale * s
    for layer in m.model.layers:
        for norm in (layer.post_attention_layernorm, layer.post_feedforward_layernorm):
            norm.weight.data = (1.0 + norm.weight.data) * s - 1.0
    return m


def greedy(m, ids, n=NEW):
    return m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, min_new_tokens=n,
                      do_sample=False, pad_token_id=tok.pad_token_id)


def logits(m, seqs):
    return [m(input_ids=s).logits.float() for s in seqs]


def timed(f):
    f(); torch.cuda.synchronize(); t = time.time(); f(); torch.cuda.synchronize()
    return time.time() - t


ids = {p: tok(p, return_tensors="pt").input_ids.cuda() for p in P["calibration"] + P["held_out"]}
with torch.no_grad():
    seqs = {p: greedy(model, x) for p, x in ids.items()}                  # float32 continuations
    cal, held = [seqs[p] for p in P["calibration"]], [seqs[p] for p in P["held_out"]]
    with Recorder(model) as r32:
        logits(model, cal)
    ref = logits(model, held)
    t32 = timed(lambda: logits(model, held))
    rfc = rfc_fix(copy.deepcopy(model)).half()
    model.half()
    with Recorder(model) as r16:
        logits(model, cal)
    plan = Plan(r32, r16)
    save_fixes(plan.fixes, os.path.join(a.out_dir, "plan.json"), model=model)
    out = {"plain_fp16": logits(model, held), "rfc_fp16": logits(rfc, held)}
    gen = {"plain_fp16": [greedy(model, ids[p]) for p in P["held_out"]], "rfc_fp16": [greedy(rfc, ids[p]) for p in P["held_out"]]}
    t_rfc = timed(lambda: logits(rfc, held))
    with Applied(model, plan.fixes):
        out["plan_fp16"] = logits(model, held)
        gen["plan_fp16"] = [greedy(model, ids[p]) for p in P["held_out"]]
        t_plan = timed(lambda: logits(model, held))
print(plan.describe())


def metrics(o, r, s, g):
    n0 = s.shape[1] - NEW                                           # the prompt's length
    a = slice(n0 - 1, s.shape[1] - 1)                               # positions that predict the continuation
    gt, ref_t = g[0, n0:], s[0, n0:]
    same = int((gt == ref_t).cumprod(0).sum()) if gt.shape == ref_t.shape else 0
    if not torch.isfinite(o[0, a]).all():
        return {"finite": False, "free_run_same_tokens": same}
    return {"finite": True, "top1": (o[0, a].argmax(-1) == r[0, a].argmax(-1)).float().mean().item(),
            "rel_err": ((o[0, a] - r[0, a]).norm() / r[0, a].norm()).item(), "free_run_same_tokens": same}


KINDS = ("plain_fp16", "plan_fp16", "rfc_fp16")
rows = [{"prompt": p, **{k: metrics(out[k][i], ref[i], held[i], gen[k][i]) for k in KINDS}} for i, p in enumerate(P["held_out"])]


def agg(k):
    ok = [r[k] for r in rows if r[k]["finite"]]
    e = sorted(x["rel_err"] for x in ok)
    return {"n": len(rows), "finite": len(ok), "top1_mean": sum(x["top1"] for x in ok) / len(ok) if ok else None,
            "rel_err_median": e[len(e) // 2] if e else None, "rel_err_max": e[-1] if e else None,
            "free_run_all_48_same": sum(r[k]["free_run_same_tokens"] == NEW for r in rows),
            "free_run_same_tokens_mean": sum(r[k]["free_run_same_tokens"] for r in rows) / len(rows)}


res = {"model": "unsloth/gemma-3-1b-pt", "transformers": transformers.__version__, "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(), "calibration_prompts": len(cal), "new_tokens": NEW, "rfc_scale": RFC_SCALE,
       "held_out_summary": {k: agg(k) for k in KINDS}, "time_s_held_out_teacher_forced": {"fp32": t32, "plan_fp16": t_plan, "rfc_fp16": t_rfc},
       "held_out": rows, "plan": plan.describe()}
json.dump(res, open(os.path.join(a.out_dir, "results.json"), "w"), indent=1, ensure_ascii=False)
for k in KINDS:
    print(k, json.dumps(res["held_out_summary"][k]))
print("time (12 held-out teacher-forced passes):", json.dumps(res["time_s_held_out_teacher_forced"]))
