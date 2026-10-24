#!/usr/bin/env python3
"""The fp16fix plan for Wan 2.2's umT5-XXL text encoder (Wan2.2-TI2V-5B-Diffusers/text_encoder), made the way it is
used: prompts encoded like diffusers' WanPipeline / wan_clip.py (padded to 512, the real tokens kept).

Calibrates on prompts.json's "calibration" prompts, writes plan.json, and checks the plan on prompts it never saw
("held_out", plus an optional --extra file of private prompts that are only reported in aggregate): relative error
and cosine vs float32 for plain float16 and float16 + plan.

    make_plan.py --model WAN_DIR [--extra private_prompts.json] [--out-dir .] [--rescale-stream]
The residual stream is kept in float32 (Plan(stream_float32=True)): rescaled in float16 it is no faster here and
less precise (5.4% worst held out vs 1.0%; see README). --rescale-stream makes the rescaled plan instead.
Run it in the environment that will use the plan: a plan names exact operations, and transformers versions differ.
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import torch
import transformers
from transformers import AutoTokenizer, UMT5EncoderModel
from fp16fix import Recorder
from fp16fix.plan import Plan, save_fixes
from fp16fix.apply import Applied

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--extra"); ap.add_argument("--out-dir", default=HERE)
ap.add_argument("--rescale-stream", action="store_true")
a = ap.parse_args()
P = json.load(open(os.path.join(HERE, "prompts.json")))
extra = json.load(open(a.extra)) if a.extra else []
tok = AutoTokenizer.from_pretrained(a.model, subfolder="tokenizer")
te = UMT5EncoderModel.from_pretrained(a.model, subfolder="text_encoder", dtype=torch.float32, device_map={"": 0}).eval()

def embed(text):
    t = tok([text], padding="max_length", max_length=512, truncation=True, add_special_tokens=True,
            return_attention_mask=True, return_tensors="pt")
    n = int(t.attention_mask.sum())
    return te(t.input_ids.cuda(), t.attention_mask.cuda()).last_hidden_state[0, :n].float().cpu()

def metrics(o, r):
    if not torch.isfinite(o).all():
        return {"finite": False}
    return {"finite": True, "rel_err": ((o - r).norm() / r.norm()).item(),
            "cos": torch.nn.functional.cosine_similarity(o.flatten(), r.flatten(), 0).item()}

with torch.no_grad():
    t0 = time.time()
    with Recorder(te) as r32:
        for p in P["calibration"]:
            embed(p)
    ref = {p: embed(p) for p in P["held_out"] + extra}
    te.half()
    with Recorder(te) as r16:
        for p in P["calibration"]:
            embed(p)
    stock = {p: embed(p) for p in ref}
    plan = Plan(r32, r16, stream_float32=not a.rescale_stream)
    save_fixes(plan.fixes, os.path.join(a.out_dir, "plan.json"))
    with Applied(te, plan.fixes):
        fixed = {p: embed(p) for p in ref}
print(plan.describe())
held = [{"prompt": p, "tokens": ref[p].shape[0], "plain_fp16": metrics(stock[p], ref[p]), "plan_fp16": metrics(fixed[p], ref[p])}
        for p in P["held_out"]]
ext = [{"plain_fp16": metrics(stock[p], ref[p]), "plan_fp16": metrics(fixed[p], ref[p])} for p in extra]

def agg(rows, key):
    e = [r[key]["rel_err"] for r in rows if r[key].get("finite")]
    c = [r[key]["cos"] for r in rows if r[key].get("finite")]
    return {"n": len(rows), "finite": len(e), "rel_err_max": max(e) if e else None, "rel_err_median": sorted(e)[len(e) // 2] if e else None,
            "cos_min": min(c) if c else None}

res = {"transformers": transformers.__version__, "torch": torch.__version__, "calibration_prompts": len(P["calibration"]),
       "held_out": held, "held_out_summary": {k: agg(held, k) for k in ("plain_fp16", "plan_fp16")},
       "extra_summary": {k: agg(ext, k) for k in ("plain_fp16", "plan_fp16")} if ext else None,
       "plan": plan.describe()}
json.dump(res, open(os.path.join(a.out_dir, "results.json"), "w"), indent=1, ensure_ascii=False)
for k in ("plain_fp16", "plan_fp16"):
    print(k, "held out:", json.dumps(res["held_out_summary"][k]), "| private extra:", json.dumps(res["extra_summary"][k]) if ext else "-")
