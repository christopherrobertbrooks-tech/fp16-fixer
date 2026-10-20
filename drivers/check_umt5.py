#!/usr/bin/env python3
"""An unseen model: Wan 2.2's umT5-XXL text encoder (Studio runs it in float32 "because T5 can overflow"). Prompts are
encoded the way Studio / diffusers' WanPipeline do (padded to 512). Float32 reference vs plain float16 (transformers
clamps over-range values to +-65,504 in float16, so it doesn't NaN -- it clips) vs float16 with fp16fix's plan.
Compared over the real tokens: relative error and cosine similarity of the encoder output."""
CWD = __import__("os").getcwd()
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "..")]
import torch
from transformers import AutoTokenizer, UMT5EncoderModel
from fp16fix import Recorder
from fp16fix.plan import Plan
from fp16fix.apply import Applied
from common import show
M = "/mnt/nvme/models/video/Wan2.2-TI2V-5B-Diffusers"; out = os.path.join(CWD, sys.argv[1])
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
            "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
            "三条腿，背景人很多，倒着走")
prompts = ["A small orange robot walks through a rainy neon city at night, reflections on the wet street, cinematic.",
           "A golden retriever runs along a beach at sunset, waves crashing, slow motion, warm light.",
           "Close-up of hands kneading bread dough on a floured wooden table, morning light from a window.",
           "An astronaut floats inside a space station, Earth visible through the window, gentle camera drift.",
           "Time-lapse of clouds rolling over snowy mountain peaks, dramatic shadows, ultra detailed.",
           "一只熊猫在竹林里吃竹子，阳光透过树叶，高清，自然纪录片风格。", NEGATIVE]
tok = AutoTokenizer.from_pretrained(M, subfolder="tokenizer")
te = UMT5EncoderModel.from_pretrained(M, subfolder="text_encoder", dtype=torch.float32, device_map={"": 0}).eval()
batch = [tok([p], padding="max_length", max_length=512, truncation=True, return_attention_mask=True, return_tensors="pt") for p in prompts]
run = lambda m: [m(b.input_ids.cuda(), b.attention_mask.cuda()).last_hidden_state[0, :int(b.attention_mask.sum())].float().cpu() for b in batch]
def timed(m, reps=3):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(reps):
        run(m)
    torch.cuda.synchronize(); return (time.time() - t) / reps
with torch.no_grad():
    t32 = timed(te); mem32 = torch.cuda.max_memory_allocated() / 2**30
    with Recorder(te) as r32:
        ref = run(te)
    te.half()                                   # in place: two copies of a 22 GB float32 model would not fit
    with Recorder(te) as r16:
        stock = run(te)
    plan = Plan(r32, r16)
    for f in plan.fixes:
        if "stream" in f["rule"]:
            print("STREAM FIX:", f["rule"], "| readers:", f.get("readers"), "| keep32:", f.get("readers_keep_float32"),
                  "| first_value:", f.get("first_value"), flush=True)
    torch.cuda.reset_peak_memory_stats()
    with Applied(te, plan.fixes):
        fixed = run(te)
        tplan = timed(te); memplan = torch.cuda.max_memory_allocated() / 2**30
    tstock = timed(te)
show(r32, "umT5-XXL (Wan 2.2) text encoder", out.replace(".json", "-find.json"), half=r16, top=30)
rows = []
for k, p in enumerate(prompts):
    def m(o):
        if not torch.isfinite(o).all():
            return {"finite": False}
        return {"finite": True, "rel_err": ((o - ref[k]).norm() / ref[k].norm()).item(),
                "cos": torch.nn.functional.cosine_similarity(o.flatten(), ref[k].flatten(), 0).item()}
    rows.append({"prompt": p[:40], "tokens": ref[k].shape[0], "stock_fp16": m(stock[k]), "plan_fp16": m(fixed[k])})
    s, f = rows[-1]["stock_fp16"], rows[-1]["plan_fp16"]
    fmt = lambda d: f"rel. error {d['rel_err']:.2e}, cosine {d['cos']:.5f}" if d["finite"] else "NaN/inf"
    print(f"{p[:34]:34s} ({ref[k].shape[0]:3d} tok) | plain float16 (clamped): {fmt(s)} | with plan: {fmt(f)}")
print(f"7 prompts: float32 {t32:.3f} s, {mem32:.1f} GB peak | float16 + plan {tplan:.3f} s, {memplan:.1f} GB peak | "
      f"plain float16 {tstock:.3f} s")
json.dump({"plan": plan.describe(), "rows": rows, "time_s": {"fp32": t32, "fp16_plan": tplan, "fp16_stock": tstock},
           "peak_gb": {"fp32": mem32, "fp16_plan": memplan}}, open(out, "w"), indent=1)
