#!/usr/bin/env python3
"""Step 4 on ACE-Step's 5 Hz LM: plan from a float32 + float16 pass over the prompts, then teacher-force each prompt
plus a 48-token float32 greedy continuation through float32, plain float16 and float16 with the plan: top-1 agreement
with float32 on the continuation positions, and logits relative error.
Hand fix for comparison (D-nanovllm-fp16.md): float16 top-token agreement with float32 100% / 80% / 100%."""
CWD = __import__("os").getcwd()
import copy, json, os, sys
ROOT = "/mnt/steam/ace-step/ACE-Step-1.5"; HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..")]; os.chdir(ROOT)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fp16fix import Recorder
from fp16fix.plan import Plan
from fp16fix.apply import Applied
ckpt = sys.argv[1]; out = os.path.join(CWD, sys.argv[2])
tok = AutoTokenizer.from_pretrained(f"checkpoints/{ckpt}")
lm = AutoModelForCausalLM.from_pretrained(f"checkpoints/{ckpt}", dtype=torch.float32, attn_implementation="eager").cuda().eval()
prompts = ["Write a song plan: an upbeat indie pop song about summer road trips, 120 BPM, female vocals.",
           "Caption: dark ambient techno with deep bass and sparse vocals. Give the BPM, key and a structure.",
           "Make lyrics for a slow country ballad about an old dog, three verses and a chorus."]
ids = [tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True),
           return_tensors="pt").input_ids.cuda() for p in prompts]
with torch.no_grad():
    seqs = [lm.generate(x, max_new_tokens=48, do_sample=False) for x in ids]          # float32 greedy continuations
    with Recorder(lm) as r32:
        ref = [lm(input_ids=s).logits.float() for s in seqs]
    half = lm.half()        # in place: a float32 copy of a 4B model would not fit next to the original
    with Recorder(half) as r16:
        stock = [half(input_ids=s).logits.float() for s in seqs]
    plan = Plan(r32, r16)
    with Applied(half, plan.fixes):
        fixed = [half(input_ids=s).logits.float() for s in seqs]
print(plan.describe())
rows = []
for k, (x, s) in enumerate(zip(ids, seqs)):
    a = slice(x.shape[1] - 1, s.shape[1] - 1)        # positions that predict the continuation
    agree = lambda o: (o[0, a].argmax(-1) == ref[k][0, a].argmax(-1)).float().mean().item() if torch.isfinite(o).all() else float("nan")
    err = lambda o: ((o[0, a] - ref[k][0, a]).norm() / ref[k][0, a].norm()).item() if torch.isfinite(o).all() else float("nan")
    rows.append({"prompt": k, "stock_finite": bool(torch.isfinite(stock[k]).all()), "plan_top1": agree(fixed[k]), "plan_rel_err": err(fixed[k])})
    print(f"prompt {k}: plain float16 {'ok' if rows[-1]['stock_finite'] else 'NaN'} | with plan: top-1 agreement "
          f"{100 * rows[-1]['plan_top1']:.0f}%, logits relative error {rows[-1]['plan_rel_err']:.2e}")
json.dump({"plan": plan.describe(), "prompts": rows}, open(out, "w"), indent=1)
