#!/usr/bin/env python3
"""An unseen model with a known fix: Gemma 3 4B-it (text only). Known: Gemma 3 overflows float16 because its
embeddings are multiplied by sqrt(hidden_size) and its post-norm weights are large; transformers RFC #37226 scaled the
embedding and the post_*_layernorm weights down (the pre-norms and the final norm make that exact).
Same check as the 5 Hz LM: teacher-force each prompt + a 48-token float32 greedy continuation through float32, plain
float16 and float16 with fp16fix's plan; top-1 agreement and logits relative error vs float32."""
CWD = __import__("os").getcwd()
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, ".."), HERE]
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from fp16fix import Recorder
from fp16fix.plan import Plan
from fp16fix.apply import Applied
from common import show
M = os.environ.get("GEMMA_MODEL", "/mnt/nvme/models/gemma-3-4b-it"); out = os.path.join(CWD, sys.argv[1])
tok = AutoTokenizer.from_pretrained(M)
model = Gemma3ForConditionalGeneration.from_pretrained(M, dtype=torch.float32, attn_implementation="eager", device_map={"": 0}).eval()
prompts = ["Explain in three sentences why the sky is blue.",
           "Write a short Python function that checks whether a string is a palindrome.",
           "What are the main causes of the French Revolution? Answer briefly.",
           "Translate into French: The train leaves at seven in the morning.",
           "Give me a haiku about an old lighthouse."]
ids = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt").to(0)
       for p in prompts]
with torch.no_grad():
    seqs = [model.generate(input_ids=x, max_new_tokens=48, do_sample=False) for x in ids]
    run = lambda: [model(input_ids=s).logits.float() for s in seqs]
    t0 = time.time(); ref = run(); t32 = time.time() - t0
    with Recorder(model) as r32:
        run()
    model.half()
    with Recorder(model) as r16:
        stock = run()
    plan = Plan(r32, r16)
    with Applied(model, plan.fixes):
        fixed = run()
        torch.cuda.synchronize(); t0 = time.time(); run(); torch.cuda.synchronize(); tplan = time.time() - t0
show(r32, "Gemma 3 4B-it (text)", out.replace(".json", "-find.json"), half=r16, top=30)
rows = []
for k, (x, s) in enumerate(zip(ids, seqs)):
    a = slice(x.shape[1] - 1, s.shape[1] - 1)
    def m(o):
        if not torch.isfinite(o[0, a]).all():
            return {"finite": False}
        return {"finite": True, "top1": (o[0, a].argmax(-1) == ref[k][0, a].argmax(-1)).float().mean().item(),
                "rel_err": ((o[0, a] - ref[k][0, a]).norm() / ref[k][0, a].norm()).item()}
    rows.append({"prompt": prompts[k], "stock": m(stock[k]), "plan": m(fixed[k])})
    fmt = lambda d: f"top-1 {100 * d['top1']:.0f}%, logits rel. error {d['rel_err']:.2e}" if d["finite"] else "NaN/inf"
    print(f"{prompts[k][:40]:40s} | plain float16: {fmt(rows[-1]['stock'])} | with plan: {fmt(rows[-1]['plan'])}")
print(f"time for the 5 teacher-forced passes: float32 {t32:.2f} s, float16 + plan {tplan:.2f} s")
json.dump({"plan": plan.describe(), "rows": rows, "time_s": {"fp32": t32, "fp16_plan": tplan}}, open(out, "w"), indent=1)
