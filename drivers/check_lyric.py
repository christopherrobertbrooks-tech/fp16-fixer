#!/usr/bin/env python3
"""Step 4 on ACE-Step's lyric encoder: plan from a float32 + float16 pass, then per song compare float32 (reference),
plain float16, and float16 with the plan applied (relative error of the encoder output, NaN or not).
Hand fix for comparison (volta-ace-step README): relative error 0.2-7% (2B), 0.2-2.4% (XL) for the float32 tail."""
CWD = __import__("os").getcwd()
import copy, json, os, sys
from env import need
ROOT = need("ACESTEP_ROOT"); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), HERE, ROOT]; os.chdir(ROOT)
import torch
from transformers import AutoModel, AutoTokenizer
from fp16fix import Recorder
from fp16fix.plan import Plan
from fp16fix.apply import Applied
ckpt = sys.argv[1]; out = os.path.join(CWD, sys.argv[2])
model = AutoModel.from_pretrained(f"checkpoints/{ckpt}", trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager")
enc = model.encoder.lyric_encoder.float().cuda().eval(); del model
te = AutoModel.from_pretrained("checkpoints/Qwen3-Embedding-0.6B", dtype=torch.bfloat16)
emb = te.embed_tokens.float().cuda(); tok = AutoTokenizer.from_pretrained("checkpoints/Qwen3-Embedding-0.6B"); del te
songs = {}
for i in ["01", "02", "03", "05", "10", "113", "118", "108"]:
    ex = json.load(open(f"examples/text2music/example_{i}.json"))
    text = f"# Languages\n{ex.get('language', 'en')}\n\n# Lyric\n{ex.get('lyrics', '')}<|endoftext|>"
    songs[i] = tok(text, return_tensors="pt", truncation=True, max_length=2048).input_ids.cuda()
half = copy.deepcopy(enc).half()
run = lambda m, ids, dt: m(inputs_embeds=emb(ids).to(dt), attention_mask=torch.ones_like(ids)).last_hidden_state.float()
with torch.no_grad():
    with Recorder(enc) as r32:
        refs = {i: run(enc, ids, torch.float32) for i, ids in songs.items()}
    with Recorder(half) as r16:
        stock = {i: run(half, ids, torch.float16) for i, ids in songs.items()}
    plan = Plan(r32, r16)
    with Applied(half, plan.fixes):
        fixed = {i: run(half, ids, torch.float16) for i, ids in songs.items()}
print(plan.describe())
rows = []
for i in songs:
    err = lambda o: ((o - refs[i]).norm() / refs[i].norm()).item() if torch.isfinite(o).all() else float("nan")
    rows.append({"song": i, "tokens": songs[i].shape[1], "stock_finite": bool(torch.isfinite(stock[i]).all()),
                 "plan_finite": bool(torch.isfinite(fixed[i]).all()), "plan_rel_err": err(fixed[i])})
    print(f"song {i}: {rows[-1]['tokens']} tokens | plain float16 {'ok' if rows[-1]['stock_finite'] else 'NaN'} | "
          f"with plan {'ok' if rows[-1]['plan_finite'] else 'NaN'}, relative error {rows[-1]['plan_rel_err']:.2e}")
json.dump({"plan": plan.describe(), "songs": rows}, open(out, "w"), indent=1)
