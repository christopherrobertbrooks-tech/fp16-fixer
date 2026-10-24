#!/usr/bin/env python3
"""Check the shipped plan.json (what Studio loads) with the current fp16fix code: float16 + plan vs float32 on the 12
held-out prompts, the plan-match check on (strict). Writes check_shipped.json.

    check_shipped.py --model WAN_DIR [--out check_shipped.json]
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import torch
import transformers
from transformers import AutoTokenizer, UMT5EncoderModel
from fp16fix import Applied, load_fixes

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out", default=os.path.join(HERE, "check_shipped.json"))
a = ap.parse_args()
P = json.load(open(os.path.join(HERE, "prompts.json")))["held_out"]
tok = AutoTokenizer.from_pretrained(a.model, subfolder="tokenizer")
te = UMT5EncoderModel.from_pretrained(a.model, subfolder="text_encoder", dtype=torch.float32, device_map={"": 0}).eval()


def embed(text):
    t = tok([text], padding="max_length", max_length=512, truncation=True, return_attention_mask=True, return_tensors="pt")
    n = int(t.attention_mask.sum())
    return te(t.input_ids.cuda(), t.attention_mask.cuda()).last_hidden_state[0, :n].float().cpu()


with torch.no_grad():
    ref = [embed(p) for p in P]
    te.half()
    with Applied(te, load_fixes(os.path.join(HERE, "plan.json"), model=te)) as ap_:
        out = [embed(p) for p in P]
        missed = [str(m) for m in ap_.missed()]
e = [((o - r).norm() / r.norm()).item() for o, r in zip(out, ref)]
s = sorted(e)
res = {"transformers": transformers.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
       "missed_planned_places": missed, "all_finite": all(bool(torch.isfinite(o).all()) for o in out),
       "rel_err_median": s[len(s) // 2], "rel_err_max": s[-1], "per_prompt": [{"prompt": p, "rel_err": x} for p, x in zip(P, e)]}
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
print(f"shipped plan, current code: median {res['rel_err_median']:.4%}, worst {res['rel_err_max']:.4%}, missed {missed}")
