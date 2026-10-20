#!/usr/bin/env python3
"""ACE-Step's 5 Hz LM (a Qwen3 causal LM) in float32 with transformers, recorded by fp16fix (answer key: volta-ace-step
pr-drafts/D-nanovllm-fp16.md -- residual stream ~2,000,000 from layer 2 at <|im_start|>; SwiGLU product into layer 2's
down_proj ~85,000; layer 27's MLP output ~68,000)."""
import os, sys
CWD = __import__("os").getcwd(); ROOT = os.environ.get("ACESTEP_ROOT", "/mnt/steam/ace-step/ACE-Step-1.5"); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), HERE]; os.chdir(ROOT)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fp16fix import Recorder
from common import show
ckpt = sys.argv[1] if len(sys.argv) > 1 else "acestep-5Hz-lm-1.7B"; out = os.path.join(CWD, sys.argv[2])
tok = AutoTokenizer.from_pretrained(f"checkpoints/{ckpt}")
lm = AutoModelForCausalLM.from_pretrained(f"checkpoints/{ckpt}", dtype=torch.float32, attn_implementation="eager").cuda().eval()
prompts = ["Write a song plan: an upbeat indie pop song about summer road trips, 120 BPM, female vocals.",
           "Caption: dark ambient techno with deep bass and sparse vocals. Give the BPM, key and a structure.",
           "Make lyrics for a slow country ballad about an old dog, three verses and a chorus."]
ids = [tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True),
          return_tensors="pt").input_ids.cuda() for p in prompts]
def run(m):
    with torch.no_grad(), Recorder(m) as r:
        for x in ids:
            m(input_ids=x)
    return r
rec = run(lm)
lm.half()          # same weights in float16: which steps really run in float16, and where it first breaks
half = run(lm)
show(rec, f"5 Hz LM, {ckpt}", out, half=half)
