#!/usr/bin/env python3
"""ACE-Step 1.5 lyric encoder in float32 on the 8 example songs, recorded by fp16fix (answer key: volta-ace-step
README -- layers[-1].mlp.down_proj peaks ~505,000; everything before it stays under ~11,500)."""
import json, os, sys
CWD = __import__("os").getcwd(); ROOT = "/mnt/steam/ace-step/ACE-Step-1.5"; HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), HERE, ROOT]; os.chdir(ROOT)
import torch
from transformers import AutoModel, AutoTokenizer
from fp16fix import Recorder
from common import show
ckpt = sys.argv[1] if len(sys.argv) > 1 else "acestep-v15-turbo"; out = os.path.join(CWD, sys.argv[2])
model = AutoModel.from_pretrained(f"checkpoints/{ckpt}", trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager")
enc = model.encoder.lyric_encoder.float().cuda().eval(); del model
te = AutoModel.from_pretrained("checkpoints/Qwen3-Embedding-0.6B", dtype=torch.bfloat16)
emb = te.embed_tokens.float().cuda(); tok = AutoTokenizer.from_pretrained("checkpoints/Qwen3-Embedding-0.6B"); del te
import copy
songs = []
for i in ["01", "02", "03", "05", "10", "113", "118", "108"]:
    ex = json.load(open(f"examples/text2music/example_{i}.json"))
    text = f"# Languages\n{ex.get('language', 'en')}\n\n# Lyric\n{ex.get('lyrics', '')}<|endoftext|>"
    songs.append(tok(text, return_tensors="pt", truncation=True, max_length=2048).input_ids.cuda())
def run(m, dtype):
    with torch.no_grad(), Recorder(m) as r:
        for ids in songs:
            m(inputs_embeds=emb(ids).to(dtype), attention_mask=torch.ones_like(ids))
    return r
half = run(copy.deepcopy(enc).half(), torch.float16)
rec = run(enc, torch.float32)
show(rec, f"lyric encoder, {ckpt}", out, half=half)
