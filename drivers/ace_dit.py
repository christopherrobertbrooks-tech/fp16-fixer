#!/usr/bin/env python3
"""ACE-Step 1.5's whole song model (lyric encoder + DiT) through the real pipeline, recorded by fp16fix.

The gateway's ACE-Step checkout carries the three hand-made float16 fixes, applied at load time only for float16; they
are switched off here, so both passes run stock code: float32 for the magnitudes, plain float16 for which steps run in
float16 and where the first inf / nan appears. Answer key (volta-ace-step README): DiT residual stream 175K (sft),
531K (xl-turbo), 789K (xl-base), 1.36M (xl-sft, layer 18); XL attention scores up to ~128,000 before scaling
(decoder.layers.0); 2B turbo's DiT stays in range; every model's lyric encoder peaks ~505,000.

    ace_dit.py CKPT STEPS CFG OUT.json
"""
CWD = __import__("os").getcwd()
import json, os, sys
ROOT = "/mnt/steam/ace-step/ACE-Step-1.5"; HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), HERE, ROOT]; os.chdir(ROOT)
import torch
import acestep.core.generation.handler.init_service_loader as isl
for name in ("apply_float16_attention_scale_fix", "apply_float16_dit_stream_fix", "apply_float16_lyric_encoder_fix"):
    setattr(isl, name, lambda model: None)          # stock behaviour: no hand-made fixes
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from fp16fix import Recorder
from common import show
ckpt, steps, cfg, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), os.path.join(CWD, sys.argv[4])
ex = json.load(open("examples/text2music/example_10.json"))       # the 3:48 English song, cut to 30 s here
params = GenerationParams(task_type="text2music", thinking=False, caption=ex.get("caption") or ex.get("prompt", ""),
                          lyrics=ex.get("lyrics", ""), vocal_language=ex.get("language", "en"), duration=30,
                          inference_steps=steps, guidance_scale=cfg, seed=1234)
dit = AceStepHandler()
msg, ok = dit.initialize_service(project_root=ROOT, config_path=ckpt, device="auto", offload_to_cpu=False)
print("INIT", ok, dit.dtype, flush=True)
def gen(tag):
    r = generate_music(dit, LLMHandler(), params=params, config=GenerationConfig(batch_size=1, audio_format="wav"),
                       save_dir=f"/mnt/steam/fp16-fixer/work/{ckpt}-{tag}")
    print("GEN", tag, r.success, (r.status_message or "")[:100].replace("\n", " "), flush=True)
with torch.no_grad(), Recorder(dit.model) as half:   # stock float16 (the V100 has no bf16)
    gen("fp16")
for part in ("model", "text_encoder", "vae"):      # the whole pipeline in float32 for the measured pass
    m = getattr(dit, part, None)
    if isinstance(m, torch.nn.Module):
        m.float()
dit.dtype = torch.float32
with torch.no_grad(), Recorder(dit.model) as rec:
    gen("fp32")
show(rec, f"song model {ckpt}, {steps} steps, CFG {cfg}", out, half=half, top=40)
