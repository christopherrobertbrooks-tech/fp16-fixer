#!/usr/bin/env python3
"""Step 4 on ACE-Step's whole song model through the real pipeline: the same song (seed 1234, 30 s of example 10) in
float32 (reference), plain float16 (stock: no hand fixes), float16 with fp16fix's plan applied as hooks, and float16
with the three hand-made fixes (re-enabled for that run) as the yardstick. Compares each to float32: waveform
correlation and log-mel correlation; NaN / failure.

    check_dit.py CKPT STEPS CFG OUT.json
"""
CWD = __import__("os").getcwd()
import json, os, sys, time
ROOT = os.environ.get("ACESTEP_ROOT", "/mnt/steam/ace-step/ACE-Step-1.5"); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), HERE, ROOT]; os.chdir(ROOT)
import torch, soundfile, torchaudio
import acestep.core.generation.handler.init_service_loader as isl
HAND = {n: getattr(isl, n) for n in ("apply_float16_attention_scale_fix", "apply_float16_dit_stream_fix", "apply_float16_lyric_encoder_fix")}
for n in HAND:
    setattr(isl, n, lambda model: None)                     # stock: the hand fixes are off unless asked for
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from fp16fix import Recorder
from fp16fix.plan import Plan
from fp16fix.apply import Applied
ckpt, steps, cfg, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), os.path.join(CWD, sys.argv[4])
ex = json.load(open("examples/text2music/example_10.json"))
params = GenerationParams(task_type="text2music", thinking=False, caption=ex.get("caption") or ex.get("prompt", ""),
                          lyrics=ex.get("lyrics", ""), vocal_language=ex.get("language", "en"), duration=30,
                          inference_steps=steps, guidance_scale=cfg, seed=1234)
dit = AceStepHandler()
dit.initialize_service(project_root=ROOT, config_path=ckpt, device="auto", offload_to_cpu=False)

def parts(dtype):
    for p in ("model", "text_encoder", "vae"):
        m = getattr(dit, p, None)
        if isinstance(m, torch.nn.Module):
            m.to(dtype)
    dit.dtype = dtype

FIRST = {}          # tag -> the decoder's output on its first call (the first denoising step: same inputs everywhere)
def first_step(tag):
    def hook(mod, args, out):
        if tag not in FIRST:
            t = out[0] if isinstance(out, (tuple, list)) else getattr(out, "sample", out)
            FIRST[tag] = t.detach().float().cpu()
    return dit.model.decoder.register_forward_hook(hook)

def gen(tag):
    h = first_step(tag)
    t0 = time.time()
    r = generate_music(dit, LLMHandler(), params=params, config=GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False, seeds=[1234]),
                       save_dir=f"{os.environ.get('FP16FIX_WORK', '/tmp/fp16fix-work')}/check-{ckpt}-{tag}")
    dt = time.time() - t0
    h.remove()
    path = r.audios[0]["path"] if r.success and r.audios else None
    print("GEN", tag, r.success, f"{dt:.1f} s", (r.status_message or "")[:80].replace("\n", " "), flush=True)
    return path, dt

with torch.no_grad():
    with Recorder(dit.model) as r16:
        stock_path, _ = gen("fp16-stock")
    parts(torch.float32)
    with Recorder(dit.model) as r32:
        rec_path, _ = gen("fp32-recorded")
    ref_path, t32 = gen("fp32")                              # timing without the recorder
    plan = Plan(r32, r16)
    print(plan.describe(), flush=True)
    parts(torch.float16)
    with Applied(dit.model, plan.fixes):
        plan_path, tplan = gen("fp16-plan")
    for n, f in HAND.items():                                # the yardstick: the hand-made fixes, as shipped
        f(dit.model)
    hand_path, thand = gen("fp16-hand")

mel = torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=2048, hop_length=512, n_mels=128)
def load(p):
    a, sr = soundfile.read(p, dtype="float32", always_2d=True)
    return torch.from_numpy(a.T).mean(0), sr
ref, sr = load(ref_path)
lm_ref = torch.log(mel(ref) + 1e-6)
def compare(p):
    if p is None:
        return {"ok": False}
    a, _ = load(p)
    n = min(len(a), len(ref)); a2, r2 = a[:n], ref[:n]
    lm = torch.log(mel(a2) + 1e-6); lr = lm_ref[:, :lm.shape[1]]
    corr = lambda x, y: torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1].item()
    return {"ok": bool(torch.isfinite(a).all()), "rms": a.pow(2).mean().sqrt().item(), "wave_corr": corr(a2, r2),
            "logmel_corr": corr(lm, lr)}
def step_err(tag):
    a, b = FIRST.get(tag), FIRST.get("fp32")
    if a is None or b is None or not torch.isfinite(a).all():
        return float("nan")
    return ((a - b).norm() / b.norm()).item()
def song_corr(p, q):
    if p is None or q is None:
        return float("nan")
    a, _ = load(p); b, _ = load(q); n = min(len(a), len(b))
    return torch.corrcoef(torch.stack([a[:n], b[:n]]))[0, 1].item()
res = {"control_fp32_vs_fp32_wave_corr": song_corr(rec_path, ref_path), "plan_vs_hand_wave_corr": song_corr(plan_path, hand_path),
       "first_step_rel_err": {k: step_err(k) for k in ("fp16-stock", "fp16-plan", "fp16-hand")},
       "ckpt": ckpt, "steps": steps, "cfg": cfg, "plan": plan.describe(), "fp32_s": t32, "plan_s": tplan, "hand_s": thand,
       "fp32": compare(ref_path), "fp16_stock": compare(stock_path), "fp16_plan": compare(plan_path), "fp16_hand": compare(hand_path)}
for k in ("fp16_stock", "fp16_plan", "fp16_hand"):
    print(k, json.dumps(res[k]))
print("control: two float32 runs, waveform corr", round(res["control_fp32_vs_fp32_wave_corr"], 4),
      "| plan vs hand fixes (both float16), waveform corr", round(res["plan_vs_hand_wave_corr"], 4))
print("first denoising step, relative error vs float32:", json.dumps(res["first_step_rel_err"]))
print(f"time: float32 {t32:.1f} s, float16 + plan hooks {tplan:.1f} s, float16 + hand fixes {thand:.1f} s")
json.dump(res, open(out, "w"), indent=1)
