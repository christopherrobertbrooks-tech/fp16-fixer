#!/usr/bin/env python3
"""fp16fix plans for the T5 family (t5-large, t5-v1_1, mT5, Flan-T5), scored on held-out prompts against float32 and
against the fix transformers ships: loading with dtype=float16 keeps every feed-forward `wo` in float32
(T5PreTrainedModel._keep_in_fp32_modules) and the layers clamp over-range values to +-65,504 (clamping is on in
every float16 run, so plain float16 clips instead of giving NaN).

Calibrates on prompts.json's "calibration" prompts (encoder input + the float32 greedy output, teacher-forced),
writes plan.json, then on the "held_out" prompts compares with float32:
- plain float16: model.half() (wo in float16, clamped);
- float16 + transformers' fix: from_pretrained(dtype=float16) (wo kept in float32, clamped);
- float16 + plan: model.half() with the plan's hooks.
Teacher-forced (the float32 output fed to the decoder: top-1 agreement, logits relative error) and free-running
(greedy, up to 64 tokens: does the output equal float32's). Timings: median of 5 runs after a warm-up.

    make_plan.py --model T5_DIR --out-dir DIR [--stream-float32]
--stream-float32 keeps the residual streams in float32 (Plan(stream_float32=True)) instead of rescaling them.
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fp16fix import Recorder, Plan, Applied, save_fixes

NEW = 64
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--stream-float32", action="store_true")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
P = json.load(open(os.path.join(HERE, "prompts.json")))
tok = AutoTokenizer.from_pretrained(a.model)
load = lambda dt: AutoModelForSeq2SeqLM.from_pretrained(a.model, dtype=dt, attn_implementation="eager",
                                                        device_map={"": 0}).eval()
model = load(torch.float32)


def greedy(m, x):
    return m.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=NEW, do_sample=False)


def logits(m, pairs):
    return [m(input_ids=x, decoder_input_ids=y[:, :-1]).logits.float() for x, y in pairs]


def timed(f, reps=5):
    """Median of `reps` timed runs after one warm-up."""
    f(); ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.time(); f(); torch.cuda.synchronize(); ts.append(time.time() - t)
    return sorted(ts)[reps // 2]


ids = {p: tok(p, return_tensors="pt").input_ids.cuda() for p in P["calibration"] + P["held_out"]}
with torch.no_grad():
    outs = {p: greedy(model, x) for p, x in ids.items()}                  # float32 outputs
    cal = [(ids[p], outs[p]) for p in P["calibration"]]
    held = [(ids[p], outs[p]) for p in P["held_out"]]
    with Recorder(model) as r32:
        logits(model, cal)
    ref = logits(model, held)
    t32 = timed(lambda: logits(model, held))
    model.half()
    with Recorder(model) as r16:
        logits(model, cal)
    plan = Plan(r32, r16, stream_float32=a.stream_float32)
    save_fixes(plan.fixes, os.path.join(a.out_dir, "plan.json"), model=model)
    res_out = {"plain_fp16": logits(model, held)}
    gen = {"plain_fp16": [greedy(model, x) for x, _ in held]}
    with Applied(model, plan.fixes):
        res_out["plan_fp16"] = logits(model, held)
        gen["plan_fp16"] = [greedy(model, x) for x, _ in held]
        t_plan = timed(lambda: logits(model, held))
    model = None                                                          # free the GPU for the next load
    torch.cuda.empty_cache()
    hf = load(torch.float16)                                              # transformers' own float16 fix
    wo_dtypes = sorted({str(m.weight.dtype) for n, m in hf.named_modules() if n.endswith(".wo")})
    res_out["hf_fp16"] = logits(hf, held)
    gen["hf_fp16"] = [greedy(hf, x) for x, _ in held]
    t_hf = timed(lambda: logits(hf, held))
print(plan.describe())
print("transformers' float16 load keeps wo in:", wo_dtypes)


def metrics(o, r, y, g):
    if not torch.isfinite(o).all():
        return {"finite": False, "same_output": False}
    return {"finite": True, "top1": (o.argmax(-1) == r.argmax(-1)).float().mean().item(),
            "rel_err": ((o - r).norm() / r.norm()).item(), "same_output": g.shape == y.shape and bool((g == y).all()),
            "output": tok.decode(g[0], skip_special_tokens=True)}


KINDS = ("plain_fp16", "plan_fp16", "hf_fp16")
rows = [{"prompt": p, "fp32_output": tok.decode(held[i][1][0], skip_special_tokens=True),
         **{k: metrics(res_out[k][i], ref[i], held[i][1], gen[k][i]) for k in KINDS}} for i, p in enumerate(P["held_out"])]


def agg(k):
    ok = [r[k] for r in rows if r[k]["finite"]]
    e = sorted(x["rel_err"] for x in ok)
    return {"n": len(rows), "finite": len(ok), "top1_mean": sum(x["top1"] for x in ok) / len(ok) if ok else None,
            "rel_err_median": e[len(e) // 2] if e else None, "rel_err_max": e[-1] if e else None,
            "same_output": sum(r[k]["same_output"] for r in rows)}


res = {"model": os.path.basename(a.model.rstrip("/")), "stream_float32": a.stream_float32, "transformers": transformers.__version__, "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(), "calibration_prompts": len(cal), "hf_fp16_wo_dtype": wo_dtypes,
       "held_out_summary": {k: agg(k) for k in KINDS},
       "time_s_held_out_teacher_forced": {"fp32": t32, "plan_fp16": t_plan, "hf_fp16": t_hf},
       "held_out": rows, "plan": plan.describe()}
json.dump(res, open(os.path.join(a.out_dir, "results.json"), "w"), indent=1, ensure_ascii=False)
for k in KINDS:
    print(k, json.dumps(res["held_out_summary"][k]))
print("time (12 held-out teacher-forced passes):", json.dumps(res["time_s_held_out_teacher_forced"]))
