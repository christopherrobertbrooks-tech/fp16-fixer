# fp16-fixer

Old GPUs without bfloat16 (V100, T4, P40, GTX 10-series) run models in float16, whose largest number is 65,504. Many
models overflow it somewhere and come out as NaN. Every fix so far has been made by hand, one model at a time. This
project automates the rule those fixes follow: find where the numbers leave float16's range, then pick an exact
power-of-two rescaling across a scale-free boundary (a norm, a softmax with its scale folded in), or a float32 island
where no such boundary exists.

**Status (2026-10-20): all four steps work on ACE-Step 1.5 and its 5 Hz LMs** -- find, plan, apply as run-time hooks,
check against float32 -- including a model nobody had fixed by hand (the 4B LM), **and on two models outside
ACE-Step: Wan 2.2's umT5-XXL text encoder and Gemma 3 4B.**

## Use

```python
import torch
from fp16fix import Recorder
from fp16fix.plan import Plan, save_fixes, load_fixes
from fp16fix.apply import Applied

with torch.no_grad():
    with Recorder(model) as r32:          # 1. the model in float32, on a few representative inputs
        run(model)
    model.half()
    with Recorder(model) as r16:          #    and the same inputs in plain float16
        run(model)
    plan = Plan(r32, r16)                 # 2. the fixes (print(plan.describe()) to read them)
    save_fixes(plan.fixes, "plan.json")

# later, at load time: the float16 model and the saved plan
with torch.no_grad(), Applied(model, load_fixes("plan.json")):   # 3. hooks on, no weight changed
    out = run(model)
```

Needs PyTorch 2.x (tested with 2.10). The drivers in `drivers/` are the runs below; they read model locations from
`ACESTEP_ROOT`, `WAN_MODEL` and `GEMMA_MODEL`. A plan is only as good as its calibration inputs: values that only
appear for other inputs are not covered, so calibrate on inputs like the ones you will run.

## Step 1: find (`fp16fix/find.py`)

One float32 pass and one plain float16 pass over the same inputs, recorded by:
- module hooks: every module's output, every `nn.Linear`'s input (a product like `act(gate) * up` only exists as the next
  projection's input), and a per-thread stack of running modules, so each operation is credited to its module;
- a `TorchFunctionMode`, one per thread (pipelines run models in worker threads): every torch function that returns a
  floating tensor -- attention scores, in-between products, residual adds -- without knowing the model's code;
  `scaled_dot_product_attention`'s scores are computed on the side.

The float32 pass gives the magnitudes; the float16 pass says which steps really run in float16 (a norm that upcasts
itself can't overflow) and where the first inf / NaN appears. Reported: everything above 65,504 / 4 that runs in
float16, deliberate mask values excepted.

## Scored against the fixes made by hand ([volta-ace-step](https://github.com/christopherrobertbrooks-tech/volta-ace-step))

Tesla V100, stock ACE-Step 1.5 code (the hand fixes switched off), the real pipeline (`drivers/`).

| model / site | hand-measured | found by fp16fix | hand fix |
| --- | ---: | ---: | --- |
| lyric encoder, `layers.7.mlp.down_proj` (turbo, xl-sft; standalone and in the pipeline) | ~505,000 | **504,584-504,864** | that projection in float32 |
| DiT residual stream, xl-sft, 50 steps, CFG 7 (30 s cut of example 10) | 1.36M at layer 18 (3:48 song) | **1,150,339 at `decoder.layers.18`** | stream / 64, exact |
| XL self-attention scores, `decoder.layers.0` | ~128,000 | **128,144** (`op:matmul`) | fold 1/sqrt(d) into `q_norm` |
| 5 Hz LM 1.7B residual stream ("massive activation") | ~2,000,000 from layer 2 | **1,958,926**, 26 layers from layer 2 | float32 residual |
| 5 Hz LM 1.7B SwiGLU product into layer 2's `down_proj` | ~85,000 | **80,024** (`down_proj` input) | `up` / 128 |
| control: 2B turbo DiT (stays in range) | in range | stream 24,026 (a within-4x warning, not an overflow); no attention site | -- |
| *not in the hand list:* 5 Hz LM 4B | -- | stream 802,706 from layer 16; product 43,630 | -- |

Plain float16 breaks first where the hand fixes say: the lyric encoder's `layers.7` (all song models), the LM's layer 2
(1.7B) / layer 16 (4B). Reports: `results/*.report.txt`, full maxima: `results/*.json`.

## Step 2: plan (`fp16fix/plan.py`)

The recorder also keeps a data-flow map (which operation produced each input; plain numbers an operation multiplies
by). The planner's one move is *shrink it where it still fits*: an over-range value is divided by a power of two at an
earlier point that still fits, walking back through steps that are proportional in that input (a linear layer, a
product in one factor, a sum in both, a multiply by a plain number). The rules, in order:

1. **Residual stream** (adds that feed each other): if only norms read it, divide the whole stream by 2^k -- shrink
   every writer, divide the reading norms' epsilon by 4^k. Exact. Otherwise keep the stream in float32.
2. **Attention scores**: move the scale that follows q @ k^T onto the q side (back to `q_norm` / `q_proj`).
3. **Product into a projection**: shrink it where it fits, multiply back in float32 after the next linear layer.
4. Otherwise a float32 island.

| model | hand fix | plan (headroom 4x) |
| --- | --- | --- |
| DiT, xl-sft | stream / 64 at `proj_in`, `o_proj`, `down_proj`; those norms' eps / 64^2 | stream / 128 at `proj_in`, `o_proj`, `down_proj` (or the gate product where it fits); eps / 128^2 on `self_attn_norm`, `cross_attn_norm`, `mlp_norm`, `norm_out` |
| XL attention, layer 0 | fold 1/sqrt(128) into `q_norm` | move the x0.08839 scale onto `q_norm`: 128,143 -> 11,326 |
| lyric encoder | last `down_proj` in float32 | stream / 32 (the hand probe's / 8 left it at ~63,000) |
| 5 Hz LM 1.7B | float32 residual; `up` / 128, x128 after `down_proj` | stream / 128, with layer 2's massive activation shrunk at `up` |

Same rules and places as the hand fixes; the factors differ only by the headroom asked for (the hand-made /64 leaves
~3x; at headroom 3 the planner picks 64 too). Turbo's DiT (peak 23,898) and XL layer 30's attention (22,772) get
"margin only" fixes: they fit float16 already.

## Steps 3 and 4: apply, check (`fp16fix/apply.py`, `drivers/check_*.py`)

`Applied(model, plan.fixes)` keeps the Recorder's module stack and operation counters, so it recognises each planned
operation at run time and multiplies its output (or moves the attention scale, or divides a norm's epsilon, or runs an
island in float32). No weight is changed; leaving the `with` block restores the model.

Tesla V100, plain float16 vs float16 with the plan vs the hand-made fixes, all against float32:

| model | plain float16 | float16 + fp16fix plan | float16 + hand fixes |
| --- | --- | --- | --- |
| lyric encoder, turbo (8 songs, output vs float32) | NaN on all 8 | all 8 finite, rel. error 0.3-2.4% | (README) 0.2-7% |
| lyric encoder, xl-sft (8 songs) | NaN on all 8 | all 8 finite, 0.2-7.1% | (README) 0.2-2.4% |
| 5 Hz LM 1.7B (3 prompts, 48-token float32 greedy continuations, teacher-forced) | NaN | top-1 agreement 100 / 98 / 96% | (PR draft) 100 / 80 / 100% (bf16: 98 / 60 / 95%) |
| **5 Hz LM 4B** (no hand fix exists) | NaN | **top-1 98 / 100 / 100%, logits 0.09% off** | -- |
| song, turbo, 8 steps (30 s, seed fixed) | NaN latents | wave corr 0.994, log-mel corr 0.997 | 0.999 / 0.997 |
| song, xl-sft, 50 steps, CFG 7 | NaN latents | wave corr 0.863, log-mel corr 0.988 | 0.816 / 0.986 |
| first denoising step vs float32, turbo / xl-sft | NaN | 3.5% / 0.35% | 4.3% / 0.35% |
| 30 s song time, turbo / xl-sft (float32: 2.2 / 30.3 s) | -- | 1.1 / 9.8 s | 1.1 / 9.8 s |

Control: two float32 runs of the pipeline give the same song (waveform correlation 1.0). An earlier attempt at these
song comparisons was invalid: ACE-Step's `GenerationConfig` defaults to `use_random_seed=True`, which ignores
`params.seed` -- every run was a different song, float32 included. `drivers/check_dit.py` sets
`use_random_seed=False, seeds=[1234]` and checks the control. The waveform correlation drops over 50 CFG steps for both
fixes alike (rounding grows through the sampler); the first step and the log-mel correlation say the two are equally
close to float32.

## An unseen model: Wan 2.2's umT5-XXL text encoder (`drivers/check_umt5.py`)

Studio-style encoding (padded to 512), 6 video prompts + Wan's default negative prompt, V100. In float16 transformers
clamps over-range values to +-65,504, so plain float16 doesn't NaN -- it silently clips.

| | rel. error vs float32 | cosine | 7 prompts | peak memory |
| --- | ---: | ---: | ---: | ---: |
| float32 | -- | -- | 3.29 s | 21.4 GB |
| plain float16 (clamped) | 4% - 78% | 0.69 - 0.999 | 1.02 s | -- |
| **float16 + fp16fix plan** | **0.16% - 0.37%** | **1.0000** | **1.15 s** | **10.9 GB** |

The plan: the residual stream (peak 374,201) stays in float32, because its feed-forward writers can't be shrunk in
time -- the gated-GELU's input reaches 250,000 in blocks 18-23, and nothing proportional sits between it and an
earlier value that fits -- and those blocks' `act` / `wo` run as float32 islands. Everything else (attention, the other
feed-forwards, all the norms) stays float16. This is the shape of the hand-made MiniMax H3 fix (float32 stream,
float16 matmuls), reached here by the rules.

## Gemma 3 4B-it (`drivers/check_gemma.py`)

Gemma 3 is known to fail in float16: its embeddings are multiplied by sqrt(hidden_size) and its post-norm weights are
large ([transformers RFC #37226](https://github.com/huggingface/transformers/pull/37226) scaled the embedding and the
`post_*_layernorm` weights down by about sqrt(hidden_size); the pre-norms and the final norm make that exact).

fp16fix's plan, from the measurements alone: **one fix -- the residual stream / 32, shrunk at the embedding's
sqrt(hidden) multiply and at every `post_attention_layernorm` / `post_feedforward_layernorm` output, the pre-norms'
epsilon / 1024.** The RFC's fix, with a power-of-two factor (peak 299,939 needs / 32 for 4x headroom).

| 5 prompts x 48 greedy tokens, teacher-forced | plain float16 | float16 + plan |
| --- | --- | --- |
| top-1 agreement with float32 | NaN on all 5 | **100% on all 5** |
| logits relative error | -- | 0.14% - 0.17% |
| 5 passes, V100 (float32: 0.36 s) | -- | 0.22 s |

Weights: Unsloth's ungated copy (`unsloth/gemma-3-4b-it`, the Gemma licence applies), text only.

A ready-made plan for Wan 2.2's umT5-XXL, calibrated and checked on held-out prompts:
[plans/wan22-umt5-xxl](plans/wan22-umt5-xxl).

Not tried: MiniMax H3 itself (its smallest 16-bit setup is ~95 GB of weights, beyond a 32 GB V100 with 16 GB of RAM).

Limits: tried on three model families (ACE-Step 1.5 + its Qwen3-based LMs, umT5, Gemma 3); the hooks run every operation through a
Python `TorchFunctionMode` (no measurable cost on these diffusion runs; not yet measured on token-by-token LLM
decoding); a plan is made per model from one calibration set (the 8 example songs, 3 prompts).

## Tests

`tests/test_find.py` (9), `tests/test_plan.py` (5, the planner on toy models with known answers) and
`tests/test_apply.py` (5: plain float16 gives NaN, float16 with the plan is finite and within 2% of float32, hooks and
epsilons restored). `test_find.py`: a toy model with planted sites -- a SwiGLU product, q.k scores (eager and SDPA), a residual
stream -- must be found and clean blocks must stay quiet; masks aren't overflows; hooks are removed; operations are seen
when the model runs in another thread. Disabling the `Linear` input hooks or the function mode makes them fail.

Made with AI assistance (Claude). MIT licence.
