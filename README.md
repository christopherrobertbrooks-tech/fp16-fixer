# fp16-fixer

Old GPUs without bfloat16 run models in float16, whose largest number is 65,504. Many models overflow it somewhere and
come out as NaN. Every fix so far has been made by hand, one model at a time. This project automates the rule those
fixes follow: find where the numbers leave float16's range, then pick a power-of-two rescaling across a scale-free
boundary (a norm, a softmax with its scale folded in) -- exact up to float16 rounding -- or a float32 island where no
such boundary exists.

**Tested on a Tesla V100 only.** It is meant for any GPU without bfloat16 (T4, P40, GTX 10-series) but hasn't been run
on one. On Pascal cards (P40, GTX 1070/1080) float16 arithmetic runs at about 1/64 of the float32 rate and there are no
tensor cores, so a plan there mainly saves memory (float16 weights and activations), not time -- measure before
switching from float32. The speed-ups below are V100 numbers.

**Status (2026-10-20): all four steps work on ACE-Step 1.5 and its 5 Hz LMs** -- find, plan, apply as run-time hooks,
check against float32 -- including a model nobody had fixed by hand (the 4B LM), **and on two models outside
ACE-Step: Wan 2.2's umT5-XXL text encoder and Gemma 3 4B.**

## Install

```bash
pip install git+https://github.com/christopherrobertbrooks-tech/fp16-fixer
```

or, from a clone, `pip install -e ".[test]"` and run the tests with `pytest`. Needs Python 3.10+ and PyTorch 2.4+; the
drivers also need transformers (`pip install -e ".[drivers]"`).

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
    save_fixes(plan.fixes, "plan.json", model=model)

# later, at load time: the float16 model and the saved plan
fixes = load_fixes("plan.json", model=model)                     # warns if made for another model / attention
with torch.no_grad(), Applied(model, fixes):                     # 3. hooks on, no weight changed
    out = run(model)
```

A plan names operations by module path and position, so **make it and use it with the same model code and the same
attention implementation** (all the runs below load with `attn_implementation="eager"`; a plan made on eager attention
doesn't fit SDPA). `Applied` checks this: the first time a module runs, every planned place inside it must have been
met, or it raises (`Applied(model, fixes, strict=False)` warns instead). A plan is also only as good as its
calibration inputs: values that only appear for other inputs are not covered, so calibrate on inputs like the ones you
will run.

Versions behind the results: the ACE-Step runs used torch 2.10 and transformers 4.57; the umT5 plan and Gemma, torch 2.8
and transformers 5.18. The drivers in `drivers/` are the runs below; they need the model locations in `ACESTEP_ROOT`,
`WAN_MODEL` and `GEMMA_MODEL`.

## Step 1: find (`fp16fix/find.py`)

One float32 pass and one plain float16 pass over the same inputs, recorded by:
- module hooks: every module's output, every `nn.Linear`'s input (a product like `act(gate) * up` only exists as the next
  projection's input), and a per-thread stack of running modules, so each operation is credited to its module;
- a `TorchFunctionMode`, one per thread (pipelines run models in worker threads): every torch function that returns a
  floating tensor -- attention scores, in-between products, residual adds -- without knowing the model's code;
  `scaled_dot_product_attention`'s scores (after its scale) are computed on the side.

The float32 pass gives the magnitudes; the float16 pass says which steps really run in float16 (a norm that upcasts
itself can't overflow) and where the first inf / NaN appears. Reported: everything above 65,504 / 4 that runs in
float16, deliberate mask values excepted.

## Scored against the fixes made by hand ([volta-ace-step](https://github.com/christopherrobertbrooks-tech/volta-ace-step))

Tesla V100, stock ACE-Step 1.5 code (the hand fixes switched off), the real pipeline (`drivers/`).

| model / site | hand-measured | found by fp16fix | hand fix |
| --- | ---: | ---: | --- |
| lyric encoder, `layers.7.mlp.down_proj` (turbo, xl-sft; standalone and in the pipeline) | ~505,000 | **504,584-504,864** | that projection in float32 |
| DiT residual stream, xl-sft, 50 steps, CFG 7 (30 s cut of example 10) | 1.36M at layer 18 (3:48 song) | **1.15-1.17M at `decoder.layers.18`** (three runs: 1,150,339 / 1,153,176 / 1,166,330) | stream / 64, exact |
| XL self-attention scores, `decoder.layers.0` | ~128,000 | **128,144** (`op:matmul`; 128,143.5 in the files) | fold 1/sqrt(d) into `q_norm` |
| 5 Hz LM 1.7B residual stream ("massive activation") | ~2,000,000 from layer 2 | **1,958,926**, 26 layers from layer 2 | float32 residual |
| 5 Hz LM 1.7B SwiGLU product into layer 2's `down_proj` | ~85,000 | **80,024** (`down_proj` input) | `up` / 128 |
| control: 2B turbo DiT (stays in range) | in range | stream 23,898-24,026 (two runs; a within-4x warning, not an overflow); no attention site | -- |
| *not in the hand list:* 5 Hz LM 4B | -- | stream 802,706 from layer 16; product 43,630 | -- |

Plain float16 breaks first where the hand fixes say: the lyric encoder's `layers.7` (all song models), the LM's layer 2
(1.7B) / layer 16 (4B). Reports: `results/*.report.txt`, full maxima: `results/*.json`.

## Step 2: plan (`fp16fix/plan.py`)

The recorder also keeps a data-flow map (which operation produced each input; plain numbers an operation multiplies
by). The planner's one move is *shrink it where it still fits*: an over-range value is divided by a power of two at an
earlier point that still fits, walking back through steps that are proportional in that input (a linear layer without
a bias, a product in one factor, a sum in both, a multiply by a plain number). Every value on the way changes, so each
may only be read by the walk itself; a value with another reader (a SwiGLU norm output that feeds gate and up, an `up`
read twice) stops the walk. The rules, in order:

1. **Residual stream** (adds that feed each other): if only scale-free norms read it (RMSNorm, LayerNorm, GroupNorm,
   with an eps attribute -- not BatchNorm, whose running statistics aren't scale-free), divide the whole stream by
   2^k -- shrink every writer, divide the reading norms' epsilon by 4^k. Exact up to float16 rounding. Otherwise
   (another layer reads it, or the model returns it) keep the stream in float32.
2. **Attention scores**: move the scale that follows q @ k^T onto the q side (back to `q_norm` / `q_proj`). Inside
   `F.scaled_dot_product_attention` the scale can't be moved, so over-range scores there make the attention module a
   float32 island.
3. **Product into a projection**: shrink it where it fits, multiply back in float32 after the next linear layer (only
   when every reader is a linear layer without a bias).
4. Otherwise a float32 island. A model input or buffer that is over range itself can't be fixed by a plan; it is
   listed as NOT COVERED.

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

Tesla V100, plain float16 vs float16 with the plan vs the hand-made fixes, all against float32. **These are on the
calibration inputs** (the plan was made from the same songs / prompts); the held-out check is the umT5 plan in
[plans/wan22-umt5-xxl](plans/wan22-umt5-xxl): 12 unseen prompts, 0.20% median / 1.0% worst error.

| model | plain float16 | float16 + fp16fix plan | float16 + hand fixes |
| --- | --- | --- | --- |
| lyric encoder, turbo (8 songs, output vs float32) | NaN on all 8 | all 8 finite, rel. error 0.3-2.4% | (README) 0.2-7% |
| lyric encoder, xl-sft (8 songs) | NaN on all 8 | all 8 finite, 0.2-7.1% | (README) 0.2-2.4% |
| 5 Hz LM 1.7B (3 prompts, 48-token float32 greedy continuations, teacher-forced) | NaN | top-1 agreement 100 / 98 / 96%, logits 15.4 / 3.9 / 3.1% off | (PR draft) 100 / 80 / 100% (bf16: 98 / 60 / 95%) |
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

## Step 5: bake (`fp16fix/bake.py`)

`Applied` runs every operation through a Python `TorchFunctionMode`, which slows token-by-token decoding.
`bake(model, plan.fixes, run)` writes the plan into the weights once instead, then the model runs as plain float16:

```python
from fp16fix import bake, load_fixes, Applied
report = bake(model, load_fixes("plan.json", model=model), run=lambda m: m(input_ids=example_ids))
print(report.describe())                  # what was baked; report.leftover: what still needs Applied (often nothing)
```

Each fix becomes a module edit: a module's output scaled -> its weights (or, for norms storing `1 + w`, the offset
weights; for a scale kept as a buffer, the buffer); the norms' epsilon set; a moved attention scale -> `q_norm`'s
weights and the attention's `scaling` set to 1; a float32 island -> that module converted to float32 behind two
module hooks. Every edit is checked on the module's own inputs against a float32 copy, weights tied to another
module (an output layer sharing the embedding) are left alone, and a final check against `Applied` on the example run
refuses the result if anything leaked. What can't be baked comes back as `report.leftover`, a smaller plan for
`Applied`.

Tesla V100, everything baked (no leftover) on all four (`drivers/check_bake_lm.py`, `drivers/check_dit.py` with
`FP16FIX_BAKE=1`; `results/bake-*.json`):

| | float32 | float16 + `Applied` | float16 baked | plain float16 |
| --- | ---: | ---: | ---: | ---: |
| Gemma 3 1B, 48-token decode | 89 tokens/s | 42 tokens/s | **82 tokens/s** | 82 tokens/s (NaN) |
| Gemma 3 1B, held-out logits error (median / worst) | -- | 0.087% / 0.12% | 0.088% / 0.17% | NaN |
| Gemma 3 4B, 48-token decode | 37 tokens/s | 31 tokens/s | **45 tokens/s** | 45 tokens/s (NaN) |
| Gemma 3 4B, held-out logits error (median / worst) | -- | 0.13% / 0.18% | 0.21% / 0.30% | NaN |
| ACE-Step xl-sft, first denoising step vs float32 | -- | 0.349% | 0.347% | NaN |
| ACE-Step xl-sft, 30 s song (50 steps, CFG 7) | 30.1 s | 9.7 s | 9.6 s | NaN |
| ACE-Step turbo, first denoising step vs float32 | -- | 3.53% | 3.65% | NaN |

Baking brings decoding to plain float16 speed (on a 1B model that is still a little slower than float32: its decoding
is launch-bound). It costs a little precision where weights are stored as an offset: Gemma's post-norms `(1 + w)`
become `(1 + w) / 32 - 1`, rounded to float16 (4B: 0.21% vs 0.13%; 11 of 12 greedy continuations identical, vs 12).
The songs: the first step matches; over 50 CFG steps the baked and hooked songs drift apart like any two float16
roundings do (waveform correlation with float32 0.825 baked, 0.863 hooks, 0.816 for the hand fixes; log-mel 0.985 /
0.988 / 0.986).

 (`drivers/check_umt5.py`)

Studio-style encoding (padded to 512), 6 video prompts + Wan's default negative prompt, V100, calibration inputs
(held out: [plans/wan22-umt5-xxl](plans/wan22-umt5-xxl)). In float16 transformers clamps over-range values to
+-65,504, so plain float16 doesn't NaN -- it silently clips.

| | rel. error vs float32 | cosine | 7 prompts | peak memory |
| --- | ---: | ---: | ---: | ---: |
| float32 | -- | -- | 3.29 s | 21.4 GB |
| plain float16 (clamped) | 4% - 78% | 0.69 - 0.999 | 1.02 s | -- |
| **float16 + fp16fix plan** | **0.24% - 0.61%** | **0.99998** | **1.15 s** | **10.9 GB** |

The plan (current code, `results/check-umt5*.json`): the residual stream (peak 374,201) / 32, shrunk at the
embedding and every attention / feed-forward output, plus float32 islands for the gated-GELU `act` of blocks 18-23
(its input reaches ~250,000). Before the dropout fix in `fp16fix/find.py` (T5 calls one dropout module twice) the
planner kept the stream in float32 instead: 0.16% - 0.37% here, at the same 1.15 s -- so for umT5 the float32 stream
is the better plan, and it is the one shipped in [plans/wan22-umt5-xxl](plans/wan22-umt5-xxl). This is the shape of
the hand-made MiniMax H3 fix (float32 stream, float16 matmuls).

## Gemma 3 4B-it (`drivers/check_gemma.py`)

Gemma 3 is known to fail in float16: its embeddings are multiplied by sqrt(hidden_size) and its post-norm weights are
large ([transformers RFC #37226](https://github.com/huggingface/transformers/pull/37226), closed without merging,
proposed scaling the embedding and the `post_*_layernorm` weights down by about sqrt(hidden_size); the pre-norms and
the final norm make that exact).

fp16fix's plan, from the measurements alone: **one fix -- the residual stream / 32, shrunk at the embedding's
sqrt(hidden) multiply and at every `post_attention_layernorm` / `post_feedforward_layernorm` output, the pre-norms'
epsilon / 1024.** The RFC's proposal, with a power-of-two factor (peak 299,939 needs / 32 for 4x headroom).

| 5 prompts x 48 greedy tokens, teacher-forced (calibration inputs) | plain float16 | float16 + plan |
| --- | --- | --- |
| top-1 agreement with float32 | NaN on all 5 | **100% on all 5** |
| logits relative error | -- | 0.13% - 0.19% |
| 5 passes, V100 (float32: 0.40 s, both warmed up) | -- | 0.24 s |

Weights: Unsloth's ungated copy (`unsloth/gemma-3-4b-it`, the Gemma licence applies), text only.

`Plan(r32, r16, stream_float32=True)` keeps residual streams in float32 even where they could be rescaled: more
precise when a stream holds huge values next to small ones (umT5), but it needs the run-time hooks (no `bake`).

A ready-made plan for Wan 2.2's umT5-XXL, calibrated and checked on held-out prompts:
[plans/wan22-umt5-xxl](plans/wan22-umt5-xxl).

## Validation sweep ([#11](https://github.com/christopherrobertbrooks-tech/fp16-fixer/issues/11))

Models with a documented float16 failure, each scored on held-out prompts against float32 and, where one exists,
against the published fix. Tesla V100.

| model | plain float16 | float16 + plan (held out) | published fix (held out) | plan |
| --- | --- | --- | --- | --- |
| Gemma 3 1B base | NaN 12/12 | logits 0.087% median, KL 6.9e-5 on WikiText, 10/12 greedy runs identical; decoding 2x slower than float32 (hooks) | RFC #37226: at its x 1/64 0.21%, KL 6.0e-4; **at the plan's x 1/8 about the same (0.090%, KL 8.1e-5)**, at float16 speed | [stream / 8](plans/gemma3-1b-pt) |
| t5-large | clamped, no NaN: logits 0.69% / 3.6%, 11/12 outputs same | logits 0.039% / 0.049%, 12/12 same; 2.6x slower than float32 (hooks); float32 streams: 0.025% / 0.029%, 3.3x slower | **transformers' wo-in-float32 + clamp: 0.024% / 0.028%, 12/12, float32 speed -- better than the plan** | [two streams / 32, / 8](plans/t5) |

Not tried: MiniMax H3 itself (its smallest 16-bit setup is ~95 GB of weights, beyond a 32 GB V100 with 16 GB of RAM).

Limits: tried on three model families (ACE-Step 1.5 + its Qwen3-based LMs, umT5, Gemma 3); the hooks run every operation through a
Python `TorchFunctionMode` (no measurable cost on these diffusion runs; on token-by-token LLM decoding it costs
~2x at 1B and ~1.4x at 4B -- `bake` removes it); a plan is made per model from one calibration set (the 8 example songs, 3 prompts).

## Tests

`pytest` (CPU, a few seconds; CI runs it on every push). `tests/test_find.py` (9): a toy model with planted sites -- a
SwiGLU product, q.k scores (eager and SDPA), a residual stream -- must be found and clean blocks must stay quiet; masks
aren't overflows; hooks are removed; operations are seen when the model runs in another thread. Disabling the `Linear`
input hooks or the function mode makes them fail. `tests/test_plan.py` (5): the planner on toy models with known
answers. `tests/test_apply.py` (5): plain float16 gives NaN, float16 with the plan is finite and within 2% of float32,
hooks and epsilons restored.

`tests/test_exact.py` (16): applying a plan to the float32 model itself must leave the output unchanged (to 1e-6),
on toys made to break each rule -- a shrink point with a second reader, biased linear layers, a BatchNorm reader,
LayerNorm / GroupNorm readers, a norm without an eps attribute, a model that returns its raw stream, a GELU MLP, a
module called twice, attention inside SDPA -- plus a plan applied to a changed model (eager -> SDPA) must raise.

Made with AI assistance (Claude). MIT licence.
