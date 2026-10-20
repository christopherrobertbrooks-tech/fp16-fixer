# fp16-fixer

Old GPUs without bfloat16 (V100, T4, P40, GTX 10-series) run models in float16, whose largest number is 65,504. Many
models overflow it somewhere and come out as NaN. Every fix so far has been made by hand, one model at a time. This
project automates the rule those fixes follow: find where the numbers leave float16's range, then pick an exact
power-of-two rescaling across a scale-free boundary (a norm, a softmax with its scale folded in), or a float32 island
where no such boundary exists.

**Status (2026-10-20): all four steps work on ACE-Step 1.5 and its 5 Hz LMs** -- find, plan, apply as run-time hooks,
check against float32 -- including a model nobody had fixed by hand (the 4B LM). Not yet tried on models outside
ACE-Step.

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

Limits: tried on one model family (ACE-Step 1.5 + its Qwen3-based LMs); the hooks run every operation through a
Python `TorchFunctionMode` (no measurable cost on these diffusion runs; not yet measured on token-by-token LLM
decoding); a plan is made per model from one calibration set (the 8 example songs, 3 prompts).

## Tests

`tests/test_find.py` (9), `tests/test_plan.py` (5, the planner on toy models with known answers) and
`tests/test_apply.py` (3: plain float16 gives NaN, float16 with the plan is finite and within 2% of float32, hooks and
epsilons restored). `test_find.py`: a toy model with planted sites -- a SwiGLU product, q.k scores (eager and SDPA), a residual
stream -- must be found and clean blocks must stay quiet; masks aren't overflows; hooks are removed; operations are seen
when the model runs in another thread. Disabling the `Linear` input hooks or the function mode makes them fail.

Made with AI assistance (Claude). MIT licence.
