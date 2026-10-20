# fp16-fixer

Old GPUs without bfloat16 (V100, T4, P40, GTX 10-series) run models in float16, whose largest number is 65,504. Many
models overflow it somewhere and come out as NaN. Every fix so far has been made by hand, one model at a time. This
project automates the rule those fixes follow: find where the numbers leave float16's range, then pick an exact
power-of-two rescaling across a scale-free boundary (a norm, a softmax with its scale folded in), or a float32 island
where no such boundary exists.

**Status (2026-10-20): steps 1 (*find*) and 2 (*plan*) work.** Steps 3-4 (apply the plan as load-time hooks, check
against float32) are next.

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

## Tests

`tests/test_find.py` (9) and `tests/test_plan.py` (5, the planner on toy models with known answers). `test_find.py`: a toy model with planted sites -- a SwiGLU product, q.k scores (eager and SDPA), a residual
stream -- must be found and clean blocks must stay quiet; masks aren't overflows; hooks are removed; operations are seen
when the model runs in another thread. Disabling the `Linear` input hooks or the function mode makes them fail.

Made with AI assistance (Claude). MIT licence.
