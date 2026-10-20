# fp16-fixer

Old GPUs without bfloat16 (V100, T4, P40, GTX 10-series) run models in float16, whose largest number is 65,504. Many
models overflow it somewhere and come out as NaN. Every fix so far has been made by hand, one model at a time. This
project automates the rule those fixes follow: find where the numbers leave float16's range, then pick an exact
power-of-two rescaling across a scale-free boundary (a norm, a softmax with its scale folded in), or a float32 island
where no such boundary exists.

**Status (2026-10-20): step 1, *find*, works.** Steps 2-4 (plan a fix, apply it as load-time hooks, check against
float32) are next.

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

## Tests

`tests/test_find.py` (7): a toy model with planted sites -- a SwiGLU product, q.k scores (eager and SDPA), a residual
stream -- must be found and clean blocks must stay quiet; masks aren't overflows; hooks are removed; operations are seen
when the model runs in another thread. Disabling the `Linear` input hooks or the function mode makes them fail.

Made with AI assistance (Claude). MIT licence.
