# The T5 family in float16

`make_plan.py` makes and scores a plan for one T5-family checkpoint (encoder-decoder, eager attention) on T5's own
task prompts in `prompts.json` (translation, summarising, grammar, similarity, questions): 20 to calibrate on, 12
held out. Each model's `plan.json` / `results.json` is in its own folder. Tesla V100, transformers 5.18, torch 2.8.

The published fix is the one transformers ships: loading with `dtype=float16` keeps every feed-forward `wo` in
float32 (`_keep_in_fp32_modules`), and the layers clamp over-range values to +-65,504. The clamp is on in every
float16 run, so plain float16 (`model.half()`, `wo` in float16) clips instead of giving NaN.

## t5-large (`google-t5/t5-large`, Apache-2.0)

Plan: **two exact stream rescales** -- the encoder's residual stream / 32 (peak 301,782), the decoder's / 8 (peak
74,930) -- shrunk at the embeddings and every attention / feed-forward output, the norms' epsilons divided to match.
No float32 islands.

12 held-out prompts vs float32, the decoder teacher-forced on float32's output (logits) and free-running (greedy):

| | top-1 agreement | logits error, median / worst | same output as float32 | 12 teacher-forced passes (median of 5) |
| --- | ---: | ---: | ---: | ---: |
| float32 | -- | -- | -- | 0.19 s |
| plain float16 (clamped) | 97.2% | 0.69% / 3.6% | 11 / 12 | -- |
| float16 + transformers' fix | 100% | **0.024% / 0.028%** | 12 / 12 | 0.19 s |
| float16 + plan | 100% | 0.039% / 0.049% | 12 / 12 | **0.48 s** |
| float16 + plan, streams kept in float32 (`--stream-float32`) | 100% | 0.025% / 0.029% | 12 / 12 | 0.63 s |

**transformers' fix is better here**: as close to float32 as the plan with float32 streams, closer than the rescaled
plan, and no slower than float32, while the plan's hooks make it 2.6-3.3x slower. `fp16fix.bake` can write the
rescaled plan into the weights, but not a float32 stream (its adds need the run-time hooks). t5-large is only mildly broken in float16 (one held-out output changes: an
STS-B score 1.6 -> 1.8). An earlier version of the planner kept the encoder stream in float32 with 20 islands
(0.008% / 0.017%, 3.6x slower); that came from a dropout module T5 calls twice, fixed in `fp16fix/find.py`.
