# Gemma 3 1B (base) in float16

`plan.json` lets Gemma 3 1B (`google/gemma-3-1b-pt`; made from Unsloth's ungated copy `unsloth/gemma-3-1b-pt`, the
Gemma licence applies) run in float16 on cards without bfloat16. Plain float16 gives NaN on every prompt
([transformers#39686](https://github.com/huggingface/transformers/issues/39686)). Made by `make_plan.py` with
transformers 5.18 / torch 2.8, eager attention, from the 20 `calibration` prompts in `prompts.json` (each with its
48-token float32 greedy continuation); Tesla V100.

The plan, from the measurements alone: **the residual stream / 8** (peak 98,223), shrunk at the embedding's
sqrt(hidden) multiply and at every `post_attention_layernorm` / `post_feedforward_layernorm` output, the pre-norms'
epsilon / 64. That is the shape of the published fix, [transformers RFC #37226](https://github.com/huggingface/transformers/pull/37226)
(closed without merging): scale the embedding and the post-norm weights by 1/64. Here the RFC is applied as that
weight edit on a float32 copy before casting to float16.

On the 12 held-out prompts (never seen by the planner), vs float32 (`make_plan.py`, `results.json`):

| | finite | top-1 agreement (teacher-forced) | logits error, median / worst | 48 greedy tokens identical to float32 |
| --- | ---: | ---: | ---: | ---: |
| plain float16 | 0 / 12 | -- | -- | 0 / 12 |
| float16 + plan | 12 / 12 | 99.65% | 0.087% / 0.12% | 10 / 12 |
| float16 + RFC #37226 (x 1/64) | 12 / 12 | 99.48% | 0.21% / 0.42% | 8 / 12 |

**The RFC at the same factor is as good.** Most of the gap above is the factor, not the method: x 1/64 pushes small
values toward float16's coarser range. `fair_check.py` (`fair.json`) runs the RFC at the plan's 1/8, at
~1/sqrt(hidden) = 1/34 and at 1/64, adds WikiText-2 test (40 windows x 1,024 tokens: perplexity and the mean KL
divergence from float32 per token) and repeats every timing 5 times:

| | held-out logits error, median | KL vs float32 (WikiText) | perplexity (float32 19.818) | 48 greedy identical | 12 teacher-forced passes | 48-token decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float32 | -- | -- | 19.818 | -- | 0.29 s | 0.54 s |
| float16 + plan (/ 8) | 0.087% | 6.9e-5 | 19.822 | 10 / 12 | 0.32 s | **1.15 s** |
| float16 + RFC x 1/8 | 0.090% | 8.1e-5 | 19.820 | 10 / 12 | 0.21 s | 0.60 s |
| float16 + RFC x 1/34 | 0.139% | 2.6e-4 | 19.824 | 11 / 12 | 0.21 s | 0.60 s |
| float16 + RFC x 1/64 | 0.211% | 6.0e-4 | 19.814 | 8 / 12 | 0.21 s | 0.60 s |

Perplexity can't tell them apart (all within 0.01 of float32); KL can. At the same factor the plan is ~15% closer
(it also divides the pre-norms' epsilon), which is small. What the planner adds is finding the fix, and its factor,
from the measurements without knowing the model. What it costs: its Python hooks make decoding ~2x slower than
float32 (0.54 s -> 1.15 s), while the RFC's weight edit runs at float16 speed (no faster than float32 on a 1B model,
whose decoding is launch-bound). `fp16fix.bake`
([#16](https://github.com/christopherrobertbrooks-tech/fp16-fixer/issues/16)) now writes this plan into the weights completely:
82 tokens/s decoding (plain float16 speed), 0.088% median logits error (`results/bake-gemma3-1b.json`).

```python
from fp16fix import load_fixes, Applied
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, attn_implementation="eager", device_map={"": 0})
with torch.no_grad(), Applied(model, load_fixes("plan.json", model=model)):
    out = model.generate(**inputs, max_new_tokens=100)
```
