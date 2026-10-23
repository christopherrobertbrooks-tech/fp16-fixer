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

On the 12 held-out prompts (never seen by the planner), vs float32:

| | finite | top-1 agreement (teacher-forced) | logits error, median / worst | 48 greedy tokens identical to float32 |
| --- | ---: | ---: | ---: | ---: |
| plain float16 | 0 / 12 | -- | -- | 0 / 12 |
| **float16 + plan** | 12 / 12 | **99.65%** | **0.087% / 0.12%** | **10 / 12** |
| float16 + RFC #37226 (x 1/64) | 12 / 12 | 99.48% | 0.21% / 0.42% | 8 / 12 |

The plan is closer to float32 than the RFC fix on every one of the 12 prompts (per-prompt numbers in `results.json`).

Speed, 12 teacher-forced passes: float32 0.29 s, float16 + plan 0.32 s, float16 + RFC 0.21 s. At 1B the plan's
Python hooks cost more than float16 saves; the RFC's edit is free at run time. Baking the plan's exact rescale into
the weights would get both ([#16](https://github.com/christopherrobertbrooks-tech/fp16-fixer/issues/16)).

```python
from fp16fix import load_fixes, Applied
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, attn_implementation="eager", device_map={"": 0})
with torch.no_grad(), Applied(model, load_fixes("plan.json", model=model)):
    out = model.generate(**inputs, max_new_tokens=100)
```
