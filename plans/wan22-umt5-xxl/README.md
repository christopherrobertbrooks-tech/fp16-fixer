# Wan 2.2 umT5-XXL text encoder in float16

`plan.json` lets Wan 2.2's umT5-XXL text encoder (`Wan2.2-TI2V-5B-Diffusers/text_encoder`) run in float16 on cards
without bfloat16, matching float32. Made by `make_plan.py` with transformers 5.18 / torch 2.8 (a plan names exact
operations, so remake it if your transformers version differs), from the 30 `calibration` prompts in `prompts.json`.

In float16, transformers clamps the encoder's over-range values to +-65,504: it doesn't NaN, it silently clips.
On the 12 held-out prompts (never seen by the planner), vs float32:

| | relative error, median / worst | cosine, worst |
| --- | ---: | ---: |
| plain float16 (clamped) | 61% / 88% | 0.63 |
| float16 + plan | 0.20% / 1.0% | 0.99995 |

Four further private prompts (aggregate only): float16 + plan worst 0.20%, cosine >= 0.999999.

The plan keeps the residual stream in float32 and runs the feed-forward `act` / `wo` of blocks 18-23 as float32
islands (their gated-GELU input reaches ~250,000); everything else runs in float16. Use:

```python
from fp16fix.plan import load_fixes
from fp16fix.apply import Applied
te = UMT5EncoderModel.from_pretrained(wan_dir, subfolder="text_encoder", dtype=torch.float16, device_map={"": 0})
with torch.no_grad(), Applied(te, load_fixes("plan.json")):
    emb = te(input_ids, attention_mask).last_hidden_state
```

Used by [wan-video-on-v100](https://github.com/christopherrobertbrooks-tech/wan-video-on-v100)'s `wan_clip.py --te-fp16-plan`.

**Remaking the plan now gives a different, worse one (2026-10-23).** `plan.json` was made before a fix to
`fp16fix/find.py`: T5 calls one dropout module on the embeddings and again on the encoder's output, and the planner
used to merge the two, so it saw the stream as read by more than norms and kept it in float32. With the fix it
rescales the stream (/ 32) in float16 instead: exact in range, faster, but on the same 12 held-out prompts 0.29%
median / **5.4% worst** (one prompt; the other eleven 0.18-0.61%). That prompt pushes no value past what calibration
saw; the loss is float16's precision: the stream holds values up to ~374,000 next to small ones, and small updates
added to a large value in float16 are lost. Keep this `plan.json` (float32 stream); it still applies and checks with
the current code (0.196% median / 1.02% worst, every planned place met).
