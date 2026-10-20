"""Applying a plan must make float16 runs finite and close to float32, on the toy models (CPU)."""
import copy, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fp16fix.find import Recorder
from fp16fix.plan import Plan
from fp16fix.apply import Applied
from test_find import Toy
from test_plan import Direct


def check(model):
    torch.manual_seed(0)
    x = torch.randn(1, 32, 64)
    with torch.no_grad():
        with Recorder(model) as r32:
            ref = model(x)
        half = copy.deepcopy(model).half()
        with Recorder(half) as r16:
            stock = half(x.half())
        plan = Plan(r32, r16)
        with Applied(half, plan.fixes):
            fixed = half(x.half())
        again = half(x.half())                          # hooks gone: back to stock
    rel = ((fixed.float() - ref).norm() / ref.norm()).item()
    return stock, fixed, again, rel, plan, half


def test_toy_breaks_in_float16_without_the_plan_and_not_with_it():
    stock, fixed, again, rel, plan, _ = check(Toy().eval())
    assert not torch.isfinite(stock).all()                   # the planted problems are real in float16
    assert torch.isfinite(fixed).all(), plan.describe()
    assert rel < 2e-2, (rel, plan.describe())
    assert not torch.isfinite(again).all()                   # nothing left behind after exit


def test_stream_toy():
    stock, fixed, again, rel, plan, half = check(Toy(stream_big=True).eval())
    assert not torch.isfinite(stock).all()
    assert torch.isfinite(fixed).all(), plan.describe()
    assert rel < 2e-2, (rel, plan.describe())
    assert all(m.eps is None for m in half.modules() if isinstance(m, torch.nn.RMSNorm))   # eps restored


def test_float32_stream_fallback():
    stock, fixed, again, rel, plan, _ = check(Direct().eval())
    assert torch.isfinite(fixed).all(), plan.describe()
    assert rel < 2e-2, (rel, plan.describe())
