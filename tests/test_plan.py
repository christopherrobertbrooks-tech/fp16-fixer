"""The planner must pick the rule the hand-made fixes used, on toy models with one planted problem each."""
import torch
import torch.nn as nn
from fp16fix.find import Recorder
from fp16fix.plan import Plan
from test_find import Toy


def plan_of(model, x=None):
    torch.manual_seed(0)
    x = torch.randn(1, 32, 64) if x is None else x
    with torch.no_grad(), Recorder(model) as rec:
        model(x)
    return Plan(rec)


def test_stream_read_only_by_norms_is_divided_by_a_power_of_two():
    torch.manual_seed(0)
    p = plan_of(Toy(stream_big=True).eval())
    s = [f for f in p.fixes if f["rule"] == "residual stream"]
    assert len(s) == 1, p.describe()
    f = s[0]
    assert f["factor"] == 2 ** (int(f["peak"] / p.lim - 1e-9).bit_length()), f["factor"]   # smallest power of two
    assert f["peak"] / f["factor"] <= p.lim < f["peak"] / (f["factor"] / 2)
    at = {(k, n[0]) for k, n in f["shrink_at"]}
    assert any(k == "output" and w == "<input>" for k, w in at), at    # shrink the model input itself (before x * 1e5)
    assert {("output", f"layers.{i}.attn.o") for i in range(4)} <= at, at
    assert {("output", f"layers.{i}.mlp.down") for i in range(4)} <= at, at
    assert {f"layers.{i}.n1" for i in range(4)} | {"norm"} <= set(f["norms"]), f["norms"]
    assert f["norms_eps_divide_by"] == f["factor"] ** 2
    rules = sorted(f["rule"] for f in p.fixes)                # the toy's other two planted problems, nothing more
    assert rules == ["move the attention scale before q @ k^T", "residual stream",
                     "shrink the product, multiply back after the next linear layer"], p.describe()


def test_swiglu_product_is_shrunk_at_up_and_multiplied_back_after_down():
    torch.manual_seed(0)
    p = plan_of(Toy().eval())
    prods = [f for f in p.fixes if f["rule"].startswith("shrink the product")]
    assert prods and prods[0]["site"][0] == "layers.1.mlp", p.describe()
    assert ("output", ("layers.1.mlp.up", "op:linear#0")) in prods[0]["shrink_at"], prods[0]
    assert prods[0]["multiply_back_at"] == ["layers.1.mlp.down"]


def test_attention_scale_is_moved_onto_q():
    torch.manual_seed(0)
    p = plan_of(Toy().eval())
    at = [f for f in p.fixes if f["rule"].startswith("move the attention scale")]
    assert at and at[0]["site"][0] == "layers.2.attn", p.describe()
    assert at[0]["move_to"] == [("output", ("layers.2.attn.q", "op:linear#0"))], at[0]
    assert abs(at[0]["scale"] - 1 / 8) < 1e-12


def test_no_island_when_a_rule_applies():
    torch.manual_seed(0)
    p = plan_of(Toy().eval())
    assert not [f for f in p.fixes if f["rule"] == "float32 island"], p.describe()


class Direct(nn.Module):
    """A stream that something other than a norm reads (a head straight on the stream) can't be rescaled."""
    def __init__(self):
        super().__init__()
        self.toy = Toy(stream_big=True)
        self.head = nn.Linear(64, 8)

    def forward(self, x):
        h = x * 1e5
        for l in self.toy.layers:
            h = l(h)
        return self.head(h)


def test_stream_read_by_a_non_norm_stays_in_float32():
    torch.manual_seed(0)
    p = plan_of(Direct().eval())
    rules = [f["rule"] for f in p.fixes]
    assert "residual stream in float32" in rules and "residual stream" not in rules, p.describe()
