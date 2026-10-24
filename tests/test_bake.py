"""bake() must give the same outputs as Applied with the same plan, and bake everything a model's own weights can carry
(issue #16). float32 toys on the CPU (outputs to 1e-6), plus a float16 run that must be finite and close to float32."""
import copy
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from fp16fix.find import Recorder
from fp16fix.plan import Plan
from fp16fix.apply import Applied
from fp16fix.bake import bake
from test_find import Toy
from test_exact import FanOut, BiasMLP, RopeAttn, SharedDropout, OtherNorms


class OffsetRMSNorm(nn.Module):
    """Gemma's norm: output * (1 + weight)."""
    def __init__(s, d, w=0.0):
        super().__init__(); s.weight = nn.Parameter(torch.full((d,), w)); s.eps = 1e-6

    def forward(s, x):
        return (F.rms_norm(x.float(), (x.shape[-1],), None, s.eps) * (1.0 + s.weight.float())).to(x.dtype)


class ScaledEmbedding(nn.Embedding):
    def __init__(s, n, d, scale):
        super().__init__(n, d); s.register_buffer("embed_scale", torch.tensor(scale), persistent=False)

    def forward(s, ids):
        return super().forward(ids) * s.embed_scale


class ScaledAttn(nn.Module):
    """Eager attention with the scale as an attribute (like transformers' `self.scaling`); large q . k^T planted."""
    def __init__(s, d=64):
        super().__init__()
        s.q, s.k, s.v, s.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        s.q_norm, s.scaling = nn.RMSNorm(d), 1 / 128       # small enough that float16 softmax isn't chaotic
        with torch.no_grad():
            for l in (s.q, s.k, s.v, s.o):
                l.weight.normal_(0, 0.02)
            s.q_norm.weight.fill_(100.0); s.k.weight.normal_(0, 1.0)

    def forward(s, x):
        q, k, v = s.q_norm(s.q(x)), s.k(x), s.v(x)
        return s.o(torch.softmax((q @ k.transpose(-2, -1)) * s.scaling, -1) @ v)


class GemmaLike(nn.Module):
    """A scaled embedding, pre-norm blocks whose post-norms (offset weights, 20,000) blow the stream up (~2e5), a final
    norm; attention scores ~2.5e4."""
    def __init__(s, d=64, n=4):
        super().__init__()
        s.embed = ScaledEmbedding(100, d, d ** 0.5 * 3000)
        s.pre = nn.ModuleList(nn.RMSNorm(d) for _ in range(n))
        s.attn = nn.ModuleList(ScaledAttn(d) for _ in range(n))
        s.post = nn.ModuleList(OffsetRMSNorm(d, 20000.0) for _ in range(n))
        s.norm = nn.RMSNorm(d)

    def forward(s, ids):
        h = s.embed(ids)
        for pre, attn, post in zip(s.pre, s.attn, s.post):
            h = h + post(attn(pre(h)))
        return s.norm(h)


def ids():
    torch.manual_seed(0)
    return torch.randint(0, 100, (1, 32))


def compare(model, x, headroom=4.0):
    """(Applied vs bake rel. difference, bake report, plan), float32."""
    with torch.no_grad():
        with Recorder(model) as r32:
            model(x)
        plan = Plan(r32, headroom=headroom)
        a = copy.deepcopy(model)
        with Applied(a, plan.fixes):
            want = a(x)
        b = copy.deepcopy(model)
        rep = bake(b, plan.fixes, lambda m: m(x))
        if rep.leftover:
            with Applied(b, rep.leftover):
                got = b(x)
        else:
            got = b(x)
    return ((got - want).norm() / want.norm()).item(), rep, plan


def test_gemma_like_bakes_completely():
    torch.manual_seed(0)
    err, rep, plan = compare(GemmaLike().eval(), ids())
    rules = sorted(f["rule"] for f in plan.fixes)
    assert "residual stream" in rules and any(r.startswith("move the attention") for r in rules), plan.describe()
    assert rep.leftover == [], (rep.describe(), plan.describe())      # nothing left for Applied (islands: module hooks)
    assert err <= 1e-6, (err, rep.describe())


def test_baked_float16_model_is_as_close_as_applied():
    """In float16 the baked model is finite and as close to float32 as Applied with the same plan (the two round
    differently -- edited weights vs a run-time multiply -- so 'as close', not 'equal')."""
    torch.manual_seed(0)
    model, x = GemmaLike().eval(), ids()
    with torch.no_grad():
        with Recorder(model) as r32:
            ref = model(x)
        half = copy.deepcopy(model).half()
        with Recorder(half) as r16:
            stock = half(x)
        fixes = Plan(r32, r16).fixes
        a = copy.deepcopy(half)
        with Applied(a, fixes):
            want = a(x).float()
        rep = bake(half, fixes, lambda m: m(x))
        got = half(x).float()
    rel = lambda o: ((o - ref).norm() / ref.norm()).item()
    assert not torch.isfinite(stock).all()
    assert rep.leftover == [] and rep.hooks == [], rep.describe()      # float16 plan: no islands, nothing left
    assert torch.isfinite(got).all() and rel(got) <= 1.5 * rel(want) + 1e-3, (rel(got), rel(want))


@pytest.mark.parametrize("make", [lambda: Toy(), lambda: Toy(stream_big=True), FanOut, BiasMLP, RopeAttn, SharedDropout,
                                  OtherNorms])
def test_bake_matches_applied(make):
    torch.manual_seed(0)
    model = make().eval()
    torch.manual_seed(0)
    err, rep, plan = compare(model, torch.randn(1, 32, 64))
    assert err <= 1e-6, (err, rep.describe(), plan.describe())


def test_toy_bakes_its_linear_writers():
    torch.manual_seed(0)
    err, rep, plan = compare(Toy(stream_big=True).eval(), torch.randn(1, 32, 64))
    # the stream's writers are attn.o / mlp.down outputs: baked; only the model input's shrink (x * 1e5 in forward()) and
    # the inline / sqrt(d) attention scale (no attribute) are left for Applied
    left = {f["rule"]: f for f in rep.leftover}
    assert set(left) <= {"residual stream", "move the attention scale before q @ k^T"}, rep.describe()
    if "residual stream" in left:
        assert all(n[0] == "<input>" for _, n in left["residual stream"]["shrink_at"]), left["residual stream"]
    assert err <= 1e-6


class TiedHead(GemmaLike):
    """GemmaLike with an output layer tied to the embedding (Gemma, T5, many LMs): scaling the embedding's weight would
    also scale the logits."""
    def __init__(s):
        super().__init__(); s.head = nn.Linear(64, 100, bias=False); s.head.weight = s.embed.weight

    def forward(s, ids):
        return s.head(super().forward(ids))


def test_tied_weights_are_not_scaled_twice():
    torch.manual_seed(0)
    model = TiedHead().eval()
    err, rep, plan = compare(model, ids())
    assert err <= 1e-6, (err, rep.describe())
    assert rep.check is not None and rep.check <= 1e-5, rep.check      # bake's own end-to-end check


def test_bake_end_to_end_check_catches_a_leak():
    """Sabotage: allow edits to shared weights; the end-to-end check must refuse the result."""
    import fp16fix.bake as B
    torch.manual_seed(0)
    model = TiedHead().eval()
    with torch.no_grad():
        with Recorder(model) as r32:
            model(ids())
        fixes = Plan(r32).fixes
    old = B._shared
    B._shared = lambda mod, owners, path: False
    try:
        with pytest.raises(ValueError, match="end-to-end"):
            bake(copy.deepcopy(model), fixes, lambda m: m(ids()))
    finally:
        B._shared = old


class ActModuleMLP(nn.Module):
    """SwiGLU with the activation as its own (weightless) module, like transformers' act_fn."""
    def __init__(s, d=64, down_std=0.02):
        super().__init__()
        s.gate, s.up, s.down = nn.Linear(d, 4 * d, bias=False), nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)
        s.act = nn.SiLU()
        with torch.no_grad():
            for l in (s.gate, s.up):
                l.weight.normal_(0, 0.3)
            s.down.weight.normal_(0, down_std)

    def forward(s, x):
        return s.down(s.act(s.gate(x)) * s.up(x))


DOWN_STD = 1000.0      # down_proj outputs ~2.5e5 into the stream; the product act(gate) * up stays ~43


class ActStream(nn.Module):
    """A residual stream whose writers are big down_proj outputs: the walk back from a writer stops at the product
    act(gate) * up, which fits -- a shrink there must move to up's weights, not to the weightless activation."""
    def __init__(s, d=64, n=3, down_std=DOWN_STD):
        super().__init__()
        s.embed = nn.Embedding(100, d)
        s.pre = nn.ModuleList(nn.RMSNorm(d) for _ in range(n))
        s.mlp = nn.ModuleList(ActModuleMLP(d, down_std) for _ in range(n))
        s.norm = nn.RMSNorm(d)

    def forward(s, ids):
        h = s.embed(ids)
        for pre, mlp in zip(s.pre, s.mlp):
            h = h + mlp(pre(h))
        return s.norm(h)


def test_product_shrink_moves_to_the_factor_with_weights():
    torch.manual_seed(0)
    err, rep, plan = compare(ActStream().eval(), ids())
    places = [n for f in plan.fixes if f["rule"] == "residual stream" for _, n in f["shrink_at"]]
    assert any(n[1].startswith("op:mul") for n in places), plan.describe()     # the walk stopped at the product
    assert rep.leftover == [], (rep.describe(), plan.describe())
    assert any(m.endswith(".up") for _, m, _ in rep.baked), rep.baked
    assert err <= 1e-6
