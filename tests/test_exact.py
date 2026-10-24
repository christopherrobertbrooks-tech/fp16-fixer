"""A plan applied to the float32 model itself must leave the output unchanged: every rule is an exact rewrite.

One toy per way a rule can be wrong (GitHub issues #1-#7): a shrink point with a second reader, biased linear layers,
a norm that isn't scale-free, a norm without an eps attribute, a returned raw stream, a plan applied to a changed model,
attention inside F.scaled_dot_product_attention. Run on the CPU in float32.
"""
import copy
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from fp16fix.find import Recorder
from fp16fix.plan import Plan, opname
from fp16fix.apply import Applied
from test_find import Toy, MLP

EXACT = 1e-6


def exactness(model, x=None, headroom=4.0):
    """float32 + plan vs float32 (relative error of the first output) and the plan."""
    torch.manual_seed(0)
    x = torch.randn(1, 32, 64) if x is None else x
    with torch.no_grad():
        with Recorder(model) as r32:
            ref = model(x)
        plan = Plan(r32, headroom=headroom)
        m = copy.deepcopy(model)
        with Applied(m, plan.fixes):
            out = m(x)
    if isinstance(ref, tuple):
        return max(((o - r).norm() / r.norm()).item() for o, r in zip(out, ref)), plan
    return ((out - ref).norm() / ref.norm()).item(), plan


def float16_check(model, headroom=4.0):
    """float16 + plan vs float32: finite, and the relative error."""
    torch.manual_seed(0)
    x = torch.randn(1, 32, 64)
    with torch.no_grad():
        with Recorder(model) as r32:
            ref = model(x)
        half = copy.deepcopy(model).half()
        with Recorder(half) as r16:
            half(x.half())
        plan = Plan(r32, r16, headroom=headroom)
        with Applied(half, plan.fixes):
            out = half(x.half()).float()
    return bool(torch.isfinite(out).all()), ((out - ref).norm() / ref.norm()).item(), plan


# ---- the toys the suite already has: still exact ------------------------------------------------------------------
@pytest.mark.parametrize("kw", [{}, {"stream_big": True}])
def test_existing_toys_are_exact(kw):
    torch.manual_seed(0)
    err, plan = exactness(Toy(**kw).eval())
    assert err <= EXACT, (err, plan.describe())


# ---- #1: a shrink point read by something off the shrink path ------------------------------------------------------
class FanOut(nn.Module):
    """SwiGLU where up's output is also read by a second linear layer."""
    def __init__(s, d=64):
        super().__init__()
        s.n = nn.RMSNorm(d)
        s.gate, s.up, s.down = nn.Linear(d, 4 * d, bias=False), nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)
        s.side = nn.Linear(4 * d, d, bias=False)
        with torch.no_grad():
            for l in (s.gate, s.up, s.down):
                l.weight.normal_(0, 0.02)
            s.gate.weight.mul_(300 / 0.02 / 4); s.up.weight.mul_(300 / 0.02 / 4); s.down.weight.mul_(1e-4)
            s.side.weight.normal_(0, 1e-3)

    def forward(s, x):
        h = s.n(x); u = s.up(h)
        return s.down(F.silu(s.gate(h)) * u) + s.side(u)


def big_qk_toy():
    torch.manual_seed(0)
    m = Toy().eval()
    with torch.no_grad():
        for b in m.layers:
            b.attn.q.weight.normal_(0, 0.3); b.attn.k.weight.normal_(0, 0.3)
    return m


def test_shrink_point_with_a_second_reader_is_not_used():
    torch.manual_seed(0)
    err, plan = exactness(FanOut().eval())
    assert err <= EXACT, (err, plan.describe())


def test_toy_with_larger_attention_at_strict_headroom_is_exact():
    for hr in (4, 16, 64):
        err, plan = exactness(big_qk_toy(), headroom=hr)
        assert err <= EXACT, (hr, err, plan.describe())


# ---- #2: biased linear layers are not proportional -----------------------------------------------------------------
class BiasMLP(nn.Module):
    """The suite's SwiGLU overflow, but up and down have biases."""
    def __init__(s, d=64):
        super().__init__()
        s.n = nn.RMSNorm(d)
        s.gate, s.up, s.down = nn.Linear(d, 4 * d, bias=False), nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        with torch.no_grad():
            for l in (s.gate, s.up, s.down):
                l.weight.normal_(0, 0.02)
            s.gate.weight.mul_(300 / 0.02 / 4); s.up.weight.mul_(300 / 0.02 / 4); s.down.weight.mul_(1e-4)
            s.up.bias.normal_(0, 50.0); s.down.bias.normal_(0, 1.0)

    def forward(s, x):
        h = s.n(x)
        return s.down(F.silu(s.gate(h)) * s.up(h))


def test_no_multiply_back_after_a_biased_linear():
    torch.manual_seed(0)
    err, plan = exactness(BiasMLP().eval())
    assert err <= EXACT, (err, plan.describe())
    assert not any("down" in m for f in plan.fixes for m in f.get("multiply_back_at", [])), plan.describe()
    ok, rel, plan16 = float16_check(BiasMLP().eval())
    assert ok and rel < 2e-2, (rel, plan16.describe())


# ---- #3: only scale-free norms may read a rescaled stream ----------------------------------------------------------
class BN(nn.Module):
    """Residual stream read at the end by an eval-mode BatchNorm (running statistics: not scale-free)."""
    def __init__(s, d=64):
        super().__init__()
        s.t = Toy(stream_big=True); s.bn = nn.BatchNorm1d(d)
        with torch.no_grad():
            s.bn.running_mean.normal_(0, 1e4); s.bn.running_var.fill_(1e9)

    def forward(s, x):
        h = x * 1e5
        for l in s.t.layers:
            h = l(h)
        return s.bn(h.transpose(1, 2)).transpose(1, 2)


def test_batchnorm_reader_keeps_the_stream_in_float32():
    torch.manual_seed(0)
    err, plan = exactness(BN().eval())
    rules = [f["rule"] for f in plan.fixes]
    assert "residual stream" not in rules and "residual stream in float32" in rules, plan.describe()
    assert err <= EXACT, (err, plan.describe())
    assert "NOT COVERED" in plan.describe() and "into bn" in plan.describe()    # running_var 1e9 can't be float16


class OtherNorms(nn.Module):
    """The stream toy with LayerNorm and GroupNorm readers: both are scale-free, so the stream is still rescaled."""
    def __init__(s, d=64):
        super().__init__()
        s.t = Toy(stream_big=True)
        for b in s.t.layers:
            b.n1, b.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.gn = nn.GroupNorm(4, d)

    def forward(s, x):
        h = x * 1e5
        for l in s.t.layers:
            h = l(h)
        return s.gn(h.transpose(1, 2)).transpose(1, 2)


def test_layernorm_and_groupnorm_readers_are_scale_free():
    torch.manual_seed(0)
    err, plan = exactness(OtherNorms().eval())
    assert "residual stream" in [f["rule"] for f in plan.fixes], plan.describe()
    assert err <= 1e-5, (err, plan.describe())     # GroupNorm / LayerNorm: eps / 4^k rounds a little in float32


# ---- #4: a norm whose eps can't be adjusted ------------------------------------------------------------------------
class MyRMSNorm(nn.Module):
    """An RMSNorm whose eps is a constant in forward(), not an attribute."""
    def __init__(s, d):
        super().__init__(); s.weight = nn.Parameter(torch.ones(d))

    def forward(s, x):
        return F.rms_norm(x.float(), (x.shape[-1],), s.weight.float(), 1e9).to(x.dtype)   # large eps: visible effect


def eps_free_toy():
    torch.manual_seed(0)
    t = Toy(stream_big=True).eval()
    for b in t.layers:
        b.n1, b.n2 = MyRMSNorm(64), MyRMSNorm(64)
    t.norm = MyRMSNorm(64)
    return t


def test_norm_without_eps_attribute_is_not_rescaled():
    err, plan = exactness(eps_free_toy())
    assert "residual stream" not in [f["rule"] for f in plan.fixes], plan.describe()
    assert err <= EXACT, (err, plan.describe())


def test_applying_an_eps_fix_to_a_norm_without_eps_raises():
    good = Toy(stream_big=True).eval()
    torch.manual_seed(0)
    with torch.no_grad(), Recorder(good) as r32:
        good(torch.randn(1, 32, 64))
    fixes = [f for f in Plan(r32).fixes if f["rule"] == "residual stream"]   # a plan made on nn.RMSNorm ...
    with pytest.raises(ValueError, match="eps"):
        with Applied(eps_free_toy(), fixes):                                 # ... applied to norms without eps
            pass


def test_eps_restored_after_exit():
    t = Toy(stream_big=True).eval()
    for m in t.modules():
        if isinstance(m, nn.RMSNorm):
            m.eps = 1e-6
    exactness(t)
    assert all(m.eps == 1e-6 for m in t.modules() if isinstance(m, nn.RMSNorm))


# ---- #5: a stream the model returns is a reader too ----------------------------------------------------------------
class RawOut(nn.Module):
    """Returns the normed output and the raw residual stream (like hidden_states)."""
    def __init__(s):
        super().__init__(); s.t = Toy(stream_big=True)

    def forward(s, x):
        h = x * 1e5
        for l in s.t.layers:
            h = l(h)
        return s.t.norm(h), h


def test_returned_raw_stream_is_not_rescaled():
    torch.manual_seed(0)
    err, plan = exactness(RawOut().eval())          # max over both returned tensors
    assert err <= EXACT, (err, plan.describe())


# ---- #6: a plan that doesn't match the model it's applied to -------------------------------------------------------
def test_plan_applied_to_sdpa_attention_reports_the_unmatched_passthrough():
    eager = big_qk_toy()
    torch.manual_seed(0)
    x = torch.randn(1, 32, 64)
    with torch.no_grad():
        with Recorder(eager) as r32:
            eager(x)
        fixes = [f for f in Plan(r32, headroom=2000).fixes if f["rule"].startswith("move the attention scale")]
        assert fixes
        m = copy.deepcopy(eager)
        with Applied(m, fixes):
            m(x)                                     # the same model: everything matches, no error
        sdpa = copy.deepcopy(eager)
        for b in sdpa.layers:
            b.attn.sdpa = True
        with pytest.raises(RuntimeError, match=r"op:div#0"):
            with Applied(sdpa, fixes):
                sdpa(x)


def test_saved_plan_records_and_checks_the_model(tmp_path):
    from fp16fix.plan import save_fixes, load_fixes
    torch.manual_seed(0)
    model = Toy().eval()
    with torch.no_grad(), Recorder(model) as r32:
        model(torch.randn(1, 32, 64))
    fixes = Plan(r32).fixes
    save_fixes(fixes, tmp_path / "plan.json", model=model)
    assert len(load_fixes(tmp_path / "plan.json", model=model)) == len(fixes)
    with pytest.warns(UserWarning, match="made for Toy"):
        load_fixes(tmp_path / "plan.json", model=FanOut())


# ---- #7: attention inside F.scaled_dot_product_attention -----------------------------------------------------------
def big_sdpa_toy():
    """Toy(sdpa=True) with scores that overflow float16 even after the 1/sqrt(d) scale."""
    torch.manual_seed(0)
    m = Toy(sdpa=True).eval()
    with torch.no_grad():
        m.layers[2].attn.q.weight.normal_(0, 24.0); m.layers[2].attn.k.weight.normal_(0, 24.0)   # scores ~1.2e5
    return m


def test_sdpa_overflow_found_is_planned():
    torch.manual_seed(0)
    m = big_sdpa_toy()
    with torch.no_grad(), Recorder(m) as r32:
        m(torch.randn(1, 32, 64))
    flagged = {p for _, p, _, k, _ in r32.report() if "sdpa" in k}
    assert flagged == {"layers.2.attn"}, r32.report()
    plan = Plan(r32)
    covered = {f.get("module") for f in plan.fixes}
    assert flagged <= covered, plan.describe()
    err, plan = exactness(big_sdpa_toy())
    assert err <= EXACT, (err, plan.describe())
    ok, rel, plan16 = float16_check(big_sdpa_toy())
    assert ok and rel < 2e-2, (rel, plan16.describe())


# ---- more shapes from the cleanups issue (#10) ---------------------------------------------------------------------
class GeluToy(nn.Module):
    """The SwiGLU-overflow toy with a GELU (tanh) MLP: up's output is the overflow, then GELU, then down."""
    def __init__(s, d=64):
        super().__init__()
        s.n, s.up, s.down = nn.RMSNorm(d), nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)
        with torch.no_grad():
            s.up.weight.normal_(0, 0.02).mul_(1e5 / 0.02 / 4); s.down.weight.normal_(0, 1e-6)

    def forward(s, x):
        return s.down(F.gelu(s.up(s.n(x)), approximate="tanh"))


class Twice(nn.Module):
    """One MLP module called twice per forward, on two different inputs (only one call overflows)."""
    def __init__(s, d=64):
        super().__init__()
        s.n, s.mlp = nn.RMSNorm(d), MLP(d, big=True)

    def forward(s, x):
        return s.mlp(s.n(x)) + s.mlp(s.n(x) * 1e-3)


@pytest.mark.parametrize("cls", [GeluToy, Twice])
def test_other_shapes_are_exact(cls):
    torch.manual_seed(0)
    err, plan = exactness(cls().eval())
    assert err <= EXACT, (err, plan.describe())


# ---- rotary embeddings: q feeds two branches that are added back together (the scale move must still happen) -------
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RopeAttn(nn.Module):
    """q_norm / k_norm, rotary embeddings, eager scores times 1/sqrt(d) (like Qwen3); q . k^T planted at ~1e5."""
    def __init__(s, d=64, leak=False):
        super().__init__()
        s.q, s.k, s.v, s.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        s.q_norm, s.k_norm = nn.RMSNorm(d), nn.RMSNorm(d)
        s.leak, s.d = leak, d
        with torch.no_grad():
            for l in (s.q, s.k, s.v, s.o):
                l.weight.normal_(0, 0.02)
            s.q_norm.weight.fill_(40.0); s.k_norm.weight.fill_(40.0)
        t = torch.arange(32)[:, None] / 10000 ** (torch.arange(d // 2) / (d // 2))
        s.register_buffer("cos", torch.cat((t, t), -1).cos()); s.register_buffer("sin", torch.cat((t, t), -1).sin())

    def forward(s, x):
        q, k, v = s.q_norm(s.q(x)), s.k_norm(s.k(x)), s.v(x)
        q2 = q * s.cos + rotate_half(q) * s.sin
        k2 = k * s.cos + rotate_half(k) * s.sin
        a = torch.softmax((q2 @ k2.transpose(-2, -1)) * s.d ** -0.5, -1) @ v
        return (s.o(a), q) if s.leak else s.o(a)        # leak: q itself is returned to the caller


def test_attention_scale_moves_through_rotary_embeddings():
    torch.manual_seed(0)
    m = RopeAttn().eval()
    err, plan = exactness(m)
    moves = [f for f in plan.fixes if f["rule"].startswith("move the attention scale")]
    assert moves and moves[0]["move_to"] == [("output", ("q_norm", "op:rms_norm#0"))], plan.describe()
    assert err <= EXACT, (err, plan.describe())


def test_attention_scale_not_moved_onto_q_when_q_is_also_returned():
    torch.manual_seed(0)
    err, plan = exactness(RopeAttn(leak=True).eval())
    moves = [f for f in plan.fixes if f["rule"].startswith("move the attention scale")]
    assert all(n[0] != "q_norm" for f in moves for _, n in f["move_to"]), plan.describe()   # k's side is fine
    assert err <= EXACT, (err, plan.describe())


# ---- one dropout module used on the way in and on the way out (T5Stack): in eval it returns its input -------------
class SharedDropout(nn.Module):
    """The stream toy with one nn.Dropout called on the embeddings and again on the final norm's output."""
    def __init__(s):
        super().__init__(); s.t = Toy(stream_big=True); s.drop = nn.Dropout(0.1); s.head = nn.Linear(64, 8, bias=False)

    def forward(s, x):
        h = s.drop(x * 1e5)
        for l in s.t.layers:
            h = l(h)
        return s.head(s.drop(s.t.norm(h)))


def test_dropout_called_twice_is_a_pass_through():
    torch.manual_seed(0)
    err, plan = exactness(SharedDropout().eval())
    assert "residual stream" in [f["rule"] for f in plan.fixes], plan.describe()   # read only by norms
    assert err <= EXACT, (err, plan.describe())
