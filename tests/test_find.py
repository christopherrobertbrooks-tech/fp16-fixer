"""The recorder must find planted overflow sites (and only those) in a toy model, run on the CPU in float32."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from fp16fix.find import Recorder, summarize


class MLP(nn.Module):
    def __init__(self, d, big):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(d, 4 * d, bias=False), nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)
        with torch.no_grad():
            for l in (self.gate, self.up, self.down): l.weight.normal_(0, 0.02)
            if big:   # planted: the SwiGLU product (= down's input) reaches ~1e5; gate/up outputs stay ~300
                self.gate.weight.mul_(300 / 0.02 / 4)
                self.up.weight.mul_(300 / 0.02 / 4)
                self.down.weight.mul_(1e-4)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attn(nn.Module):
    def __init__(self, d, big, sdpa):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.sdpa, self.d = sdpa, d
        with torch.no_grad():
            for l in (self.q, self.k, self.v, self.o): l.weight.normal_(0, 0.02)
            if big:   # planted: q . k^T ~ 1e5 before the 1/sqrt(d) scale
                self.q.weight.normal_(0, 3.0); self.k.weight.normal_(0, 3.0)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        if self.sdpa:
            a = F.scaled_dot_product_attention(q, k, v)
        else:
            a = torch.softmax((q @ k.transpose(-2, -1)) / self.d ** 0.5, -1) @ v
        return self.o(a)


class Block(nn.Module):
    def __init__(self, d, mlp_big=False, attn_big=False, sdpa=False):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn, self.mlp = Attn(d, attn_big, sdpa), MLP(d, mlp_big)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class Toy(nn.Module):
    def __init__(self, d=64, stream_big=False, sdpa=False):
        super().__init__()
        self.layers = nn.ModuleList([Block(d), Block(d, mlp_big=True), Block(d, attn_big=True, sdpa=sdpa), Block(d)])
        self.norm = nn.RMSNorm(d)
        self.stream_big = stream_big

    def forward(self, x):
        if self.stream_big:
            x = x * 1e5          # planted: the residual stream itself at ~1e5 (scale-free through the RMSNorms)
        for l in self.layers:
            x = l(x)
        return self.norm(x)


def run(model):
    torch.manual_seed(0)
    x = torch.randn(1, 32, 64)
    with torch.no_grad(), Recorder(model) as rec:
        model(x)
    return rec, rec.report()


def test_finds_planted_sites_and_nothing_else():
    torch.manual_seed(0)
    rec, rows = run(Toy().eval())
    paths = {(p, k) for _, p, _, k, _ in rows}
    assert ("layers.1.mlp.down", "input") in paths, paths          # the product into down_proj
    assert ("layers.1.mlp.gate", "output") not in paths             # its factors stay in range
    assert any(p == "layers.2.attn" and k == "op:matmul" for p, k in paths), paths   # q @ k^T
    assert not any(p.startswith(("layers.0.", "layers.3.")) for p, _ in paths), paths  # clean blocks stay quiet


def test_sdpa_scores_are_seen():
    torch.manual_seed(0)
    m = Toy(sdpa=True).eval()
    with torch.no_grad():     # the scores are measured after SDPA's 1/sqrt(d) scale: plant a larger q . k^T
        m.layers[2].attn.q.weight.normal_(0, 12.0); m.layers[2].attn.k.weight.normal_(0, 12.0)
    rec, rows = run(m)
    assert any(p == "layers.2.attn" and k == "op:sdpa_scores" for _, p, _, k, _ in rows), rows


def test_stream_overflow_is_seen_at_block_level():
    torch.manual_seed(0)
    rec, rows = run(Toy(stream_big=True).eval())
    paths = {(p, k) for _, p, _, k, _ in rows}
    assert all(("layers.%d" % i, "output") in paths for i in range(4)), paths
    assert ("norm", "output") not in paths      # the final norm brings it back into range


def test_summary_collapses_layers():
    torch.manual_seed(0)
    rec, rows = run(Toy(stream_big=True).eval())
    g = summarize(rows)
    assert ("layers.N", "Block", "output") in g and g[("layers.N", "Block", "output")]["n"] == 4


def test_hooks_removed_after_exit():
    m = Toy().eval()
    run(m)
    assert all(not mod._forward_hooks and not mod._forward_pre_hooks for mod in m.modules())


class Masked(nn.Module):
    """An attention mask built with finfo.min must not be reported as an overflow."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        mask = torch.full((x.shape[1], x.shape[1]), torch.finfo(x.dtype).min).triu(1)
        s = (x @ x.transpose(-2, -1)) + mask
        return self.lin(torch.softmax(s, -1) @ x)


def test_masks_are_not_overflows():
    m = Masked().eval()
    with torch.no_grad(), Recorder(m) as rec:
        m(torch.randn(1, 6, 8))
    assert rec.report() == [], rec.report()


def test_ops_seen_when_the_model_runs_in_another_thread():
    """Pipelines often run the model in a worker thread; operations inside it must still be recorded."""
    import threading
    torch.manual_seed(0)
    m = Toy().eval()
    x = torch.randn(1, 32, 64)
    with torch.no_grad(), Recorder(m) as rec:
        t = threading.Thread(target=lambda: m(x)); t.start(); t.join()
    rows = rec.report()
    assert any(p == "layers.2.attn" and k == "op:matmul" for _, p, _, k, _ in rows), rows


def test_dataflow_graph_links_the_swiglu_product():
    torch.manual_seed(0)
    rec, rows = run(Toy().eval())
    e = rec.edges
    mul = ("layers.1.mlp", "op:mul#0")
    assert (("layers.1.mlp.up", "op:linear#0"), mul) in e, sorted(x for x in e if x[1] == mul)
    assert (("layers.1.mlp", "op:silu#0"), mul) in e
    assert (mul, ("layers.1.mlp.down", "op:linear#0")) in e
    # the two residual adds of a block are told apart, and the second reads the first through the MLP's norm
    assert (("layers.1", "op:add#0"), ("layers.1", "op:add#1")) in e
    assert any(src == ("layers.1", "op:add#0") and dst[0] == "layers.1.n2" for src, dst in e)


def test_attention_scale_is_recorded():
    torch.manual_seed(0)
    rec, rows = run(Toy().eval())
    divs = {n: v for n, v in rec.node_scalars.items() if n[0] == "layers.2.attn"}
    assert any(abs(x - 8.0) < 1e-9 for v in divs.values() for x in v), divs     # / sqrt(64)
