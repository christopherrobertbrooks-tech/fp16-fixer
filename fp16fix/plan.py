"""Step 2 of the FP16 fixer: turn the find results into a fix plan, by the rules the hand-made fixes followed.

Input: a float32 Recorder (magnitudes + data-flow map) and, optionally, a float16 Recorder of the same inputs (which
operations really run in float16). Every operation above FP16_MAX / headroom that runs in float16 must be covered.

The one move behind every rule -- "shrink it where it still fits": a value that is too big is divided by a power of
two at an earlier point of its computation that still fits, walking back through operations that are proportional
in that input (a linear layer without a bias in its input, a product in one factor, a sum in both, a multiply or
divide by a plain number), until a point at or below the limit is reached. A non-proportional step (an activation, a
softmax, a norm, a biased linear layer) stops the walk. Every value on the walk changes, and so does everything
downstream of it: that must carry the change through unchanged back to the place the rule fixes (rotary embeddings'
q * cos + rotate(q) * sin does; a SwiGLU norm output that also feeds the gate's activation doesn't), or the walk fails.

1. Residual stream: the residual adds that feed each other form a stream. If everything that reads it is a scale-free
   norm with an eps attribute (RMSNorm / LayerNorm / GroupNorm ignore overall scale; BatchNorm doesn't), the whole
   stream can be divided by 2^k: every writer into it is shrunk (walked back as above) and every reading norm's
   epsilon divided by 4^k. Exact up to float16 rounding. If something else reads it (another layer, or the caller,
   when the model returns it), the stream is kept in float32 instead.
2. Attention scores: an over-range q @ k^T followed by a multiply / divide by a plain number (the 1/sqrt(d) scale)
   gets that scale moved in front of the product, onto the q side (walked back to q_norm / q_proj).
3. Product into a projection: an over-range value read only by linear layers without a bias is shrunk where it
   still fits and multiplied back, in float32, at the output of those linear layers.
4. Anything else: a float32 island for the module that computes it. Attention scores inside
   F.scaled_dot_product_attention can't be reached (the scale is inside the fused call), so an over-range one makes
   its module an island.
"""
import collections
import math
import re

import torch

from .find import FP16_MAX, OUTPUT

ADD = {"add", "__add__", "add_", "__iadd__", "__radd__"}
MUL = {"mul", "__mul__", "mul_", "__imul__", "__rmul__"}
DIV = {"div", "__truediv__", "div_", "__itruediv__"}
LIN = {"linear"}
MM = {"matmul", "__matmul__", "bmm", "mm", "baddbmm"}
EMB = {"embedding"}


def opname(node):
    return node[1][3:].split("#")[0] if node[1].startswith("op:") else node[1]


def pow2_at_least(x):
    """The smallest power of two >= x (x >= 1)."""
    return 2 ** max(0, math.ceil(math.log2(x) - 1e-12))


class Plan:
    def __init__(self, rec32, rec16=None, headroom=4.0, stream_float32=False):
        """stream_float32=True keeps every residual stream in float32 even where it could be rescaled: slower in
        principle, more precise where a stream holds huge values next to small ones (float16 loses small updates
        added to a large value; umT5: 1.0% worst held out vs 5.4% rescaled, at the same speed)."""
        self.lim = FP16_MAX / headroom
        self.headroom = headroom
        self.stream_float32 = stream_float32
        self.val = rec32.node_values()
        self.modout = {k[0]: v for k, v in rec32.values().items() if k[1] == "output"}   # module path -> output peak
        self.order = rec32.node_order
        self.mtype = rec32.mtype
        self.scalars = rec32.node_scalars
        self.biased = rec32.biased
        self.repeated = rec32.repeated
        self.scale_free = rec32.scale_free
        self.sdpa = {k[0]: v for k, v in rec32.values().items() if k[1] == "op:sdpa_scores"}   # module -> scores peak
        dt16 = rec16.node_dtypes if rec16 is not None else {}
        self.half = lambda n: n not in dt16 or "float16" in dt16[n]
        self.preds, self.succs = collections.defaultdict(set), collections.defaultdict(set)
        for a, b in rec32.edges:
            self.preds[b].add(a)
            self.succs[a].add(b)
        sentinel = 0.99 * torch.finfo(torch.float32).max
        self.flagged = sorted((n for n, v in self.val.items() if self.lim < v < sentinel and self.half(n)),
                              key=lambda n: self.order[n])
        self.covered = {}            # node -> the fix that covers it
        self.fixes = []
        self.uncovered = []          # over-range values no fix can reach (a model input or buffer itself)
        self._make()

    # ---- helpers ------------------------------------------------------------------------------------------------
    def v(self, n):
        return self.val.get(n, 0.0)

    def is_input(self, n):
        return n[0] == "<input>"

    def is_norm(self, n):
        """A scale-free norm with an eps attribute (find.scale_free_eps): it may read a rescaled stream."""
        return n[0] in self.scale_free

    def named_norm(self, n):
        return "norm" in self.mtype.get(n[0], "").lower()

    def scale_reaches_only(self, chain, allowed):
        """Scaling the walk's values is exact only if the change can't leak: follow everything downstream of the walk
        (an input node is scaled only where it goes into the walked op, so start after it) until `allowed` (the
        places the rule handles). Every other value reached must carry the scale through unchanged -- a sum whose
        inputs are all scaled, a product or matmul with exactly one scaled factor, a negation, a divide by a plain
        number, a linear layer without a bias -- and nothing reached may be returned to the caller. The walk's own
        steps are proportional by construction (a sum on the walk shrinks all its inputs, in their own branches)."""
        on_walk = set(chain)
        region = {c for c in chain if not self.is_input(c)}
        todo = list(region)
        while todo:
            for s in self.succs.get(todo.pop(), ()):
                if s in allowed or s in region:
                    continue
                if s == OUTPUT:
                    return False
                region.add(s)
                todo.append(s)
        for x in region - on_walk:
            op, ps = opname(x), self.preds.get(x, set())
            inside = sum(p in region for p in ps)
            if op in ADD:
                ok = inside == len(ps)
            elif op in MUL or op in MM:
                ok = inside == 1 and x not in self.repeated     # x * x squares the scale
            elif op in DIV:
                ok = len(ps) == 1
            elif op in LIN:
                ok = x not in self.biased
            else:
                ok = op == "neg"
            if not ok:
                return False
        return True

    def _done(self, places, chain, allowed):
        return (places, chain) if self.scale_reaches_only(chain, allowed) else (None, chain)

    def module_of_linear(self, n):
        return n[0] if opname(n) in LIN else None

    def shrink(self, n, factor, chain=None, allowed=frozenset()):
        """Places where dividing by `factor` makes `n` (and everything between) fit: a list of
        ("output", node) / ("input", node) placements, or None if a non-proportional step is in the way, or a value on
        the way is also read by something off the walk. `allowed`: readers that may see the change."""
        chain = [] if chain is None else chain
        chain.append(n)
        if self.v(n) <= self.lim or self.is_input(n):
            return self._done([("output", n)], chain, allowed)
        op, ps = opname(n), sorted(self.preds.get(n, ()), key=lambda p: self.order.get(p, 0))
        if op in LIN and n in self.biased:              # W x + b is not proportional in x
            return None, chain
        if op in LIN or op in EMB:
            if not ps:
                return self._done([("input", n)], chain, allowed)   # a model input (inputs_embeds) feeds it
            return self._shrink_one(ps, factor, chain, allowed)
        if (op in MUL or op in MM) and n in self.repeated:   # x * x: not proportional in x
            return None, chain
        if op in MUL or op in MM:                       # proportional in one factor: try the linear-layer factor first
            ps = sorted(ps, key=lambda p: (opname(p) not in LIN, self.v(p)))
            if not ps:
                return self._done([("input", n)], chain, allowed)
            for p in ps:
                r = self.shrink(p, factor, list(chain), allowed)
                if r[0] is not None:
                    return r
            return None, chain
        if op in DIV:
            return self._shrink_one(ps, factor, chain, allowed) if ps else self._done([("input", n)], chain, allowed)
        if op in ADD:                                   # proportional in both inputs: every input has to shrink
            out, ch = [], chain
            for p in ps:
                r, ch2 = self.shrink(p, factor, list(chain), allowed)
                if r is None:
                    return None, chain
                out += r
                ch += [x for x in ch2 if x not in ch]
            if len(ps) < 2:
                out.append(("input", n))                # one addend is a model input or a constant
                return self._done(out, ch, allowed)
            return out, ch
        return None, chain                              # activation, softmax, norm, ...: not proportional

    def _shrink_one(self, ps, factor, chain, allowed):
        for p in ps:
            r = self.shrink(p, factor, list(chain), allowed)
            if r[0] is not None:
                return r
        return None, chain

    # ---- the rules ----------------------------------------------------------------------------------------------
    def _make(self):
        # streams first (a stream's writers are often flagged before its adds), then attention, products, islands
        for path in sorted(self.sdpa, key=lambda p: self.order.get((p, "op:sdpa_scores"), 0)):
            fix = self._sdpa_island(path)
            if fix is not None:
                self.fixes.append(fix)
        for rule in (self._stream, self._attention, self._product, self._island):
            for n in self.flagged:
                if n in self.covered:
                    continue
                if rule.__name__ == "_stream" and opname(n) not in ADD or rule.__name__ == "_attention" and opname(n) not in MM:
                    continue
                fix = rule(n)
                if fix is not None:
                    self.fixes.append(fix)

    def _island(self, n):
        if self.is_input(n):                            # a model input or a buffer: it is too big before any op runs
            self.uncovered.append(n)
            return None
        fix = {"rule": "float32 island", "module": n[0], "peak": self.v(n), "site": n,
               "keep_float32": self.modout.get(n[0], self.v(n)) > self.lim,   # output past the headroom: float32
               "why": "no proportional path back to a value that fits"}
        self.covered[n] = fix
        return fix

    def _sdpa_island(self, path):
        ran16 = [n for n in self.order if n[0] == path and opname(n) == "scaled_dot_product_attention"]
        if self.sdpa[path] <= self.lim or not any(self.half(n) for n in ran16):
            return None
        if any(path == f["module"] or path.startswith(f["module"] + ".") for f in self.fixes if "module" in f):
            return None
        fix = {"rule": "float32 island", "module": path, "peak": self.sdpa[path], "site": (path, "op:sdpa_scores"),
               "keep_float32": self.modout.get(path, 0.0) > self.lim,
               "why": "attention scores inside F.scaled_dot_product_attention (the scale can't be moved there)"}
        for n in self.order:
            if n[0] == path or n[0].startswith(path + "."):
                self.covered[n] = fix
        return fix

    def _stream(self, n):
        comp, todo = set(), [n]
        while todo:                                     # residual adds connected add -> add
            x = todo.pop()
            if x in comp:
                continue
            comp.add(x)
            for y in set(self.preds.get(x, ())) | set(self.succs.get(x, ())):
                if opname(y) in ADD and y not in comp:       # an add feeding, or fed by, a stream add
                    todo.append(y)
        if len(comp) < 2:
            return None
        writers = {p for c in comp for p in self.preds.get(c, ()) if p not in comp}
        # the stream's tensors: every add's output, and what the writers feed in (the stream's first value is read
        # by the first block's norm before any add)
        readers = {s for c in comp for s in self.succs.get(c, ()) if s not in comp}
        readers |= {s for w in writers for s in self.succs.get(w, ()) if s not in comp}
        returned = OUTPUT in readers
        readers.discard(OUTPUT)
        externals = []      # model inputs now have their own nodes ("<input>", ...), so they are ordinary writers
        peak = max(self.v(c) for c in comp)
        factor = pow2_at_least(peak / self.lim)
        non_norm = sorted({r[0] for r in readers if not self.is_norm(r)}) + (["the caller (returned)"] if returned else [])
        fix = {"rule": "residual stream", "nodes": sorted(comp, key=lambda c: self.order.get(c, 0)), "peak": peak,
               "peak_at": max(comp, key=self.v), "factor": factor}
        if non_norm or self.stream_float32:
            fix.update(rule="residual stream in float32",
                       why=f"read by modules that aren't scale-free norms with an eps attribute: {non_norm[:4]}" if non_norm
                       else "kept in float32 by request (stream_float32=True)",
                       readers=sorted({r[0] for r in readers}),
                       readers_keep_float32=sorted({r[0] for r in readers if self.modout.get(r[0], 0.0) > FP16_MAX}),
                       first_value=sorted(w for w in writers if len(self.succs.get(w, ())) > 1 and self.v(w) > self.lim))
            for c in list(comp) + [w for w in writers if len(self.succs.get(w, ())) > 1]:   # + its first value
                self.covered[c] = fix
            return fix
        places, chains, ok = [], [], True
        allowed = frozenset(comp | readers)
        for w in sorted(writers, key=lambda x: self.order.get(x, 0)):
            r, ch = self.shrink(w, factor, allowed=allowed)
            if r is None:
                ok = False
                break
            places += r
            chains += ch
        if externals:
            places += [("input", c) for c in externals]
        if not ok:
            fix.update(rule="residual stream in float32", why="a writer has no proportional path to a value that fits",
                       readers=sorted({r[0] for r in readers}),
                       readers_keep_float32=sorted({r[0] for r in readers if self.modout.get(r[0], 0.0) > FP16_MAX}),
                       first_value=sorted(w for w in writers if len(self.succs.get(w, ())) > 1 and self.v(w) > self.lim))
        else:
            fix["shrink_at"] = sorted(set(places), key=lambda p: self.order.get(p[1], 0))
            fix["norms_eps_divide_by"] = factor * factor
            fix["norms"] = sorted({r[0] for r in readers})
        for c in list(comp) + chains:
            self.covered[c] = fix
        return fix

    def _attention(self, n):
        scale_ops = [s for s in self.succs.get(n, ()) if opname(s) in MUL | DIV and self.scalars.get(s)]
        if not scale_ops:
            return None
        s = scale_ops[0]
        if set(self.succs.get(n, ())) != {s} or len(self.scalars[s]) != 1:
            return None                                 # the unscaled scores are read elsewhere too
        c = next(iter(self.scalars[s]))
        mult = c if opname(s) in MUL else 1.0 / c       # the factor the scores get after the product
        if mult >= 1:
            return None
        need = 1.0 / mult
        for q in sorted(self.preds.get(n, ()), key=lambda p: ("q" not in p[0].split(".")[-1], self.order.get(p, 0))):
            r, ch = self.shrink_by(q, need, allowed={n})
            if r is not None:
                fix = {"rule": "move the attention scale before q @ k^T", "site": n, "peak": self.v(n),
                       "scale": mult, "move_from": s, "move_to": r, "after": self.v(n) * mult}
                if self.v(n) * mult > self.lim:
                    fix["note"] = "still over range after the move: add a power of two as well"
                for x in [n] + ch:
                    self.covered[x] = fix
                return fix
        return None

    def shrink_by(self, n, factor, allowed=frozenset()):
        """Like shrink, but always walks to the first module boundary (a norm or linear layer output) on the way."""
        chain, x = [], n
        while True:
            chain.append(x)
            if opname(x) in LIN or self.named_norm(x):
                return self._done([("output", x)], chain, allowed)
            ps = [p for p in self.preds.get(x, ()) if opname(x) in MUL | DIV | ADD or opname(x) in {"cat", "neg"}]
            ps = sorted(ps, key=lambda p: ("q" not in p[0].split(".")[-1], self.order.get(p, 0)))
            if not ps:
                return None, chain
            x = ps[0]

    def _product(self, n):
        readers = self.succs.get(n, set())
        consumers = [s for s in readers if opname(s) in LIN]
        # the multiply-back restores the value at each reader's output: every reader must be a linear module
        # without a bias (W (x / f) * f = W x, but (W (x / f) + b) * f != W x + b)
        if not consumers or len(consumers) != len(readers) or any(
                c in self.biased or "Linear" not in self.mtype.get(c[0], "") for c in consumers):
            return None
        factor = pow2_at_least(self.v(n) / self.lim)
        r, ch = self.shrink(n, factor, allowed=frozenset(consumers))
        if r is None or r == [("output", n)]:
            return None
        fix = {"rule": "shrink the product, multiply back after the next linear layer", "site": n, "peak": self.v(n),
               "factor": factor, "shrink_at": r, "multiply_back_at": sorted({c[0] for c in consumers})}
        for x in ch:
            self.covered[x] = fix
        return fix

    # ---- output ------------------------------------------------------------------------------------------------
    def describe(self):
        def pat(p):
            return re.sub(r"\.(\d+)(\.|$)", r".N\2", p)

        def places(ps):
            g = collections.OrderedDict()
            for kind, node in ps:
                g.setdefault((kind, pat(node[0]), opname(node)), set()).add(node[0])
            return "; ".join(f"{kind} of {where} ({op}, {len(v)}x)" for (kind, where, op), v in g.items())

        lines = [f"{len(self.flagged)} operations above {self.lim:,.0f} that run in float16 -> {len(self.fixes)} fix(es)"
                 " (\"margin only\" = the peak fits float16 already; the fix only buys the requested headroom)"]
        for f in self.fixes:
            r = f["rule"]
            if f.get("peak", 0) <= FP16_MAX:
                lines.append("  [margin only]")
            if r == "residual stream":
                lines.append(f"- residual stream ({len(f['nodes'])} adds, peak {f['peak']:,.0f} at {f['peak_at'][0]}): "
                             f"divide by {f['factor']} -> shrink at: {places(f['shrink_at'])}; "
                             f"norms' eps / {f['norms_eps_divide_by']}: {', '.join(sorted({pat(x) for x in f['norms']}))}")
            elif r == "residual stream in float32":
                lines.append(f"- residual stream ({len(f['nodes'])} adds, peak {f['peak']:,.0f}): keep in float32 ({f['why']})")
            elif r.startswith("move the attention scale"):
                lines.append(f"- attention scores at {f['site'][0]} (peak {f['peak']:,.0f}): move the x{f['scale']:.4g} scale "
                             f"to {places(f['move_to'])} -> scores {f['after']:,.0f}" + (f" ({f['note']})" if "note" in f else ""))
            elif r.startswith("shrink the product"):
                lines.append(f"- product at {f['site'][0]} {opname(f['site'])} (peak {f['peak']:,.0f}): divide by {f['factor']} at "
                             f"{places(f['shrink_at'])}; multiply back in float32 after {', '.join(f['multiply_back_at'])}")
            else:
                lines.append(f"- float32 island: {f['module']} (peak {f['peak']:,.0f}; {f['why']})")
        for n in self.uncovered:
            lines.append(f"- NOT COVERED: {n[1]} (peak {self.v(n):,.0f}): a model input or buffer is over range itself; "
                         "it has to stay float32 (pass it in float32, or keep that buffer float32)")
        return "\n".join(lines)


def made_with(model):
    """What a plan depends on: the model class, its attention implementation, and the library versions."""
    import sys
    cfg = getattr(model, "config", None)
    tf = sys.modules.get("transformers")
    return {"model": type(model).__name__, "attn_implementation": getattr(cfg, "_attn_implementation", None),
            "torch": torch.__version__, "transformers": getattr(tf, "__version__", None)}


def save_fixes(fixes, path, model=None):
    """Write a plan's fixes as JSON (made once from a calibration run, reused at load time). With the model, the file
    also records what the plan was made with (made_with), and load_fixes warns when it's used on something else."""
    import json
    data = fixes if model is None else {"made_with": made_with(model), "fixes": fixes}
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else list(o))


def load_fixes(path, model=None):
    """The fixes of a saved plan. With the model: warn if the plan was made for a different model class or attention
    implementation (the plan names operations by position, so it only fits the model code it was made on)."""
    import json
    import warnings
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if model is not None:
        was, now = data.get("made_with", {}), made_with(model)
        for k in ("model", "attn_implementation"):
            if was.get(k) != now[k]:
                warnings.warn(f"plan {path} was made for {was.get('model')} (attention {was.get('attn_implementation')}), "
                              f"used on {now['model']} (attention {now['attn_implementation']})", UserWarning, stacklevel=2)
                break
    return data["fixes"]
