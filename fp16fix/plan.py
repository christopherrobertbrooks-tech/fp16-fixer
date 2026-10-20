"""Step 2 of the FP16 fixer: turn the find results into a fix plan, by the rules the hand-made fixes followed.

Input: a float32 Recorder (magnitudes + data-flow map) and, optionally, a float16 Recorder of the same inputs (which
operations really run in float16). Every operation above FP16_MAX / headroom that runs in float16 must be covered.

The one move behind every rule -- "shrink it where it still fits": a value that is too big is divided by a power of
two at an earlier point of its computation that still fits, walking back through operations that are proportional
in that input (a linear layer in its input, a product in one factor, a sum in both, a multiply or divide by a plain
number), until a point at or below the limit is reached. A non-proportional step (an activation, a softmax, a norm)
stops the walk.

1. Residual stream: the residual adds that feed each other form a stream. If everything that reads it is a norm
   (RMSNorm / LayerNorm ignore overall scale), the whole stream can be divided by 2^k: every writer into it is shrunk
   (walked back as above) and every reading norm's epsilon divided by 4^k. Exact.
   If something else reads it, the stream is kept in float32 instead.
2. Attention scores: an over-range q @ k^T followed by a multiply / divide by a plain number (the 1/sqrt(d) scale)
   gets that scale moved in front of the product, onto the q side (walked back to q_norm / q_proj).
3. Product into a projection: an over-range value is shrunk where it still fits and multiplied back, in float32, at
   the output of the next linear layer.
4. Anything else: a float32 island for the module that computes it.
"""
import collections
import math
import re

import torch

from .find import FP16_MAX

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
    def __init__(self, rec32, rec16=None, headroom=4.0):
        self.lim = FP16_MAX / headroom
        self.headroom = headroom
        self.val = rec32.node_values()
        self.order = rec32.node_order
        self.mtype = rec32.mtype
        self.scalars = rec32.node_scalars
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
        self._make()

    # ---- helpers ------------------------------------------------------------------------------------------------
    def v(self, n):
        return self.val.get(n, 0.0)

    def is_input(self, n):
        return n[0] == "<input>"

    def is_norm(self, n):
        return "norm" in self.mtype.get(n[0], "").lower()

    def module_of_linear(self, n):
        return n[0] if opname(n) in LIN else None

    def shrink(self, n, factor, chain=None):
        """Places where dividing by `factor` makes `n` (and everything between) fit: a list of
        ("output", node) / ("input", node) placements, or None if a non-proportional step is in the way."""
        chain = [] if chain is None else chain
        chain.append(n)
        if self.v(n) <= self.lim or self.is_input(n):
            return [("output", n)], chain
        op, ps = opname(n), sorted(self.preds.get(n, ()), key=lambda p: self.order.get(p, 0))
        if op in LIN or op in EMB:
            if not ps:
                return [("input", n)], chain           # a model input (inputs_embeds) feeds it
            return self._shrink_one(ps, factor, chain)
        if op in MUL or op in MM:                       # proportional in one factor: try the linear-layer factor first
            ps = sorted(ps, key=lambda p: (opname(p) not in LIN, self.v(p)))
            if not ps:
                return [("input", n)], chain
            for p in ps:
                r = self.shrink(p, factor, list(chain))
                if r[0] is not None:
                    return r
            return None, chain
        if op in DIV:
            return self._shrink_one(ps, factor, chain) if ps else ([("input", n)], chain)
        if op in ADD:                                   # proportional in both inputs: every input has to shrink
            out, ch = [], chain
            for p in ps:
                r, ch2 = self.shrink(p, factor, list(chain))
                if r is None:
                    return None, chain
                out += r
                ch += [x for x in ch2 if x not in ch]
            if len(ps) < 2:
                out.append(("input", n))                # one addend is a model input or a constant
            return out, ch
        return None, chain                              # activation, softmax, norm, ...: not proportional

    def _shrink_one(self, ps, factor, chain):
        for p in ps:
            r = self.shrink(p, factor, list(chain))
            if r[0] is not None:
                return r
        return None, chain

    # ---- the rules ----------------------------------------------------------------------------------------------
    def _make(self):
        # streams first (a stream's writers are often flagged before its adds), then attention, products, islands
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
        fix = {"rule": "float32 island", "module": n[0], "peak": self.v(n), "site": n,
               "why": "no proportional path back to a value that fits"}
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
        externals = []      # model inputs now have their own nodes ("<input>", ...), so they are ordinary writers
        peak = max(self.v(c) for c in comp)
        factor = pow2_at_least(peak / self.lim)
        non_norm = sorted({r[0] for r in readers if not self.is_norm(r)})
        fix = {"rule": "residual stream", "nodes": sorted(comp, key=lambda c: self.order.get(c, 0)), "peak": peak,
               "peak_at": max(comp, key=self.v), "factor": factor}
        if non_norm:
            fix.update(rule="residual stream in float32", why=f"read by non-norm modules: {non_norm[:4]}",
                       readers=sorted({r[0] for r in readers}),
                       readers_keep_float32=sorted({r[0] for r in readers if self.v(r) > FP16_MAX}),
                       first_value=sorted(w for w in writers if len(self.succs.get(w, ())) > 1 and self.v(w) > self.lim))
            for c in list(comp) + [w for w in writers if len(self.succs.get(w, ())) > 1]:   # + its first value
                self.covered[c] = fix
            return fix
        places, chains, ok = [], [], True
        for w in sorted(writers, key=lambda x: self.order.get(x, 0)):
            r, ch = self.shrink(w, factor)
            if r is None:
                ok = False
                break
            places += r
            chains += ch
        if externals:
            places += [("input", c) for c in externals]
        if not ok:
            fix.update(rule="residual stream in float32", why="a writer has no proportional path to a value that fits")
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
        c = next(iter(self.scalars[s]))
        mult = c if opname(s) in MUL else 1.0 / c       # the factor the scores get after the product
        if mult >= 1:
            return None
        need = 1.0 / mult
        for q in sorted(self.preds.get(n, ()), key=lambda p: ("q" not in p[0].split(".")[-1], self.order.get(p, 0))):
            r, ch = self.shrink_by(q, need)
            if r is not None:
                fix = {"rule": "move the attention scale before q @ k^T", "site": n, "peak": self.v(n),
                       "scale": mult, "move_from": s, "move_to": r, "after": self.v(n) * mult}
                if self.v(n) * mult > self.lim:
                    fix["note"] = "still over range after the move: add a power of two as well"
                for x in [n] + ch:
                    self.covered[x] = fix
                return fix
        return None

    def shrink_by(self, n, factor):
        """Like shrink, but always walks to the first module boundary (a norm or linear layer output) on the way."""
        chain, x = [], n
        while True:
            chain.append(x)
            if opname(x) in LIN or self.is_norm(x):
                return [("output", x)], chain
            ps = [p for p in self.preds.get(x, ()) if opname(x) in MUL | DIV | ADD or opname(x) in {"cat", "neg"}]
            ps = sorted(ps, key=lambda p: ("q" not in p[0].split(".")[-1], self.order.get(p, 0)))
            if not ps:
                return None, chain
            x = ps[0]

    def _product(self, n):
        consumers = [s for s in self.succs.get(n, ()) if opname(s) in LIN]
        if not consumers:
            return None
        factor = pow2_at_least(self.v(n) / self.lim)
        r, ch = self.shrink(n, factor)
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
        return "\n".join(lines)
