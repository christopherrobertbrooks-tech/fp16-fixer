"""Bake a plan into the model: write the fixes into the weights once, so the model runs at plain float16 speed.

    report = bake(model, plan.fixes, run)       # run(model): one example forward pass, used to check every edit
    if report.leftover:                          # what couldn't be baked still needs the run-time hooks
        with Applied(model, report.leftover): ...

What is baked (each edit is checked on the module's own recorded inputs; an edit that doesn't reproduce the scaled
output is not made, and the fix stays in `leftover`):
- "shrink at the output of X" / "multiply back after X", where X's last operation produces the module's output:
  scale the module's parameters (weights and biases: Linear, Conv, Embedding, RMSNorm / LayerNorm / GroupNorm); for
  norms that store their weight as an offset, output * (1 + w) (Gemma), w -> (1 + w) * f - 1 instead;
- a shrink at an operation inside a module whose only reader is a multiply that leads to a module's output (a norm's
  normalised value before its weight, an embedding before its scale): that module; a shrink at a product: the factor
  that a module computes, if that factor is read only by the product (scaling one factor scales the product);
- norms' epsilon: the attribute is set;
- a moved attention scale: q_norm / q_proj's output as above, and the attention module's own scale attribute
  (`scaling` / `scale` / `softmax_scale`, equal to the moved number) set to 1;
- float32 islands: the module is converted to float32, with module hooks that cast its inputs to float32 and its
  output back to the model's dtype (unless the plan keeps it float32). These are plain module hooks, not the
  per-operation function mode.
What stays: a shrink at a model input or at an operation inside a module that isn't any module's output, a moved
scale with no attribute to set, the operations of a float32 stream. `report.leftover` is a plan with only those.
"""
import collections
import copy
import math

import torch

from .find import EPS_ATTRS, Recorder, _first_tensor, _tensors
from .plan import MUL, DIV, LIN, EMB, opname

_SCALE_ATTRS = ("scaling", "scale", "softmax_scale")


def _node(n):
    return (n[0], n[1]) if isinstance(n, (list, tuple)) else n


class BakeReport:
    def __init__(self):
        self.baked = []        # (what, where, detail)
        self.leftover = []     # fixes (in plan format) that still need Applied
        self.hooks = []        # handles of the island hooks bake() added
        self.check = None      # end to end: baked vs Applied on the example run, relative difference

    def describe(self):
        c = collections.Counter(w for w, _, _ in self.baked)
        lines = [f"baked: {', '.join(f'{n} {w}' for w, n in c.items()) or 'nothing'}"
                 + (f"; end-to-end check vs Applied {self.check:.1e}" if self.check is not None else "")]
        for f in self.leftover:
            lines.append(f"left for Applied: {f['rule']}: {_leftover_size(f)}")
        return "\n".join(lines)


def _leftover_size(f):
    return {k: len(v) for k, v in f.items() if isinstance(v, list) and k in ("shrink_at", "move_to", "nodes",
                                                                              "multiply_back_at", "first_value")} or "1"


def _mods(model):
    return {(n or "<root>"): m for n, m in model.named_modules()}


def _capture(model, run, paths, fixes):
    """The first call's inputs of each module in `paths`, from a run with the plan applied (without it a float16 model
    overflows before reaching them), and that run's output."""
    from .apply import Applied
    seen, hooks, mods = {}, [], _mods(model)

    def hook(path):
        def f(mod, args, kwargs):
            if path not in seen:
                det = lambda o: o.detach().clone() if isinstance(o, torch.Tensor) else o
                seen[path] = (tuple(det(a) for a in args), {k: det(v) for k, v in kwargs.items()})
        return f
    for p in paths:
        if p in mods:
            hooks.append(mods[p].register_forward_pre_hook(hook(p), with_kwargs=True))
    try:
        with torch.no_grad(), Applied(model, fixes, strict=False):
            out = _first_tensor(run(model))
    finally:
        for h in hooks:
            h.remove()
    return seen, (out.detach().float().clone() if isinstance(out, torch.Tensor) else None)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def _owners(model):
    """id(parameter) -> the module paths that hold it directly (tied weights have more than one)."""
    own = collections.defaultdict(set)
    for path, m in _mods(model).items():
        for p in m.parameters(recurse=False):
            own[id(p)].add(path)
    return own


def _inside(path, module):
    return module == "<root>" or path == module or path.startswith(module + ".")


def _shared(mod, owners, path):
    """Does mod hold a parameter that a module outside it also uses (an output layer tied to the embedding)?"""
    return any(not all(_inside(o, path) for o in owners[id(p)]) for p in mod.parameters())


def _untie(mod, owners, path):
    """Give mod its own copies of the parameters it shares with modules outside it."""
    for name, p in list(mod.named_parameters()):
        if not all(_inside(o, path) for o in owners[id(p)]):
            sub, _, attr = name.rpartition(".")
            setattr(mod.get_submodule(sub) if sub else mod, attr, torch.nn.Parameter(p.data.clone(), requires_grad=p.requires_grad))


def _scale_module(mod, f, sample, tol, shared=False, untie=None):
    """Scale mod's output by f by editing it; check it on the recorded inputs against a float32 copy of the unedited
    module (times f). Candidate edits, in order: every parameter x f; offset weights ((1 + w) f - 1, Gemma's norms);
    every floating buffer x f (a scale kept as a buffer, like Gemma's embed_scale). A module whose parameters are
    shared with modules outside it (tied weights) tries its buffers first, then gets its own copies (`untie`).
    Returns (relative error, what was edited), or None if no edit reproduces it within tol."""
    args, kwargs = sample
    up = lambda o: o.float() if isinstance(o, torch.Tensor) and o.is_floating_point() else o
    ref = copy.deepcopy(mod).float()       # the unedited output in float32: in float16 it is what overflows
    with torch.no_grad():
        want = _first_tensor(ref(*[up(a) for a in args], **{k: up(v) for k, v in kwargs.items()}))
    del ref
    if not isinstance(want, torch.Tensor) or not torch.isfinite(want).all():
        return None
    want = want.float() * f

    def tensors():
        return [("p", n, p) for n, p in mod.named_parameters() if p.is_floating_point()] + \
               [("b", n, b) for n, b in mod.named_buffers() if b.is_floating_point()]

    def every_param(k, n, t):
        return (t.float() * f).to(t.dtype) if k == "p" else None

    def offset(k, n, t):           # output * (1 + w): w -> (1 + w) f - 1
        if k != "p":
            return None
        return ((1.0 + t.float()) * f - 1.0).to(t.dtype) if n.endswith("weight") else (t.float() * f).to(t.dtype)

    def buffers(k, n, t):
        return (t.float() * f).to(t.dtype) if k == "b" else None

    def attempt(edit, name):
        old = [(t, t.data.clone()) for _, _, t in tensors()]
        changed = False
        for k, n, t in tensors():
            new = edit(k, n, t)
            if new is not None:
                t.data, changed = new, True
        if not changed:
            return None
        with torch.no_grad():
            got = _first_tensor(mod(*args, **kwargs))
        err = _rel(got, want) if isinstance(got, torch.Tensor) and torch.isfinite(got).all() else math.inf
        if err <= tol:
            return err, name
        for t, d in old:
            t.data = d
        return None

    order = [(buffers, "buffers")] if shared else [(every_param, "weights"), (offset, "offset weights"), (buffers, "buffers")]
    for edit, name in order:
        r = attempt(edit, name)
        if r is not None:
            return r
    if shared and untie is not None:
        untie()
        for edit, name in ((every_param, "weights (untied)"), (offset, "offset weights (untied)")):
            r = attempt(edit, name)
            if r is not None:
                return r
    return None


def bake(model, fixes, run, tol=None):
    """Write `fixes` into `model` where it can be done exactly (see the module docstring). `run(model)` must run one
    representative forward pass. Returns a BakeReport; its `leftover` fixes still need `Applied`."""
    dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.float32)
    tol = tol if tol is not None else (1e-5 if dtype == torch.float32 else 1e-2)
    mods = _mods(model)
    with torch.no_grad(), Recorder(model) as rec:
        run(model)
    preds, succs = collections.defaultdict(set), collections.defaultdict(set)
    for a, b in rec.edges:
        preds[b].add(a)
        succs[a].add(b)
    out_of = {}           # node -> the innermost module whose output it is (a wrapper may return its child's output)
    for p, n in rec.module_out.items():
        if n is not None and p in mods and (n not in out_of or len(p) > len(out_of[n])):
            out_of[n] = p
    rep = BakeReport()

    def editable(m):
        return any(t.is_floating_point() for t in list(mods[m].parameters()) + list(mods[m].buffers()))

    def module_for(node, forward=True):
        """The module whose output scaling equals scaling `node`, or None: the module node is the output of; or, when
        node's only reader is a multiply (or a divide by a plain number) that is a module's output, that module (one
        step only: `node` itself is then left unscaled, and the planner checked it fits); or, for a product, a module
        with weights that computes one factor read only by the product."""
        node = _node(node)
        if node in out_of and editable(out_of[node]):
            return out_of[node]
        nxt = succs.get(node, set())
        if forward and len(nxt) == 1:
            (s,) = nxt
            if s not in rec.repeated and (opname(s) in MUL or opname(s) in DIV and preds.get(s) == {node}):
                m = module_for(s, forward=False)
                if m is not None:
                    return m
        if opname(node) in MUL and node not in rec.repeated:   # one factor of a product, computed by a module, read only here
            for p in sorted(preds.get(node, ()), key=str):
                if p in out_of and succs.get(p) == {node} and editable(out_of[p]):
                    return out_of[p]
        return None

    # 1. collect the module scalings each fix needs; anything that can't map stays in a leftover fix
    scale = collections.defaultdict(lambda: 1.0)
    pending = []          # (fix index, key, factor, module)
    left = [dict(f) for f in fixes]
    for i, f in enumerate(fixes):
        r = f["rule"]
        places = []
        if r in ("residual stream",) or r.startswith("shrink the product"):
            places = [("shrink_at", k, _node(n), 1.0 / f["factor"]) for k, n in f["shrink_at"]]
        elif r.startswith("move the attention scale"):
            places = [("move_to", k, _node(n), f["scale"]) for k, n in f["move_to"]]
        keep = collections.defaultdict(list)
        for field, kind, n, fac in places:
            if n[0] == "<input>":         # an external tensor into (path, op): scaling it scales a proportional op's output
                path, op = n[1][len("into "):].rsplit(" ", 1)
                target, ok = (path, op), op.split("#")[0][3:] in LIN | MUL | DIV
            elif kind == "input":         # the op's external inputs: the same as its output for a proportional op
                target, ok = n, opname(n) in LIN | EMB | MUL | DIV
            else:
                target, ok = n, True
            m = module_for(target) if ok else None
            if m is None:
                keep[field].append([kind, list(n)])
            else:
                scale[m] *= fac
                pending.append((i, field, [kind, list(n)], m))
        for field in ("shrink_at", "move_to"):
            if field in f:
                left[i][field] = keep[field]
        if r.startswith("shrink the product"):
            for m in f["multiply_back_at"]:
                scale[m] *= f["factor"]
            left[i]["multiply_back_at"] = []
    # 2. edit each module once (its factors multiplied), checked on its recorded inputs
    samples, want = _capture(model, run, [m for m in scale if scale[m] != 1.0], fixes)
    owners = _owners(model)
    failed = set()
    for m, fac in scale.items():
        if fac == 1.0:
            continue
        sh = _shared(mods[m], owners, m)
        r = _scale_module(mods[m], fac, samples[m], tol, sh, lambda: _untie(mods[m], owners, m)) if m in samples else None
        if r is None:
            failed.add(m)
        else:
            rep.baked.append(("module scalings", m, f"x{fac:g}, {r[1]} (check {r[0]:.1e})"))
    if failed:            # a module that couldn't be edited: its places go back to the leftover plan
        for i, field, place, m in pending:
            if m in failed:
                left[i].setdefault(field, []).append(place)
        for i, f in enumerate(fixes):
            if f["rule"].startswith("shrink the product"):
                left[i]["multiply_back_at"] = [m for m in f["multiply_back_at"] if m in failed]
    # 3. norms' epsilon, attention scale attributes, islands
    for i, f in enumerate(fixes):
        r = f["rule"]
        if r == "residual stream":
            for path in f["norms"]:
                mod = mods.get(path)
                attr = next((a for a in EPS_ATTRS if mod is not None and hasattr(mod, a)), None)
                if attr is None:
                    raise ValueError(f"the plan divides the eps of {path}, which has no eps attribute")
                old = getattr(mod, attr)
                old = torch.finfo(next((p.dtype for p in mod.parameters()), torch.float32)).eps if old is None else old
                setattr(mod, attr, old / f["norms_eps_divide_by"])
            rep.baked.append(("eps", f"{len(f['norms'])} norms", f"/ {f['norms_eps_divide_by']}"))
            left[i]["norms"] = []
        elif r.startswith("move the attention scale"):
            src = _node(f["move_from"])
            mod = mods.get(src[0])
            c = next(iter(rec.node_scalars.get(src, {None})))
            attr = next((a for a in _SCALE_ATTRS if mod is not None and isinstance(getattr(mod, a, None), (int, float))
                         and c is not None and math.isclose(getattr(mod, a), c, rel_tol=1e-6)), None)
            if attr is not None and not left[i]["move_to"]:
                setattr(mod, attr, 1.0)
                rep.baked.append(("attention scales", src[0], f"{attr} {c:g} -> 1"))
                left[i] = None
        elif r == "float32 island":
            m = f["module"]
            if m in mods and m != "<root>":                   # a whole-model island stays with Applied
                rep.hooks += _island(mods[m], dtype, f.get("keep_float32", f["peak"] > 65504.0))
                rep.baked.append(("float32 islands", m, ""))
                left[i] = None
        elif r == "residual stream in float32":                # its reading modules become islands; its adds stay
            keep = set(f.get("readers_keep_float32", []))
            for m in f.get("readers", []):
                if m in mods and m != "<root>":
                    rep.hooks += _island(mods[m], dtype, m in keep)
                    rep.baked.append(("float32 islands", m, ""))
            left[i]["readers"], left[i]["readers_keep_float32"] = [], []
    # a leftover fix that has nothing left to do is dropped
    rep.leftover = [f for f in left if f is not None and _still_needed(f)]
    # 4. end to end: the baked model (with any leftover applied) must give what Applied gave on the example run --
    #    an edit can leak where the module check can't see it (weights tied to another module)
    if want is not None:
        from .apply import Applied
        with torch.no_grad():
            if rep.leftover:
                with Applied(model, rep.leftover, strict=False):
                    got = _first_tensor(run(model))
            else:
                got = _first_tensor(run(model))
        rep.check = _rel(got, want) if torch.isfinite(got).all() else math.inf
        # float16 rounds differently with edited weights than with a run-time multiply; a noisy model (ACE-Step turbo's
        # first denoising step: 3.5% from float32 either way) differs by a few percent. A leak is a factor 2^k: >= 50%.
        limit = 1e-4 if dtype == torch.float32 else 0.1
        if not rep.check <= limit:
            raise ValueError(f"bake's end-to-end check failed: baked vs Applied differ by {rep.check:.3g} (> {limit:g}) "
                             "on the example run; the model was edited -- reload it")
    return rep


def _still_needed(f):
    r = f["rule"]
    if r == "residual stream" or r.startswith("shrink the product"):
        return bool(f.get("shrink_at")) or bool(f.get("multiply_back_at")) or bool(f.get("norms"))
    if r.startswith("move the attention scale"):
        return True
    if r == "residual stream in float32":
        return bool(f.get("nodes")) or bool(f.get("first_value"))
    return True


def _island(mod, dtype, keep32):
    """Run mod in float32: its parameters converted, its floating inputs cast up, its output cast back to dtype."""
    mod.float()
    up = lambda o: o.float() if isinstance(o, torch.Tensor) and o.is_floating_point() else o

    def pre(m, args, kwargs):
        return tuple(up(a) for a in args), {k: up(v) for k, v in kwargs.items()}

    def post(m, args, out):
        if keep32:
            return out
        if isinstance(out, torch.Tensor):
            return out.to(dtype) if out.is_floating_point() else out
        if isinstance(out, tuple):
            return tuple(o.to(dtype) if isinstance(o, torch.Tensor) and o.is_floating_point() else o for o in out)
        return out
    return [mod.register_forward_pre_hook(pre, with_kwargs=True), mod.register_forward_hook(post)]
