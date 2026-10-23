"""Step 3 of the FP16 fixer: apply a plan to a model at run time, with hooks -- no weight is changed.

The plan names operations as (module path, "op:<name>#<n>"), counted the way the Recorder counted them; the applier
keeps the same per-thread module stack and counters, so it recognises each planned operation and acts on its output:
- "shrink at": multiply that operation's output by 1 / factor (several fixes on one place multiply together);
- a model input (an "<input>" node): multiply the external tensor going into that operation;
- a moved attention scale: multiply the target (q_norm / q_proj output) by the scale, and turn the original multiply
  (or divide) after q @ k^T into a pass-through;
- a product fix: multiply the next linear layer's output back by the factor;
- norms: divide their epsilon by factor^2 (an attribute; restored on exit; a norm without one is an error);
- a float32 island: run every operation inside that module in float32; hand its output back in the model's dtype
  (or keep it in float32 when its peak doesn't fit float16).

    with Applied(model, plan.fixes):
        model(...)

A plan only fits the model code it was made on (the same module paths, the same operations in the same order: the
same attention implementation, eager or SDPA). So the first time a module runs, every planned place inside it must
have been met; if not, Applied raises (strict=False: warns) instead of silently doing part of the plan.
"""
import collections
import threading
import weakref

import torch
from torch.overrides import TorchFunctionMode

from .find import EPS_ATTRS as _EPS_ATTRS, FP16_MAX, _SKIP, _first_tensor, _tensors


def _inside(path, module):
    """path is module or runs inside it ("<root>" is the whole model)."""
    return module == "<root>" or path == module or path.startswith(module + ".")


def _node(n):
    return (n[0], n[1]) if isinstance(n, (list, tuple)) else n


class Applied:
    def __init__(self, model, fixes, strict=True):
        self.model, self.strict = model, strict
        self.out_scale = collections.defaultdict(lambda: 1.0)    # node -> multiply its output
        self.in_scale = collections.defaultdict(lambda: 1.0)     # node -> multiply its external tensor inputs
        self.passthrough = set()                                  # nodes whose scalar multiply / divide was moved
        self.eps_div = collections.defaultdict(lambda: 1.0)      # module path -> divide eps by
        self.mult_back = collections.defaultdict(lambda: 1.0)    # Linear module path -> multiply its output (float32)
        self.islands = {}                                         # module path -> keep its output in float32?
        self.fp32_ops = set()                                     # ops computed in float32 and left in float32
        for f in fixes:
            r = f["rule"]
            if r == "residual stream":
                for kind, node in f["shrink_at"]:
                    self._place(kind, _node(node), 1.0 / f["factor"])
                for m in f["norms"]:
                    self.eps_div[m] *= f["norms_eps_divide_by"]
            elif r.startswith("move the attention scale"):
                for kind, node in f["move_to"]:
                    self._place(kind, _node(node), f["scale"])
                self.passthrough.add(_node(f["move_from"]))
            elif r.startswith("shrink the product"):
                for kind, node in f["shrink_at"]:
                    self._place(kind, _node(node), 1.0 / f["factor"])
                for m in f["multiply_back_at"]:
                    self.mult_back[m] *= f["factor"]
            elif r == "float32 island":                              # an output that can't fit float16 stays float32
                self.islands[f["module"]] = f.get("keep_float32", f["peak"] > FP16_MAX)
            elif r == "residual stream in float32":
                for n in list(f.get("nodes", [])) + list(f.get("first_value", [])):
                    self.fp32_ops.add(_node(n))                       # the stream's adds (and first value), in float32
                keep = set(f.get("readers_keep_float32", []))
                for m in f.get("readers", []):
                    if m != "<output>":
                        self.islands[m] = m in keep                  # norms hand back float16
        # every place the plan names, to check it is met: operations, and modules (islands, multiply-backs)
        self._planned_ops = set(self.out_scale) | set(self.in_scale) | self.passthrough | self.fp32_ops
        self._planned_mods = set(self.islands) | set(self.mult_back)
        self._met, self._checked = set(), set()
        self._tl = threading.local()
        self._hooks, self._saved_eps, self._active = [], [], False

    def _place(self, kind, node, factor):
        if node[0] == "<input>":                                  # ("<input>", "into <path> <op>")
            path, op = node[1][len("into "):].rsplit(" ", 1)
            self.in_scale[(path, op)] *= factor
        elif kind == "output":
            self.out_scale[node] *= factor
        else:
            self.in_scale[node] *= factor

    # ---- per-thread state, as in the Recorder ------------------------------------------------------------------
    def _st(self):
        tl = self._tl
        if not hasattr(tl, "stack"):
            tl.stack, tl.counters, tl.mode, tl.owner, tl.busy = [], [], None, 0, False
            tl.made = {}                       # id(tensor) -> weakref: tensors produced inside the model
        return tl

    def _pre(self, path):
        def f(mod, args):
            tl = self._st()
            if self._active and tl.mode is None:
                tl.mode = self._make_mode()
                tl.mode.__enter__()
                tl.owner = len(tl.stack)
            tl.stack.append(path)
            tl.counters.append(collections.Counter())
            if path in self._planned_mods:
                self._met.add(path)
        return f

    def _post(self, path):
        def f(mod, args, out):
            tl = self._st()
            if tl.stack and tl.stack[-1] == path:
                tl.stack.pop()
                tl.counters.pop()
            if path in self.islands and not self.islands[path]:
                tl.busy = True                    # the island's output, back in the model's dtype
                try:
                    out = self._cast(out, self.dtype)
                finally:
                    tl.busy = False
            if path in self.mult_back:
                tl.busy = True                    # our own multiply must not be counted as one of the model's ops
                try:
                    out = self._scale(out, self.mult_back[path], float32=True)
                finally:
                    tl.busy = False
            if tl.mode is not None and len(tl.stack) == tl.owner:
                tl.mode.__exit__(None, None, None)
                tl.mode = None
                if path not in self._checked:     # the first time this module ran as the outermost call
                    self._checked.add(path)
                    self._check(path)
            return out
        return f

    def missed(self, under="<root>"):
        """Planned places inside module `under` that haven't been met yet."""
        return sorted([k for k in self._planned_ops if _inside(k[0], under) and k not in self._met], key=str) + \
            sorted(m for m in self._planned_mods if _inside(m, under) and m not in self._met)

    def _check(self, path):
        miss = self.missed(path)
        if not miss:
            return
        msg = (f"the plan doesn't match this model: {len(miss)} planned place(s) inside {path} never ran, e.g. "
               f"{miss[:4]}. A plan only fits the model code it was made on (same module paths, same attention "
               f"implementation -- eager vs SDPA). Make a new plan for this model.")
        if self.strict:
            raise RuntimeError(msg)
        import warnings
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

    @staticmethod
    def _cast(out, dtype):
        if isinstance(out, torch.Tensor):
            return out.to(dtype) if out.is_floating_point() else out
        if isinstance(out, tuple):
            return tuple(Applied._cast(o, dtype) for o in out)
        if hasattr(out, "last_hidden_state"):
            out.last_hidden_state = out.last_hidden_state.to(dtype)
        return out

    @staticmethod
    def _scale(out, factor, float32=False):
        if factor == 1.0:
            return out
        if isinstance(out, torch.Tensor):
            # multiply back in float32, hand back the dtype the model expects (the plan checked it fits)
            return (out.float() * factor).to(out.dtype) if float32 else out * factor
        if isinstance(out, tuple):
            return (Applied._scale(out[0], factor, float32),) + out[1:]
        return out

    def _make_mode(self):
        ap = self

        class Mode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                tl = ap._st()
                name = getattr(func, "__name__", str(func))
                if tl.busy or not tl.stack or name in _SKIP:
                    return func(*args, **kwargs)
                path = tl.stack[-1]
                cnt = tl.counters[-1]
                key = (path, f"op:{name}#{cnt[name]}")
                if key in ap._planned_ops:
                    ap._met.add(key)
                if key in ap.in_scale:                            # scale external tensors going into this op
                    f = ap.in_scale[key]
                    args = tuple(a * f if ap._external(a) else a for a in args)
                isl = next((m for m in ap.islands if _inside(path, m)), None)
                island = isl is not None or key in ap.fp32_ops
                keep32 = key in ap.fp32_ops or (isl is not None and ap.islands[isl])
                if island:
                    in_dt = next((a.dtype for a in _tensors(args) if a.is_floating_point()), None)
                    args = tuple(a.float() if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args)
                    kwargs = {k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in kwargs.items()}
                if key in ap.passthrough:                         # the scale was moved onto q: x * c -> x
                    out = next(a for a in args if isinstance(a, torch.Tensor))
                else:
                    out = func(*args, **kwargs)
                ft = _first_tensor(out)
                if isinstance(ft, torch.Tensor) and ft.is_floating_point():
                    cnt[name] += 1
                    if key in ap.out_scale:
                        out = Applied._scale(out, ap.out_scale[key])
                    # inside an island everything stays float32; only the island module's own output is handed
                    # back in the model's dtype (in _post), so a norm can square a float32 stream safely
                    for o in _tensors(out):
                        try:
                            tl.made[id(o)] = weakref.ref(o)
                        except TypeError:
                            pass
                    if len(tl.made) > 200000:
                        tl.made = {k: v for k, v in tl.made.items() if v() is not None}
                return out

        return Mode()

    def _external(self, a):
        if not isinstance(a, torch.Tensor) or not a.is_floating_point() or isinstance(a, torch.nn.Parameter):
            return False
        r = self._st().made.get(id(a))
        return r is None or r() is not a

    # ---- enter / exit ------------------------------------------------------------------------------------------
    def __enter__(self):
        self.dtype = next((p.dtype for p in self.model.parameters() if p.is_floating_point()), torch.float32)
        mods = dict(self.model.named_modules())
        for path, div in self.eps_div.items():
            mod = mods.get("" if path == "<root>" else path)
            if mod is None:
                self._restore_eps()
                raise ValueError(f"the plan divides the eps of norm {path}, which this model doesn't have")
            if not any(hasattr(mod, a) for a in _EPS_ATTRS):
                self._restore_eps()
                raise ValueError(f"the plan divides the eps of {path} ({type(mod).__name__}), but it has no eps "
                                 f"attribute ({', '.join(_EPS_ATTRS)}): the rescaled stream would be wrong there")
            for attr in _EPS_ATTRS:
                if hasattr(mod, attr):
                    old = getattr(mod, attr)
                    if old is None:          # nn.RMSNorm(eps=None) uses finfo(dtype).eps
                        dt = next((p.dtype for p in mod.parameters()), torch.float32)
                        old_val = torch.finfo(dt).eps
                    else:
                        old_val = old
                    self._saved_eps.append((mod, attr, old))
                    setattr(mod, attr, old_val / div)
                    break
        for name, mod in mods.items():
            path = name or "<root>"
            self._hooks.append(mod.register_forward_pre_hook(self._pre(path)))
            self._hooks.append(mod.register_forward_hook(self._post(path)))
        self._active = True
        return self

    def __exit__(self, *exc):
        self._active = False
        tl = self._st()
        if tl.mode is not None:
            tl.mode.__exit__(None, None, None)
            tl.mode = None
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._restore_eps()
        return False

    def _restore_eps(self):
        for mod, attr, old in reversed(self._saved_eps):
            setattr(mod, attr, old)
        self._saved_eps.clear()
