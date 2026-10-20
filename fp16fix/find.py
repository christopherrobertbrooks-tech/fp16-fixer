"""Step 1 of the FP16 fixer: run a model once in float32 and record the largest magnitude every operation produces.

Two kinds of observer run together:
- module hooks: each module's output, each nn.Linear's input (a product such as act(gate) * up only exists as the
  next projection's input), and a stack of the modules currently running, so every operation is credited to the
  innermost module that ran it;
- a TorchFunctionMode: every torch function and tensor method that returns a floating tensor (matmul, add, mul,
  softmax inputs, ...), so attention scores and in-between products are seen without knowing the model's code.
  F.scaled_dot_product_attention never materialises its scores, so for it the scores q @ k^T are computed on the side.

Maxima are kept on the GPU (torch.maximum) and read once at the end, so recording costs no syncs.
"""
import collections
import re
import threading
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

FP16_MAX = 65504.0
_SKIP = {"size", "dim", "shape", "numel", "dtype", "device", "is_floating_point", "__get__", "__len__", "item", "tolist",
         "detach", "to", "float", "half", "bfloat16", "contiguous", "view", "reshape", "transpose", "permute", "expand",
         "unsqueeze", "squeeze", "flatten", "unflatten", "__getitem__", "split", "chunk", "narrow", "clone", "type_as",
         "masked_fill", "zeros_like", "ones_like", "new_zeros", "new_ones", "repeat", "repeat_interleave", "cat", "stack"}


def _first_tensor(o):
    if isinstance(o, torch.Tensor):
        return o
    if isinstance(o, (tuple, list)):
        for x in o:
            t = _first_tensor(x)
            if t is not None:
                return t
    if hasattr(o, "last_hidden_state"):
        return o.last_hidden_state
    return None


class Recorder:
    """with Recorder(model) as rec: model(...); then rec.report()."""

    def __init__(self, model, ops=True):
        self.model, self.ops = model, ops
        self.maxima = {}                  # (module path, kind) -> 0-d tensor
        self._bad = {}                    # (module path, kind) -> 0-d bool tensor: saw inf / nan
        self.order = {}                   # (module path, kind) -> first time seen
        self.mtype = {}                   # module path -> class name
        self._hooks, self._active = [], False
        self._tl = threading.local()      # per thread: the module stack, the function mode, and a "busy" flag --
                                          # pipelines often run the model in a worker thread, and torch's function
                                          # modes are per thread, so each thread gets its own, opened by its first
                                          # module call and closed when that call returns
        self.dtypes = {}                  # (module path, kind) -> set of dtype names it produced
        self.nonfinite = {}               # (module path, kind) -> first time a non-finite value appeared

    # ---- recording -------------------------------------------------------------------------------------------
    @property
    def _stack(self):
        st = getattr(self._tl, "stack", None)
        if st is None:
            st = self._tl.stack = []
        return st

    @property
    def _busy(self):
        return getattr(self._tl, "busy", False)

    @_busy.setter
    def _busy(self, v):
        self._tl.busy = v

    def _rec(self, path, kind, t):
        if self._busy or not isinstance(t, torch.Tensor) or not t.is_floating_point() or t.numel() == 0:
            return
        self._busy = True
        try:
            with torch.no_grad():
                a = t.detach().abs()
                m = a.amax().float()
                bad = ~torch.isfinite(a).all()
            k = (path, kind)
            if k in self.maxima:
                self.maxima[k] = torch.maximum(self.maxima[k], m)
                self._bad[k] = self._bad[k] | bad
            else:
                self.maxima[k] = m
                self._bad[k] = bad
                self.order[k] = len(self.order)
            self.dtypes.setdefault(k, set()).add(str(t.dtype).replace("torch.", ""))
        finally:
            self._busy = False

    def _pre(self, path, is_linear):
        def f(mod, args, kwargs=None):
            if self.ops and self._active and getattr(self._tl, "mode", None) is None:
                self._tl.mode = self._make_mode()
                self._tl.mode.__enter__()
                self._tl.owner_depth = len(self._stack)
            self._stack.append(path)
            if is_linear and args:
                self._rec(path, "input", args[0])
        return f

    def _post(self, path):
        def f(mod, args, out):
            self._rec(path, "output", _first_tensor(out))
            if self._stack and self._stack[-1] == path:
                self._stack.pop()
            mode = getattr(self._tl, "mode", None)
            if mode is not None and len(self._stack) == self._tl.owner_depth:
                mode.__exit__(None, None, None)      # the call that opened this thread's mode has returned
                self._tl.mode = None
        return f

    def _make_mode(self):
        rec = self

        class Mode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                out = func(*args, **kwargs)
                name = getattr(func, "__name__", str(func))
                if name in _SKIP or not rec._stack or rec._busy:
                    return out
                where = rec._stack[-1]
                if func is F.scaled_dot_product_attention:
                    q, k = args[0], args[1]
                    rec._busy = True
                    try:
                        with torch.no_grad():
                            s = q.float() @ k.float().transpose(-2, -1)
                    finally:
                        rec._busy = False
                    rec._rec(where, "op:sdpa_scores_unscaled", s)
                rec._rec(where, "op:" + name, _first_tensor(out))
                return out

        return Mode()

    def __enter__(self):
        for name, mod in self.model.named_modules():
            path = name or "<root>"
            self.mtype[path] = type(mod).__name__
            lin = isinstance(mod, torch.nn.Linear)
            self._hooks.append(mod.register_forward_pre_hook(self._pre(path, lin)))
            self._hooks.append(mod.register_forward_hook(self._post(path)))
        self._active = True
        return self

    def __exit__(self, *exc):
        self._active = False
        mode = getattr(self._tl, "mode", None)
        if mode is not None:
            mode.__exit__(None, None, None)
            self._tl.mode = None
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        return False

    # ---- reporting -------------------------------------------------------------------------------------------
    def values(self):
        return {k: float(v) for k, v in self.maxima.items()}

    def first_nonfinite(self):
        """Keys that produced inf / nan, in the order they first ran (the first one is where a float16 run breaks)."""
        return sorted((k for k, b in self._bad.items() if bool(b)), key=lambda k: self.order[k])

    def report(self, headroom=4.0, top=40, half_ops=None):
        """Everything above FP16_MAX / headroom, in the order it first ran. Returns the rows.

        half_ops: the dtypes seen per key in a float16 pass of the same model (Recorder.dtypes); keys that ran in
        float32 there (a norm upcasting itself, say) are dropped, since they can't overflow float16."""
        lim = FP16_MAX / headroom
        vals = self.values()
        def runs_in_half(k):
            return half_ops is None or k not in half_ops or "float16" in half_ops[k]
        # masks: "minus the largest number" on purpose (torch.finfo(float32).min, or float16's -65504 in a half pass)
        sentinel = 0.99 * torch.finfo(torch.float32).max
        rows = [(self.order[k], k[0], self.mtype.get(k[0], "?"), k[1], v) for k, v in vals.items()
                if lim < v < sentinel and runs_in_half(k)]
        rows.sort()
        return rows


def summarize(rows, collapse_layers=True):
    """Group rows like 'decoder.layers.12.mlp.down_proj' across layer numbers: one line per site pattern."""
    groups = collections.OrderedDict()
    for order, path, mtype, kind, v in rows:
        pat = re.sub(r"\.(\d+)(\.|$)", r".N\2", path) if collapse_layers else path
        g = groups.setdefault((pat, mtype, kind), {"max": 0.0, "where": None, "n": 0, "first": order})
        g["n"] += 1
        if v > g["max"]:
            g["max"], g["where"] = v, path
    return groups
