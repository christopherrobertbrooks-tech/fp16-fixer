"""Step 1 of the FP16 fixer: run a model once in float32 and record the largest magnitude every operation produces.

Two kinds of observer run together:
- module hooks: each module's output, each nn.Linear's input (a product such as act(gate) * up only exists as the
  next projection's input), and a stack of the modules currently running, so every operation is credited to the
  innermost module that ran it;
- a TorchFunctionMode: every torch function and tensor method that returns a floating tensor (matmul, add, mul,
  softmax inputs, ...), so attention scores and in-between products are seen without knowing the model's code.
  F.scaled_dot_product_attention never materialises its scores, so for it the scaled scores q @ k^T * scale are
  computed on the side ("op:sdpa_scores").

Maxima are kept on the GPU (torch.maximum) and read once at the end, so recording costs no syncs.

With graph=True (the default) it also keeps a data-flow map for the planner: every operation call is a node
(module path, "op:<name>#<n>", n = its count of that op within one call of the module, so the two residual adds of a
decoder layer are #0 and #1), with an edge from the node that produced each of its tensor inputs. Shape and dtype
operations (view, transpose, to, cat, ...) pass their input's producer through. Plain numbers an operation multiplies
or divides by are kept too (node_scalars), so an attention scale after q @ k^T can be found. Tensors the model
returns get an edge to the OUTPUT node, so the planner knows the caller reads them.
"""
import collections
import re
import threading
import weakref
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

FP16_MAX = 65504.0
EPS_ATTRS = ("variance_epsilon", "eps", "epsilon")
OUTPUT = ("<output>", "output")      # the graph node that stands for "returned from the model"
_SKIP = {"size", "dim", "shape", "numel", "dtype", "device", "is_floating_point", "__get__", "__len__", "item", "tolist",
         "detach", "to", "float", "half", "bfloat16", "contiguous", "view", "reshape", "transpose", "permute", "expand",
         "unsqueeze", "squeeze", "flatten", "unflatten", "__getitem__", "split", "chunk", "narrow", "clone", "type_as",
         "masked_fill", "zeros_like", "ones_like", "new_zeros", "new_ones", "repeat", "repeat_interleave", "cat", "stack"}


def _tensors(o):
    """Every tensor inside nested tuples / lists / dicts."""
    if isinstance(o, torch.Tensor):
        yield o
    elif isinstance(o, (tuple, list)):
        for x in o:
            yield from _tensors(x)
    elif isinstance(o, dict):
        for x in o.values():
            yield from _tensors(x)


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


def scale_free_eps(mod):
    """The name of mod's epsilon attribute if mod is a norm that ignores its input's overall scale (RMSNorm, LayerNorm,
    GroupNorm and the transformers *RMSNorm / *LayerNorm classes), else None. A norm with running statistics
    (BatchNorm) is not scale-free; one without an eps attribute can't have its eps adjusted, so it doesn't count."""
    if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) or getattr(mod, "running_mean", None) is not None:
        return None
    known = isinstance(mod, (torch.nn.RMSNorm, torch.nn.LayerNorm, torch.nn.GroupNorm))
    if not (known or re.search(r"(RMS|Layer|Group)Norm", type(mod).__name__)):
        return None
    return next((a for a in EPS_ATTRS if hasattr(mod, a)), None)


def _sdpa_scores(args, kwargs):
    """The scaled scores q @ k^T * scale an F.scaled_dot_product_attention call computes (without its mask)."""
    q, k = args[0], args[1]
    scale = kwargs.get("scale", args[6] if len(args) > 6 else None)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if q.dim() >= 3 and k.dim() == q.dim() and k.shape[-3] != q.shape[-3]:       # grouped-query attention
        k = k.repeat_interleave(q.shape[-3] // k.shape[-3], dim=-3)
    return (q.float() @ k.float().transpose(-2, -1)) * scale


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
        self.graph = True
        self.node_max, self.node_order, self.node_dtypes = {}, {}, {}   # node -> 0-d tensor / first seen / dtypes
        self.edges = set()                # (producer node, consumer node)
        self.node_scalars = {}            # node -> set of plain numbers in its arguments
        self.biased = set()               # linear-op nodes called with a bias (not proportional in their input)
        self.repeated = set()             # nodes with one producer's tensor in two arguments (x * x)
        self.scale_free = {}              # module path -> eps attribute name, for scale-free norms (scale_free_eps)
        self._prod = {}                   # id(tensor) -> (weakref, node): who produced a live tensor

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

    @property
    def _counters(self):
        c = getattr(self._tl, "counters", None)
        if c is None:
            c = self._tl.counters = []
        return c

    def _producer(self, t):
        e = self._prod.get(id(t))
        return e[1] if e is not None and e[0]() is t else None

    def _set_producer(self, t, node):
        try:
            self._prod[id(t)] = (weakref.ref(t), node)
        except TypeError:
            return
        if len(self._prod) > 200000:                # drop entries whose tensor is gone
            self._prod = {k: v for k, v in self._prod.items() if v[0]() is not None}

    def _rec_node(self, node, t):
        with torch.no_grad():
            m = t.detach().abs().amax().float()
        if node in self.node_max:
            self.node_max[node] = torch.maximum(self.node_max[node], m)
        else:
            self.node_max[node] = m
            self.node_order[node] = len(self.node_order)
        self.node_dtypes.setdefault(node, set()).add(str(t.dtype).replace("torch.", ""))

    def _pre(self, path, is_linear):
        def f(mod, args, kwargs=None):
            if self.ops and self._active and getattr(self._tl, "mode", None) is None:
                self._tl.mode = self._make_mode()
                self._tl.mode.__enter__()
                self._tl.owner_depth = len(self._stack)
            self._stack.append(path)
            self._counters.append(collections.Counter())
            if is_linear and args:
                self._rec(path, "input", args[0])
        return f

    def _post(self, path):
        def f(mod, args, out):
            self._rec(path, "output", _first_tensor(out))
            if path == "<root>" and self.graph:        # what the model returns is read by its caller
                for t in _tensors(out if not hasattr(out, "to_tuple") else out.to_tuple()):
                    src = self._producer(t)
                    if src is not None:
                        self.edges.add((src, OUTPUT))
            if self._stack and self._stack[-1] == path:
                self._stack.pop()
                if self._counters:
                    self._counters.pop()
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
                if not rec._stack or rec._busy:
                    return out
                if name in _SKIP:
                    if rec.graph:                     # shape / dtype ops: the output comes from the input's producer
                        src = next((rec._producer(a) for a in _tensors(args) if rec._producer(a) is not None), None)
                        if src is not None:
                            for o in _tensors(out):
                                rec._set_producer(o, src)
                    return out
                where = rec._stack[-1]
                if rec.graph and isinstance(_first_tensor(out), torch.Tensor) and _first_tensor(out).is_floating_point():
                    rec._busy = True
                    try:
                        cnt = rec._counters[-1] if rec._counters else collections.Counter()
                        node = (where, f"op:{name}#{cnt[name]}")
                        cnt[name] += 1
                        srcs = []
                        for a in _tensors((args, kwargs)):
                            src = rec._producer(a)
                            if src is None and a.is_floating_point() and not isinstance(a, torch.nn.Parameter) and a.numel():
                                src = ("<input>", f"into {node[0]} {node[1]}")   # a model input (or a buffer)
                                rec._set_producer(a, src)
                                rec._rec_node(src, a)
                            if src is not None:
                                rec.edges.add((src, node))
                                srcs.append(src)
                        if len(srcs) != len(set(srcs)):
                            rec.repeated.add(node)
                        nums = [x for x in list(args) + list(kwargs.values()) if isinstance(x, (int, float)) and not isinstance(x, bool)]
                        if nums:
                            rec.node_scalars.setdefault(node, set()).update(float(x) for x in nums)
                        if name == "linear" and (kwargs.get("bias") if len(args) < 3 else args[2]) is not None:
                            rec.biased.add(node)
                        for o in _tensors(out):
                            rec._set_producer(o, node)
                        ft = _first_tensor(out)
                        if ft.numel():
                            rec._rec_node(node, ft)
                    finally:
                        rec._busy = False
                if func is F.scaled_dot_product_attention:
                    rec._busy = True
                    try:
                        with torch.no_grad():
                            s = _sdpa_scores(args, kwargs)
                    finally:
                        rec._busy = False
                    rec._rec(where, "op:sdpa_scores", s)
                rec._rec(where, "op:" + name, _first_tensor(out))
                return out

        return Mode()

    def __enter__(self):
        for name, mod in self.model.named_modules():
            path = name or "<root>"
            self.mtype[path] = type(mod).__name__
            eps = scale_free_eps(mod)
            if eps is not None:
                self.scale_free[path] = eps
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

    def node_values(self):
        return {k: float(v) for k, v in self.node_max.items()}

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
