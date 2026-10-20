"""Shared bits for the drivers: print a find report and save it as JSON."""
import json
from fp16fix.find import FP16_MAX, summarize
from fp16fix.plan import Plan


def show(rec, title, out_json, headroom=4.0, top=25, half=None):
    """rec: the float32 pass; half: the float16 pass of the same inputs (optional)."""
    rows = rec.report(headroom, half_ops=half.dtypes if half else None)
    groups = summarize(rows)
    allv = rec.values()
    print(f"\n=== {title}: {len(rows)} tensors above {FP16_MAX / headroom:,.0f} (FP16 max / {headroom:g}); "
          f"largest anywhere (masks aside) {max((v for v in allv.values() if v < 3e38), default=0):,.0f}")
    print(f"{'site (layers collapsed)':58s} {'module':22s} {'what':26s} {'peak':>12s}  at")
    for (pat, mtype, kind), g in sorted(groups.items(), key=lambda kv: kv[1]["first"])[:top]:
        print(f"{pat[:58]:58s} {mtype[:22]:22s} {kind[:26]:26s} {g['max']:12,.0f}  {g['where']} ({g['n']} layer(s))")
    if half is not None:
        bad = half.first_nonfinite()
        print(f"plain float16 run: {'first non-finite at ' + bad[0][0] + ' ' + bad[0][1] if bad else 'no inf / nan'}"
              f"{f' ({len(bad)} sites in all)' if bad else ''}")
    plan = Plan(rec, half, headroom)
    print("PLAN:", plan.describe(), sep="\n")
    json.dump([{k: (v if not isinstance(v, (set, tuple)) else list(v)) for k, v in f.items() if k != "nodes"}
               for f in plan.fixes], open(out_json.replace(".json", ".plan.json"), "w"), indent=1, default=str)
    json.dump({"title": title, "headroom": headroom,
               "flagged": [{"path": p, "module": m, "kind": k, "max": v} for _, p, m, k, v in rows],
               "all": {f"{p}|{k}": v for (p, k), v in allv.items()},
               "fp16_first_nonfinite": [f"{p}|{k}" for p, k in (half.first_nonfinite() if half else [])]}, open(out_json, "w"), indent=1)
    return rows
