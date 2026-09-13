#!/usr/bin/env python3
"""Table of HumanEval+ / MBPP+ pass@1 across models, with paired per-task comparisons.

usage: qcode.py <label>=<prefix> ...    where <prefix> resolves to
       eval/score-<prefix>-humanevalplus.json and eval/score-<prefix>-mbppplus.json
       (and eval/genr-<prefix>-*.json for generation stats)
"""
import json, math, os, sys

BENCH = ["humanevalplus", "mbppplus"]
SHORT = {"humanevalplus": "HumanEval+", "mbppplus": "MBPP+"}
CEIL = {"humanevalplus": 163, "mbppplus": 378}   # canonical solutions that pass their own tests

models = []
for arg in sys.argv[1:]:
    label, _, pre = arg.partition("=")
    m = {"label": label, "score": {}, "gen": {}}
    for b in BENCH:
        sp, gp = f"eval/score-{pre}-{b}.json", f"eval/genr-{pre}-{b}.json"
        if os.path.exists(sp):
            m["score"][b] = json.load(open(sp))
        if os.path.exists(gp):
            m["gen"][b] = json.load(open(gp))
    models.append(m)

w = max(len(m["label"]) for m in models) + 1
print(f"{'model':{w}s} " + "  ".join(f"{SHORT[b]:>18s}" for b in BENCH) + f" {'combined':>10s}")
for m in models:
    row, tp, tn = f"{m['label']:{w}s} ", 0, 0
    for b in BENCH:
        d = m["score"].get(b)
        if not d:
            row += f"{'-':>20s}"; continue
        tp += d["passed"]; tn += d["n"]
        row += f"{d['passed']:7d}/{d['n']:<4d}{100*d['pass_rate']:6.1f}%"
    row += f" {100*tp/max(1,tn):9.1f}%"
    print(row)
print(f"{'(canonical ceiling)':{w}s} " + "  ".join(
    f"{CEIL[b]:7d}/{ {'humanevalplus':164,'mbppplus':378}[b]:<4d}{100*CEIL[b]/{'humanevalplus':164,'mbppplus':378}[b]:6.1f}%" for b in BENCH))

print("\ngeneration cost (after retrying anything that hit the first token cap):")
for m in models:
    for b in BENCH:
        g = m["gen"].get(b)
        if not g:
            continue
        sols = g["solutions"]
        tok = sum(s.get("completion_tokens", 0) for s in sols)
        capped = sum(1 for s in sols if s.get("finish_reason") == "length")
        print(f"  {m['label']:{w}s} {SHORT[b]:11s} {tok//max(1,len(sols)):5d} tok/task median-ish, "
              f"{capped:3d} still capped, {g.get('seconds',0)/60:5.1f} min wall")

have = [m for m in models if all(b in m["score"] for b in BENCH)]
if len(have) > 1:
    print("\npaired over the same tasks (row minus column, points; * = McNemar p<0.05):")
    print(" " * w + " " + " ".join(f"{m['label'][:11]:>12s}" for m in have))
    res = {m["label"]: {r["task_id"]: r["pass"] for b in BENCH for r in m["score"][b]["results"]}
           for m in have}
    for a in have:
        row = f"{a['label']:{w}s} "
        for bm in have:
            if a is bm:
                row += f"{'-':>12s}"; continue
            A, B = res[a["label"]], res[bm["label"]]
            common = sorted(set(A) & set(B))
            aw = sum(1 for t in common if A[t] and not B[t])
            bw = sum(1 for t in common if B[t] and not A[t])
            disc = aw + bw
            p = min(1.0, 2 * sum(math.comb(disc, i) for i in range(min(aw, bw) + 1)) / 2 ** disc) if disc else 1.0
            row += f"{100*(aw-bw)/len(common):+8.1f}{'*' if p < 0.05 else ' '}   "
        print(row)
