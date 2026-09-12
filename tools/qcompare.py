#!/usr/bin/env python3
"""Compare two quality.py runs: perplexity deltas and a paired (McNemar) MMLU test.
usage: qcompare.py <base.txt> <reap.txt> [base-mc.json reap-mc.json]"""
import json, math, sys, re


def metrics(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line.startswith("{"):
            d = json.loads(line)
            out[d.get("file", d["metric"])] = d
    return out


b, r = metrics(sys.argv[1]), metrics(sys.argv[2])
print(f"{'metric':28s} {'unpruned':>12s} {'REAP-272E':>12s} {'delta':>12s}")
for k in b:
    if k in r and "ppl" in b[k]:
        db, dr = b[k], r[k]
        assert db["per_window_tokens"] == dr["per_window_tokens"], f"token counts differ for {k}"
        print(f"{'ppl ' + k:28s} {db['ppl']:12.4f} {dr['ppl']:12.4f} "
              f"{100*(dr['ppl']/db['ppl']-1):+11.2f}%   ({db['tokens']} tokens)")
        # paired over windows: mean per-window nll difference and its standard error
        d = [x - y for x, y in zip(dr["per_window_nll"], db["per_window_nll"])]
        m = sum(d) / len(d)
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
        se = sd / math.sqrt(len(d))
        print(f"{'  per-window nll delta':28s} {'':12s} {'':12s} {m:+12.4f}   "
              f"+-{1.96*se:.4f} (95% CI over {len(d)} windows)")
for k in ("mmlu",):
    if k in b and k in r:
        print(f"{'MMLU accuracy':28s} {b[k]['accuracy']:12.4f} {r[k]['accuracy']:12.4f} "
              f"{100*(r[k]['accuracy']-b[k]['accuracy']):+11.2f} pts  (n={b[k]['n']})")
        print(f"{'MMLU mean margin (nats)':28s} {b[k]['mean_margin']:12.3f} {r[k]['mean_margin']:12.3f} "
              f"{r[k]['mean_margin']-b[k]['mean_margin']:+12.3f}")

if len(sys.argv) > 4:
    B = {d["i"]: d for d in json.load(open(sys.argv[3]))}
    R = {d["i"]: d for d in json.load(open(sys.argv[4]))}
    common = sorted(set(B) & set(R))
    b_only = sum(1 for i in common if B[i]["hit"] and not R[i]["hit"])
    r_only = sum(1 for i in common if R[i]["hit"] and not B[i]["hit"])
    both = sum(1 for i in common if R[i]["hit"] and B[i]["hit"])
    same_pred = sum(1 for i in common if R[i]["pred"] == B[i]["pred"])
    n = len(common)
    print(f"\npaired MMLU over {n} identical questions")
    print(f"  both correct {both}, unpruned only {b_only}, REAP only {r_only}, "
          f"both wrong {n - both - b_only - r_only}")
    print(f"  same predicted letter: {same_pred}/{n} = {same_pred/n:.3f}")
    disc = b_only + r_only
    if disc:
        # exact binomial two-sided p on the discordant pairs (McNemar)
        k = min(b_only, r_only)
        p = min(1.0, 2 * sum(math.comb(disc, i) for i in range(k + 1)) / 2 ** disc)
        print(f"  McNemar on {disc} discordant pairs: p = {p:.4f}"
              f"{'  (no significant difference at 0.05)' if p > 0.05 else '  (significant at 0.05)'}")
    # which subjects lose most (paired, only subjects with enough questions)
    per = {}
    for i in common:
        a = per.setdefault(B[i]["subject"], [0, 0, 0])
        a[0] += B[i]["hit"]; a[1] += R[i]["hit"]; a[2] += 1
    worst = sorted(((v[1] - v[0]) / v[2], k, v) for k, v in per.items() if v[2] >= 12)
    print("  biggest paired swings (subjects with >=12 questions):")
    for d, k, v in worst[:5]:
        print(f"    {k:38s} {v[0]:3d} -> {v[1]:3d} of {v[2]:3d}   {100*d:+6.1f} pts")
    for d, k, v in worst[-3:]:
        print(f"    {k:38s} {v[0]:3d} -> {v[1]:3d} of {v[2]:3d}   {100*d:+6.1f} pts")
    mb = sum(B[i]["lp"][B[i]["gold"]] - max(v for t, v in B[i]["lp"].items() if t != B[i]["gold"])
             for i in common if len(B[i]["lp"]) == 4) / n
    mr = sum(R[i]["lp"][R[i]["gold"]] - max(v for t, v in R[i]["lp"].items() if t != R[i]["gold"])
             for i in common if len(R[i]["lp"]) == 4) / n
    print(f"  mean correct-vs-best-distractor margin: unpruned {mb:.3f}, REAP {mr:.3f} ({mr-mb:+.3f} nats)")
