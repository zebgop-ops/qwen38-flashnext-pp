#!/usr/bin/env python3
"""Cross-model quality table from quality.py runs.

Perplexity is per token, so it is NOT comparable across tokenizers; bits per byte is, and
is what the table ranks on. MMLU is scored on the same 1000 questions for every model, so
the pairwise agreement and McNemar tests below are paired.

usage: qtable.py <label>=<eval.txt>[:<mc-dump.json>] ...
"""
import json, math, sys

def ensure_bpb(d, corpus_dir="eval", window=8192):
    """Older runs predate the bits_per_byte field; windows are deterministic character
    offsets, so it can be recomputed from the corpus."""
    if "bits_per_byte" in d:
        return d
    text = open(f"{corpus_dir}/{d['file']}").read().lstrip("\n ")
    n = len(d["per_window_tokens"])
    by = sum(len(text[i * window:(i + 1) * window].encode()) for i in range(n))
    nats = sum(t * l for t, l in zip(d["per_window_tokens"], d["per_window_nll"]))
    d["bits_per_byte"] = nats / math.log(2) / by
    d["bytes"] = by
    return d


models = []
for arg in sys.argv[1:]:
    label, _, paths = arg.partition("=")
    evalf, _, dumpf = paths.partition(":")
    m = {"label": label, "ppl": {}, "mmlu": None, "dump": None}
    for line in open(evalf):
        line = line.strip()
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        if d["metric"] == "perplexity":
            m["ppl"][d["file"]] = ensure_bpb(d)
        elif d["metric"] == "mmlu" and d["n"] >= 1000:
            m["mmlu"] = d
    if dumpf:
        m["dump"] = {r["i"]: r for r in json.load(open(dumpf))}
    models.append(m)

corpora = [c for c in ("wikitext2-test.txt", "stdlib-code.txt") if all(c in m["ppl"] for m in models)]
w = max(len(m["label"]) for m in models) + 1
print(f"{'model':{w}s} " + " ".join(f"{c.split('-')[0]:>12s} bpb" for c in corpora)
      + f" {'MMLU':>7s} {'margin':>7s}")
for m in models:
    row = f"{m['label']:{w}s} "
    for c in corpora:
        row += f"{m['ppl'][c]['bits_per_byte']:16.4f}"
    if m["mmlu"]:
        row += f" {100*m['mmlu']['accuracy']:6.1f}% {m['mmlu']['mean_margin']:7.2f}"
    print(row)

print("\nper-token perplexity (only comparable within a tokenizer family):")
for m in models:
    print(f"  {m['label']:{w}s} " + "  ".join(
        f"{c.split('-')[0]}: {m['ppl'][c]['ppl']:7.4f} over {m['ppl'][c]['tokens']:6d} tokens" for c in corpora))

have = [m for m in models if m["dump"]]
if len(have) > 1:
    print("\npaired MMLU over the same questions (row beats column, by points):")
    print(" " * w + " " + " ".join(f"{m['label'][:11]:>12s}" for m in have))
    for a in have:
        row = f"{a['label']:{w}s} "
        for b in have:
            if a is b:
                row += f"{'-':>12s}"; continue
            common = sorted(set(a["dump"]) & set(b["dump"]))
            aw = sum(1 for i in common if a["dump"][i]["hit"] and not b["dump"][i]["hit"])
            bw = sum(1 for i in common if b["dump"][i]["hit"] and not a["dump"][i]["hit"])
            disc = aw + bw
            p = min(1.0, 2 * sum(math.comb(disc, i) for i in range(min(aw, bw) + 1)) / 2 ** disc) if disc else 1.0
            row += f"{100*(aw-bw)/len(common):+8.1f}{'*' if p < 0.05 else ' ':1s}   "
        print(row)
    print("  (* significant at p<0.05, McNemar exact on the discordant pairs)")
    base = have[0]
    print("\nagreement with " + base["label"] + " (same predicted letter):")
    for b in have[1:]:
        common = sorted(set(base["dump"]) & set(b["dump"]))
        same = sum(1 for i in common if base["dump"][i]["pred"] == b["dump"][i]["pred"])
        print(f"  {b['label']:{w}s} {same/len(common):.3f}")
