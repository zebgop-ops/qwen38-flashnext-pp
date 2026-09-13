#!/usr/bin/env python3
"""Render the cross-model quality report from the saved JSON, so no number is hand-copied.

usage: qreport.py > CROSS-MODEL.md
"""
import json, math, os

MODELS = [("Qwen3.8-Flash-Next FP8", "qwen38", "eval/qwen38.txt", "eval/mc-qwen38-1000.json"),
          ("GLM-5.3-Flash W4A16", "glm53", "eval/glm53.txt", "eval/mc-glm53-1000.json"),
          ("DeepSeek-V4.1-Flash (unpruned)", "base", "eval/base.txt", "eval/mc-base-1000.json"),
          ("DeepSeek-V4.1-Flash REAP-272E", "reap", "eval/reap.txt", "eval/mc-reap-1000.json")]
SPEED = {"qwen38": ("74 tok/s", "~10k tok/s", "262k"),
         "glm53": ("66-70 tok/s", "~3.2k tok/s", "512k"),
         "base": ("14-22 tok/s", "~300 tok/s", "131k"),
         "reap": ("~30 tok/s", "~3k tok/s", "512k")}
BENCH = [("humanevalplus", "HumanEval+", 164), ("mbppplus", "MBPP+", 378)]


def bpb(evalfile, corpus):
    for line in open(evalfile):
        line = line.strip()
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        if d.get("metric") == "perplexity" and d["file"] == corpus:
            if "bits_per_byte" in d:
                return d["bits_per_byte"]
            text = open("eval/" + corpus).read().lstrip("\n ")
            n = len(d["per_window_tokens"])
            by = sum(len(text[i * 8192:(i + 1) * 8192].encode()) for i in range(n))
            nats = sum(t * l for t, l in zip(d["per_window_tokens"], d["per_window_nll"]))
            return nats / math.log(2) / by
    return None


def mmlu(evalfile):
    best = None
    for line in open(evalfile):
        line = line.strip()
        if line.startswith("{"):
            d = json.loads(line)
            if d.get("metric") == "mmlu" and d["n"] >= 1000:
                best = d
    return best


def code(tag):
    out = {}
    for b, _, total in BENCH:
        p = f"eval/score-{tag}-{b}.json"
        if os.path.exists(p) and os.path.getsize(p):
            d = json.load(open(p))
            out[b] = (d["passed"], d["n"])
    return out


rows = []
for label, tag, evalfile, dump in MODELS:
    rows.append({"label": label, "tag": tag,
                 "wiki": bpb(evalfile, "wikitext2-test.txt"),
                 "code_bpb": bpb(evalfile, "stdlib-code.txt"),
                 "mmlu": mmlu(evalfile), "code": code(tag),
                 "dump": {r["i"]: r for r in json.load(open(dump))} if os.path.exists(dump) else {}})

print("# Four models on this box, one harness, identical inputs\n")
print("Measured 2026-09-12. Each model served by its own stack on the four CMP 170HX cards, one at a")
print("time. Perplexity per token is not comparable across tokenizers, so language modelling is")
print("reported as **bits per byte**. MMLU uses the **same 1000 questions** for every model, cloze-scored")
print("through the raw completions API (no chat template, so no thinking, 2-shot format anchor).")
print("HumanEval+ / MBPP+ are pass@1 with the generated code executed against the EvalPlus tests in a")
print("sandboxed container; anything that hit the first 4k token budget was retried at 16k so a verbose")
print("reasoner is not scored down for thinking past the cap.\n")

print("| model | wikitext bits/byte | code bits/byte | MMLU (1000q) | HumanEval+ | MBPP+ |")
print("|---|---|---|---|---|---|")
for r in rows:
    he = r["code"].get("humanevalplus")
    mb = r["code"].get("mbppplus")
    print(f"| {r['label']} | {r['wiki']:.4f} | {r['code_bpb']:.4f} | "
          f"{100*r['mmlu']['accuracy']:.1f}% | "
          + (f"{100*he[0]/he[1]:.1f}% ({he[0]}/{he[1]})" if he else "-") + " | "
          + (f"{100*mb[0]/mb[1]:.1f}% ({mb[0]}/{mb[1]})" if mb else "-") + " |")
print("| *canonical ceiling* | | | | 99.4% (163/164) | 100% (378/378) |")

print("\n## With speed (single stream, from each repo's results)\n")
print("| model | decode | prefill | context |")
print("|---|---|---|---|")
for r in rows:
    d, p, c = SPEED[r["tag"]]
    print(f"| {r['label']} | {d} | {p} | {c} |")


def mcnemar(A, B, key):
    common = sorted(set(A) & set(B))
    aw = sum(1 for t in common if key(A[t]) and not key(B[t]))
    bw = sum(1 for t in common if key(B[t]) and not key(A[t]))
    disc = aw + bw
    p = (min(1.0, 2 * sum(math.comb(disc, i) for i in range(min(aw, bw) + 1)) / 2 ** disc)
         if disc else 1.0)
    return 100 * (aw - bw) / len(common), p, len(common)


print("\n## Paired tests (row minus column, in points; * = McNemar p < 0.05)\n")
print("### MMLU, same 1000 questions\n")
have = [r for r in rows if r["dump"]]
print("| | " + " | ".join(r["label"].split()[0] for r in have) + " |")
print("|---" * (len(have) + 1) + "|")
for a in have:
    cells = []
    for b in have:
        if a is b:
            cells.append("-"); continue
        d, p, _ = mcnemar(a["dump"], b["dump"], lambda x: x["hit"])
        cells.append(f"{d:+.1f}{'*' if p < 0.05 else ''}")
    print(f"| **{a['label'].split()[0]}** | " + " | ".join(cells) + " |")

code_res = {}
for r in rows:
    acc = {}
    for b, _, _ in BENCH:
        p = f"eval/score-{r['tag']}-{b}.json"
        if os.path.exists(p) and os.path.getsize(p):
            for x in json.load(open(p))["results"]:
                acc[b + ":" + str(x["task_id"])] = x["pass"]
    if acc:
        code_res[r["label"]] = acc
cl = [r for r in rows if r["label"] in code_res]
if len(cl) > 1:
    print("\n### HumanEval+ and MBPP+ combined, same tasks\n")
    print("| | " + " | ".join(r["label"].split()[0] for r in cl) + " |")
    print("|---" * (len(cl) + 1) + "|")
    for a in cl:
        cells = []
        for b in cl:
            if a is b:
                cells.append("-"); continue
            d, p, _ = mcnemar(code_res[a["label"]], code_res[b["label"]], lambda x: x)
            cells.append(f"{d:+.1f}{'*' if p < 0.05 else ''}")
        print(f"| **{a['label'].split()[0]}** | " + " | ".join(cells) + " |")
