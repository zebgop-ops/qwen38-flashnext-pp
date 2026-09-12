#!/usr/bin/env python3
"""Quality comparison between two checkpoints served by the same stack (stdlib only).

Both metrics are prefill-only and deterministic, so a run is reproducible and the two
checkpoints see byte-identical inputs (same tokenizer -> same token sequences).

  ppl  <url> <textfile> [max_tokens_total] [window_chars]
       Perplexity and bits-per-byte from the server's `prompt_logprobs`: sum log P(token | prefix) over a
       fixed set of windows. Windows are cut by character offset, so they are identical
       across models; the per-window token counts are printed and must match.

  mc   <url> <mmlu.json> [n] [seed]   (DSV41_MC_DUMP=<file> writes per-question records,
       so two runs can be compared question by question with a paired test)
       Multiple choice via the raw completions API (no chat template, so no thinking):
       the prompt ends in "Answer:" and the first generated token's top-k logprobs are
       read for " A".." D". Reports accuracy, the mean logprob margin between the correct
       option and the best distractor, and how often all four letters were in the top-k.

usage: quality.py ppl|mc <url> <file> [...]   (DSV41_MODEL names the served model)
"""
import json, math, os, sys, random, time, urllib.request

MODEL = os.environ.get("DSV41_MODEL", "DSv41Flash")


def post(url, path, body, timeout=3600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        # servers started without a raised --max-logprobs reject large logprobs asks
        if e.code == 400 and body.get("logprobs", 0) > 5:
            return post(url, path, dict(body, logprobs=5), timeout)
        detail = e.read()[:300].decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} from {path}: {detail}") from None


def perplexity(url, path, budget=65536, window_chars=8192):
    text = open(path).read()
    # skip the leading blank/header lines some corpora start with
    text = text.lstrip("\n ")
    windows, i = [], 0
    while i + window_chars <= len(text) and sum(w[1] for w in windows) < budget * 4:
        windows.append((i, window_chars))
        i += window_chars
    total_lp, total_tok, per_window = 0.0, 0, []
    t0 = time.time()
    for k, (off, n) in enumerate(windows):
        chunk = text[off:off + n]
        r = post(url, "/v1/completions", {
            "model": MODEL, "prompt": chunk, "max_tokens": 1, "temperature": 0,
            "prompt_logprobs": 0, "echo": False})
        pl = r["choices"][0]["prompt_logprobs"] or []
        lps = [next(iter(d.values()))["logprob"] for d in pl if d]
        total_lp += sum(lps); total_tok += len(lps)
        per_window.append((len(lps), sum(lps)))
        if (k + 1) % 8 == 0 or k + 1 == len(windows):
            print(f"  window {k+1:3d}/{len(windows)}: {total_tok:7d} tokens, "
                  f"ppl so far {math.exp(-total_lp/total_tok):8.4f}  ({time.time()-t0:5.0f} s)",
                  flush=True)
    ppl = math.exp(-total_lp / total_tok)
    # bits per byte is tokenizer-independent, so it compares across model families
    total_bytes = sum(len(text[o:o + n].encode()) for o, n in windows)
    bpb = (-total_lp / math.log(2)) / total_bytes
    print(json.dumps({"metric": "perplexity", "file": os.path.basename(path), "model": MODEL,
                      "tokens": total_tok, "windows": len(windows), "nll": -total_lp / total_tok,
                      "bytes": total_bytes, "bits_per_byte": bpb,
                      "ppl": ppl, "per_window_tokens": [w[0] for w in per_window],
                      "per_window_nll": [-w[1] / w[0] for w in per_window]}))
    return ppl


LETTERS = ["A", "B", "C", "D"]


def mc_prompt(row, shots):
    p = ""
    for s in shots:
        p += (f"The following is a multiple choice question about "
              f"{s['subject'].replace('_',' ')}.\n\n{s['q'].strip()}\n"
              + "".join(f"{L}. {c}\n" for L, c in zip(LETTERS, s["choices"]))
              + f"Answer: {LETTERS[s['answer']]}\n\n")
    p += (f"The following is a multiple choice question about "
          f"{row['subject'].replace('_',' ')}.\n\n{row['q'].strip()}\n"
          + "".join(f"{L}. {c}\n" for L, c in zip(LETTERS, row["choices"]))
          + "Answer:")
    return p


def multiple_choice(url, path, n=400, seed=0):
    rows = json.load(open(path))
    dump_path, dump = os.environ.get("DSV41_MC_DUMP"), []
    rnd = random.Random(seed)
    idx = list(range(len(rows)))
    rnd.shuffle(idx)
    shots = [rows[i] for i in idx[:2]]            # 2-shot format anchor, excluded from scoring
    sample = [rows[i] for i in idx[2:2 + n]]
    correct = margins = 0.0
    seen_all = ok = 0
    per_subject = {}
    t0 = time.time()
    for k, row in enumerate(sample):
        r = post(url, "/v1/completions", {
            "model": MODEL, "prompt": mc_prompt(row, shots), "max_tokens": 1,
            "temperature": 0, "logprobs": 20})
        top = r["choices"][0]["logprobs"]["top_logprobs"][0]
        lp = {}
        for tok, v in top.items():
            t = tok.strip()
            if t in LETTERS and t not in lp:
                lp[t] = v
        if len(lp) == 4:
            seen_all += 1
        if not lp:
            continue
        ok += 1
        pred = max(lp, key=lp.get)
        gold = LETTERS[row["answer"]]
        hit = pred == gold
        correct += hit
        if gold in lp and len(lp) > 1:
            best_wrong = max(v for t, v in lp.items() if t != gold)
            margins += lp[gold] - best_wrong
        s = per_subject.setdefault(row["subject"], [0, 0])
        s[0] += hit; s[1] += 1
        if dump_path:
            dump.append({"i": idx[2 + k], "subject": row["subject"], "gold": gold,
                         "pred": pred, "hit": bool(hit),
                         "lp": {t: round(v, 4) for t, v in sorted(lp.items())}})
        if (k + 1) % 50 == 0:
            print(f"  {k+1:4d}/{len(sample)}: acc {correct/max(1,ok):.3f}  ({time.time()-t0:5.0f} s)", flush=True)
    if dump_path:
        json.dump(dump, open(dump_path, "w"))
    print(json.dumps({"metric": "mmlu", "model": MODEL, "n": ok, "requested": len(sample),
                      "accuracy": correct / max(1, ok), "mean_margin": margins / max(1, ok),
                      "all_four_letters_in_topk": seen_all / max(1, ok),
                      "per_subject": {k: v for k, v in sorted(per_subject.items())}}))
    return correct / max(1, ok)


if __name__ == "__main__":
    mode, url, path = sys.argv[1], sys.argv[2], sys.argv[3]
    args = [int(x) for x in sys.argv[4:]]
    if mode == "ppl":
        perplexity(url, path, *args)
    elif mode == "mc":
        multiple_choice(url, path, *args)
    else:
        sys.exit(__doc__)
