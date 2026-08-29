# Qwen3.8-Flash-Next: PP3 + MTP + prefix caching on vLLM

Patches, root-cause forensics, benchmarks, and a runnable serving setup for
**Qwen/Qwen3.8-Flash-Next-FP8** on vLLM with **pipeline parallelism (PP=3),
MTP speculative decoding, and prefix caching — all enabled at once**.
Developed and validated on 3× sm_80 GPUs (CMP 170HX, 200 W power limit, PCIe
Gen2 x4, no NVLink/P2P), on top of the `release/qwen38next_offload` branch
(vllm#53896 / vllm#46994 lineage).

Headline numbers (details in **[RESULTS.md](RESULTS.md)**): 74 tok/s decode
single-stream / 440 tok/s at 8-way, ~10k tok/s prefill, 3.8× TTFT from
prefix caching at ~60% hit rate, 262k context verified — with **zero**
output corruption across 300+-request soaks and 48/48 exact planted-value
recalls (before the fix here, 14–27% of requests degenerated into
constant-token loops whenever caching + MTP + PP ≥3 slots were combined).

## Quickstart

```bash
./serve/make-patched-tree.sh          # extract image files + apply patches
QWEN38_HF=/path/to/hf-cache ./serve/run-pp3-mtp.sh
```

`serve/run-pp3-mtp.sh` is the exact production launcher — every flag, env
var, and bind-mount, with inline commentary on why each load-bearing flag is
load-bearing. The builder is verified to reproduce the production tree
byte-for-byte. Tunables: `QWEN38_SPEC` (MTP depth, default 3 = measured
throughput optimum), `QWEN38_SEQS`, `QWEN38_MAXLEN`, `QWEN38_PARTITION`,
`QWEN38_YARN` (2.0 → verified 524k context).

## The bug this repo fixes

With PP + MTP + prefix caching and ≥3 concurrent requests, generation
degenerates into constant-token loops (`ductductduct…`). Root cause
(FINDINGS.md, Addenda 2–19 — the loop token is 1023, which is what the
sampler emits for **all-NaN logits**):

- Under PP, `pp_size+1` steps are in flight; the per-step *gathered* block
  tables are persistent buffers that later steps re-gather into.
- The mamba spec-decode GPU context captures **raw pointers** to those
  tables once and indexes them by **batch row**. On non-last ranks its
  deferred postprocess runs `pp_size` steps late — so it walks the
  *current* tables with a *stale* batch mapping, copying recurrent state
  through **other requests' freed/reallocated block ids**.
- In the CSA unified layout, every cache tensor (main KV, GDN conv/SSM
  state, PLE conv state) aliases the same physical pages, distinguished only
  by block-id ownership. The misdirected copies land inside other requests'
  live state: **silent NaN corruption on sm_80**, **illegal memory access on
  sm_121** (the vllm#54173 reports are the loud variant of the same fault).

**The fix (patch 0010, ablation-verified necessary and sufficient):** the
ctx captures the **source per-request-slot** block tables (stable pointers,
req-indexed, mutated only by stream-ordered staged writes) and its copy
kernels index rows by `req_idx`.

## Patch series

Enablement — make PP=3 serve this model at all (each fix gated a startup
crash or a measured defect):

| # | Fix |
|---|-----|
| 0001 | `load_weights`: skip `hyper_connection_mixer.*` on non-last ranks (module is None there) |
| 0002 | MTP drafter: branch on first-**or-last** PP rank (drafter lives on the last rank, gets hidden states directly) |
| 0003/0004 | Replace blanket "PLE requires PP=1" guards with the real invariant: every PLE layer confined to rank 0 (the only rank receiving `input_ids`) |
| 0005 | PLE offload worker is not a pipeline rank: clear `VLLM_PP_LAYER_PARTITION` in its `proc_main` |
| 0006 | PLE connector: staging-complete handshake (under PP the next forward outruns staging → `queue.Full` kills rank 0; deeper queues would silently stage stale inputs) |
| 0007 | Skip PLE connector on ranks without a `PleOffloadLayer` |

Correctness — the prefix-caching corruption and its relatives:

| # | Fix |
|---|-----|
| **0010** | **THE FIX** — mamba ctx captures source req-indexed block tables; copy kernels index by `req_idx` |
| 0009 | Round-robin pool of gathered-table sets (`VLLM_BT_POOL ≥ pp+2`) — optional defense-in-depth; does **not** fix the bug alone |
| 0011 | Port of unmerged vllm#48375 (`MambaManager` honors `drop_eagle_block`) — resume-path state poisoning |
| 0012 | Fix for vllm#53142 (**no upstream PR exists**): state-seed divisor must be the mamba group's block size |
| 0013 | Zero-init the PLE spec-extension state columns on prefill (uninit-VRAM NaN hardening) |

Also shipped: `ported-files/` (the vllm#46994 MTP-under-PP relay port —
`pp_utils.py` + the V2 runner — and vllm#53877's fp32 GDN beta), which are
wholesale replacements rather than diffs. **Do not apply
`patches/0008-WITHDRAWN-*`** — a falsified experiment kept as history; it
also silently caps concurrency-1 MTP at 1 token/step.

## Repo layout

| path | contents |
|---|---|
| `serve/` | production launcher + patched-tree builder |
| `patches/` | numbered diffs against the `vllm/vllm-openai:qwen38-flash-next` image tree |
| `patches-rebased-head/` | the four model-directory patches rebased onto branch head `a5530b90c` (upstream path `vllm/models/qwen4_exp/`); the other eight apply to head unchanged, dry-run verified |
| `ported-files/` | wholesale-ported files the diffs don't reconstruct |
| `tools/` | validated loop detector + soak harnesses (UUID-first allocation-path soak; conversation-shaped hit-path soak with planted-code recall) |
| `RESULTS.md` | throughput, caching, MTP depth sweep, fix ablation, long-context |
| `FINDINGS.md` | full investigation log, 19 addenda — every hypothesis, including the falsified ones, and which instrument decided each |

## Related work

Upstream: [vllm#54173](https://github.com/vllm-project/vllm/issues/54173)
(sm_121 IMA — same fault family, loud variant),
[vllm#50021](https://github.com/vllm-project/vllm/pull/50021) (sm_86 report;
its `--no-async-scheduling` workaround deadlocks under PP),
[vllm#54199](https://github.com/vllm-project/vllm/issues/54199),
[vllm#48375](https://github.com/vllm-project/vllm/pull/48375),
[vllm#53142](https://github.com/vllm-project/vllm/issues/53142),
[vllm#46994](https://github.com/vllm-project/vllm/pull/46994),
[vllm#53896](https://github.com/vllm-project/vllm/pull/53896).

Prior art: [allover326/vllm-dsa-mtp-sm80](https://github.com/allover326/vllm-dsa-mtp-sm80)
— MTP-under-PP for DeepSeek on sm_80, whose setup notes
(`VLLM_USE_FLASHINFER_SAMPLER=0`, drafter placement on the last PP rank)
informed this work.

## License

Apache-2.0 (patches and ported files contain code derived from vLLM).
