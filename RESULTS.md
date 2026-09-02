# Results

Model: **Qwen/Qwen3.8-Flash-Next-FP8** — the official FP8 checkpoint
(block-wise FP8 weights, ~126 GiB incl. the ~50 GiB FP8 n-gram PLE table;
KV cache stays bf16 — `--kv-cache-dtype fp8` is refused by the QSA layers).

All measurements below the *Update* section on **3× NVIDIA CMP 170HX** (GA100, sm_80, 64 GiB each),
**power-limited to 200 W per card** (`nvidia-smi -pl 200`), **PCIe Gen2 x4**,
no NVLink, no P2P (`NCCL_P2P_DISABLE=1`) — about as hostile an interconnect
as PP ever sees. Stock-power cards should do somewhat better on the
compute-bound numbers (prefill, C8 decode). Serving config: PP=3, MTP
`num_speculative_tokens=3`, **prefix caching ON**, HUMMING MoE backend (the
only FP8-block-quant MoE path that works on sm_80: Marlin faults in
`gptq_marlin_repack` and Triton declines block-wise FP8 outright), PIECEWISE
CUDA graphs, `mamba_ssm_cache_dtype=float32`, FP8 PLE table served from CPU
(`VLLM_PLE_CPU_OFFLOAD=1`), `max_model_len=262144`, `max_num_seqs=8`. All
patches from this repo applied.

Two partitions appear below, both current: the **balanced `16,16,16`**
(throughput-first, used for the decode/prefill tables) and the **capacity-first
`16,17,15`** (production default; +22% KV for −6% single-stream decode). See
*KV capacity* for both pool sizes.

## Update 2026-09-02: four cards, PP=4, full CUDA graphs

A 4th CMP 170HX went in (bus 01/21/41/42, all PCIe Gen2 x4, no P2P — every pair
host-bounces at 0.78 GB/s). New production default: **partition `12,12,12,12`,
`cudagraph_mode=FULL_AND_PIECEWISE`, per-rank KV budgets, `max_num_seqs=8`,
MTP N=3, PLE table still on CPU.** Everything else as above.

| | PP3 `16,17,15` PIECEWISE (old default) | PP4 `12,12,12,12` PIECEWISE | **PP4 `12,12,12,12` FULL_AND_PIECEWISE** |
|---|---|---|---|
| C1 decode (tok/s) / step | 70–74 / 34.8 ms | 59 / 44.2 ms | **89 / 30.0 ms** |
| C4 / C8 decode (tok/s) | 248 / 371–440 | 199–254 / 372 | **254 / 429–433** |
| 32k single-prompt prefill | 9.6k | 11.2k | **11.1–11.3k** |
| 250k needle prefill | 8.6k | 10.3k | **10.6k** |
| KV pool (tokens) | 1,359,009 (5.18×) | 2,537,837 | **2,537,837 (9.68×)** |

**Where the PP4 single-stream loss came from.** Adding a stage cost +9.4 ms per
decode step with identical tokens/step. A torch-profiler cross-rank timeline
(all ranks share the host clock; only CPU-side events are captured on these
cards) put it beyond doubt: at C1 each 12-layer stage occupied ~9.2 ms, the
drafter rank 10.0 ms (its three draft forwards cost <1 ms), all three hops
together 0.4 ms, and the relay's three NCCL broadcasts 0.13 ms (a 3-vs-4-rank
microbenchmark in the same image showed the wire is identical). With 4 tokens
per step the GPU work per stage is a few ms of expert reads, so **the stages
are CPU-dispatch-bound — Python launching 12 layers piecewise — and a fourth
stage adds a fourth helping.** Full CUDA graphs replay a whole stage's decode
step in one launch: 44.2 → 30.0 ms, better than PP3 ever was. (The launcher had
pinned `-cc.cudagraph_mode=PIECEWISE`; a later `--compilation-config` is
silently overridden by the dotted form — check the engine's `Initializing`
line, not the CLI.) Things that did *not* fix it: MTP N=4 (+6% C1, −12% C8,
−2.6% pool), moving layers off the drafter rank (`13,13,13,9` — nothing to
gain at C1, where stages run serially, and <1 ms at C8).

**Full graphs × the block-table pool (patch 0009) corrupt state.** With
`VLLM_BT_POOL=6` and full graphs the UUID soak produced 10/144 'duct' loops —
the original NaN-state signature — after ~10 minutes. A captured graph bakes
the pointer of the *one* pool slot current at capture time, so five of every
six replays run the GDN/QSA/PLE state kernels against stale block tables. The
pool was always optional (0010 alone is necessary and sufficient, see the
ablation); the launcher now forces `VLLM_BT_POOL=1` whenever the mode is
`FULL*`. With that: **UUID soak 0/168, hit-path 16 conversations × 8 turns 0
loops (46/48 planted-code recalls), 250k needle exact, 8 concurrent ~241k
needles 8/8 exact with 1.93M tokens resident (10.6k tok/s aggregate prefill).**

**KV sizing on 4 cards.** Discovery boot at uniform 0.94 utilization gave
2,309,701 tokens (drafter rank binds); vLLM's per-rank "fully utilize" maxima
minus margins (0.5 GiB on ranks 0–2, 1.0 GiB on the drafter rank) →
`28089402880 / 30123884544 / 30123884544 / 24216759808` bytes → **2,537,837
tokens**, within 0.1% of the projection. Measured per-rank footprint: 12
layers 34.0–34.3 GiB, 12 layers + drafter 37.9 GiB, plus ~1.6 GiB transient
HUMMING repack scratch at load.

**Putting the n-gram table in VRAM is architecturally blocked.** A `2,…`
partition (rank 0 = embedding + PLE only) has no attention layer, so (a) its
mamba specs arrive un-normalized (`block_size` 16 vs 1600 — patch 0015 fixes
that comparison) and then (b) the CSA layout planner refuses: "pipeline stage
has mamba cache owners but no main_kv tensor slots" — the GDN/PLE state pages
are carved inside the same-stage QSA tensors, and rank 0 cannot also fit a QSA
layer next to the 50 GiB table. The CPU offload costs ~2% of a step; it stays.

**Host CPU governor.** With stages dispatch-bound, the host's `powersave`
governor (amd-pstate-epp, `balance_performance`) was a suspect. A/B on the
same running server: `performance` took C1 from 84.9 to 89.2 tok/s (29.9 →
29.3 ms/step) and tightened the run spread (83.8–90.4 → 87.6–91.8); C8 429 →
443. A 2–5% effect — the busy core was already boosting under load — but free.

**Cold numbers lie.** Right after boot the first 32k prefill read 5.4k tok/s
(JIT) and the first C8 run 216; warm they are 11.2k and 372+. Warm up before
judging any configuration.

## Decode throughput (tok/s)

Median of 3+ runs, 200–300 generated tokens, unique (cache-defeating)
prompts. C≥4 aggregate has large run-to-run variance from MoE routing; the
first run after boot reads 20–40% low (warmup) and is discarded.

| Concurrency | 1 | 2 | 3 | 4 | 8 |
|---|--|--|--|--|--|
| aggregate | 74.0 | ~120–152 | ~121–200 | 247.8 | 440.5 |
| per stream | 74.0 | ~50–76 | ~40–67 | ~62 | ~55 |

Story-style prompts at temperature 1.0 run ~66 tok/s single-stream (lower
draft acceptance); the table uses the standard expository prompt.

## Prefill

~**9,600–10,500 tok/s** for 32k-token unique prompts at concurrency 1–3
(saturated at C=1; one 32k prompt fills the pipeline). Earlier profiling on
the same stack: ~5.7k @8k, ~9.2k @32k, ~10.0k @110k, ~8.6k @250k prompt
tokens.

## Prefix caching

- TTFT for a 20k-token prompt: **5.12 s cold → 1.35 s warm (3.8×)**.
- Conversation-shaped traffic (16 growing conversations × 8 turns over a 40k
  shared document, concurrency 8): **~60% token hit rate**, **0 degenerate
  loops**, **48/48 exact planted 5-digit code recalls** (codes planted one
  turn, recalled three turns later from deep inside the cached region — the
  check that catches silent state corruption that liveness tests miss).
- Allocation-path soak (UUID-first prompts, 0% hit rate by construction,
  8-way, ~7k-token prompts, 1500-token generations): **0/160 and 0/168 clean**
  across independent runs. Before patch 0010, this soak looped on 14–27% of
  requests in every one of ~15 instrumented configurations.

## MTP depth sweep (`num_speculative_tokens`)

| N | C1 tok/s | C8 tok/s (median of 4) | acceptance |
|---|---|---|---|
| 1 | 61.3 | 213.9 | 78.9% |
| 2 | 70.5 | 266.7 | 67.5% |
| **3** | 74.6 | **370.8** | 57.4% |
| 4 | **78.8** | 273.9 | 49.5% |
| 5 | won't boot: QSA ring capacity 12 must divide the attention block size | | |

Single-stream keeps improving with depth (deeper drafts amortize the
per-step pipeline traversal), but at 8-way the longer speculation window
turns into wasted verify work. N=3 is the throughput sweet spot.

## Fix ablation (patch 0010 vs 0009)

UUID-first 8-way soak, prefix caching on:

| configuration | result |
|---|---|
| 0010 only (table pool disabled) | **0/160 clean** |
| 0009 only (pool=5, ctx fix reverted) | **10/24 corrupt by round 3** |
| both (production) | 0/160, 0/168 clean |

0010 is necessary and sufficient. 0009 alone cannot help because the mamba
ctx captures raw pointers to one gathered set while the pool rotates through
the others; it is kept as cheap defense-in-depth.

## KV capacity (per-rank budgets, patch 0014)

`--kv-cache-memory` is one value for every worker, but heterogeneous PP
ranks (PLE on rank 0, drafter on the last rank, differing per-token KV
costs) strand ~3 GiB/rank under a uniform budget. Patch 0014 lets each rank
take its own budget from `VLLM_KV_CACHE_MEMORY_RANK<i>`; set to vLLM's own
suggested maxima minus 0.5/0.5/1.0 GiB margins:

- balanced `16,16,16`: pool **935,216 → 1,116,591 tokens (+19.4%)**, 4.26×
  full-context (262k) concurrency — validated with a clean 0/160 soak and an
  exact 253k-token needle recall with no OOM at the activation peak.
- capacity-first `16,17,15` + rebalanced budgets (**production default**):
  **1,359,009 tokens, 5.18×** full-context — i.e. **+45% over the uniform
  baseline**, costing ~6% single-stream decode (74 → ~70 tok/s) because the
  17-layer rank sets the pipeline step time; 8-way aggregate is unaffected.
  Validated at `max_num_seqs=5`: **5 concurrent ~241k-token needles, 5/5
  exact** (1.21M tokens resident, ~8.3k tok/s aggregate prefill), 0/115 soak,
  48/48 hit-path recalls.
- Note the two knobs interact: under a *uniform* budget, shifting layers only
  makes some rank a worse limiter. It is per-rank budgets that make the
  `16,17,15` shift pay, by letting the freed rank actually claim the memory.

## Context length

- **262,144 native context verified**: 260k-token needle recall in 35.7 s.
- Optional static YaRN factor 2.0 → **524,288**: 500k-token needle recall
  verified, short-context spot checks unchanged (off by default per Qwen's
  static-YaRN guidance).
- **1M context verified** (enabled by patch 0014's larger pool): YaRN factor
  4.0 + `max_model_len=1048576` boots with a 1,220,499-token pool (1.16×
  one full sequence) and recalled a needle from a **1,041,971-token prompt
  exactly** — 189.6 s wall, ~5.5k tok/s prefill at the 1M scale. Opt-in:
  `QWEN38_YARN=4.0 QWEN38_MAXLEN=1048576`.

## Cost of the CPU PLE offload

The ~50 GiB FP8 n-gram (PLE) table is served from CPU RAM via
`VLLM_PLE_CPU_OFFLOAD=1`, which raises the obvious question of what that
costs. It cannot be A/B'd on 3 cards — with the table resident on GPU, rank 0
is fully consumed by it (L0+L1 = 55.2 GiB) and the remaining 46 layers do not
fit on two cards (they OOM inside `transform_humming_weight`, which needs
~1.6 GiB of transient repack scratch on top of resident weights, with or
without MTP). So the offload worker was instrumented instead.

Measured at concurrency-1 decode (medians over 200 forwards):

| | ms |
|---|---:|
| forward period | 34.81 |
| **total offload service** | **0.69 (2.0% of the step)** |
| ├ CPU n-gram lookup | 0.51 |
| ├ H2D copy enqueue | 0.09 |
| └ CPU blocked waiting on GPU | 0.04 |

**The offload costs at most ~2% of a decode step**, and that is an upper
bound — it assumes the GPU stalls for the entire service window rather than
overlapping it. The near-zero "blocked waiting on GPU" term shows the CPU
worker runs ahead of the model rather than gating it. Serving this table from
host RAM is essentially free; it is not a reason to add GPUs.

*Method note:* the GPU-side wait is a `cuStreamWaitValue32` **stream stall**
that is captured *into* the piecewise CUDA graph, so instrumenting the Python
impl of `vllm::ple_offload_wait` measures nothing on replay and raises
`cudaErrorStreamCaptureInvalidated` if events are queried there. The
CPU-side offload worker is the only viable vantage point.

## Why PP wins on this interconnect

Compared with expert-parallel (DEP3) on the same box, PP3 is ~5× faster at
prefill and has ~4× the KV capacity: with expert parallel, every one of 48
layers does an all-to-all over PCIe Gen2 x4 with no P2P; under PP each stage
owns all experts for its layers, so routing stays on-GPU and only hidden
states cross stage boundaries, twice per forward.
