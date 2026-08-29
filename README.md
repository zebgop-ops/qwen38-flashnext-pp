# Qwen3.8-Flash-Next: PP3 + MTP + prefix caching on vLLM — patches & forensics

Patches, root-cause forensics, and repro tools for running
**Qwen3.8-Flash-Next (FP8)** on vLLM with **pipeline parallelism (PP=3), MTP
speculative decoding, and prefix caching** — developed and validated on
3x sm_80 GPUs (no NVLink, PCIe Gen2 x4), on top of the
`release/qwen38next_offload` branch (vllm#53896 / vllm#46994 lineage).

**Headline fix (patch 0010):** the mamba spec-decode GPU context captures raw
pointers to the per-step *gathered* block tables and indexes them by batch
row; on non-last PP ranks its deferred postprocess runs `pp_size` steps late,
so it walks the *current* tables with a *stale* batch mapping — copying
recurrent state through other requests' freed/reallocated block ids. In the
CSA unified layout every cache tensor aliases the same pages, so this
silently corrupts live state (constant-token loops from NaN logits on sm_80)
or faults loudly (the sm_121 illegal-memory-access reports in vllm#54173).
Ablation-verified necessary and sufficient (FINDINGS.md, Addendum 19).

Related upstream: vllm#54173, vllm#50021, vllm#54199, vllm#48375 (ported
here as 0011), vllm#53142 (fix here as 0012 — no upstream PR exists yet).

Layout: `patches/` apply to the `vllm/vllm-openai:qwen38-flash-next` image
tree; `patches-rebased-head/` are the four model-directory patches rebased
onto branch head `a5530b90c` (upstream path `vllm/models/qwen4_exp/`); the
other eight apply to head unchanged. `tools/` has the validated loop
detector and soak harnesses. `FINDINGS.md` is the full investigation log
(19 addenda). **Do not apply `0008-WITHDRAWN-*`** — kept only as history.

License: Apache-2.0 (patches contain code derived from vLLM).

---

# [Spec][PP] Qwen3.8-Flash-Next: make pipeline parallelism (+ MTP) usable

Applies on top of #53896/#53899 (model support) and #46994 (MTP under PP).

## What this series does
Patches 0001-0007 are **validated fixes** (each gated a startup failure or a
measured correctness/stability defect) that make PP=3 serve this model:

| # | Fix | Evidence |
|---|-----|----------|
|0001| load_weights: skip `hyper_connection_mixer.*` on non-last ranks (module is None there) | startup crash -> fixed |
|0002| MTP drafter: branch on first-OR-last PP rank (drafter lives on last rank, gets hidden states directly) | matches #46994's own qwen3_5 fix |
|0003/0004| Replace blanket "PLE requires PP=1" guards with the real invariant: every PLE layer confined to rank 0 (which alone receives input_ids; offload worker already spawns from rank 0, num_workers = dp*tp excludes PP by design) | PP serves; 260k-token needle recall verified |
|0005| PLE offload worker is not a pipeline rank: clear VLLM_PP_LAYER_PARTITION in proc_main (else len(partitions)!=pp_size=1) | startup crash -> fixed |
|0006| PLE connector: staging-complete handshake. maxsize=1 + put_nowait guards a copy from the runner's LIVE input buffers; under PP the next forward outruns staging -> queue.Full kills rank 0 (ranks 1-2 spin at 100%). Deeper queue or blocking put would silently stage stale inputs; the event serializes only the small copy | queue.Full: was fatal in minutes -> 0 in 248+ requests |
|0007| Skip PLE connector on ranks without a PleOffloadLayer (registration count = dp*tp) | startup crash -> fixed |

Also recommended alongside: upstream #53877 (fp32 GDN decode beta) — applies
cleanly, kept in our deployment.

## The 'duct'-loop bug: ROOT-CAUSED AND FIXED (patches 0009-0013)
The >=3-slot constant-token loops ("duct" x1000+) were fully root-caused on
2026-08-29 (0008's placeholder-commit framing was falsified along the way — that patch is
renamed 0008-WITHDRAWN and must not be applied: it also silently caps
concurrency-1 MTP decode at 1 token/step, Addendum 18;
FINDINGS.md Addenda 2-16 document the complete chain). Mechanism:
'duct' = token 1023 = what the sampler emits for ALL-NaN logits; the NaN is
born in the PLE layer's short-conv state, which receives stripes of foreign
bytes because mamba/PLE state kernels walked block tables that were mutated
or mis-indexed while their step was still in flight under PP:
- the persistent gathered input_block_tables are re-gathered by later
  in-flight steps (PP holds pp_size+1 steps in flight);
- the mamba spec-decode GPU context captures raw pointers to them ONCE and
  its deferred postprocess (runs pp_size steps late on non-last ranks)
  indexed them by STALE batch rows — walking other requests' freed and
  reallocated block ids. In the CSA unified layout every cache tensor
  aliases the same pages, so those writes land inside other requests' live
  recurrent state.
The fixes:
|0010| mamba_utils.py + runner: ctx captures SOURCE req-indexed tables; copy kernels index by req_idx | THE FIX — ablation-verified necessary and sufficient (Addendum 19) |
|0009| block_table.py: round-robin POOL of gathered-table sets (VLLM_BT_POOL >= pp+2); stable per-slot data_ptrs | optional defense-in-depth; does NOT fix the bug alone |
|0011| port of unmerged vllm#48375: MambaManager honors drop_eagle_block | resume-path state poisoning (the sm_121 crash lane, #54173) |
|0012| vllm#53142 fix (no upstream PR): state-seed divisor = mamba group block size | OOB block_table column on cached resume |
|0013| ple_layer.py: zero-init the spec-extension state columns on prefill | uninit-VRAM NaN reads on fresh blocks (hardening) |
Related upstream: #54173 (sm_121 illegal memory access — the loud variant of
the same stale-id family), #50021 (sm_86 report; its --no-async-scheduling
workaround deadlocks under PP and is not available), #54199.

## Rebase status (branch head a5530b90c, release/qwen38next_offload)
patches/0004-0007 and 0009-0012 apply to head unchanged (dry-run verified).
0001/0002/0003/0013 touch the model directory, which upstream keeps at
vllm/models/qwen4_exp/ (the release image renames it) and which moved with
the NVFP4 work: rebased, apply-verified variants are in
patches-rebased-head/. 0008 is WITHDRAWN (do not apply).

## Verified configurations on 3x sm_80 (PCIe Gen2 x4)
- PP3 + PLE offload, no MTP, 8 slots: 0/120 clean, ~10,150 tok/s prefill @32k
- PP3 + MTP, 8 slots, prefix caching OFF: 0/168 clean, 40.2% acceptance,
  445 tok/s aggregate
- PP3 + MTP, 8 slots, PREFIX CACHING ON + patches 0009-0013: 0/120 clean
  (UUID soak); hit-path 16 conversations x 8 turns: 0 loops, 48/48 planted
  5-digit code recalls exact, 60.6% prefix-cache hit rate, acceptance
  36.6-38.4%. Before the patches this configuration looped on 14-27% of
  requests in every one of ~15 instrumented cells.

## Reproducer & instrument suite
tools/ has the validated loop detector and soak harnesses (UUID-first
allocation-path soak + conversation-shaped hit-path soak with planted-code
recall). The forensic probes used for the root cause are env-gated
(VLLM_NAN_DIAG, VLLM_QNAN, VLLM_PLENAN, VLLM_RUN_DIAG, VLLM_GDN_DIAG,
VLLM_QSA_DIAG) and live in the investigation tree; FINDINGS.md maps each
probe to the addendum it decided.
