# MTP-under-PP corruption: evidence dossier (Qwen3.8-Flash-Next, vLLM #46994)

## Symptom
With PP=3 + MTP and max_num_seqs >= 3: 14-27% of requests degenerate into a
constant 4-char token loop ("duct"/rotations, ~1150 reps), independent of
prompt. Spec acceptance drops 44-52% -> 26-35%. Threshold is exactly 2->3
slots; rate is flat above. Absent with MTP alone (DEP: 0/80) and PP alone
(0/120).

## Direct measurement of the mechanism
Integer per-step checksums of each rank's input token IDs (immune to the
stack's run-to-run nondeterminism, which we proved exists even at
max_num_seqs=1, temp=0, no MTP — no output-differential oracle is possible):
rank 0 embeds sum=0 (zero-init draft rows) at ~1 request-onset step per
request while the last rank holds real values. An immediate-scatter patch
turned those zeros into real tokens (causality confirmed) but did not stop
the loops and serialized the pipeline 2.3x — because the commitment of
verification decisions, not the staleness itself, is the defect.

## Fifteen falsified hypotheses (all by measurement)
draft depth (1 vs 3: 21% vs 20%) · batch-size-gated spec (spec ran only at
batch<=2, still 14%) · stale scatter rows (0.4%) · relay tensor width (fixed
by construction) · scatter-vs-postprocess ordering (unchanged) · gate
asymmetry (send/recv counts match exactly) · NCCL op-pairing shift (counter
placement artifact) · queue depth = pp_size (slots=3 corrupt 27%) · cross-rank
draft-state checksums x2 (confounded by pipeline delay / call-site clocks) ·
temp-0 differential oracle (stack nondeterministic) · sync scheduling
(deadlocks step 1: relay collectives assume async cadence) · fp32 GDN beta
#53877 alone (25%) · mamba align PRs #53798/#53803 (already in image /
schema-mismatch) · --mamba-cache-mode align (active, 19%) ·
disable_padded_drafter_batch (see addendum).

## Surviving theory
Async-PP placeholder forwards are expected; plain decode discards their
samples via need_sampled and rolls state back. MTP verification runs on the
same poisoned logits and COMMITS accept/reject. Accepted garbage converges on
the drafter's unconditioned argmax -> constant token. Fix: never commit
verification from placeholder-input forwards (patch 0008 RFC).

## Addendum: disable_padded_drafter_batch under PP
Incompatible, not merely ineffective: trips
`assert (num_scheduled_tokens_np >= num_logits).all()` on rank 0 within
minutes at 8 slots (scheduler still ships padded spec-token counts), wedging
the engine while the API frontend stays responsive. 16th falsified approach;
also worth an upstream note that this flag + PP should be rejected at config
validation instead of asserting mid-flight.

## Addendum 2: commit-time verification gate (patch 0008 as implemented) — FALSIFIED
Implemented and soaked at 8 slots: per-request staleness (< pp_size sample-steps
since last output) -> all drafts force-rejected on those steps. Ran clean
(no scheduler-ledger asserts after restricting to shrink-only), fired ~once per
request at onsets, acceptance 29.2% -- loops persisted (6/40). Conclusion: the
corruption is committed DURING the placeholder forward, not at verification.
Strongest surviving candidate: GDN/mamba recurrent state written by
placeholder-input forwards is never rewound (rejection cannot undo it; align
mode alone measured insufficient). A fix must gate state WRITES at forward
time for placeholder steps -- kernel/state-manager level, i.e. inside the PR.

## RESOLUTION (2026-08-28)
Hypothesis 18 — found via web search, not instrumentation: the corruption is
prefix caching x MTP x PP (GDN prefix-cache state under spec decode; upstream
issue #54173, fix family PR #50021). Every corrupt run had prefix caching
enabled (vLLM default, never varied); trigger is shared-prefix prompts with
differing suffixes, which both our soak and real agent workloads produce, and
which byte-identical benchmark prompts mask. With --no-enable-prefix-caching:
0/168 loops at 8 slots, 40.2% acceptance, 445 tok/s aggregate (faster than the
no-MTP config). This retroactively explains the 2->3 slot threshold (>=3
distinct-suffix requests exercise shared/split cache paths) and the constant
token (a mis-associated cached state block is the same bytes every time).
Patch 0008's commit-time gate remains falsified and is superseded by this
finding; the durable fix is porting PR #50021.

## Addendum 3: PR #50021 port — insufficient as ported
Five core files (causal_conv1d, mamba_ssm, fused_recurrent, fused_sigmoid_gating,
mamba_utils incl. hand-ported src_block_id bounds guard) mounted with prefix
caching re-enabled: 27/104 loops (26%), acceptance 26.1%. Key refinement: the
soak's prompts share NO prefix (cache hit rate 0.0%) yet caching-ON corrupts and
caching-OFF is clean (0/168). The defect is therefore in the prefix-cache
ALLOCATION/precopy path for GDN state (cf. upstream #54199, precopy_mamba_align
faults), not in cross-request block sharing, and this image's variant needs
guards beyond #50021's. Production keeps --no-enable-prefix-caching.

## Addendum 4: num_accepted staleness clamp (our patch) — FALSIFIED
Theory: align precopy's accepted-token bias (src_off = num_accepted - 1) reads
stale values on non-last PP ranks. Implemented freshness tracking in
update_pp_decode_requests + clamp-to-1 before preprocess_state. Ran clean as
code; loops unchanged (25/112, 22%; acceptance 27.8%). Either the bias is not
the poisoned read, or the clamp misses the true read site (e.g. the bias is
also consumed inside forward-time kernels we cannot re-point). Net standing:
prefix-cache-ON GDN state path corrupts under PP+MTP against every reachable
intervention (stock / align / #50021 guards / fp32-beta / verification gate /
na-clamp). --no-enable-prefix-caching remains the correct production setting;
the durable fix requires the model/state authors.

## Addendum 5: state_idx deterministic reseed (ours) — FALSIFIED
Recomputed _mamba_state_idx_gpu for every scheduled row from
num_computed_tokens ((nct-1)//mamba_block_size, the add_request seeding
formula) before preprocess_state on non-last ranks, stacked on the
num_accepted clamp and #50021 guards. Cache-ON at 8 slots: 18/120 loops
(15%), acceptance 31.3%. Together with Addendum 4 this exonerates BOTH the
accepted-token bias and the absolute state-block index as the poisoned
inputs. The corruption therefore lives in the precopy/state CONTENTS under
concurrent cache-block allocation (forward-time, kernel-level). Five
falsified fixes on this path now bound the bug tightly for the state-
machinery authors; nothing reachable by bind-mount remains untested.

## Addendum 6: #53802 hit-boundary fix falsified; compute-sanitizer unavailable
- #53802 (scheduler-side hit-boundary alignment) tested in isolation, cache-ON
  slots=8: 20/112 loops (18%). Sixth falsified fix on this path.
- compute-sanitizer memcheck (CUDA 13.0, version-matched) wrapped the full
  server; corruption reproduced under it (3/16) but the tool reported
  "GPU debugging features are disabled" on every context: CMP 170HX does not
  support sanitizer instrumentation. No memcheck/initcheck/racecheck evidence
  is obtainable on this hardware; OOB-vs-logic remains undecided here and
  needs a supported GPU (the #54173 reporter's Xid faults suggest OOB).
Reachable surface on this hardware is now fully exhausted for the caching bug.

## Addendum 7: terminal localization — QSA compressed-cache integration, by exhaustion
Two decisive cross-experiments with cache ON at 8 slots:
- --mamba-block-size 32768 (no boundaries, no checkpoints, no precopy
  migrations => mamba caching machinery degenerate): 22/88 loops (25%).
  MAMBA SIDE EXONERATED WHOLESALE.
- --block-size 1600 (pinning the geometry every clean cache-OFF run
  negotiates; cache ON vs OFF DO negotiate different sizes — OFF logs
  "Setting attention block size to 1600 ... >= mamba page size" while the
  corrupt-era CSA diagnostics measured 816): 21/80 loops (26%).
  BLOCK-SIZE/TILING CONFOUND EXONERATED.
With sharing (0% hits), mamba machinery, and geometry all excluded, the only
subsystem prefix caching still activates is the model's own QSA/CSA
compressed-cache prefix-caching integration (compressed + compressor_state
groups; prefix_match_unit is matching-granularity only per its docs and is
not a lever). Eight falsified experiments bound the defect there. This code
is model-specific and 2 days old; it needs its authors.

## Addendum 8: deferred block-free (upstream's own hazard fix) — FALSIFIED
scheduler.py contains a fully-plumbed writer-after-free deferral whose comment
names our exact topology ("with overlapping batches (async scheduling or PP),
a step may still be writing a freed request's KV blocks") but arms it only for
KV-connector consumers. We widened the gate to all multi-inflight configs
(max_concurrent_batches>1; verified armed: pp+1=4). Cache-ON slots=8:
26/112 loops (23%), acceptance 26.9%. NINTH falsified intervention. Same-step
block recycling is thereby exonerated as the mechanism (though the gate is
arguably still a real latent bug upstream should widen for #54199's fault).
Cumulative exclusion for the cache-ON corruption: sharing, mamba machinery,
block geometry, block lifecycle/recycling, kernel bounds (portable subset),
GDN gate dtype, acceptance bias, state index, verification commitment, draft
relay. The QSA compressed-cache integration remains implicated by exhaustion;
no reachable lever exists on this hardware.

## Addendum 9: external replication names two real bugs; ports fix resume path, not our 0-hit lane
2026-08-28: a replication report landed on vllm#54173 (GB10/sm_121, same model,
MTP, prefix caching): crashes resolved completely by exactly two patches —
unmerged vllm#48375 (MambaManager.find_longest_cache_hit accepts
drop_eagle_block and ignores it, so a cache-hit resume lands on a page whose
recurrent-state snapshot was taken over rejected draft tokens) plus a fix for
vllm#53142 with no upstream PR (state-seed divisor uses cache_config.block_size
instead of the mamba group's block size; a resume then seeds an out-of-range
block_table column and the align precopy reads a garbage block id — IMA on
sm_121, silent wrong state where the read stays mapped). Both bugs VERIFIED
PRESENT in our image (drop_eagle_block dead parameter in MambaManager;
wrong divisor in v1/worker/gpu/model_states/mamba_hybrid.py). Ported both
(max_length-lowering variant of #48375, which also bounds this branch's
fine-grained partial-unit search; the reporter's implementation note).
Result, cache-ON UUID-first soak slots=8: 16/72 loops (22%) — UNCHANGED.
Consistent with mechanism: both fixes act on the hit/resume path, and our
corruption occurs at 0% hit rate. CONCLUSION: two distinct defects. Theirs
(resume-path poisoning/OOB-seed) is real, present here, and our port should be
kept — it will bite the moment hits occur. Ours is hit-independent and lives in
what cache-ON enables unconditionally: mamba_cache_mode flips none->align,
activating state-in-block indexing (_mamba_state_idx_gpu -> block_table column)
and the align precopy under PP overlap (max_concurrent_batches=4) + MTP.
Next lever (untried, motivated): --mamba-cache-mode all, which keeps prefix
caching on while bypassing the align machinery.

## Addendum 10: geometry definitively exonerated; align-specific machinery too
2026-08-28, three cells (all slots=8, PP3+MTP, both Addendum-9 ports mounted):
1. cache ON, default (align, mamba-block 1600): 16/72 loops (22%) — baseline
   reproduced with the resume-path fixes in place.
2. cache ON, --mamba-cache-mode all: 6/32 (19%) — align-SPECIFIC machinery
   (state_idx precopy/postprocess realign) exonerated; what align and all
   share remains in play.
3. cache OFF, --block-size 1600: 0/104 CLEAN — the complementary geometry
   cell. Caching-forced block size 1600 is NOT the mechanism; rounds also ran
   ~40s vs ~60s (cache ON is slower, consistent with degraded acceptance).
Corruption appears in round 1 on a fresh pool (0 hits, no reuse, no eviction)
for cells 1-2. Audit note: the earlier --mamba-block-size 32768 null (soak
seqs <=9k never cross a boundary, align kernels fast-exit, yet corruption) was
only ever observed from round 9 onward — its round-1 behavior is unverified,
so boundary-keyed mamba machinery is NOT yet excluded for the fresh-pool
regime. Rerunning that cell watched from round 1 to split: corrupt-from-round-1
=> mamba excluded entirely (mechanism structural/QSA); clean-early-corrupt-late
=> mamba state-block lifecycle (e.g. remove_skipped_blocks recycling,
which the defer_block_free gate does NOT cover) becomes the prime suspect.

## Addendum 10b: mode=all cell VOID; GDN state-slot assignment verified clean
The Addendum-10 cell 2 result is void: Qwen3_8FlashNextMTP raises
NotImplementedError for mamba_cache_mode=all, so a booted MTP server proves
config.py fell back to align ("Hybrid or mamba-based model detected without
support ... falling back to 'align' mode", model_executor/models/config.py).
Only ALIGN has ever run with cache ON. However the align per-step kernels are
all boundary-gated (precopy fast-exits on src==dst; postprocess_align's
needs_copy = aligned_new_computed >= num_tokens_running_state is False below
the first boundary), so the 32768 no-boundary null still exonerates them —
and that null is now verified corrupt FROM ROUND 1 (1/8 in the first batch,
fresh pool, 0 hits, no boundary events possible).
New instrument: GDN-DIAG in gdn_attn.py build() (env VLLM_GDN_DIAG) checks
spec_state_indices/non_spec_state_indices for cross-request, intra-row, and
spec-vs-nonspec block-id collisions plus zero ids. 1359 samples across all 3
ranks: ZERO violations. State-slot assignment is collision-free; if the
corruption is state-related it is content/timing, not slot identity.
Next cell: --mamba-block-size 262144 (= max_model_len) makes the mamba
subsystem geometrically identical to mode none (1 position block + spec
columns, gather pinned to col 0, boundaries unreachable). Corrupt => mamba
excluded by construction; clean => the 32768-vs-262144 delta pins it.

## Addendum 11: ROOT MECHANISM FOUND — 'duct' loops are NaN logits
2026-08-28, instrument chain (all cells: cache ON default align, PP3+MTP,
slots=8, UUID soak):
1. RUN-DIAG (runner): per-step slot-mapping collision check across ALL 6 KV
   groups + per-request position-continuity audit. 14,000 steps: ZERO
   violations. Indices/positions are provably correct.
2. FD-DIAG (runner): fed-token invariant — the token written into input_ids
   for every decode row equals the last committed token recorded at
   postprocess for that slot. ZERO mismatches. The feed/relay/scatter path is
   correct. BUT: SAMPLER-RUN events show every looper commits THE SAME token
   id 1023 for 256+ consecutive steps, on every rank, every round.
3. Token 1023 IS 'duct' (tokenizer-confirmed; 1021='line', 1022='/*').
   1023 = last index of a 1024 tile — suspicious, and explained by (4).
4. LG-DIAG (runner): when 1023 is committed, dump logits rows. THE LOOPING
   REQUEST'S LOGITS ROWS ARE ALL-NaN (max=nan, every value nan) while the
   co-batched healthy request's rows are normal. The sampler deterministically
   converts an all-NaN row into 1023. Later in the same soak ALL rows of the
   step are NaN (a second request went NaN too).
CONCLUSION: the corruption is NaN generation in the target model's forward
under prefix caching. 'duct' loops, 26-31% acceptance, and per-request
all-or-nothing behavior are all downstream symptoms. Once a request's state/KV
holds NaN it never recovers (loops until max_tokens).
Also excluded today by construction: the entire mamba align subsystem
(--mamba-block-size 262144 = mode-none-identical geometry still corrupts,
5/32), and QSA side-cache slot mapping (QSA-DIAG: zero collisions, zero ring
regressions on real traffic; the only violations were dummy/capture batches).
Next: NAN-DIAG logs the onset step of each request's first NaN with
num_computed alignment mod 1600 / mod 4 / mod 8 to name the boundary machinery.

## Addendum 12: NaN birth localized to the PLE short-conv (layer 1, rank 0)
Instrument chain, all on cache-ON default align PP3+MTP slots=8:
- NAN-DIAG law: every logits-NaN onset lands at num_computed %1600 in
  [1536,1600) — the final FLA_CHUNK_SIZE=64 tokens before an ATTENTION
  1600-block boundary. Law holds identically at --mamba-block-size 262144
  (mamba boundaries unreachable) => keyed to the attention/storage block, not
  mamba geometry. Falsified along the way: postprocess self-copy-with-bias
  skip (exact-boundary in-place hazard is real code but not the writer);
  state_idx no-regress clamp; --no-async-scheduling (deadlocks under PP3 —
  the #50021 sm_86 workaround is unavailable with PP).
- QNAN (qsa.py _run_qsa): ZERO "born-here" events across all 12 QSA layers +
  MTP QSA; QSA outputs are NaN-free even with NaN inputs (sparse attention
  washes NaN); on rank 0 hidden is already NaN entering layer 3 (first QSA)
  => birth in layers 0-2 (GDN, PLE, GDN).
- PLENAN (ple_layer.py short-conv custom op): "CONV-BORN" — inputs CLEAN,
  conv output NaN, repeatedly, only at language_model.model.layers.1.ple.
  THE NaN IS BORN IN THE PLE LAYER'S SHORT-CONV STATE PATH. Clean inputs +
  NaN output = the kernel reads NaN from its conv-state cache; the poisoned
  output is then appended back to the state => permanent per-request NaN.
Note: the PLE mamba-group state page is padded to the main_kv page size
(kv_cache_utils csa_linear grouping pads with page_size_padded=main_kv_page);
the padding is never-written VRAM — a natural NaN reservoir for any in-block
offset that escapes the real state extent. The PLE spec path uses a single
state slot (block_table[spec_req_idx, 0]) with num_accepted-driven in-block
offsets (short_conv_attn.py PleShortConvAttentionMetadataBuilder).

## Addendum 13: the poison is foreign bytes written into the PLE conv state
Instrument chain (cache-ON default, PP3+MTP slots=8, all cells):
- STATE-VAL at first NaN sighting: healthy columns at |x|=6-15; corrupted
  columns contain values like 3.3762e38 (= bf16 0x7F7E) plus ~0.4% NaN
  density — the statistics of RANDOM/FOREIGN BYTES, not numerical overflow
  (PLENAN MAG tracker: max|state| flat at 9-15 forever; no growth).
- The corruption arrives as 6-column stripes ([5:11), [6:12)...) in the
  12-column PLE conv state — 6 cols x 10240 x 2B = 122,880 bytes = exactly
  the GDN kv0 per-block extent, and PLE cols [6:12) = bytes [122880:245760)
  = exactly where GDN kv1 begins in its own page layout.
- BLK-HISTORY: the corrupted block had been in continuous use by the PLE spec
  path for 172 consecutive steps — an in-place clobber of a live block, not
  an identity change.
- ADDR-DIAG (with strides): the CSA unified layout is page-strided views
  (1,638,400-element pages) where main-KV, GDN conv (6x10240 @0), GDN SSM
  (fp32 @byte 122880), and PLE conv (12x10240 @0) FULLY ALIAS per page;
  global block-id disjointness is the only safety mechanism.
- RUN-DIAG GLOBAL-XGROUP: within-step, all six groups' gathered tables are
  globally id-disjoint (0 violations) — the tables the runner prepares are
  clean; the writer consumes something temporally stale at kernel-execution
  time.
Falsified fixes: FIFO free-queue (removing LIFO immediate reuse), PLE
prefill tail-zero, gathered-row tail-zero in _gather_block_tables_kernel
(upstream's own get_dummy_block_tables comment acknowledges the stale-id
hazard for dummy rows; real rows still don't zero tails — arguably still an
upstream bug), postprocess self-copy skip, state_idx no-regress clamp.
--no-async-scheduling deadlocks under PP3 (the #50021 sm_86 workaround is
structurally unavailable with PP).

## Addendum 14: remaining lane = in-flight persistent-buffer mutation
The persistent gathered input_block_tables (and sibling per-step buffers) are
re-written by later steps' prepare while earlier steps' kernels may still be
in flight under PP (max_concurrent_batches=4). UVA machinery (UvaBufferPool
round-robin depth = max_concurrent_batches; UvaBackedTensor num_blocks)
appears correctly sized; engine batch_queue pops before reuse. A full-table
per-step clone crashed with an illegal memory access (likely a CUDA-graph
pointer contract); a mamba-groups-only clone is under test. If the clone
suppresses the corruption, the writer is confirmed to read gathered tables
after a later step's re-gather — an upstream PP-only hazard in the V2
runner's persistent-buffer design.

## Addendum 15: clone counter-test confirms pointer-capture coupling; fix direction
Per-step clones of the gathered block tables (full set, then mamba-only)
both die with an illegal memory access: the mamba spec-decode GPU context
captures RAW DATA POINTERS to the persistent input_block_tables at init
("stable data_ptr ... captured once and reused across steps",
MambaSpecDecodeGPUContext.initialize_from_forward_context) — per-step
temporaries dangle or diverge from the captured pointers. This confirms the
persistent-buffer + raw-pointer-capture design is load-bearing, and that the
per-step gathered tables CANNOT be snapshotted without reworking the capture.
CONCLUSION / upstream fix direction: under PP (max_concurrent_batches > 1),
the V2 runner needs a round-robin POOL of pp+1 persistent gathered-table
sets (stable pointers per slot, rotated per step) — mirroring what
UvaBufferPool already does for the CPU-side staging — so that a step's
kernels never observe a later step's re-gather. Until then, PP + MTP +
prefix caching on this model corrupts recurrent state via in-flight table
mutation, and --no-enable-prefix-caching remains the production mitigation
(with cache OFF the mamba tables are 4 columns wide, allocated once per
request, and effectively immutable in flight — which is why OFF is clean).

## Addendum 16: SOLVED — prefix caching + PP + MTP works
2026-08-29. Two changes, applied together, eliminate the corruption entirely:
1. patches/0009 (block_table.py): a round-robin POOL of gathered
   input_block_table sets (VLLM_BT_POOL env, we run pp_size+2=5). With a
   single persistent set, a later step's gather mutates the tables while an
   earlier in-flight step's kernels may still read them (PP holds
   max_concurrent_batches=pp+1 steps in flight). Per-slot tensors have
   stable data_ptrs, so downstream raw-pointer captures remain valid —
   unlike per-step clones, which die with an illegal memory access.
2. patches/0010 (mamba_utils.py) + the v2 runner edit: the mamba spec-decode
   GPU context now captures the SOURCE per-request-slot block tables
   (stable pointers, req-indexed, mutated only by stream-ordered staged
   writes) and its copy kernels index rows by req_idx. Previously the ctx
   captured the per-step gathered tables ONCE and indexed them by batch row;
   on a non-last PP rank the deferred postprocess runs pp_size steps after
   its batch was gathered, so it walked the CURRENT tables with a STALE
   batch mapping — reading/writing mamba state through other requests'
   (freed/reallocated) block ids. In the CSA unified layout those pages
   alias every cache tensor, which is how random bytes landed inside the
   PLE conv state.
Plus hardening/ports: 0011 (#48375 eagle-drop, unmerged upstream),
0012 (#53142 divisor, no upstream PR), 0013 (PLE spec-extension column
zero-init).
VALIDATION (instrumented build): UUID-first soak slots=8: 0/120 loops
(baseline 14-27%), zero NaN onsets, zero PLE conv-born events, acceptance
36.6-38.4%. Hit-path: 16 conversations x 8 turns at concurrency 8 over a
40k shared document: 0 loops, 48/48 planted 5-digit code recalls exact,
prefix-cache hit rate 60.6% (4.22M/6.97M tokens). Clean production build
(probe-free) re-validation in progress.
Note for upstream: an ablation (0009 alone vs 0010 alone) has not been run
yet; the deferred-postprocess row mismatch (0010) is proven by inspection,
while 0009 additionally covers any late gathered-table read. #54173 (sm_121
IMA) and #50021's sm_86 report are the loud variants of the same stale-id
family; --no-async-scheduling cannot help under PP (deadlocks).

## Addendum 17: production validation (clean build, probes off)
run-pp3-mtp.sh now ships --enable-prefix-caching + VLLM_BT_POOL=5 + the
0009-0013 mounts. Validation on the clean build:
- UUID-first soak, slots=8: 0/160 loops over 20 rounds — rounds ~40s vs
  ~58s for the old cache-OFF production (~30% higher throughput on this
  workload, from warm-prefix prefills within rounds).
- Hit-path (16 convs x 8 turns, concurrency 8): 0 loops, 47/48 planted-code
  recalls (the 1 miss is temp-1.0 compliance noise; instrumented build
  scored 48/48; no loop/NaN signature).
- TTFT on a 20k-token prompt: 5.12s cold -> 1.35s warm (3.8x).
- Cumulative hit rate across both validations: 3.82M/7.14M queried tokens
  (hit-path traffic alone ran ~60%).

## Addendum 18: C1 decode regression was the leftover 0008 gate; removed
Benchmarking after the caching fix showed C1 decode at ~30 tok/s vs the
historical 76-84.5. Bisect: no-MTP C1 = 56.5 tok/s (pipeline healthy);
CPU/PCIe/clocks/page-cache all nominal; identical-file A/B reproduced 30.
Transcript archaeology found the 84.5 run used v2_model_runner.py.pregate2 —
the runner WITHOUT the experimental "placeholder verification gate" from the
falsified 0008 hypothesis. The gate (still present in production
v2_model_runner.py) rejected every draft for any request rescheduled within
pp_size sample-steps of its last — at concurrency 1 that is EVERY step, so
MTP degraded to 1 token/step while the pre-gate acceptance metric still read
~40%. Removed (its premise was falsified; the real corruption is fixed by
0009/0010). Also restored the #53877 fp32-GDN-beta quality kernel
(fused_recurrent.py) which the fast build had mounted.
Revalidation without the gate, prefix caching ON: speedrun 74.0 C1 /
247.8 C4 / 440.5 C8 (historical parity); UUID soak 0/168 (fastest rounds
yet, ~38s); hit-path 16x8 turns 0 loops, 48/48 recalls. The gate was not
load-bearing for stability.

## Addendum 19: ablation — 0010 is necessary and sufficient; 0009 is optional
Cache-ON UUID soaks, slots=8, production build otherwise:
- 0010 only (VLLM_BT_POOL=1, pool disabled): 0/160 CLEAN.
- 0009 only (pool=5, req-idx ctx reverted to stock): 10/24 corrupt by round 3
  (immediate, same 'duct' signature).
CONCLUSION: the essential fix is 0010 — the mamba spec-decode ctx must
capture the SOURCE per-request-slot block tables and index rows by req_idx.
The pool (0009) cannot fix the ctx path alone because the ctx captures raw
pointers to ONE gathered set while gathers rotate through the others; with
0010 applied the ctx never touches the gathered tables at all and the pool
is defense-in-depth for any OTHER late gathered-table read (none observed:
0010-only soaked clean). Production keeps VLLM_BT_POOL=5 as cheap hardening;
upstream should land 0010 as the fix and may take 0009 as optional.
