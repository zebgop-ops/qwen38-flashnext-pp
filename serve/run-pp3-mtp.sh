#!/bin/bash
# Qwen3.8-Flash-Next-FP8 on 4x CMP 170HX (200 W): PIPELINE PARALLEL 4 + MTP + prefix caching.
# DEFAULT (since 2026-09-02): partition 12,12,12,12, FULL_AND_PIECEWISE cudagraphs,
# per-rank KV budgets -> 2,537,837 KV tokens (9.68x full 262k context), seqs 8, MTP 3.
# Speeds (2026-09-02, warm): decode 89 tok/s C1 (30.0 ms/step) / 254 C4 / 429-433 C8;
#   prefill 11.1-11.3k tok/s (32k single), 10.6k at 250k and for 8x241k concurrent.
#   Validated: UUID soak 0/168, hit-path 16x8 turns 0 loops 46/48 recalls, 250k needle
#   exact, 8 CONCURRENT 241k needles 8/8 exact (1.93M tokens resident).
# WHY FULL CUDAGRAPHS (the PP4 speed hunt): at C1 a 12-layer stage occupied ~9 ms of
#   which only a few ms is GPU work -- stages are CPU-dispatch-bound under PIECEWISE
#   (profiler cross-rank timeline; hops 0.4 ms/step, relay 0.13 ms, wire exonerated).
#   PIECEWISE PP4 = 59 tok/s (44.2 ms/step) vs PP3 74 (34.8). FULL_AND_PIECEWISE replays
#   the whole decode step per stage: 30.0 ms -> 89 tok/s. MTP N=4 only bought +6% C1 for
#   -12% C8; 13,13,13,9 is pointless (drafter forwards cost <1 ms).
# FULL GRAPHS x BLOCK-TABLE POOL: patch 0009's rotating gathered-table pool (VLLM_BT_POOL)
#   CORRUPTS under full graphs (10/144 'duct' loops): a captured graph bakes ONE pool
#   slot's pointer, so 5/6 steps read stale tables. Pool is forced to 1 when CG=FULL*
#   (0010 alone is the proven-sufficient fix; ablation 0/160). Do not raise it.
# 4-CARD TOPOLOGY: bus 01/21/41/42, PCIe Gen2 x4 each, no P2P, 0.78 GB/s host-bounce
#   for every pair. PLE n-gram table stays on CPU (VLLM_PLE_CPU_OFFLOAD=1, ~2% of a
#   step): an attention-less rank cannot host it (cache layout needs a QSA tensor slot).
# Per-rank sizing (measured): 12L 34.0-34.3 GiB weights+overhead, 12L+drafter 37.9;
#   humming repack needs ~1.6 GiB transient scratch at load. Over-committing a rank
#   faults (IMA) instead of OOM and wedges GSP firmware -> reboot.
# 3-CARD RECIPES still supported (budgets keyed by partition below):
#   QWEN38_PARTITION=16,17,15 QWEN38_GPUS=0,1,2 ./run-pp3-mtp.sh   (1,359,009 tokens, 70 C1)
#   QWEN38_PARTITION=16,16,16 QWEN38_GPUS=0,1,2 ./run-pp3-mtp.sh   (1,116,591 tokens, 74 C1)
#   (PP3 numbers above were measured with PIECEWISE; FULL_AND_PIECEWISE should help there too.)
# 1M CONTEXT (verified on PP3): QWEN38_YARN=4.0 QWEN38_MAXLEN=1048576 -> 1,041,971-token
#   needle exact. Opt-in via those env vars.
# MTP depth (QWEN38_SPEC): PP3 sweep C1/C8/acc N=1 61/214/79% N=2 71/267/68% N=3 75/371/57%
#   N=4 79/274/50%; N=5 refuses (QSA ring divisibility; upstream #54912 lifts it). PP4: N=4 63/327.
# HISTORY: prefix-caching 'duct' loops were NaN corruption of recurrent state under PP
#   (stale gathered block tables walked by the deferred mamba spec-decode ctx); fixed by
#   patch 0010 (ctx captures SOURCE req-indexed tables; kernels index by req_idx). A
#   falsified 'placeholder verification gate' (0008) once capped C1 at ~30 tok/s; removed.
# Knobs: QWEN38_PARTITION QWEN38_GPUS QWEN38_KV0..3 QWEN38_CG QWEN38_BT_POOL QWEN38_SPEC
#   QWEN38_SEQS QWEN38_MAXLEN QWEN38_YARN QWEN38_EXTRA_ARGS (e.g. --max-num-batched-tokens 8192)
#   Profiling: QWEN38_EXTRA_ARGS='--profiler-config {"profiler":"torch","torch_profiler_dir":"/tmp/profiles"}'
#   then POST /start_profile, /stop_profile (CPU-side events only on these cards).


NAME=${QWEN38_NAME:-qwen38-pp}
PORT=${QWEN38_PORT:-8001}
IMG=${QWEN38_IMG:-vllm/vllm-openai:qwen38-flash-next}
HFCACHE=${QWEN38_HF:-/home/r/.cache/huggingface}
SNAPSHOT=${QWEN38_SNAPSHOT:-bcd9f01ddc9cff2316eb84281bebcd5b058bddce}
PARTITION=${QWEN38_PARTITION:-12,12,12,12}
PP=$(echo "$PARTITION" | tr "," "\n" | wc -l)   # ranks = partition entries
GPU_ORDER=${QWEN38_GPUS:-all}                     # e.g. 0,1,2,3 to pin rank->device
UTIL=${QWEN38_UTIL:-0.94}
MAXLEN=${QWEN38_MAXLEN:-262144}
# QWEN38_YARN=<factor> extends context past the native 262,144 via Qwen's own
# YaRN recipe. Verified: factor 2.0 + --max-model-len 524288 recalls a needle at
# 500,128 tokens in 67s, and short-prompt answers stayed correct. OFF by default
# because Qwen warn static YaRN "remains constant regardless of input length,
# potentially impacting performance on shorter texts" -- a real cost on a server
# that mostly sees short requests. factor 4.0 (their 1M example) does NOT fit:
# the KV pool is ~957k tokens, short of one 1,048,576-token sequence.
YARN=${QWEN38_YARN:-}
SEQS=${QWEN38_SEQS:-8}
SPEC_N=${QWEN38_SPEC:-3}
EXTRA_ARGS=${QWEN38_EXTRA_ARGS:-}
CG=${QWEN38_CG:-FULL_AND_PIECEWISE}
# gathered-table pool (patch 0009) rotates buffers per step; a FULL cudagraph bakes
# one slot's pointer at capture -> stale tables -> state corruption. Pool=1 under FULL*.
BT_POOL=${QWEN38_BT_POOL:-$( [[ $CG == FULL* ]] && echo 1 || echo $((PP+2)) )}   # e.g. --profiler-config ... or --max-num-batched-tokens 8192

# Per-rank KV budgets (patch 0014) are measured per PARTITION; a partition must never
# inherit another's numbers (over-committing a rank faults at load and wedges GSP firmware).
case "$PARTITION" in
  12,12,12,12) D0=28089402880; D1=30123884544; D2=30123884544; D3=24216759808 ;;  # 4 cards: pool 2,537,837 (9.68x)
  16,17,15)    D0=18253611008; D1=16106127360; D2=16642998272; D3= ;;             # 3 cards: pool 1,359,009 (5.18x)
  16,16,16)    D0=17212851712; D1=19231211520; D2=13323913216; D3= ;;             # 3 cards throughput-first: 1,116,591
  *)           D0=; D1=; D2=; D3= ;;                                              # unknown: discovery run (util 0.94)
esac
KV0=${QWEN38_KV0:-$D0}; KV1=${QWEN38_KV1:-$D1}; KV2=${QWEN38_KV2:-$D2}; KV3=${QWEN38_KV3:-$D3}
KV_ENV="${KV0:+-e VLLM_KV_CACHE_MEMORY_RANK0=$KV0} ${KV1:+-e VLLM_KV_CACHE_MEMORY_RANK1=$KV1} ${KV2:+-e VLLM_KV_CACHE_MEMORY_RANK2=$KV2} ${KV3:+-e VLLM_KV_CACHE_MEMORY_RANK3=$KV3}"
PATCHDIR=${QWEN38_PATCH:-/home/r/qwen38-run/patch}

V=/usr/local/lib/python3.12/dist-packages/vllm
MODEL="/hf/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/$SNAPSHOT"

for f in model.py mtp.py model_state.py v2_model_runner.py pp_utils.py \
         gpu_worker.py ple_worker.py connector.py \
         block_table.py mamba_utils.py single_type_kv_cache_manager.py \
         mamba_hybrid.py ple_layer.py fused_recurrent.py; do
  [ -f "$PATCHDIR/$f" ] || { echo "missing patch $PATCHDIR/$f" >&2; exit 1; }
done

if docker inspect -f '{{.State.Running}}' dsv4-a100 2>/dev/null | grep -q true; then
  echo "dsv4-a100 is running and holds the GPUs. Stop it first." >&2; exit 1
fi

docker stop -t 60 "$NAME" >/dev/null 2>&1
docker rm "$NAME" >/dev/null 2>&1

docker run -d --name "$NAME" --gpus "$([ "$GPU_ORDER" = all ] && echo all || echo "\"device=$GPU_ORDER\"")" --ipc=host --shm-size=32g \
  -e HF_HOME=/hf \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=5400 -e VLLM_BT_POOL=$BT_POOL \
  $KV_ENV \
  ${YARN:+-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1} \
  -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 \
  -e VLLM_PP_LAYER_PARTITION="$PARTITION" \
  -v "$HFCACHE":/hf:ro \
  -v "$PATCHDIR/model.py":$V/models/qwen3_8_flash_next/nvidia/model.py:ro \
  -v "$PATCHDIR/mtp.py":$V/models/qwen3_8_flash_next/nvidia/mtp.py:ro \
  -v "$PATCHDIR/model_state.py":$V/models/qwen3_8_flash_next/nvidia/model_state.py:ro \
  -v "$PATCHDIR/v2_model_runner.py":$V/v1/worker/gpu/model_runner.py:ro \
  -v "$PATCHDIR/pp_utils.py":$V/v1/worker/gpu/pp_utils.py:ro \
  -v "$PATCHDIR/gpu_worker.py":$V/v1/worker/gpu_worker.py:ro \
  -v "$PATCHDIR/ple_worker.py":$V/v1/ple_offload/worker.py:ro \
  -v "$PATCHDIR/connector.py":$V/v1/ple_offload/connector.py:ro \
  -v "$PATCHDIR/kv_cache_utils.py":$V/v1/core/kv_cache_utils.py:ro \
  -v "$PATCHDIR/block_table.py":$V/v1/worker/gpu/block_table.py:ro \
  -v "$PATCHDIR/mamba_utils.py":$V/v1/worker/mamba_utils.py:ro \
  -v "$PATCHDIR/single_type_kv_cache_manager.py":$V/v1/core/single_type_kv_cache_manager.py:ro \
  -v "$PATCHDIR/mamba_hybrid.py":$V/v1/worker/gpu/model_states/mamba_hybrid.py:ro \
  -v "$PATCHDIR/ple_layer.py":$V/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
  -v "$PATCHDIR/fused_recurrent.py":$V/third_party/flash_linear_attention/ops/fused_recurrent.py:ro \
  -p "$PORT":8000 \
  "$IMG" "$MODEL" --served-model-name qwen38 \
  --pipeline-parallel-size "$PP" --moe-backend humming \
  --enable-prefix-caching \
  --mamba-ssm-cache-dtype float32 \
  --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" \
  ${YARN:+--hf-overrides "{\"text_config\":{\"rope_parameters\":{\"mrope_interleaved\":true,\"mrope_section\":[11,11,10],\"rope_type\":\"yarn\",\"rope_theta\":10000000,\"partial_rotary_factor\":0.25,\"factor\":$YARN,\"original_max_position_embeddings\":262144}}}"} \
  -cc.cudagraph_mode=$CG --no-enable-flashinfer-autotune \
  $EXTRA_ARGS \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_N}" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  >/dev/null

echo "launched $NAME on :$PORT  (PP$PP $PARTITION, cudagraph $CG pool $BT_POOL, maxlen $MAXLEN, seqs $SEQS, mtp $SPEC_N)"
echo "startup ~5 min. follow:  docker logs -f $NAME"
