#!/bin/bash
# Qwen3.8-Flash-Next-FP8, PIPELINE PARALLEL 3 + MTP, on 3x CMP 170HX.
# Speeds (2026-08-29, this build, prefix caching ON):
#   decode  74 tok/s C1 / 248 C4 / 440 C8 (speedrun.py, ocean-currents prompt)
#           story-style prompts ~66 C1; C>=4 aggregate varies with MoE routing
#   prefill 9.6-10.5k tok/s (32k unique prompts, C1-C3)
# PER-RANK KV BUDGETS (2026-08-29, gpu_worker.py patch): the global 0.94
# utilization strands ~3 GiB/rank because PP ranks are heterogeneous.
# VLLM_KV_CACHE_MEMORY_RANK{0,1,2} below set each rank to vLLM's own
# suggested max minus margin (0.5/0.5/1.0 GiB). Pool: 935,216 -> 1,116,591
# tokens (+19.4%, 4.26x full-ctx). Validated: 0/160 soak, 253k needle exact
# with no OOM at the activation peak.
# 1M CONTEXT VERIFIED: QWEN38_YARN=4.0 QWEN38_MAXLEN=1048576 boots with a
# 1,220,499-token pool (1.16x) and recalled a needle from a 1,041,971-token
# prompt exactly (189.6s, ~5.5k tok/s prefill). Opt-in via those env vars.
# MTP depth sweep 2026-08-29 (QWEN38_SPEC): C1/C8-median/acceptance:
#   N=1 61/214/79%  N=2 71/267/68%  N=3 75/371/57%  N=4 79/274/50%
#   N=5 refuses to boot (QSA ring capacity 12 must divide the block size).
#   Default 3 = best throughput; QWEN38_SPEC=4 gains ~5% single-stream at a
#   ~26% cost to 8-way aggregate.
#   TTFT    20k prompt: 5.1s cold -> 1.35s warm (3.8x via prefix cache)
# NOTE: an experimental 'placeholder verification gate' (falsified hypothesis
# 0008) lingered in v2_model_runner.py until 08-29 and silently rejected all
# drafts for requests rescheduled within pp_size sample-steps — at C1 that is
# every step, capping decode at ~30 tok/s. Removed; do not re-add.
# PREFIX CACHING: ENABLED as of 2026-08-29 — root cause found and fixed.
# The old 'duct' loops (14-27%% of requests, upstream #54173's silent variant)
# were NaN corruption of recurrent state: under PP the persistent gathered
# block tables are re-gathered by later in-flight steps, and the mamba
# spec-decode ctx walked them with stale batch rows. Fixed by patches:
#   block_table.py     round-robin POOL of gathered-table sets (VLLM_BT_POOL=5)
#   mamba_utils.py     ctx copy kernels index SOURCE tables by req_idx
#   v2_model_runner.py ctx captures source req-indexed tables
#   single_type_kv_cache_manager.py  ported unmerged vllm#48375 (eagle drop)
#   mamba_hybrid.py    vllm#53142 fix (mamba-group block-size divisor)
#   ple_layer.py       zero-init spec-extension state columns (hardening)
# Verified 2026-08-29 (clean production build): UUID soak 0/160 loops
# (~30%% faster rounds than cache-OFF); hit-path 16 convs x 8 turns 0 loops,
# 47-48/48 planted-code recalls, ~60%% hit rate on conversation traffic;
# TTFT 20k prompt 5.12s cold -> 1.35s warm (3.8x).
# Verified 2026-08-27: 262,144 ctx (model native max), 935k KV tokens (3.57x
# concurrency at full length), ~10,100 tok/s prefill @32k, 260k needle recall in 35.7s,
# ~69 tok/s decode single / ~434 tok/s at 8 concurrent, MTP ~56% acceptance.
#
# WHY PP BEATS DEP ON THIS BOX (measured ~5x prefill):
#   With data+expert parallel, each token's top-10 of 512 experts are scattered
#   across all three GPUs, so EVERY one of the 48 layers does an all-to-all over
#   a PCIe Gen2 x4 link with no working P2P. Under PP each stage owns ALL experts
#   for its own layers, so expert routing is local and only hidden states cross
#   stage boundaries -- twice per forward instead of 48 times. PP also shards KV
#   by layer instead of replicating per rank: 904k KV tokens vs 227k under DEP.
#
# Upstream does not support PP+MTP for this model. ./patch holds 8 fixes:
#   model.py          hyper_connection_mixer is None on non-last ranks; None is not
#                     an nn.Module so AutoWeightsLoader cannot place its weights.
#   mtp.py            drafter consulted the parent PP group, so is_first_rank was
#                     False on the last rank and it demanded intermediate_tensors
#                     that never arrive. Branch on first-OR-last (per PR #46994).
#   model_state.py    blanket "PLE requires PP=1" guard -> allow when the PLE
#                     layers are confined to rank 0 (the only rank with input_ids).
#   gpu_worker.py     PLE offload rejected PP outright; same containment check.
#   ple_worker.py     the offload worker is not a pipeline rank but still parsed
#                     VLLM_PP_LAYER_PARTITION -> "len(partitions)=3 vs pp_size=1".
#   v2_model_runner.py  skip the PLE connector on ranks with no PleOffloadLayer
#                     (the worker expects exactly dp_size*tp_size registrations,
#                     a count that deliberately excludes PP). NOTE: the ACTIVE
#                     runner is v1/worker/gpu/model_runner.py, not gpu_model_runner.py.
#   connector.py      PLE offload stages inputs by copying from the runner's LIVE
#                     buffers, guarded only by a maxsize=1 queue + put_nowait so
#                     it fails loudly rather than staging stale inputs. That holds
#                     at PP=1; under PP rank 0 issues the next forward before the
#                     sender has staged the previous one -> queue.Full kills the
#                     worker (rank 0 at 0% GPU, ranks 1-2 spinning at 100%). A
#                     deeper queue or a blocking put would BOTH silently corrupt
#                     inputs: the sender dequeues before it stages. Fix adds a
#                     staging-complete event so a launch waits for the previous
#                     copy to finish. Serialises only the small input copy.
#   pp_utils.py       PORTED PR #46994: the last rank must relay draft tokens as a
#   + model_runner.py THIRD broadcast so every rank agrees on the per-step
#                     collective count. Without it ranks desync and a worker dies
#                     by signal during decode warmup ("Connection closed by peer").
#
# --mamba-ssm-cache-dtype float32 is REQUIRED, not tuning: without it rank 0's GDN
# state resolves to bfloat16 while rank 2's is float32, so main_kv page requirements
# differ per stage (816 vs 1584) and the CSA validator rejects the mismatch. float32
# matches the model's own config (mamba_ssm_dtype). Do NOT "fix" that with a huge
# --block-size: 1600 unified the pages but produced 3.27MB blocks and a segfault.
set -u

NAME=${QWEN38_NAME:-qwen38-pp}
PORT=${QWEN38_PORT:-8001}
IMG=${QWEN38_IMG:-vllm/vllm-openai:qwen38-flash-next}
HFCACHE=${QWEN38_HF:-/home/r/.cache/huggingface}
SNAPSHOT=${QWEN38_SNAPSHOT:-bcd9f01ddc9cff2316eb84281bebcd5b058bddce}
PARTITION=${QWEN38_PARTITION:-16,16,16}
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
PATCHDIR=${QWEN38_PATCH:-./patched}   # produce with serve/make-patched-tree.sh

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

docker run -d --name "$NAME" --gpus all --ipc=host --shm-size=32g \
  -e HF_HOME=/hf \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=5400 -e VLLM_BT_POOL=5 \
  -e VLLM_KV_CACHE_MEMORY_RANK0=17212851712 -e VLLM_KV_CACHE_MEMORY_RANK1=19231211520 \
  -e VLLM_KV_CACHE_MEMORY_RANK2=13324086784 \
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
  --pipeline-parallel-size 3 --moe-backend humming \
  --enable-prefix-caching \
  --mamba-ssm-cache-dtype float32 \
  --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" \
  ${YARN:+--hf-overrides "{\"text_config\":{\"rope_parameters\":{\"mrope_interleaved\":true,\"mrope_section\":[11,11,10],\"rope_type\":\"yarn\",\"rope_theta\":10000000,\"partial_rotary_factor\":0.25,\"factor\":$YARN,\"original_max_position_embeddings\":262144}}}"} \
  -cc.cudagraph_mode=PIECEWISE --no-enable-flashinfer-autotune \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_N}" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  >/dev/null

echo "launched $NAME on :$PORT  (PP3 $PARTITION, maxlen $MAXLEN, seqs $SEQS, mtp $SPEC_N)"
echo "startup ~5 min. follow:  docker logs -f $NAME"
