#!/bin/bash
# Extract the files touched by patches/ from the vLLM image and apply the
# patches, producing the flat ./patched/ directory that run-pp3-mtp.sh
# bind-mounts. Run from the repo root.
set -eu
IMG=${QWEN38_IMG:-vllm/vllm-openai:qwen38-flash-next}
V=/usr/local/lib/python3.12/dist-packages/vllm
OUT=${1:-patched}
mkdir -p "$OUT" work/vllm

# repo-relative source path for each mounted file (image tree layout)
declare -A SRC=(
  [model.py]=models/qwen3_8_flash_next/nvidia/model.py
  [mtp.py]=models/qwen3_8_flash_next/nvidia/mtp.py
  [model_state.py]=models/qwen3_8_flash_next/nvidia/model_state.py
  [ple_layer.py]=models/qwen3_8_flash_next/nvidia/ple_layer.py
  [v2_model_runner.py]=v1/worker/gpu/model_runner.py
  [pp_utils.py]=v1/worker/gpu/pp_utils.py
  [gpu_worker.py]=v1/worker/gpu_worker.py
  [ple_worker.py]=v1/ple_offload/worker.py
  [connector.py]=v1/ple_offload/connector.py
  [kv_cache_utils.py]=v1/core/kv_cache_utils.py
  [block_table.py]=v1/worker/gpu/block_table.py
  [mamba_utils.py]=v1/worker/mamba_utils.py
  [single_type_kv_cache_manager.py]=v1/core/single_type_kv_cache_manager.py
  [mamba_hybrid.py]=v1/worker/gpu/model_states/mamba_hybrid.py
  [fused_recurrent.py]=third_party/flash_linear_attention/ops/fused_recurrent.py
)
for out in "${!SRC[@]}"; do
  p=${SRC[$out]}
  mkdir -p "work/vllm/$(dirname "$p")"
  docker run --rm --entrypoint cat "$IMG" "$V/$p" > "work/vllm/$p"
done
( cd work
  for f in ../patches/00*.patch; do
    case "$f" in *WITHDRAWN*) echo "skip $(basename "$f")"; continue;; esac
    patch -p1 --forward < "$f" && echo "applied $(basename "$f")"
  done )
for out in "${!SRC[@]}"; do cp "work/vllm/${SRC[$out]}" "$OUT/$out"; done
# Overlay the wholesale-ported files (see ported-files/README.md): these are
# complete replacements the numbered patches do not reconstruct from stock.
for f in ported-files/*.py; do cp "$f" "$OUT/$(basename "$f")"; done
for f in "$OUT"/*.py; do python3 -m py_compile "$f"; done
echo "patched tree ready in $OUT/ ($(ls "$OUT" | wc -l) files)"
echo "NOTE: if a patch hunk fails against a newer image, see"
echo "patches-rebased-head/ and FINDINGS.md."
