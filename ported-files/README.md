# Wholesale-ported files

Three files are complete replacements rather than small diffs, so they ship
whole (the numbered patches document their headline deltas, but do not
reconstruct them from stock):

- **`pp_utils.py`** — the vllm#46994 MTP-under-PP relay port: sampled-token
  broadcast on a sibling communicator plus the third broadcast relaying the
  proposed draft tokens, with generation-counted invalidation of freed
  request slots.
- **`v2_model_runner.py`** — the matching V2 runner port (per-step relay
  consume/deferred postprocess), plus the patch 0007 PLE-connector skip and
  the patch 0010 runner-side change (mamba ctx gets the source req-indexed
  block tables).
- **`fused_recurrent.py`** — upstream vllm#53877 (fp32 GDN decode beta),
  kept for output quality.

`serve/make-patched-tree.sh` overlays these after applying `patches/`.
