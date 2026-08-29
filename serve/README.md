# Serving

- **`run-pp3-mtp.sh`** — the exact production launcher: every engine flag,
  env var, and bind-mount, with comments explaining why each load-bearing
  flag is load-bearing (HUMMING vs Marlin/Triton on sm_80, PIECEWISE vs FULL
  graphs, fp32 SSM cache dtype, `VLLM_BT_POOL`, the PLE offload envs, the
  optional YaRN block for 524k context). Tunables are env-overridable:
  `QWEN38_SPEC` (MTP depth, default 3), `QWEN38_SEQS`, `QWEN38_MAXLEN`,
  `QWEN38_PARTITION`, `QWEN38_YARN`, `QWEN38_HF`, `QWEN38_PATCH`.
- **`make-patched-tree.sh`** — extracts the touched files from the
  `vllm/vllm-openai:qwen38-flash-next` image and applies `patches/`,
  producing the flat `patched/` directory the launcher bind-mounts.

```bash
./serve/make-patched-tree.sh
QWEN38_HF=/path/to/huggingface-cache ./serve/run-pp3-mtp.sh
```
