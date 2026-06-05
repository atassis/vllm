#!/bin/bash
# Run the E3 harness on gpu-wb. Kept as a FILE (not an inline ssh string) so the
# `pkill -f e3_run.py` below does not match — and kill — the launching shell.
# Env knobs (with defaults): MODE OUT QUANT_BITS CPU_OFFLOAD_GB GPU_MEM_UTIL.
cd /root/vllm-dev || exit 9
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -9 "$p" 2>/dev/null
done
pkill -9 -f e3_run.py 2>/dev/null
sleep 2
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=INFO CUDA_DEVICE_ORDER=PCI_BUS_ID
export QUANT_BITS="${QUANT_BITS:-4}"
export CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MODE="${MODE:-spec}"
OUT="${OUT:-spec_int4.json}"
LOG="${LOG:-run.log}"
echo "[run_a] MODE=$MODE OUT=$OUT QUANT_BITS=$QUANT_BITS CPU_OFFLOAD_GB=$CPU_OFFLOAD_GB GPU_MEM_UTIL=$GPU_MEM_UTIL" > "$LOG"
/root/vllm-dev/.venv/bin/python e3_run.py "$MODE" "$OUT" >> "$LOG" 2>&1
echo "[run_a] python exit=$?" >> "$LOG"
