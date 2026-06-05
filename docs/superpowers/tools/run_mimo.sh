#!/bin/bash
# Map the MTP+PP execution cascade on MiMo-7B (cheap vehicle). Kept as a FILE so
# the `pkill -f e3_run.py` below cannot match/kill the launching ssh shell.
# Unlike run_a.sh, this does NOT default QUANT_BITS -> e3_run.py sees it UNSET
# (None) and passes NO draft_embed_quant_bits (MiMoMTP has no load-time quant path;
# A1c is Qwen3.5-only). MiMo fits PP=2 with margin -> no offload, no quant needed.
cd /root/vllm-dev || exit 9
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -9 "$p" 2>/dev/null
done
pkill -9 -f e3_run.py 2>/dev/null
sleep 2
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=INFO CUDA_DEVICE_ORDER=PCI_BUS_ID
export MODEL="${MODEL:-/models/MiMo-7B-Base}"
export CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
unset QUANT_BITS
MODE="${MODE:-spec}"
OUT="${OUT:-mimo_spec.json}"
LOG="${LOG:-mimo_run.log}"
echo "[run_mimo] MODEL=$MODEL MODE=$MODE OUT=$OUT CPU_OFFLOAD_GB=$CPU_OFFLOAD_GB GPU_MEM_UTIL=$GPU_MEM_UTIL" > "$LOG"
/root/vllm-dev/.venv/bin/python e3_run.py "$MODE" "$OUT" >> "$LOG" 2>&1
echo "[run_mimo] python exit=$?" >> "$LOG"
