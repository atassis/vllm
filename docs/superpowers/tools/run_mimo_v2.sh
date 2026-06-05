#!/bin/bash
# s8 V2-viability probe: same MiMo-7B PP=2 + MTP async run as run_mimo.sh, but
# FORCE the V2 model runner via VLLM_USE_V2_MODEL_RUNNER=1. Goal: does V2's
# native PPHandler path run MiMo PP=2+MTP WITHOUT break#2 (the V1 non-last-rank
# `-1` cascade we hand-fix in C4)? Tested on a clean-for-MiMo tree (B1a/C3/C4
# stashed; committed A1c/standalone-flag are Qwen3.5-only -> inert for MiMo).
# Kept as a FILE so the `pkill -f e3_run.py` below cannot match/kill the ssh shell.
cd /root/vllm-dev || exit 9
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -9 "$p" 2>/dev/null
done
pkill -9 -f e3_run.py 2>/dev/null
sleep 2
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=INFO CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_V2_MODEL_RUNNER=1          # <-- the whole point of this probe
export CUDA_LAUNCH_BLOCKING=1              # crisp tracebacks at the crash site
export MODEL="${MODEL:-/models/MiMo-7B-Base}"
export CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
unset QUANT_BITS
MODE="${MODE:-spec}"
OUT="${OUT:-mimo_spec_v2.json}"
LOG="${LOG:-mimo_run_v2.log}"
echo "[run_mimo_v2] V2_RUNNER=1 MODEL=$MODEL MODE=$MODE OUT=$OUT CPU_OFFLOAD_GB=$CPU_OFFLOAD_GB GPU_MEM_UTIL=$GPU_MEM_UTIL" > "$LOG"
/root/vllm-dev/.venv/bin/python e3_run.py "$MODE" "$OUT" >> "$LOG" 2>&1
echo "[run_mimo_v2] python exit=$?" >> "$LOG"
