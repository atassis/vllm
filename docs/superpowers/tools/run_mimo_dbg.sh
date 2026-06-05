#!/bin/bash
# s8 trajectory probe: MiMo-7B PP=2 + MTP async on V1 (NOT V2) with
# VLLM_PP_SPEC_DEBUG=1 to capture the multi-step write-pos (receiver) vs
# read-pos (_prepare_inputs) trajectory leading to break#2, so the holistic C4
# receiver formula is grounded in real data instead of guessed (B1b history).
# Working tree under test = current (B1a + C3 + C4(A) single-slot + the PPDBG
# probe). The probe only logs; behaviour is the still-insufficient C4(A).
cd /root/vllm-dev || exit 9
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -9 "$p" 2>/dev/null
done
pkill -9 -f e3_run.py 2>/dev/null
sleep 2
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=INFO CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=1
export VLLM_PP_SPEC_DEBUG=1
export MODEL="${MODEL:-/models/MiMo-7B-Base}"
export CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
unset QUANT_BITS
MODE="${MODE:-spec}"
OUT="${OUT:-mimo_spec_dbg.json}"
LOG="${LOG:-mimo_run_dbg.log}"
echo "[run_mimo_dbg] PPDBG=1 MODEL=$MODEL MODE=$MODE OUT=$OUT" > "$LOG"
/root/vllm-dev/.venv/bin/python e3_run.py "$MODE" "$OUT" >> "$LOG" 2>&1
echo "[run_mimo_dbg] python exit=$?" >> "$LOG"
