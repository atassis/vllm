#!/bin/bash
# Lean acceptance measurement: SPEC arm only (baseline tok/s already known),
# short length, with disable_log_stats=False -> SpecDecoding metrics in log.
cd /root/vllm-dev || exit 9
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
sleep 3
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=INFO CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_LAUNCH_BLOCKING VLLM_PP_SPEC_DEBUG
export PP_SIZE=2 GPU_MEM_UTIL=0.90 MAX_NUM_SEQS=1
export MODEL="$MODEL" CPU_OFFLOAD_GB="${OFFLOAD:-0}" MAX_TOKENS="${MAXT:-80}" MAX_MODEL_LEN="${MML:-256}"
TAG="${TAG:-acc}"; LOG="accept_${TAG}.log"
[ -n "$QUANT" ] && export QUANT_BITS="$QUANT"
echo "[accept $TAG] start maxt=$MAX_TOKENS" > "$LOG"
/root/vllm-dev/.venv/bin/python e3_run.py spec accept_${TAG}.json >> "$LOG" 2>&1
echo "[exit=$?] DONE" >> "$LOG"
