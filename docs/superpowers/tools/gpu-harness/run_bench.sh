#!/bin/bash
# [[CMP]] benchmark: baseline vs spec, tokens/s + acceptance rate, per model.
# Usage: MODEL=... OFFLOAD=.. QUANT=.. TAG=.. MAXT=.. bash run_bench.sh
cd /root/vllm-dev || exit 9
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
sleep 2
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=INFO CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_LAUNCH_BLOCKING VLLM_PP_SPEC_DEBUG
export PP_SIZE=2 GPU_MEM_UTIL=0.90 MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}" MAX_NUM_SEQS=1
export MODEL="$MODEL" CPU_OFFLOAD_GB="${OFFLOAD:-0}" MAX_TOKENS="${MAXT:-200}"
TAG="${TAG:-bench}"; LOG="bench_${TAG}.log"
echo "[bench $TAG] MODEL=$MODEL offload=$CPU_OFFLOAD_GB maxt=$MAX_TOKENS quant=$QUANT" > "$LOG"
echo "===== BASELINE =====" >> "$LOG"
unset QUANT_BITS
/root/vllm-dev/.venv/bin/python e3_run.py baseline bench_${TAG}_base.json >> "$LOG" 2>&1
echo "[exit base=$?]" >> "$LOG"
echo "===== SPEC =====" >> "$LOG"
[ -n "$QUANT" ] && export QUANT_BITS="$QUANT"
/root/vllm-dev/.venv/bin/python e3_run.py spec bench_${TAG}_spec.json >> "$LOG" 2>&1
echo "[exit spec=$?]" >> "$LOG"
echo "===== SUMMARY ($TAG) =====" >> "$LOG"
grep -E "PERF (baseline|spec)" "$LOG" >> "$LOG.summary" 2>/dev/null
grep "SpecDecoding metrics" "$LOG" | tail -1 >> "$LOG.summary" 2>/dev/null
echo "[bench $TAG] DONE" >> "$LOG"
