#!/bin/bash
# num_speculative_tokens sweep on MiMo PP=2: greedy-equiv (vs mimo_base.json) +
# speedup + acceptance, for k=1,2,3. Tests whether the s9 PP+spec fixes generalize
# past k=1 (broadcast width = num_spec+1; valid count v in 1..k+1).
cd /root/vllm-dev || exit 9
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
sleep 3
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=INFO CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_LAUNCH_BLOCKING VLLM_PP_SPEC_DEBUG QUANT_BITS
export MODEL=/models/MiMo-7B-Base PP_SIZE=2 CPU_OFFLOAD_GB=0 GPU_MEM_UTIL=0.90
export MAX_TOKENS=120 MAX_MODEL_LEN=512 MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS=32
LOG=kspec.log; : > "$LOG"
for K in 1 2 3; do
  echo "===== NUM_SPEC=$K =====" >> "$LOG"
  NUM_SPEC=$K /root/vllm-dev/.venv/bin/python e3_run.py spec kspec_$K.json >> "$LOG" 2>&1
  echo "[exit k=$K=$?]" >> "$LOG"
done
echo DONE >> "$LOG"
