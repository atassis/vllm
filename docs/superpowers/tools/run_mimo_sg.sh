#!/bin/bash
# s8 isolation: single-GPU MiMo MTP spec decode (PP_SIZE=1, no PP). If spec ==
# greedy baseline here, the greedy-equiv divergence seen under PP=2 is
# PP-specific (spec-position hidden-state feeding on the non-last rank). If it
# ALSO diverges single-GPU, the bug is in MTP/the model itself, not PP.
# Runs on ONE GPU (gpu0); leaves gpu1 free. No PPDBG / no broadcast (pp=1).
cd /root/vllm-dev || exit 9
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -9 "$p" 2>/dev/null
done
sleep 2
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=INFO CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PP_SIZE=1
# MiMo-7B (~15 GiB bf16) does not fit on one 16GB GPU with KV cache (PP=2 splits
# it across two). Offload weights to CPU to free room for KV + the MTP draft.
# cpu_offload changes only WHERE weights live, not the math -> greedy-equiv valid.
export MODEL="${MODEL:-/models/MiMo-7B-Base}"
export CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-7}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
unset QUANT_BITS VLLM_PP_SPEC_DEBUG
LOG="${LOG:-mimo_sg.log}"
echo "[run_mimo_sg] PP_SIZE=1 single-GPU MODEL=$MODEL" > "$LOG"
/root/vllm-dev/.venv/bin/python e3_run.py baseline sg_base.json >> "$LOG" 2>&1
echo "[run_mimo_sg] baseline exit=$?" >> "$LOG"
/root/vllm-dev/.venv/bin/python e3_run.py spec sg_spec.json >> "$LOG" 2>&1
echo "[run_mimo_sg] spec exit=$?" >> "$LOG"
