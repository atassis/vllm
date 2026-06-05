"""E3 harness — Qwen3.5-27B PP=2 greedy, baseline vs MTP spec, with A1c knobs.

Env-configurable (so one file covers the whole sweep):
  MODE              "baseline" | "spec"  (arg1)
  OUT               output json path     (arg2)
  QUANT_BITS        draft_embed_quant_bits: unset|4|8  (A1c; spec only)
  CPU_OFFLOAD_GB    cpu_offload_gb (default 0 = test the real fit, no offload)
  GPU_MEM_UTIL      gpu_memory_utilization (default 0.90)
  PP_PARTITION      sets VLLM_PP_LAYER_PARTITION upstream (optional)

Prints clear markers + per-rank load is visible in vLLM logs. Oracle =
greedy baseline (base.json); spec must be token-identical (ignore_eos honored).
Run artifact (untracked, lives under docs/superpowers/tools), scp'd to gpu-wb.
"""

import json
import os
import sys

import torch

from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL", "/models/Qwen3.5-27B-AWQ")
PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue:",
    "def quicksort(arr):",
    "List three prime numbers:",
    "Translate 'good morning' to Spanish:",
]


def main():
    mode = sys.argv[1]
    out = sys.argv[2]
    bits = os.environ.get("QUANT_BITS")
    offload = int(os.environ.get("CPU_OFFLOAD_GB", "0"))
    util = float(os.environ.get("GPU_MEM_UTIL", "0.90"))

    print(
        f"[CFG] model={MODEL} mode={mode} quant_bits={bits} cpu_offload_gb={offload} "
        f"gpu_mem_util={util} pp_partition={os.environ.get('VLLM_PP_LAYER_PARTITION')}",
        flush=True,
    )

    max_tokens = int(os.environ.get("MAX_TOKENS", "40"))
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "128"))
    max_num_seqs = int(os.environ.get("MAX_NUM_SEQS", "1"))
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    kw = dict(
        model=MODEL,
        tensor_parallel_size=1,
        pipeline_parallel_size=int(os.environ.get("PP_SIZE", "2")),
        enforce_eager=True,
        gpu_memory_utilization=util,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "16")),
        trust_remote_code=True,
        cpu_offload_gb=offload,
        # Enable the periodic stat logger so SpecDecoding metrics (acceptance
        # rate, mean acceptance length) appear in the INFO log (offline LLM
        # disables stats by default).
        disable_log_stats=False,
    )
    # ASYNC_SCHED=0 forces SYNC scheduling (else vLLM auto-enables async for
    # MTP+PP). Sync uses the scheduler ship-back path, not the GPU broadcast.
    async_sched = os.environ.get("ASYNC_SCHED")
    if async_sched is not None:
        kw["async_scheduling"] = async_sched not in ("0", "false", "False", "")
    if mode == "spec":
        spec = {
            "method": "mtp",
            "num_speculative_tokens": int(os.environ.get("NUM_SPEC", "1")),
        }
        if bits:
            spec["draft_embed_quant_bits"] = int(bits)
        kw["speculative_config"] = spec

    try:
        llm = LLM(**kw)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL@LOAD] {type(e).__name__}: {e}", flush=True)
        raise

    print("[OK] engine constructed (model + KV fit)", flush=True)
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(
            f"[MEM] gpu{i} used={(total - free) / 2**30:.2f} "
            f"total={total / 2**30:.2f} GiB",
            flush=True,
        )

    import time

    # Warm up one short generation so the timed run excludes one-off CUDA/graph
    # init and the comparison is steady-state.
    llm.generate(["Hello"], SamplingParams(temperature=0.0, max_tokens=4))

    t0 = time.perf_counter()
    try:
        outs = llm.generate(PROMPTS, sp)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL@GENERATE] {type(e).__name__}: {e}", flush=True)
        raise
    elapsed = time.perf_counter() - t0

    res = [list(o.outputs[0].token_ids) for o in outs]
    with open(out, "w") as f:
        json.dump(res, f)
    total_out = sum(len(r) for r in res)
    print(f"[{mode}] wrote {len(res)} sequences -> {out}", flush=True)
    print(
        f"[PERF {mode}] elapsed={elapsed:.3f}s out_tokens={total_out} "
        f"tok/s={total_out / elapsed:.2f} "
        f"max_tokens={max_tokens} max_num_seqs={max_num_seqs}",
        flush=True,
    )
    for i, r in enumerate(res):
        print(f"  seq{i}: len={len(r)} first8={r[:8]}", flush=True)


if __name__ == "__main__":
    main()
