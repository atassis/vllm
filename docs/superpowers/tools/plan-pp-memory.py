#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plan a PP layer partition + gpu_memory_utilization for PP>1 + MTP spec decode.

Work artifact (untracked). Answers "do I have to hand-tune memory per GPU every
time?" — NO. This computes VLLM_PP_LAYER_PARTITION and a safe
gpu_memory_utilization from the model config + the per-GPU free memory, accounting
for the Design-C draft's extra footprint on the LAST rank (its own vocab embedding;
the MTP lm_head is shared with the target, so it is NOT double-counted).

It is a *heuristic* planner: CUDA context and activation peak are empirical, so it
reports headroom and a feasibility verdict, then you do ONE verification run — far
better than blind sweeps. Calibrated against the gpu-wb 27B-AWQ/2x16GB runs
(2026-06-04): per-rank CUDA+framework overhead ~= 2.0 GiB; embedding/lm_head are
fp16 (vocab*hidden*2B) even under AWQ.

Usage:
  python plan-pp-memory.py --config /path/to/config.json \
      --gpu-free 15.5,15.4 --checkpoint-gib 20.35 [--bytes-per-weight 0.5]

  # or hardcode dims if no config.json handy:
  python plan-pp-memory.py --num-layers 64 --vocab 248320 --hidden 5120 \
      --gpu-free 15.5,15.4 --checkpoint-gib 20.35
"""

import argparse
import json

GIB = 1024**3
CTX_OVERHEAD_GIB = 2.0  # empirical CUDA context + framework per rank (gpu-wb)
MIN_KV_GIB = 0.4  # minimum KV headroom we insist each rank keeps


def fp16_table_gib(vocab: int, hidden: int) -> float:
    return vocab * hidden * 2 / GIB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="path to HF config.json")
    ap.add_argument("--num-layers", type=int)
    ap.add_argument("--vocab", type=int)
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--checkpoint-gib", type=float, required=True,
                    help="on-disk checkpoint size in GiB (target weights)")
    ap.add_argument("--gpu-free", required=True,
                    help="comma-separated free GiB per GPU, in PP-rank order")
    ap.add_argument("--draft-layers", type=int, default=1)
    ap.add_argument("--ctx-overhead-gib", type=float, default=CTX_OVERHEAD_GIB)
    args = ap.parse_args()

    if args.config:
        c = json.load(open(args.config))
        tc = c.get("text_config", c)
        num_layers = tc["num_hidden_layers"]
        vocab = tc["vocab_size"]
        hidden = tc["hidden_size"]
    else:
        num_layers, vocab, hidden = args.num_layers, args.vocab, args.hidden

    gpu_free = [float(x) for x in args.gpu_free.split(",")]
    pp = len(gpu_free)
    assert pp >= 2, "PP>=2 required"

    embed = fp16_table_gib(vocab, hidden)         # on rank 0
    lm_head = fp16_table_gib(vocab, hidden)        # on last rank (target)
    # per-transformer-layer weight (target), excluding embed/lm_head
    per_layer = (args.checkpoint_gib - embed - lm_head) / num_layers
    # Design-C draft on the LAST rank: its own embed (+ tiny layer/fc); lm_head shared
    draft_extra = embed + args.draft_layers * per_layer + 0.1

    print(f"# model: {num_layers} layers, vocab {vocab}, hidden {hidden}")
    print(f"# embed/lm_head (fp16): {embed:.2f} GiB each; per-layer ~{per_layer:.3f} GiB")
    print(f"# Design-C draft extra on last rank: {draft_extra:.2f} GiB "
          f"(own embed {embed:.2f} + {args.draft_layers} layer + fc)")
    print(f"# per-rank fixed overhead assumed: {args.ctx_overhead_gib:.1f} GiB")

    # Budget for *weights* on each rank = free - ctx - min_kv
    budget = [g - args.ctx_overhead_gib - MIN_KV_GIB for g in gpu_free]
    # Fixed (non-layer) weight on each rank:
    fixed = [0.0] * pp
    fixed[0] += embed
    fixed[-1] += lm_head + draft_extra
    # Greedy: assign layers to balance (budget_i - fixed_i)/per_layer capacity.
    cap = [max(0.0, (budget[i] - fixed[i]) / per_layer) for i in range(pp)]
    total_cap = sum(cap)
    if total_cap < num_layers:
        print(f"\n## VERDICT: DOES NOT FIT. Layer capacity {total_cap:.1f} < "
              f"{num_layers} needed. Short by ~{(num_layers-total_cap)*per_layer:.2f} GiB.")
        print("## Options: kv-cache fp8 won't help (weights-bound); reduce model, "
              "add a GPU, or share the draft embed across ranks (code change).")
    # Proportional integer split honoring capacity
    raw = [cap[i] / total_cap * num_layers for i in range(pp)]
    split = [int(x) for x in raw]
    while sum(split) < num_layers:
        # give the remaining layer to the rank with most slack
        slack = [(budget[i] - fixed[i] - split[i] * per_layer) for i in range(pp)]
        split[slack.index(max(slack))] += 1
    while sum(split) > num_layers:
        slack = [(budget[i] - fixed[i] - split[i] * per_layer) for i in range(pp)]
        split[slack.index(min(slack))] -= 1

    print(f"\nVLLM_PP_LAYER_PARTITION=\"{','.join(map(str, split))}\"")
    # Report per-rank physical usage + headroom, and a safe util
    worst_free = 1e9
    for i in range(pp):
        wt = fixed[i] + split[i] * per_layer
        phys = wt + args.ctx_overhead_gib
        free_after = gpu_free[i] - phys
        worst_free = min(worst_free, free_after)
        print(f"  rank{i}: {split[i]} layers, weights {wt:.2f} GiB, "
              f"+ctx -> {phys:.2f}/{gpu_free[i]:.2f} GiB, KV headroom {free_after:.2f} GiB"
              + ("  <-- TIGHT" if free_after < MIN_KV_GIB else ""))
    # util sized so KV reservation never starves the draft load on the last rank
    # (draft loads before KV reservation, so keep (1-util)*total >= draft slack)
    safe_util = round(min(0.95, max(0.70,
                  1 - (draft_extra) / max(gpu_free))), 2)
    print(f"\n# suggested gpu_memory_utilization ~= {safe_util} "
          f"(leaves physical room for the draft weight load on the last rank)")
    if worst_free < MIN_KV_GIB:
        print("# WARNING: tightest rank has <0.4 GiB KV headroom -> reduce "
              "max_model_len / max_num_batched_tokens, or it may OOM in profiling.")


if __name__ == "__main__":
    main()
