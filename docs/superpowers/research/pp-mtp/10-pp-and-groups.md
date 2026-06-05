# Brick 10 — Pipeline Parallelism & Process Groups in vLLM

Status: **DONE** · Answers Q1 · Verified against code (paths relative to repo root).

> How vLLM splits a model across GPUs by stages, what a "PP group" is, and what
> is / isn't shared across TP vs PP ranks. This is the foundation the whole
> PP+MTP task sits on.

---

## 1. GroupCoordinator — the object behind `get_pp_group()`

Defined in `vllm/distributed/parallel_state.py:290`. Wraps a torch
`ProcessGroup` and owns all comms for one group of processes.

Key properties (parallel_state.py:455-487):
- `world_size` = `len(ranks)` — size **of this group**, not the global world.
- `rank_in_group` = `ranks.index(self.rank)` — position within the group.
- `is_first_rank` = `rank == ranks[0]`; `is_last_rank` = `rank == ranks[-1]`.
- `device_group` (NCCL) for GPU comms; `cpu_group` (gloo) for coordination.

A model's `forward` asks this object "am I the first/last stage?" to decide
whether to embed input or read activations from the previous stage.

## 2. The group singletons — one of each, per process

Module-level globals in `parallel_state.py:1257-1318`, each a **single
singleton per process**, with an accessor that asserts it's initialized:

| Global | Accessor | Axis |
|---|---|---|
| `_TP` | `get_tp_group()` | tensor parallel |
| `_PP` | `get_pp_group()` | pipeline parallel |
| `_DP` | `get_dp_group()` | data parallel |
| `_EP` / `_EPLB` | `get_ep_group()` | expert parallel (MoE) |
| `_PCP` / `_DCP` | `get_pcp_group()` / `get_dcp_group()` | context parallel |

**→ implication (the crux of E2):** there is exactly **one** `_PP` per process.
The draft model and the target model both call the same `get_pp_group()`. There
is no separate "draft PP group" unless one is explicitly created. (Reopened
Design-A would have to create/redirect one — see Q4.)

## 3. Group creation & the rank math

`initialize_model_parallel(tp, pp, pcp, dcp, backend)` (parallel_state.py:1522)
is called **once**, with the **target** model's sizes, from the worker:
`ensure_model_parallel_initialized(...)` ← `gpu_worker.py:1171`.
`world_size = ExternalDP × DP × PP × PCP × TP` (parallel_state.py:1588-1603).

Ranks are reshaped into a grid and grouped per axis:
- TP groups: `all_ranks.view(-1, tp).unbind(0)` (1605-1620).
- PP groups: `all_ranks.transpose(2,4).reshape(-1, pp).unbind(0)` (1663-1679).

Example (docstring 1531-1551): 8 GPUs, TP=2, PP=4 → TP groups
`[g0,g1][g2,g3][g4,g5][g6,g7]`, PP groups `[g0,g2,g4,g6][g1,g3,g5,g7]`.

**Our config (2 GPU, PP=2, TP=1):** one PP group `[0,1]`, no TP groups.

## 4. What is / isn't shared — TP vs PP

### Tensor parallel (TP) — splits *within* a layer
- **Weights: sharded.** `ColumnParallelLinear` (linear.py:407) shards output dim
  (`output_size/tp_size`); `RowParallelLinear` (linear.py:1389) shards input dim.
- **Input tokens: replicated** (all TP ranks see the same `input_ids`).
- **Activations: all-reduced/all-gathered** within the TP group
  (`gather_output` / `reduce_results`). This is the per-layer PCIe-heavy traffic
  we avoid by not using TP across the no-NVLink pair.

### Pipeline parallel (PP) — splits *across* layers
- **Layers: disjoint per rank.** `make_layers` (models/utils.py:617) builds
  `[PPMissingLayer]*start + real[start:end] + [PPMissingLayer]*rest`, with
  `start,end = get_pp_indices(...)`.
- **`embed_tokens`: only on the first rank.** Canonical guard (llama.py:368):
  `if is_first_rank or (tie_word_embeddings and is_last_rank): VocabParallelEmbedding(...) else: PPMissingLayer()`.
- **`norm` + `lm_head`: only on the last rank** (llama.py:383).
- **What crosses the PP boundary:** `IntermediateTensors` = a dict
  `{hidden_states, residual}` (sequence.py:12), sent stage→stage via
  `get_pp_group().send_tensor_dict(...)` (gpu_model_runner.py:4317) /
  `irecv_tensor_dict(...)` (gpu_worker.py). Small tensors (per-token hidden
  vectors), unlike TP's big per-layer all-reduce.

**→ implication for the MTP draft:** the draft's `embed_tokens` matters only on
the first rank by this convention; its `norm`/`lm_head` only on the last. Any
design that runs the draft *only on the last rank* (Design A) collides with the
"embed lives on first rank" convention → must replicate embed onto the last rank
or bypass the convention. This is exactly Q2/Q3.

## 5. PPMissingLayer & weight loading

`PPMissingLayer(nn.Identity)` (models/utils.py:604) is a no-op placeholder for
layers not on this rank; its `forward` returns its first arg unchanged.
Weight loading skips it: `_load_module` early-returns for `PPMissingLayer`
(models/utils.py:265); `is_pp_missing_parameter(name, model)` (models/utils.py:674)
filters params whose layer isn't local.

**→ implication:** under PP, each rank only loads the weights for *its* layers +
its embed (first) or norm/lm_head (last). For the draft, this governs where the
`mtp.*` and the borrowed `embed_tokens` weights physically land (Q3/brick 50).

## 6. Layer split — `get_pp_indices` (and uneven splits)

`get_pp_indices(num_layers, pp_rank, pp_size)` (distributed/utils.py:109):
even split `num_layers // pp_size`; remainder goes to the *rightmost ranks
except the last* (the last holds the output norm). Overridable via
`VLLM_PP_LAYER_PARTITION`.

**→ implication (PR #16568 review concern):** code that computes a draft layer
offset as `target_layers × pp_size` assumes an even split and breaks on uneven
ones. For our models the MTP is a single layer, so the *target's* split is what
matters for where hidden states are produced, not the draft's. (Q7.)

## 7. Canonical PP forward (the pattern MTP mimics)

`LlamaModel.forward` (llama.py:395-434):
```python
if get_pp_group().is_first_rank:
    hidden_states = inputs_embeds or self.embed_input_ids(input_ids)
    residual = None
else:
    hidden_states = intermediate_tensors["hidden_states"]   # from prev stage
    residual = intermediate_tensors["residual"]
for layer in self.layers[start:end]:
    hidden_states, residual = layer(positions, hidden_states, residual)
if not get_pp_group().is_last_rank:
    return IntermediateTensors({"hidden_states": ..., "residual": ...})  # to next
hidden_states, _ = self.norm(hidden_states, residual)
return hidden_states
```
`make_empty_intermediate_tensors_factory(["hidden_states","residual"], hidden)`
(models/utils.py:685) provides the profiling/placeholder path for non-first ranks.

`Qwen3_5MultiTokenPredictor.forward` (qwen3_5_mtp.py:132-159) follows the **same
pattern** — which is why it currently keys off the global `_PP` group, and why
naive Design A breaks (on the last rank `is_first_rank=False`).

---

## Open questions spawned / touched by this brick
- Q2/Q3 → brick 20 (embeddings): where the draft's embed weights live under PP.
- Q4 → can we redirect `get_pp_group()` for the draft forward only?
- Q5/Q6 → brick 30 (spec arch): what feeds the draft, and on which rank.
- Q7 → uneven split impact (likely nil for a 1-layer draft).
