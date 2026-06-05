# Brick 30 — Spec-decode dataflow & KV cache (draft input, sharing, rollback)

Status: **DONE** · Answers Q5, Q9, Q10, Q11; nuances Q12 · Verified against code
(V1 runner path, which Qwen3.5 uses).

> What feeds the MTP draft at runtime, on which rank, and how the draft's KV
> cache relates to the target's (own vs shared, block tables, rejection
> rollback). Directly informs Design C and the efficiency lens.

---

## Part A — what feeds the draft (Q5, Q9)

**The drafter runs only on the last PP rank** (gpu_model_runner.py:542):
```python
if self.speculative_config and get_pp_group().is_last_rank:
    self.drafter = EagleProposer(...)  # / DraftModelProposer / MTP path
```

**`propose()` is called with the target's final hidden states** already on GPU
(gpu_model_runner.py:5050-5061): `target_hidden_states`, `target_token_ids`,
`target_positions`, `next_token_ids` (last sampled token), `common_attn_metadata`
(block_table, seq_lens, slot_mapping), `num_rejected_tokens_gpu`, `slot_mappings`.

`target_hidden_states` come from the target model forward output
(`hidden_states = model_output`, gpu_model_runner.py:4290), sliced at
:5002 (first proposal) or :5019 (post-rejection via `token_indices`). The draft
forward then consumes them (llm_base_proposer.py `propose()` → `self.model(**kw)`),
looping for `num_speculative_tokens>1`.

**→ Q9 = YES (decisive for Design C).** On the last PP rank the target's final
hidden state is already resident and handed to the drafter with **zero cross-rank
communication**. A draft-standalone-on-last-rank design (C) has its input for
free — no PCIe hop. This is the most PCIe-friendly arrangement possible.

## Part B — the draft's KV cache (Q10, Q11)

**Own KV tensors, same KV-cache group as the target.** The proposer assigns the
draft layers into the target's existing `kv_cache_group` (llm_base_proposer.py:1561):
```python
for gid, group in enumerate(kv_cache_config.kv_cache_groups):
    if self._draft_attn_layer_names & set(group.layer_names):
        self.kv_cache_gid = gid   # draft joins the SAME group id
```
The draft has its own KV tensors (its own attention layer) but lives in the same
group/budget as the target.

**Block tables & slot mapping are SHARED** (llm_base_proposer.py:1015-1016): the
proposer builds `CommonAttentionMetadata` reusing the target's
`block_table_tensor` and `slot_mapping`. So draft and target address the same
per-request block allocation.

**Rejection rollback is unified, not a physical clear** (Q11):
- scheduler decrements `num_computed_tokens -= num_rejected` (scheduler.py:1406)
  and `num_output_placeholders` (1410).
- the next draft step reduces `seq_lens -= num_rejected_tokens_gpu`
  (llm_base_proposer.py:563); rejected KV slots are simply **overwritten** next
  step, never explicitly cleared.
- because draft and target share `seq_lens`/block tables, **both caches roll back
  together** — no separate draft cleanup.

**→ implication:** the stakeholder's "confirmed segments reused, rejected keys
cleared" is essentially already the behavior — confirm-and-prune via shared
`seq_lens`, with overwrite-on-reuse instead of explicit clears.

## Part C — cross-model KV sharing: precedent & why Qwen3.5 differs (Q12)

**Precedent exists.** Gemma4 MTP and Step3.5 share KV *across models*: draft
attention reuses the target's KV tensors via `kv_sharing_target_layer_name`
(gemma4.py:334) → `shared_kv_cache_layers` → `kv_caches[draft] = kv_caches[target]`
(gpu_model_runner.py:7232, literally the same tensor). This is exactly the
"unified memory" idea — already shipped for those models.

**But it does not transfer to Qwen3.5 MTP.** Gemma4's draft is **Q-only** (no own
K/V → can borrow the target's). Qwen3.5 MTP has its **own** `self_attn.k_proj` /
`v_proj` (checkpoint: `mtp.layers.0.self_attn.{q,k,v}_proj`), so it computes its
own K/V which cannot be substituted by the target's (different projections).

**And it barely matters for our case.** The Qwen draft is **1 layer** vs 32+
target layers (~3% of target KV), already shares block tables/slot mapping, and
rolls back for free. The KV-memory prize is small; the real efficiency win is
Design C's zero cross-stage traffic (Part A) + MTP's free draft compute (it
reuses the target hidden state as input). Record cross-model KV sharing as a
**known lever for other models / future num_spec>1**, not a Qwen3.5 win.

## Efficiency-lens ledger (running)

| Lever | Applies to Qwen3.5 MTP? | Note |
|---|---|---|
| Reuse target hidden state as draft input (no big draft forward) | ✅ already | core MTP mechanism |
| Draft input resident on last rank (no PCIe hop) | ✅ via Design C | Q9 |
| Share block tables / slot mapping | ✅ already | llm_base_proposer.py:1015 |
| Unified rejection rollback (shared seq_lens) | ✅ already | :563 |
| Cross-model KV sharing (draft borrows target KV) | ❌ (own k/v proj) | works for Gemma4/Step3.5; not Qwen |
| Avoid separate draft KV alloc | ➖ small (1 layer) | low value for Qwen |

## Items to verify at runtime (E3)
- Exact `target_hidden_states` slice in the post-rejection path for num_spec>1.
- `parallel_drafting` branch (llm_base_proposer.py:507) — parallel vs sequential.
- Whether rejected KV blocks are reclaimed or "wasted" until sequence end.

---

## Open questions updated
- **Q5 — ANSWERED:** draft fed `target_hidden_states` (+ block/slot metadata).
- **Q9 — ANSWERED (YES):** last-rank drafter has target hidden state resident,
  zero cross-rank fetch → Design C input is free.
- **Q10 — ANSWERED:** own KV tensors, same group; Gemma4/Step3.5 share cross-model.
- **Q11 — ANSWERED:** shared block tables/slot mapping; unified rollback via seq_lens.
- **Q12 — NUANCED:** KV-sharing idea is real (Gemma4 precedent) but inapplicable
  to Qwen3.5 (own k/v projections) and low-value (1-layer draft). Logged as a
  lever for other models.
- **Q4 — nearly closed:** the only PP-group dependence in the draft path is the
  MTP model forward's is_first_rank/is_last_rank branching (brick 20); the draft
  attention/KV runs single-rank on the last rank already. One confirmation left
  in brick 60.
