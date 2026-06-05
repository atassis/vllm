# Brick 20 — Embeddings under PP & the MTP draft

Status: **DONE** · Answers Q2, Q3 · Verified against code.

> How token embeddings work in vLLM, where their weights live under PP, and —
> the key result — that the MTP draft already holds a *weight-loaded* embedding
> on every rank, including the last. This reshapes the A/B decision.

---

## 1. VocabParallelEmbedding — the embedding layer

`vllm/model_executor/layers/vocab_parallel_embedding.py:192`. Shards the **vocab
dimension across TP ranks** (`num_embeddings_per_partition = vocab/tp_size`,
`__init__` ~233-319).

- `forward` (477-497): with `tp_size>1`, masks ids outside this rank's vocab
  slice, looks up, zeros out-of-range rows, then `tensor_model_parallel_all_reduce`
  to assemble the full embedding. **With `tp_size==1` (our case): no mask, no
  all-reduce — a plain lookup.**
- `weight_loader` (424-475): narrows the full `[vocab, hidden]` weight to this
  rank's vocab slice. (TP-only; PP doesn't shard the embedding.)

**→ implication:** the embedding shards along **TP**, not PP. PP decides *whether*
a rank has the embedding module at all (next section), not how it's sliced. With
TP=1 the embedding is a trivial, self-contained lookup on whatever rank owns it.

## 2. Embedding placement under PP (base models) + tie_word_embeddings

Base models gate the module by PP rank (llama.py:368, qwen2.py:334):
```python
if is_first_rank or (tie_word_embeddings and is_last_rank):
    self.embed_tokens = VocabParallelEmbedding(...)
else:
    self.embed_tokens = PPMissingLayer()
```
`lm_head` lives on the last rank (llama.py:583, qwen3.py:288); when
`tie_word_embeddings` it *is* `self.model.embed_tokens` (the same Parameter),
which is why the embed is also instantiated on the last rank in the tied case.

**Our target (Qwen3.5-27B-AWQ): `tie_word_embeddings=False`** → base embed only
on the first rank; separate `lm_head` on the last rank.

## 3. THE KEY FINDING — the MTP draft's embedding is unconditional

`Qwen3_5MultiTokenPredictor.__init__` (qwen3_5_mtp.py:75) creates the embedding
**with no PP guard**:
```python
self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)
```
Same pattern in `NemotronHMTP` (nemotron_h_mtp.py:235) and `Glm4MoeLiteMTP`
(glm4_moe_lite_mtp.py:163) — **every MTP draft creates `embed_tokens` on every
rank**, unlike base models.

Consequence for weight loading (utils.py): `is_pp_missing_parameter` (674) only
treats a param as missing if its module is a `PPMissingLayer`. The draft's
`embed_tokens` is a *real* `VocabParallelEmbedding` on every rank → **not**
skipped → its weights are loaded on **every rank, including the last**.

Weight source: `Qwen3_5MTP.load_weights.remap_weight_names` (qwen3_5_mtp.py:446)
accepts `embed_tokens`/`lm_head` keys from the checkpoint stream (the base
model's `embed_tokens.weight` in the same checkpoint; the `mtp.*` tensors have no
embed of their own).

| Rank (PP=2) | base model `embed_tokens` | MTP draft `embed_tokens` |
|---|---|---|
| 0 (first) | real, weights loaded | real, weights loaded |
| 1 (last) | `PPMissingLayer`, skipped | **real, weights loaded** |

## 4. load_eagle_model — embedding sharing is PP-aware (skips under PP)

`vllm/v1/worker/gpu/spec_decode/eagle/utils.py:27` (MTP reaches it via
`MTPSpeculator.load_draft_model`, mtp/speculator.py:12):
```python
# Skip embedding sharing under PP — each rank owns its own embedding.
if get_pp_group().world_size == 1:
    ... draft_inner.embed_tokens = target_embed   # share only when no PP
```
With PP (`world_size>1`) the sharing block is **skipped** → the draft keeps its
own, independently-weight-loaded `embed_tokens` (per §3). `_should_share`
(utils.py:11) only matters in the non-PP path.

## 5. Definitive answer to Q3 (and what it unlocks)

**Q3 — Does the MTP draft's `embed_tokens` have valid weights on the last PP
rank under PP=2? → YES.** Created unconditionally (§3), not pp-missing, loaded
from the base embed in the checkpoint, sharing skipped under PP (§4).

**→ implication (reshapes the decision):** the embedding obstacle I attributed to
Design A **does not exist** — the draft already has a working embedding on the
last rank. The *only* remaining reason a draft-on-last-rank design fails today is
that `Qwen3_5MultiTokenPredictor.forward` branches on the **global** PP group
(`is_first_rank` is False on the last rank → it skips the embed path and waits
for intermediate tensors). That is a **localized forward-logic issue**, fixable
with a "this draft runs standalone (first==last)" flag — **no separate process
group, no embed replication, no global-state surgery, and no Design-B backward
dependency.** This is the basis for **Design C** (see `00-map.md`).

## 6. Remaining uncertainties (need execution — E3)

1. `VocabParallelEmbedding.forward` on a non-first PP rank with TP=1 should be a
   plain lookup (no mask/all-reduce) — verify nothing implicitly assumes "first
   rank." (Low risk by code reading; confirm at runtime.)
2. Confirm the draft's layer/attention path on the last rank doesn't itself call
   `get_pp_group()` in a way that breaks when treated as standalone. (→ brick 60.)
3. Confirm the remap actually loads the base `embed_tokens` weights into the
   draft on the last rank at runtime (not just that the module exists). (→ E3 /
   brick 50.)

---

## Open questions updated by this brick
- **Q2 — ANSWERED:** embed shards over TP, placed per-PP-rank for base models;
  for MTP drafts it's unconditional.
- **Q3 — ANSWERED (YES):** draft embed is weight-loaded on the last rank.
- **Q4 — narrowed:** no need to redirect `get_pp_group()`; a forward flag making
  the draft behave as standalone first==last suffices (verify layer/attn don't
  depend on the group → brick 60).
- Spawns **Design C** in the solution space.
