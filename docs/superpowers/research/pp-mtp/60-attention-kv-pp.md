# Brick 60 — Draft attention/KV path & PP dependencies (closing Q4)

Status: **DONE** · Answers Q4 · Verified against code.

> Complete inventory of where the draft execution path depends on PP rank, to
> confirm that a standalone-on-last-rank draft (Design C) is safe under PP=2.

---

## Verdict

**Design C is safe.** The **only** PP-rank dependency that affects the draft is
the MTP model forward's `is_first_rank`/`is_last_rank` branching
(`qwen3_5_mtp.py:132-160`). Everything else in the draft path is PP-neutral, and
there is **no deadlock risk** (non-last ranks have no drafter and never wait on
the draft). One small localized change (a "standalone draft" flag in the forward)
makes C work.

## Inventory (confirmed-safe unless noted)

| Component | file:line | PP usage | Verdict |
|---|---|---|---|
| Proposer base | `llm_base_proposer.py:1282` | `if get_pp_group().world_size == 1:` gates target↔draft embed/lm_head **sharing** | **safe** — under PP it's False → draft keeps its own (already loaded, brick 20) |
| EagleProposer / DraftModelProposer | `eagle.py`, `draft_model.py` | none | safe |
| Attn metadata build | `llm_base_proposer.py:863` | none — built from `common_attn_metadata` | safe |
| Slot mapping | `llm_base_proposer.py:362` | none — reuses target's slot mapping | safe |
| KV writes / attn backend | v1 attention | none — no PP send/recv in attn | safe (no deadlock) |
| `set_forward_context` | `forward_context.py` | DP-only, no PP-rank logic | safe |
| Draft layer construction | `qwen3_5_mtp.py:99` | builds `self.layers` directly (NOT `make_layers`/`get_pp_indices`) | safe — all draft layers exist locally; no PPMissingLayer for the draft |
| `lm_head` / `compute_logits` | `qwen3_5_mtp.py:377`, `llm_base_proposer.py:400` | `lm_head` on `is_last_rank` | safe — draft runs on the last rank, so it exists |
| Decoder layer / norm / linear | qwen3_5 layers | no all-reduce/send/recv (TP=1) | safe (no deadlock) |
| Drafter init | `gpu_model_runner.py:542` | `if ... is_last_rank:` | safe — drafter only on last rank |
| Non-last rank | `gpu_model_runner.py:~1257` | no drafter, skips spec | safe — never calls `propose()`, never blocks on it |
| `propose()` call | `gpu_model_runner.py:~5050` | inside last-rank drafter guard | safe |

## The one change — the standalone-draft flag (Q4 detail)

The draft model forward (`Qwen3_5MultiTokenPredictor.forward`) consults the
**global** `_PP` group (there is **no** separate draft PP group — brick 10 / E2).
On a target-PP=2 run the drafter executes on the **last** global rank, where:
- `get_pp_group().is_first_rank == False` → the forward takes the `else` branch
  and tries to read `intermediate_tensors` (which the proposer does not supply)
  → **wrong path / assertion**, and the embed/fc step is skipped even though the
  embed is present (brick 20).
- `get_pp_group().is_last_rank == True` → the norm/return path is fine.

> Note: an earlier analysis claimed the draft sees a size-1 group on the last
> rank (`is_first_rank==True`). That is **incorrect** — per brick 10/E2 the group
> is the global size-2 singleton, so `is_first_rank==False` on the last rank.
> The fix (a flag) is the same either way; only the rationale differs.

**Fix:** make the MTP forward behave as **first==last** when running as a
standalone draft (always: embed→fc→layer→norm→return tensor; never read/return
`IntermediateTensors`). Gate this on the draft running with `draft_pp=1` (its own
`parallel_config.pipeline_parallel_size == 1`), not on the global group. Since
`Qwen3_5MTP` is *only ever* a speculative draft, and Design C always uses
`draft_pp=1`, this branch is effectively unconditional for our path.

**Consequence for the earlier SupportsPP commit:** under Design C
(`draft_pp=1`), the `model.py` guard never fires (`pipeline_parallel_size > 1`
is False for the draft), so the `SupportsPP` declaration added in E3-step1 is
**not required for C** (it was for Design B). Harmless to keep, but the C
implementation doesn't depend on it. Revisit when locking the design.

## → implications
- **Design C feasibility: confirmed.** Input present (Q9), embed present (Q3),
  KV/rollback shared (Q10/Q11), no PP deadlock (this brick). Only the forward
  flag is needed.
- **Remaining work for C is NOT feasibility — it's correctness plumbing**
  (brick 40 / Q8): spec tokens flowing through PP's `batch_queue` pipelined
  execution (the #39704 zone). That gate is independent of draft placement.
- **New risk to weigh (memory):** C concentrates the whole draft (embed ≈
  vocab×hidden, plus its layer + lm_head) on the **last** rank's GPU, which is
  already the fuller one in prod. Assess headroom at E3; this is C's main
  downside vs B. (→ new question Q13.)

## Open questions updated
- **Q4 — CLOSED:** only the MTP forward branching matters; a standalone flag
  fixes it; no other PP dependency, no deadlock.
- Spawns **Q13** (memory headroom on the last-rank GPU under Design C).
- Points to **brick 40 / Q8** as the next (and main) remaining gate — correctness
  of spec output under PP `batch_queue`.
