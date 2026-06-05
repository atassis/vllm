# Brick 40 — Spec output under PP `batch_queue` (the real remaining work)

Status: **DONE (research)** · Answers Q8 · Cross-referenced with PR #39704 + current code.
This is the **design-independent correctness gate** — it applies to A/B/C alike.

> Under PP>1 vLLM pipelines microbatches; spec tokens and accepted tokens must
> flow correctly across that pipelined (delayed) execution. This is where the
> gibberish risk (#36872) lives, and where most of the real work remains.

---

## The delay (confirmed)

PP>1 enables `batch_queue` (engine/core.py:188-198); `step_with_batch_queue`
(core.py:484-598) schedules step N (`schedule()` → `execute_model(non_block)` →
`appendleft`) and only later `pop()`s + calls `update_from_output()` — **k =
pp_size−1 steps later**. The `SchedulerOutput` enqueued at N is a **snapshot**;
`request.spec_token_ids`, `num_output_placeholders`, `is_prefill_chunk` can
mutate in between. (Confirmed against current code.)

## How spec tokens flow today (confirmed facts)

- `ModelRunnerOutput` (outputs.py:234-281) has **no `spec_token_ids` field**;
  draft tokens travel via a separate `DraftTokenIds` (outputs.py:311) pulled by
  `take_draft_token_ids()` → `scheduler.update_draft_token_ids()`
  (scheduler.py:1716) → `request.spec_token_ids`.
- `schedule()` snapshots spec tokens into `scheduled_spec_decode_tokens`
  (scheduler.py:516-531) then clears `request.spec_token_ids`.
- `update_from_output()` does rejection accounting from that snapshot
  (scheduler.py:1393-1410): `num_rejected = num_draft − (len(generated)−1)`;
  decrements `num_computed_tokens` / `num_output_placeholders`.

> Note: PR #39704 proposed adding `spec_token_ids` to `ModelRunnerOutput` and
> reworking this. Current main uses the `DraftTokenIds`/`update_draft_token_ids`
> path instead — so #39704's diff won't apply verbatim; it's a **map of the
> hazards**, not a patch to cherry-pick.

## The gaps (LEADS — confirm at E3 / by deeper reading)

These are the areas #39704 had to touch; the agent's read of current main
suggests they are still open, but each is a **lead to verify**, not a settled
fact:

1. **Draft-token retrieval in the *normal* batch_queue path.** In
   `step_with_batch_queue`, draft tokens are pulled in the
   `deferred_scheduler_output` branch (core.py:577-589). **CORRECTED (this
   session):** the normal path is *also* covered, but by a DIFFERENT mechanism
   depending on `async_scheduling` — see "Two spec-under-PP plumbing paths"
   below. The open part is the **timing** under batch_queue, not "no-op", and it
   diverges by config. (Still confirm end-to-end at layer-2/E3.)
2. **Stale-snapshot validation.** `update_from_output()` trusts the snapshot's
   spec count; no check that `is_prefill_chunk`/placeholder state still matches k
   steps later. #39704 added a `new_token_ids`-vs-`is_prefill_chunk` guard for
   exactly this ("stale spec_token_ids on prefill chunks in batch_queue mode").
   **VERIFIED (this session) — does NOT reproduce as a scheduler-level bug on
   current main; see "Verified: scheduler accounting" below.**
3. **Non-last-rank accepted-draft accounting.** On non-last ranks
   `output_token_ids` excludes accepted draft tokens (gpu_model_runner.py:1332-
   1357, comment: "doesn't include 'unverified' tokens like spec tokens"), so
   `token_ids_cpu` positions can drift. #39704 added a `token_ids_cpu` fix-up.
   **CORRECTED (this session) — current main ALREADY handles this**, see
   "Current main already plumbs async+PP+spec" below. The #39704-era "unhandled"
   read is stale.

## Two spec-under-PP plumbing paths (CORRECTED — this session)

`step_fn` is chosen once at init (`core.py:217`): `step` if no batch_queue, else
`step_with_batch_queue`. PP>1 enables batch_queue **regardless of async** —
`max_concurrent_batches` (`vllm/config/vllm.py:497`) returns `pp_size` even with
`async_scheduling=False` (the final `return pp_size`). The main loop calls
`step_fn()` **then `post_step()`** every iteration (`core.py:1261-1266`). So the
spec-token plumbing splits in two by config:

- **PP>1, NOT async (the prod default — `vllm serve … --pp 2`, no async flag):**
  batch_queue is active, and `post_step` (`core.py:474-482`) pulls the drafter
  output via `take_draft_token_ids()` → `scheduler.update_draft_token_ids()`
  (the **sync** path) — but **only because it is gated on `not async_scheduling`**.
  The hazard is **timing**: under batch_queue, `schedule()` for the next step can
  run before `post_step` injects the drafter tokens for the just-popped step
  (lead #1 in its real form — a draft-token *latency*, not a no-op).
- **PP>1 AND async (`--async-scheduling`):** `post_step` early-returns (its
  `not self.async_scheduling` guard), and the **worker** injects draft tokens
  directly into the input batch — `update_async_spec_token_ids`
  (`gpu_model_runner.py:3546`) — while the `AsyncScheduler` reserves spec slots
  optimistically with `[-1]*num_spec` placeholders
  (`async_scheduler.py:_update_after_schedule`, :33-36). The scheduler never sees
  real draft *values*; its accounting is purely count-based.

**Critical caveat for our V1-runner target:** `vllm/config/vllm.py:504` comment —
*"V1 Model Runner does not fully support async scheduling with PP"* — and
`max_concurrent_batches` does NOT give async its `pp_size+1` (that's V2 only).
Qwen3.5 uses the **V1 runner** (spike-plan §3). So the realistic path to validate
is **PP>1 sync** (prod default), where lead #1's draft-token *timing* under
batch_queue is the live question. Async-at-pp=1 is still a useful local proxy for
the AsyncScheduler accounting, but it exercises a *different* spec plumbing than
prod PP-sync. (→ revisit whether prod should run `--async-scheduling` at all.)

## Verified: scheduler accounting under 2-deep pipelining (this session, local CPU)

Local unit tests (`tests/v1/core/test_pp_spec_batch_queue.py`) drive the
`AsyncScheduler` through engine-core's `step_with_batch_queue` ordering with a
faithful synthetic worker (prompt-aware: incomplete prefill chunks emit no token;
decode steps emit `[accepted…, bonus]`), asserting the request stops at **exactly
`max_tokens`** (`ignore_eos`). The driver also asserts the in-flight queue truly
reached depth ≥2 (a real schedule-ahead, not lockstep — confirmed via an
instrumented trace showing the boundary step truncating +2→+1 and discarding the
last in-flight batch).

**Result — the invariant holds across the whole grid:** `num_spec ∈ {1,2,3}` ×
`accept ∈ {0…num_spec}` × `max_tokens ∈ {1,2,3,5,8}`, plus **chunked prefill
spanning 3 in-flight batches** and **a stop-token fired mid-pipeline**. No
over/under-generation. → **Lead #2 (stale-snapshot accounting) does NOT reproduce
as a scheduler-level bug on current main.** The AsyncScheduler's
placeholder/`num_output_placeholders` bookkeeping already handles the k-step
delay correctly for the stop boundary.

**Scope of this result (calibration):** it validates **count** accounting (no
over/under-generation), which is the heart of lead #2. It does **not** validate
token-**value** correctness (greedy ≡ non-spec) — the scheduler never sees token
values; that needs a real model (E3). And it tests the **async** plumbing path,
not the prod **PP-sync** path (lead #1 timing). → The remaining correctness risk
is narrowed to **engine-core lead #1** (draft-token retrieval *timing* under
batch_queue) and **model-runner lead #3** (non-last-rank `token_ids_cpu` drift) —
**not** the scheduler.

## Current main already plumbs async+PP+spec (CLAIM PARTLY WRONG — see session-3 note)

> ⚠️ **Session-3 correction (the real E3 run disproved this section's optimism).**
> The code paths below EXIST, but they were **never actually run** for MTP+PP+async
> and are **buggy when exercised**. Session 3 found a cascade: 5 non-last-rank
> `self.drafter` AttributeErrors (fixed), and — directly contradicting "lead #3 is
> handled" — `_pp_broadcast_prev_sampled_token_ids` (`gpu_model_runner.py:4653`)
> asserts `sampled_token_ids` shape `[num_reqs, 1]` while spec emits `[num_reqs,
> num_spec+1]`. So **lead #3 is NOT handled**; "code exists" ≠ "works". Full record:
> `../../archive/2026-06-04-e3-execution-log.md`. Read the below as "the intended
> design that still has real bugs", not "done".

The brick-40 leads were read from PR #39704 (a DeepSeek-MTP+PP patch). Since then,
current main has landed method-agnostic async+PP+spec plumbing *scaffolding* in the
V1 runner — but per the session-3 note above it is unfinished/buggy:

- `use_async_spec_decode = use_async_scheduling and num_spec_tokens > 0`
  (`gpu_model_runner.py:628`) — **method-agnostic** (MTP/eagle/ngram_gpu alike);
  the `is_ngram_gpu` branches are just extra ngram optimizations within it.
- **Non-last-rank accounting EXISTS** (`gpu_model_runner.py:1332-1340`): under
  async-scheduled PP, sampled tokens are propagated by **GPU broadcast**
  (`new_token_ids == []` path); under non-async PP, the scheduler ships them back.
  Both branches are present — lead #3 is handled, not missing.
- **Accepted-draft drift is corrected** via `prev_num_draft_len` +
  optimistic-extend-then-correct (`gpu_model_runner.py:1287-1328`,
  `update_scheduler_for_invalid_drafts` :1270) and the last-rank realignment
  (:1353-1363). This is exactly the `token_ids_cpu` fix-up #39704 was reaching for.

**Runner reality (decisive for our config):** async auto-enables for MTP+PP
(`vllm/config/vllm.py:969-997`: method `mtp ∈ EagleModelTypes`, MultiprocExecutor
`supports_async_scheduling()==True`). The **V2 runner** (#42187, "avoid PP
bubbles", 2026-06-02) is the *performance* path for async+PP — but it is
**unavailable to us**: `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` = {Llama, Mistral,
Qwen3ForCausalLM} only, and `_is_default_v2_model_runner_model` also requires
`not is_moe and not is_quantized` — our checkpoints are `Qwen3_5…` **and AWQ/GPTQ
quantized** (one MoE). So we are firmly on the **V1 runner**, whose async+PP is
"not fully supported" only in the **bubble/perf** sense (the #42187 comment),
while being **functionally plumbed** for spec per the points above.

**→ Re-scoping the remaining work (calibrated):** the bulk is NOT re-implementing
#39704's plumbing — most of it is already on main. What actually remains for
Qwen3.5 PP+MTP is: **(1)** the Design-C draft-forward flag (the one model change),
and **(2)** an E3 greedy-equivalence run to confirm the existing async+PP+spec
plumbing is correct for Qwen3.5 MTP end-to-end (perf bubbles aside). Confidence
that this is a *small, mostly-existing-infra* task — not a core-weeks plumbing
rewrite — rises materially. **Caveat:** "the code exists" ≠ "it's correct for our
model"; (2) is still gated on the window. Verify the GPU-broadcast non-last-rank
path actually carries MTP accepted-drafts at E3.

## Session 5 — B1a DONE, break #2 mapped on MiMo (2026-06-05)

**Vehicle:** MiMo-7B PP=2+MTP on gpu-wb (loads ~40s; MiMoMTP already standalone).
**Q15 CLOSED empirically:** MiMo runs PP=2+MTP under `draft_pp=1` with no `SupportsPP`
— the `model.py:1199` guard short-circuits on `pp>1==False` (log reached "Loading
drafter model..."). Engine constructs, KV fits (gpu0 12.5/gpu1 14.4 GiB).

**B1a — broadcast transport width: FIXED + unit-tested + validated past `:4653`.**
- New CUDA-free module `vllm/v1/worker/pp_spec_broadcast.py`:
  `broadcast_sampled_token_ids` (sender, drops the `[num_reqs,1]` assert),
  `receive_sampled_token_ids` (receiver allocs `[num_reqs, num_spec+1]`),
  `count_valid_sampled_tokens_per_req` (`(t != -1).sum(dim=1)`; B1c building block,
  not yet wired).
- `tests/v1/spec_decode/test_pp_spec_broadcast.py` — 3 green incl. a **2-rank gloo
  CPU** variable-width broadcast round-trip (the A3.5 local-proxy TDD).
- Wired into `gpu_model_runner._pp_broadcast_prev_sampled_token_ids` (sender) and
  `_pp_receive_prev_sampled_token_ids_to_input_batch` (receiver width = `num_spec+1`).
  **MiMo now gets PAST `:4653`** → break moved downstream. ✅
- **Calibration (TDD minimalism):** I first also changed the receiver *accounting*
  (advance `num_tokens_no_spec` by per-req valid count + extend `output_token_ids`
  by N). That crashed at break #2 — but break #2 **also reproduces with the ORIGINAL
  `+1` accounting** + the width fix alone. So the accounting change was speculative
  and unnecessary for B1a; **reverted to width-only** (minimal, evidence-forced).

**Break #2 (the next B1c gate, MAPPED — independent of accounting):**
- Root: **CUDA device-side assert `indexSelectSmallIndex: srcIndex < srcSelectDimSize`
  on rank0 (the NON-last rank) in the model forward** — an embedding index OOB (a
  `-1`/stale index reaching a vocab lookup). Fires first (log line 67, first forward
  step), under `CUDA_LAUNCH_BLOCKING=1`.
- Downstream **`KeyError` at `scheduler.py:1388`** (`update_from_output`:
  `model_runner_output.req_id_to_index[req_id]`) — the req is in the scheduler's
  running set but absent from the crashed worker's (malformed) output. **Fallout of
  the worker crash, not an independent scheduler bug** (req is not finished, :1378).
- **Root cause PINNED (env-gated `input_ids` instrumentation, `VLLM_PP_SPEC_DEBUG`):**
  on an *irregular* decode step the request goes **non-common** (`prev_positions=[-1]`,
  `tot_spec=0` — no drafts scheduled), so `_prepare_input_ids` (`:1708`) takes the
  early-return path (`:1784`) and feeds **`input_ids_cpu`** instead of overwriting
  position-0 from `prev_sampled_token_ids[:,0]`. `input_ids_cpu` holds a **`-1`
  placeholder** the non-last-rank receiver wrote (the receiver stores `-1` as a length
  marker, relying on the *common* path to backfill the real value each step). On the
  non-common step the `-1` is embedded → OOB. (Confirmed `bad_val=-1`, `prev_sampled`
  col0 was a *valid* token — so the `-1` is from the local buffer, not the broadcast.)
- **B1b TESTED + FAILED (don't re-walk):** extending the `is_ngram_gpu` gates at
  `:1330`/`:1490` to MTP (advance/correct `num_tokens_no_spec`) did **not** fix it — it
  *moved* the `-1` to a different position and made `num_tokens_no_spec` **over-advance**
  (ntns ran ahead of num_computed: 14 vs 11) because it double-counts with the
  receiver's `+1`. So B1b as a standalone edit is wrong; the non-last-rank token
  accounting (receiver `-1` placeholder + the `:1314` optimistic-extend + B1b) must be
  reconciled **holistically**, not gate-by-gate. Reverted to the B1a baseline.
- **Real fix direction (F2, not yet done):** the non-last rank must hold **real**
  sampled-token values in its local token buffer (`token_ids_cpu`/`output_token_ids`)
  rather than `-1` placeholders — the values are in the broadcast
  (`prev_sampled_token_ids`, `count_valid_sampled_tokens_per_req` already built) — so
  the non-common fallback path can never embed a placeholder. Subtlety: receive-step
  vs consume-step position aliasing under the pipeline, and which `prev_sampled` column
  is the request's latest token. **Needs holistic non-last-rank accounting work.**
- **F2 (receiver writes real `recv[:,0]` instead of `-1`) — PARTIAL:** fixed the
  normal-step position 0 (`input_ids_cpu0` went `[-1,..]`→`[12095,..]`), but the
  crash persists — the non-common step reads a DIFFERENT `token_ids_cpu` position
  whose `-1` comes from another source (the optimistic-extend `:1314` and/or
  confirmed positions never backfilled with real values). Writing ONE column at the
  receiver's single `pos` is insufficient; a correct fix must backfill ALL confirmed
  positions with real values — holistic non-last-rank accounting.
- **SYNC path TESTED (`async_scheduling=False`) — DEADLOCKS.** Forcing sync (the
  prod-default-without-spec path; `vllm/config/vllm.py:957-997` makes async opt-out)
  was the candidate "better way" (sidesteps broadcast/`-1`/non-common entirely). But
  MiMo PP=2+MTP **hangs**: engine constructs, then GPU util 0% + repeating
  `shm_broadcast.py:705 "No available shared memory broadcast block found in 60s"` —
  a PP-rank deadlock (a rank waiting on spec output that never arrives; brick-40
  lead #1 draft-token timing under batch_queue). So **sync is NOT a free win**.

**CONCLUSION (the real picture): MTP+PP+spec on the V1 runner is unfinished in BOTH
modes** — async crashes (break #2, non-last-rank `-1` leak), sync deadlocks. This is
genuine multi-bug engine work, not a one/two-bug fix. → strategic decision point
(blocker): how deep to invest, and STRONGLY consider reading **PR #39704** (DeepSeek
MTP+PP) as the reference map of the full fix set, per AGENTS.md (coordinate, don't
reinvent). B1a + A1c remain valid standalone upstream deliverables regardless.

## Session 6 — Q16 RESOLVED (NEGATIVE): C3+B1a do NOT close break #2 (2026-06-05)

**Run:** MiMo-7B PP=2+MTP, async, gpu-wb, `CUDA_LAUNCH_BLOCKING=1`, no QUANT_BITS/offload.
Integrated working set = **B1a (broadcast width) + C3 (scheduler placeholder discipline,
≈#40768)**, both verified present on the remote tree before the run. Engine constructs
(gpu0 12.50 / gpu1 14.42 GiB, KV 72,368 tokens), reaches `generate`.

**Result: break #2 STILL fires on the FIRST forward step, on rank0 (the NON-last rank).**
Exact crash frame (full Python traceback captured, not just the device-assert spam):
```
gpu_model_runner.py:4286 execute_model → :3747 _model_forward → self.model(...)
qwen2.py:550 (MiMoModel.forward) → mimo.py:73  hidden_states = self.embed_input_ids(input_ids)
qwen2.py:387 embed_input_ids → self.embed_tokens(input_ids)
vocab_parallel_embedding.py:491 forward → :78  F.embedding(input_, layer.weight)
→ indexSelectSmallIndex: srcIndex < srcSelectDimSize  (input_ids carries a -1 / OOB id)
```
So the OOB index is in the non-last rank's **`input_ids`** fed to `embed_tokens` — exactly
the F2/C4 target. (TP=1 → `VocabParallelEmbedding.forward` does no masking, so a `-1` in
`input_ids` reaches `F.embedding` raw → OOB.)

**Calibration confirmed (the session-5/Q16 tension is resolved empirically):** C3 fixes the
**scheduler-side** `-1` source (AsyncScheduler no longer emits `[-1]*k` for a request absent
from `prev_step_scheduled_req_ids`). But break #2's proximate `-1` is **worker-side** — the
non-last-rank receiver's local `input_ids_cpu`/`token_ids_cpu` buffer, read on the non-common
early-return path of `_prepare_input_ids` (`:1708`/`:1784`). C3's scheduler discipline does
**not** touch that buffer, so it cannot close break #2 alone. C3 stays a valid, green,
standalone hardening (≈#40768) — necessary-but-not-sufficient here, not the fix.

**→ Q16 = NO.** The next gate is unchanged from the session-5 map: **F2/C4 — holistic
non-last-rank input reconstruction.** The non-last rank must hold the **real** sampled-token
value in its local buffer (available in the broadcast `prev_sampled_token_ids`,
`count_valid_sampled_tokens_per_req` already built) for **all** confirmed positions, so the
non-common fallback can never embed a `-1`. Prior F2 attempt (session 5) was PARTIAL —
writing one column at the receiver's single `pos` is insufficient; must backfill all
confirmed positions. This is the real worker-side accounting work (W3/W5, brick 80 §3, Q17).
GPU freed after the run (gpu-wb clean).

### F2/C4 — grounded diagnosis (session 6, code-read, every site pinned)

The `-1` that reaches `embed_tokens` on the non-last rank has **two producer sites** and a
**missing back-write**, all in `gpu_model_runner.py`:

1. **Optimistic extend** (`_update_states`, `:1318`): `req_state.output_token_ids.extend(
   [-1] * optimistic_num_accepted)` — method-agnostic; reserves slots for the drafts the
   last rank *might* accept. `num_tokens_no_spec` advance for these is `is_ngram_gpu`-gated
   (`:1334`) → **NOT applied for MTP** (the Q17/W5 asymmetry).
2. **Receiver placeholder** (`_pp_receive_prev_sampled_token_ids_to_input_batch`, `:4694-4698`):
   `req_state.output_token_ids.append(-1)`, then `is_token_ids[i, pos] = True` and
   `num_tokens_no_spec[i] = pos + 1` — it advances the count and flags the slot **but never
   writes the real token value** into the persistent `token_ids_cpu[i, pos]`. The real value
   is sitting in `recv` = `self.input_batch.prev_sampled_token_ids` (`[num_reqs, num_spec+1]`,
   B1a width).
3. **The only place the real value is applied** is `_prepare_input_ids` (`:1707`), which
   scatters `prev_sampled_token_ids[:, 0]` into the **transient GPU `input_ids`** — and only
   on the **common** path (`prev_index >= 0`). On non-last ranks `_draft_token_ids is None`
   (`:1813` early-return) so columns `1..k` are never placed; and the write goes to the GPU
   buffer, **never back into the persistent CPU `token_ids_cpu`**.

**→ break #2 mechanism, exactly:** when a request goes **non-common** (`prev_index < 0` — not
in this step's `prev_req_id_to_index`, e.g. not scheduled in the immediately-previous step or
moved by `condense()`), `_prepare_input_ids` hits `num_common_tokens == 0` (`:1783`) and uses
`input_ids_cpu` as-is. That slot (`token_ids_cpu[i, pos]`) was count-advanced by the receiver
(site 2) but **never value-written**, so it holds `-1` → `F.embedding(-1)` → `indexSelectSmallIndex`
OOB. (Confirmed at runtime: rank0, first forward, `mimo.py:73` → `vocab_parallel_embedding.py:491`.)

**The fix (C4) — two coupled pieces, both worker-side, no scheduler change:**
- **(A) Value back-write:** in the receiver (`:4694` loop), write the request's real confirmed
  token(s) from `recv` into the persistent `token_ids_cpu[i, pos..]` (not just flag + count).
  Then the non-common fallback can never read a `-1`. The "next input" token for a request that
  advanced by `v = count_valid_sampled_tokens_per_req(recv)[i]` positions is the LAST valid
  column `recv[i, v-1]` (reject `v=1` → `recv[i,0]`; accept+bonus `v=2` → `recv[i,1]`).
- **(B) Count reconciliation (Q17):** the per-request advance must equal `v`, reconciled with
  site-1's optimistic `[-1]*optimistic_num_accepted` + site-2's `+1`, so `num_tokens_no_spec`
  and `output_token_ids` neither double-count nor under-count. (Session-5 B1b failed precisely
  because it advanced `num_tokens_no_spec` *in addition to* the receiver's `+1` → over-advance
  14-vs-11. So (B) must be done **at one site**, not added on top.)

**TDD plan (red→green, not blind):**
- **C4-unit (CPU, no GPU):** extract a pure helper, e.g.
  `reconstruct_non_last_rank_tokens(recv, valid_counts, prev_positions, ...) ->
  (writes: list[(req_i, pos, value)], advances: list[(req_i, delta)])`. Unit-test over the
  grid: `num_spec=1 × accept∈{0,1}` (reject→v=1, accept→v=2), **non-common** (`prev_index=-1`)
  and **re-added** requests, asserting (i) every written value is a real token (never `-1`),
  (ii) `sum(advances) == sum(valid_counts)`, (iii) the chosen "next input" column is `v-1`.
  Mirrors the `pp_spec_broadcast.py` micro-component style (brick 81 C1/C4).
- **C4-integration:** wire the helper into the receiver + delete the now-redundant bare `-1`
  append at `:4695`; re-run MiMo PP=2+MTP async on gpu-wb → break #2 gone → greedy-equiv.

This keeps C4 a small typed/tested worker-side component (brick 81 §4), composable with C3
(scheduler discipline) which stays green/standalone.

### F2/C4 — full accounting trace (session 6, deeper read before coding)

Traced the whole non-last-rank async-PP accounting machinery to remove blind spots before any
fix. Call order within one step on the **non-last** rank:
`execute_model:4052 _update_states` (start) → forward (→ IntermediateTensors) →
`sample_tokens:4392` early-returns (no `execute_model_state`) → `:4397`
`_pp_receive_prev_sampled_token_ids_to_input_batch`. The deferred correction returned by
`_update_states` is `correct_spec_decode_token_counts` (`:1470-1498`).

**The decisive structural finding — NO value back-write site exists for non-last async-PP:**
- `_update_states` branch-1 (`:1342-1362`): touches only `req_state.output_token_ids`, and for
  async PP `new_token_ids == []` → it appends nothing.
- `_update_states` branch-2 (`:1410-1433`): advances `num_tokens_no_spec` to `num_computed_tokens`
  and flags `is_token_ids = True`, but the `token_ids_cpu` write is gated `if new_token_ids:`
  (`:1423`) → **False for async PP → values NOT written**.
- receiver (`:4694-4698`): `output_token_ids.append(-1)`, `is_token_ids[i,pos]=True`,
  `num_tokens_no_spec[i]=pos+1` → flag + count, **no value**.
- `_prepare_input_ids` (`:1707`): writes the **GPU `input_ids`** (common path, col0 only), never
  the persistent `token_ids_cpu`.
→ So under async PP the real sampled-token value is **never persisted** into the non-last-rank
`token_ids_cpu`; it lives only transiently in GPU `input_ids` (common path) and one step in the
broadcast `recv`. **ngram_gpu populates/accounts via its `is_ngram_gpu`-gated lines (`:1335`
advance, `:1494` correct) + `update_ngram_gpu_tensors_incremental` (`:1458`); MTP has no
equivalent** → break #2 is the first symptom of that missing MTP value-population. (Likely the
non-last-rank receiver path is effectively untested for *every* method — local greedy-equiv was
pp=1, which has no broadcast.)

**Count machinery (the (B) part), now mapped:** `num_computed_tokens` is corrected
**method-agnostically** (`:1487` `req_state.num_computed_tokens -= correction`,
`correction = optimistic_num_accepted - (valid_count - 1)`); `num_tokens_no_spec` correction is
**ngram-gated** (`:1494`). For MTP, branch-2 (`:1418`, method-agnostic) re-derives
`num_tokens_no_spec` from the corrected `num_computed_tokens` each step — so the count MAY already
be coherent for MTP via branch-2 + the num_computed correction, making (B) possibly unnecessary.
**This is the one thing pure reading can't pin with confidence** (the exact `num_tokens_no_spec`
trajectory across the k-step delay, branch-2's `+to-num_computed` vs receiver's `+1`).

**→ Decision before coding: ONE targeted instrumentation run on MiMo** (env-gated, like
session-5's `VLLM_PP_SPEC_DEBUG`): per non-last-rank step, log per req `{prev_index,
num_computed_tokens, num_tokens_no_spec (pre/post branch-2 and pre/post receiver), pos, recv row,
valid_count}`. That turns the (A)-only-vs-(A+B) question into data: confirm (A) value-back-write
is the fix and observe whether the count already self-corrects (→ (A) alone) or drifts (→ which
single site does (B)). Then implement fully grounded. (`_update_states_after_model_execute` `:1513`
is hybrid-only → fires for 27B, not MiMo — so MiMo isolates the non-hybrid accounting cleanly.)

## Session 7 — de-risk: the V2 runner is the C4 BLUEPRINT, not a moot-maker (2026-06-05)

Before sinking sessions into C4, checked two cheap things (read-only). Both verified by code-read.

**(1) C4 is NOT obsoleted by the V2 runner.** Quantized models are hard-gated to V1:
`_is_default_v2_model_runner_model` ends `return not model_config.is_moe and not
model_config.is_quantized` (`vllm/config/vllm.py:558`); the V2 allowlist is
`{LlamaForCausalLM, MistralForCausalLM, Qwen3ForCausalLM}` (`vllm/config/vllm.py:69-75`).
Our target is **Qwen3.5 AWQ/GPTQ → permanently V1.** So fixing V1's non-last-rank
reconstruction is the only path for our model; not throwaway.

**(2) But V2 already solves this problem the RIGHT way → it's the blueprint for C4.** The V2
runner (`vllm/v1/worker/gpu/model_runner.py`, selected in `gpu_worker.py:317-322`) uses a
`PPHandler` (`vllm/v1/worker/gpu/pp_utils.py`) that, on the non-last rank, broadcasts **two**
tensors: `sampled_tokens` (real int64 values) **and** `combined = [num_sampled, num_rejected]`
per req (`pp_utils.py:receive`, ~:145-160) — **counts travel SEPARATELY from values; no `-1`
is ever packed into the token grid.** The deferred `PendingRecv` is consumed k steps later; the
`post_update` Triton kernel (`vllm/v1/worker/gpu/input_batch.py:~467-472`) skips rows whose
`req_state_idx < 0` and writes exactly `num_sampled` **real** token values into `all_token_ids`.
→ This is precisely C4 done right: **(A) write real values + (B) advance by an explicit per-req
count carried alongside, never inferred from `-1` positions.** It answers the session-6
instrumentation question — (A)-only vs (A)+(B) — **by reference: BOTH.** The remaining V1-specific
risk is integration (V1's receiver `+1`, branch-2 re-derive, and the `:1318` optimistic-extend
are different plumbing), so MiMo stays the integration oracle; but the *design* is no longer a
guess. **Plan refinement:** model C4's receiver on the V2 PPHandler pattern (real values +
separate counts + skip-negative), TDD the pure helper, then MiMo greedy-equiv. The diagnostic
instrumentation run becomes *optional confirmation* rather than the primary path.

**(3) #40768 is scheduler-only → C4 is ours, no upstream collision.** `gh pr view 40768`:
OPEN, updated 2026-06-03, changed files = `{test_async_scheduler.py, utils.py,
async_scheduler.py, scheduler.py, request.py}` — **exactly our C3's 5 files; it does NOT touch
`gpu_model_runner.py`/worker.** So upstream is fixing the scheduler-side `-1`, not the worker-side
reconstruction. **Implication:** C3 overlaps #40768 directly → coordinate (help land it / build on
it, don't open a competing PR — AGENTS.md); C4 (worker-side) has no upstream duplicate → our
distinct contribution.

### C4 (A) implemented + MiMo run → root cause CONFIRMED (the `-1` is the optimistic-extend)

**C4 (A) value-back-write** shipped via TDD: new `select_latest_sampled_token_per_req` helper
(`pp_spec_broadcast.py`, red→green, 3 unit tests) + the receiver
(`_pp_receive_prev_sampled_token_ids_to_input_batch`) writes the real `recv[i, v-1]` into
`req_state.output_token_ids` **and** `token_ids_cpu[i, pos]` instead of `-1` (chunked-prefill
branch preserved). Local: ruff clean, 20 tests green.

**MiMo PP=2+MTP async run (s7, gpu-wb, `CUDA_LAUNCH_BLOCKING=1`): break #2 STILL fires** — exactly
as brick-40 §F2 (s5) predicted for a single-position write. Evidence: **Worker_PP0** (the non-last
rank) asserts `indexSelectSmallIndex` in its forward; **Worker_PP1** (last) merely blocks in
`gpu_worker.py:831 get_pp_group().irecv_tensor_dict` and reports gloo "Connection closed by peer"
(it was waiting for PP0's IntermediateTensors); the `scheduler.py:1431` KeyError is downstream
fallout. So PP0 crashes embedding a `-1` during the forward, which runs **before** that step's
`_pp_receive…` — i.e. the `-1` was placed by an **earlier** site, not this step's receiver.

**Root cause CONFIRMED (static read, Phase-1 complete):** the embedded `-1` originates from the
**optimistic-extend** — `_update_states` `gpu_model_runner.py:1318-1319`:
`optimistic_num_accepted = req_state.prev_num_draft_len; output_token_ids.extend([-1] *
optimistic_num_accepted)` (optimistically assumes all drafts accepted; a deferred
`correct_spec_decode_token_counts` fixes the count after the forward). Those placeholder `-1`s land
in `output_token_ids` → copied into `token_ids_cpu` (the `gpu_input_batch.py:372` idiom) at the
speculative positions → read on the **non-common** path of `_prepare_input_ids` → embedded → OOB.
The receiver's single-`pos` write (my C4 (A)) doesn't cover those `optimistic_num_accepted`
positions. Note the MTP asymmetry: `:1335` advances `num_tokens_no_spec` by `optimistic_num_accepted`
**only for `is_ngram_gpu`**, not MTP (Q17).

**→ Holistic C4 (next):** backfill **all v** confirmed positions with the real broadcast values
`recv[i, 0:v]` (accepted drafts + bonus), aligned with the positions the optimistic-extend touched,
+ reconcile the count at one site (s5 B1b over-advanced 14-vs-11 by double-counting — so the count
(B) is the part that still needs runtime confirmation; instrument `{num_computed_tokens,
num_tokens_no_spec, output_token_ids tail, embedded input_ids positions/values, optimistic_num_accepted,
recv, v}` on PP0, or implement the value backfill and use the run as the count oracle). C4 (A) stays
(it correctly writes the receiver's pos); it's necessary-but-insufficient on its own. (Mirrors V2's
`PPHandler`: write `num_sampled` real values, never a `-1`.)

### Instrumentation run (s7) — the position trajectory, DATA (then reverted)

Re-added a throwaway env-gated `VLLM_PP_SPEC_DEBUG` block in `execute_model` (logged, on the
embedding/first rank before the forward: `input_ids`, neg positions, and per-req
`{num_computed_tokens nc, num_tokens_no_spec ntns, prev_num_draft_len, token_ids_cpu window,
output_token_ids tail}`), ran MiMo, **reverted it** (working set stays B1a+C3+C4(A)+tests).
(First attempt died on a `NameError: os` — the env-gate used `os` which isn't imported in
gpu_model_runner; fixed with a local `import os as _os`. Lesson: guard even the gate in try, or
import locally.)

**Crash step (PP0):** `input_ids=[-1, 0]` (neg at pos 0) with `nc=13, ntns=13, pndl=1,
token_ids_cpu[11:16]=[-1,0,-1,0,0], output tail=[-1,315,-1,-1]`. Three grounded conclusions:

1. **The `-1` is the optimistic-extend placeholder** (`:1319`), confirmed: `output_token_ids`
   tails consistently alternate real tokens and `-1` (`[315,-1,576,-1]`, `[-1,315,-1,-1]`) — the
   draft-slot placeholders are **never backfilled** with the real accepted value.
2. **The non-common read indexes `token_ids_cpu[num_computed_tokens]`, but C4(A) writes at
   `token_ids_cpu[num_tokens_no_spec]`.** On the crash step `nc==ntns==13` yet `tic[13]==-1`,
   because the value at the read position was set (to `-1`) by a *previous* step's optimistic
   extend and no step wrote the real value at `nc`. (On *common* steps `input_ids[0]` comes from
   the `prev_sampled` GPU overlay, so they don't crash — only non-common steps read raw `tic`.)
3. **num_spec=1 → each step adds 2 slots** (bonus + optimistic draft); the bonus slot gets a real
   value, the draft slot keeps `-1`. The non-common read lands on a draft `-1`.

**→ Holistic C4 (grounded design):** in the receiver, backfill the real broadcast values
`recv[i, 0:v]` into the `v` confirmed `token_ids_cpu` positions **indexed by num_computed_tokens**
(the request's true length), not the single `num_tokens_no_spec` slot — covering exactly the
positions the optimistic-extend tentatively filled with `-1`. Reconcile with the deferred
`correct_spec_decode_token_counts` (the (B) part: when drafts are rejected the optimistic count
shrinks — those slots must not be read). This is the V2 `PPHandler`/`post_update` shape ported to
V1's CPU buffers. (B1b over-advanced by double-counting; the fix must set the count at ONE site,
consistent with the value writes.)

## Session 8 — Q18 RESOLVED: V2 is the "right" arch but DEADLOCKS on our config → stay V1 (2026-06-05)

The s7 reframe (Q18) asked whether to pivot to the V2 runner instead of finishing C4. s7's
*verdict* rested on "27B quant-locked to V1." This session **overturned that premise statically,
then killed the pivot empirically.**

**Static read (all by code, the cheap ~1h investigation):**
1. **(a) V2 spec supports MTP — YES, explicitly.** `qwen3_5_mtp` ∈ `MTPModelTypes`
   (`config/speculative.py:46`) → normalized to `method="mtp"` (`:565-569`); `method=="mtp"` →
   `MTPSpeculator` (`gpu/spec_decode/__init__.py:17-20`, loads draft via `load_eagle_model`); the
   V2 hard-gate **explicitly allows** `"mtp"` (`config/vllm.py:2011`) and — decisively — bars
   `eagle3+PP>1` (`:2018-2022`) but has **NO `mtp`+PP>1 exclusion**. Our exact config is on the
   intended-supported list. Bonus: V2 returns `max_concurrent_batches = pp_size+1` for async
   (`:502-503`), i.e. it's *built* for async+PP, unlike V1 (`:504` "does not fully support").
2. **(b) The quant gate is "not validated", NOT "not implemented" — liftable.** Quant exclusion
   lives ONLY in `_is_default_v2_model_runner_model` (`config/vllm.py:558`
   `not is_moe and not is_quantized`) which governs the **default auto-select**. It is **absent**
   from the hard list `_get_v2_model_runner_unsupported_features` (`:1982-2060`). `use_v2_model_runner`
   (`:519-522`): if `VLLM_USE_V2_MODEL_RUNNER` is set it returns that **directly**, bypassing the
   default gate; `_validate_v2_model_runner` then checks only the hard list (no quant). No
   quant-specific incompatibility inside the V2 runner (quant is layer-level). git-blame frames it:
   gate introduced as **"[1/N] Oracle for model runner v2 — qwen3 dense by default"** (#39337),
   allowlist later widened to Llama/Mistral (#43458) → a **phased validate-one-arch rollout**, not
   a capability gap.
3. **(c) Arch gate soft, but the model class is the real unknown.** The allowlist
   (`{Llama, Mistral, Qwen3}`, `config/vllm.py:69-75`) is also default-only, bypassable. No
   `SupportsV2` protocol gates models. BUT our target is `Qwen3_5ForConditionalGeneration` /
   `Qwen3_5MoeForConditionalGeneration` (registry `:565-568`) — a conditional-generation/MM arch,
   richer than the allowlisted `Qwen3ForCausalLM`; whether it *executes* under V2 is unverified.

**Empirical kill-shot (runs.md s8 row; clean-for-MiMo tree, B1a/C3/C4 stashed; A1c/standalone-flag
are Qwen3.5-only → inert for MiMo):** forced `VLLM_USE_V2_MODEL_RUNNER=1`, MiMo-7B PP=2 + MTP async
on gpu-wb. **`Using V2 Model Runner` confirmed**; both ranks load (PP0 7.11 / PP1 8.66 GiB), KV
cache sized (72,448 tok), engine-core inits — then **DEADLOCKS during construction**: `[OK]`
(construction-complete marker in `e3_run.py`) **never prints**; EngineCore loops
`shm_broadcast.py:705 No available shared memory broadcast block` 5×60s (12:05→12:11); **Worker_PP1
in `futex_wait_queue`**, PP0 wchan=0; killed (`exit=137`, "Worker proc died unexpectedly"). **No
`indexSelectSmallIndex`, no traceback** — a *hang*, not V1's OOB crash. Signature = a PP collective
step-sync mismatch at warmup/first-step: the V2 analog of break#2, manifesting as deadlock instead
of an out-of-bounds embedding. (gdb `py-bt`/native `bt` yielded nothing — no debug syms; py-spy not
installed.)

**Verdict → stay on V1, finish C4.** V2 is architecturally the *right* home for MTP+PP+async and the
quant gate is liftable — but it does **not** run our config out of the box; it trades V1's
*pinned-root-cause, design-grounded* break#2 for a **fresh, unattributed V2 deadlock**. For the
**speedup** goal, C4-on-V1 is strictly shorter. V2 remains the **longer-lived contribution** track
([[CMP]] north-star / a future RFC), to be reopened only deliberately. **Honest caveat:** this is one
run with our specific knobs (enforce_eager, async auto-on, `num_spec=1`, MiMo `mimo_mtp`); the
deadlock *could* be a config artifact (async batch_queue pipeline-fill, or the "unbatched P2P op /
lazy-NCCL-init" interaction flagged in the log) rather than a deep V2 limitation — but pinning that
is itself a multi-session effort comparable to C4, so it does not change the near-term fork.

### C4 holistic implemented — (A) value/position DONE + hw-confirmed; (B) count is the last gate

After the V2 fork was settled (stay V1), implemented the holistic C4 and ground-truthed it with a
multi-step PPDBG probe (env-gated, remove-before-PR) on MiMo. Three runs (runs.md s8-traj/holA/holB):

**Ground truth (s8-traj, still-single-slot C4(A) + probe):** the read in `_prepare_inputs` gathers
`token_ids_cpu` at `positions = num_computed_tokens_cpu[req] + query_pos` (`:1913-1944`), and
**`num_computed_tokens` grows by EXACTLY the previous step's valid count `v`** (measured Δnct =
+2,+2,+1,+1,+2 == the receiver's per-step `v`). The single-slot C4(A) wrote the bonus
(`select_latest` = `recv[v-1]`) at the *early* slot `num_tokens_no_spec` and left the accepted-draft
positions as the optimistic `-1` → read → `indexSelectSmallIndex`.

**(A) value+position fix — DONE, hardware-confirmed.** New pure helper
`gather_valid_sampled_tokens_per_req(recv) -> recv[i, 0:v]` (TDD, 4 tests; generalises
`select_latest`). Receiver (`_pp_receive…`) now writes **all `v` real tokens in order** into
`token_ids_cpu[i, ntns : ntns+v]` and advances `num_tokens_no_spec` by `v`, so the write cursor
stays in lockstep with the `v`-per-step read cursor. Result: reads on **every clean decode step
(s1–s5) are now real** (were `-1`); ordering correct. Strictly better than C4(A); local suite stays
green (async_scheduler + broadcast 24).

**(B) count reconciliation — the ONLY remaining break#2 cause, now precisely pinned.** The crash now
lands at the SAME step the original C4(A) died at (s6 for this prompt) — and the probe shows the
ORIGINAL run ALSO went `gathered=None` (all-chunked) at s6, so **s6-chunked is pre-existing, not a
regression.** Mechanism: the optimistic-extend (`:1319`) appends `prev_num_draft_len` `-1`s to
`output_token_ids` each step; on the **non-last rank these are never corrected** (the deferred
`correct_spec_decode_token_counts` `:1471` runs only where the sampler is — the last rank). They
accumulate → `num_tokens = num_prompt + len(output_token_ids)` inflates → the discard mask
`discard_request_mask = (num_computed_tokens + num_scheduled) < num_tokens` (`:2045`) eventually
fires spuriously → the request is mis-classified all-chunked-prefill → `_pp_receive` takes the
`gathered=None` branch → writes a `-1` → next read embeds it → break#2. (Confirmed by the s8-holA
detour: coupling `output_token_ids` length to the one-step-ahead `ntns` via del+extend tripped the
same discard mask early, at s3; decoupling pushed it back to the inherent s6 boundary.)

**The (B) fix (next):** at the receiver, trim `output_token_ids` by
`correction = prev_num_draft_len - (v - 1)` — the non-last-rank analogue of
`correct_spec_decode_token_counts` — so `num_tokens` tracks the true committed length and the discard
mask stops misfiring. This is the **exact B1b double-count site** (s5 over-advanced 14-vs-11): get
`prev_num_draft_len` provenance right (reset/restored at `_update_states` 1438-1442). Oracle: MiMo
generate completes → greedy-equiv vs a `mode=baseline` MiMo run → 27B vs `base.json`.

**(B) SHIPPED → break#2 CLOSED (s8-holB2).** Receiver now trims the `prev_num_draft_len` optimistic
placeholders then extends `recv[i,0:v]`. Result: **MiMo PP=2+MTP async runs END-TO-END (first ever)**,
`exit=0`, 5×40 tok; the discard probe shows `discard=False` every step (`num_tokens` tracks
`optimistic_seq`); reads clean through s19+ (negatives only on the draft/`query_pos=1` slot, which is
overlaid). `prev_num_draft_len` was the right value at the receiver (provenance OK for MTP). Local
suite stays green (24).

**BUT greedy-equivalence FAILS — the remaining (correctness) gate.** `mimo_spec_dbg.json` vs the no-spec
`mimo_base.json`: all 5 sequences diverge at ~token 2 (seq0 base `[12095,13,1084,…]` vs spec
`[12095,13,315,…]`). Lead: the receiver wrote `recv[s1] col1 = 0` into pos 6, but the true 2nd
committed token (from the last rank's own output) is `13` → the non-last target got a wrong-context
token → it wrongly accepted draft `315` as out[2] (baseline `1084`). So **the broadcast grid's
non-bonus columns are NOT all genuinely-committed** (col may be the *next*, uncommitted draft), OR the
PPDBG `recv s{n}` labels (incremented at the read site) are misaligned with the sender's engine step.
The broadcast is `sampler_output.sampled_token_ids` (sender `_pp_broadcast_prev_sampled_token_ids`
`:4470`/`:4684`; doc'd "accepted drafts + bonus, -1 padded" — but the data contradicts a naive
all-committed read). **NEXT: instrument the SENDER (last rank)** — per-req `sampled_token_ids` row +
`num_computed_tokens` + engine step — so recv rows map 1:1 to committed baseline tokens, then write
ONLY the genuinely-committed values. Keep (A)+(B) (they close the crash, locally green); this refines
*which* values to write. (cf. gibberish #36872 — greedy-equiv is THE bar.)

**Sender width-pad — one more real bug found + fixed; greedy-equiv divergence is NOT in the
reconstruction.** Sender-side PPDBG showed the broadcast grid is variable-width: `send s0`
(first decode, no scheduled spec) = `[[12095]]` width-1, but every later step is width-2, while the
receiver always reads `num_spec+1` → it read uninitialised buffer garbage (`0`) in col1 and committed
it. **Fix (B1a completion):** the sender pads `sampled_token_ids` to `num_spec+1` with `-1` before
broadcast (`_pp_broadcast_prev_sampled_token_ids`). Verified: `recv s1` went `[12095,0]`→`[12095,-1]`,
`gathered`→`[12095]` (v=1). **But the engine output is BYTE-IDENTICAL across C4(A) → A+B →
width-pad** — i.e. **invariant to every non-last reconstruction change.** Baseline is deterministic
(×3 token-identical), so the divergence is real, and its invariance to the reconstruction is
**decisive**: in PP the non-last rank's *GPU* inputs are overlaid correctly in the common case
(`prev_sampled` overlay + draft scatter in `_prepare_input_ids`), so the forward is correct there;
the cpu reconstruction only matters on the *non-common* (crash) path. **⇒ The greedy-equiv divergence
(`[12095,13,315,…]` vs greedy `[…,1084,…]`) is the spec MECHANISM emitting non-greedy tokens — a
separate, likely PRE-EXISTING MTP+PP verification/acceptance bug, NOT the input reconstruction my
A/B/width-pad work fixed.** Net of session 8: **break#2 CLOSED, MiMo PP=2+MTP async RUNS end-to-end**;
greedy-equiv remains, now correctly scoped to the verification layer. Lead: the read trajectory shows
the draft (`query_pos=1`) slot = `-1` on the non-last rank — if its hidden states for the spec
positions are wrong, the last rank's verification logits (hence acceptance) are wrong. **Isolation
test (next): single-GPU MTP (no PP) must be greedy-equiv; if it is, the bug is PP-specific** (spec-
position hidden-state feeding / draft embedding on the non-last rank). Likely a separate area/PR.

### greedy-equiv ROOT CAUSE — FULLY PINNED (s8-iso + s8-spectok): drafts never reach the non-last rank

**Isolation (s8-iso):** single-GPU MiMo MTP (pp=1, cpu_offload) is ~greedy-equiv — 4/5 sequences
token-identical to baseline, seq0 diverges only at tok29 (a minor near-tie/offload edge). PP=2
diverges at tok2 on ALL sequences. ⇒ **the gross divergence is PP-specific; MTP itself is correct.**

**Mechanism (s8-spectok):** a probe in `update_req_spec_token_ids` shows
`scheduled_spec_decode_tokens = [-1]` every step on BOTH ranks — the scheduler carries a `-1`
placeholder for the draft (the async/C3 design; the real token arrives via the GPU path). The real
draft lives in `_draft_token_ids`, set by `propose_draft_token_ids`, which runs **only on the last
rank** (the drafter is gated `if self.speculative_config and get_pp_group().is_last_rank:`,
`gpu_model_runner.py:547`). On the non-last rank `_draft_token_ids is None`, so the GPU draft-scatter
in `_prepare_input_ids` (`:1814` `if self._draft_token_ids is None: return`) is **skipped** → the
non-last rank keeps the `-1` placeholder at the spec position and embeds it → its verification-forward
hidden states for the draft position are garbage → the last rank's verification logits (computed from
those PP hidden states) are wrong → it accepts a non-greedy token (out[2]=315 vs greedy 1084). The
divergence is BYTE-invariant to the C4 reconstruction because that fixes the *sample* position / crash
path, not the *draft* embedding.

**FIX (a new piece, analogous to B1a's sampled-token broadcast — distinct from C4):** the last rank
must broadcast its proposed `_draft_token_ids` to the PP group, and the non-last rank must scatter
them into the spec positions in `_prepare_input_ids` (replacing the `-1` placeholder), so its
hidden states match the single-GPU path. Oracle: PP=2 spec == `base_a.json` (modulo the tok29-class
edge). **Net of session 8: break#2 CLOSED (C4 A+B + width-pad → MiMo PP=2+MTP RUNS e2e); greedy-equiv
root cause fully pinned with a concrete fix design (draft broadcast), which is the next, separate
contribution.**

## → implications

- **C makes the *draft side* clean (1 forward flag); this brick is the *bulk* of
  the real work, and it is design-independent.** Choosing C doesn't shrink brick
  40 — but it doesn't grow it either, and C adds no cross-stage traffic on top.
- **Strategy:** borrow #39704's proven approach for these three areas (it
  empirically hit and fixed them), reimplemented against current main's
  `DraftTokenIds` path. Coordinate with its author rather than competing — this
  is the part where they've already done the hard debugging.
- **Correctness/equivalence (greedy ≡ non-spec) is gated on getting these three
  right.** This is the heart of E3, not the draft placement.

## Q13 — memory mitigation (confirmed mechanism)

`VLLM_PP_LAYER_PARTITION` (envs.py:51,823; honored by `get_pp_indices`,
distributed/utils.py:125) lets us set an explicit per-stage layer count, e.g.
`[36,28]` for a 2-stage split. Giving the **last** rank fewer target layers frees
VRAM there for the whole draft (embed + layer + lm_head). Zero code. This turns
Q13 (Design C's last-rank memory load) from a blocker into a **tuning knob**, and
also balances compute on the heterogeneous pair (give the faster Blackwell GPU1,
which also runs the draft, fewer target layers). → Q13 mitigated; confirm headroom
numerically at E3.

## Testing strategy — brick 40 is locally TDD-able (no GPU window)

The batch_queue correctness leads are **scheduler/engine-core state logic**, which
is unit-testable with synthetic `ModelRunnerOutput`s and hand-computed expected
state transitions — **no real model, no 2 GPUs** for most of it.

Confirmed enablers:
- `async_scheduling` turns on `batch_queue` even at pp=1 (`vllm/config/vllm.py:497`
  `max_concurrent_batches`) → the k-step delay is reproducible locally.
- `load_format="dummy"` (random weights) → tiny models with no checkpoint.
- **Existing harness already covers the pieces separately:**
  - `tests/v1/core/test_scheduler.py:131`
    `test_async_scheduling_pp_allows_rescheduling_with_output_placeholders` —
    `create_scheduler(async_scheduling=True, pipeline_parallel_size=2)`, pure
    scheduler logic, no GPU.
  - `test_scheduler.py:337+` — spec tests with `scheduled_spec_decode_tokens`,
    `DraftTokenIds`, `update_draft_token_ids`, and
    `test_schedule_spec_decoding_stats(spec_tokens, output_tokens, expected)`
    (the hand-computed-expected pattern).
  - `_make_model_runner_output(...)` builds synthetic outputs.
  - **Gap = the combination:** spec under PP/async batch_queue delay. Write those
    by combining the two configs + delayed synthetic outputs.

Layered plan (cheap → complete):
1. **Scheduler unit tests** (this harness): drive `create_scheduler(async=True,
   pp=2, num_speculative_tokens=k)` through a pipelined sequence of synthetic
   outputs; assert `spec_token_ids` / `num_computed_tokens` / `output_token_ids`
   match hand-computed expectations. **Confirms or refutes the 3 leads on current
   main** — turning TDD-red into the actual bug list. No GPU.
2. **Tiny target + ngram spec + async batch_queue on 1 GPU** (dummy weights):
   end-to-end delay path, oracle = greedy ≡ non-spec. Exercises engine-core
   `step_with_batch_queue` (lead #1) for real. Ngram needs no draft model, so it
   isolates brick-40 plumbing from the MTP draft.
3. **2 ranks (CPU or 2 GPU)** for lead #3 (non-last-rank accounting) — the only
   genuinely PP-rank-specific lead.
4. **Real Qwen3.5 PP=2 on gpu-wb (E3)** — final greedy-equivalence + memory.

**Harness validated locally (2026-06-04):** pp=1 spec scheduler tests pass on
CPU out of the box (only `tblib` needed). The pp=2 tests failed only on the
`ParallelConfig` world-size>GPU-count check (environmental, not logic); fixed by
making `create_scheduler` (tests/v1/core/utils.py) build the config with
`distributed_executor_backend="mp"` for pp>1. After that, the spec+async+pp
subset is 20/20 green — so brick-40 tests (pp=2 + async + spec with delayed
synthetic outputs) are writable locally on this harness. (utils.py tweak is an
uncommitted working change.)

**→ implication:** the prod window is needed only at step 4, for the real model.
Steps 1–3 de-risk the bulk of brick-40 correctness locally, and turn "implement 3
fixes then iterate" into "write failing tests that pin the exact bugs → fix to
green" — disciplined, not symptom-chasing.

## Open questions updated
- **Q8 — ANSWERED (with leads):** spec-under-PP needs three plumbing fixes
  (draft-token retrieval in batch_queue, stale-snapshot guard, non-last-rank
  token accounting); #39704 is the reference. Confirm each at E3.
- **Q13 — MITIGATED:** `VLLM_PP_LAYER_PARTITION` rebalances layers off the last
  rank to fit the draft. Stakeholder idea, built-in knob.
- **Q7 — now actionable:** uneven split is exactly the lever for Q13.

## Session 9 — greedy-equiv FIRST divergence ROOT-CAUSED: non-last rank skips the async position correction (2026-06-05)

**Reproduced** (run_mimo_dbg MODE=spec, no-offload): per-seq first divergence vs
`mimo_base.json` = seq4@out3, seq3@5, seq1@6, seq0@8, seq2@25 (exactly as s8).
Deterministic. Dissected the EARLIEST + cleanest: **seq4@out3** (prompt 8 tok).

**Probe trajectory (max_num_seqs=1 → seqs run one-at-a-time, clean per-seq):**
```
s93 prefill pos[0..7]              -> sample 4710        (out0 = pos8)
s94 pos[8,9]  feed[4710, draft61797] reject(≠13708)     -> commit 13708 (out1 = pos9)
s95 pos[10,11] feed[13708,draft766]  accept(766) bonus29 -> commit 766(out2),29(out3) ✗
```
Base out3 = 397, spec = 29. **Smoking gun:** token 13708 is the true sequence
pos **9** (out1), but at s95 it is fed at rope position **10**, and the draft 766
(true pos 10) at position **11** — every position shifted **+1**. The bonus is then
predicted for "766 at pos 11" = 29 instead of "766 at pos 10" = 397.

**Confirming pattern — `num_computed_tokens` advance is INVERTED on the non-last rank:**
reject (1 tok committed) → nct += 2; accept (2 tok) → nct += 1. (Should be the
reverse: advance by the *valid* count.) The +1 error appears on the step *after* a
rejection, corrupts exactly one bonus token, then "self-heals" on the next accept —
which is why each seq diverges at a different out-index (wherever its first
rejection-then-accept lands).

**ROOT CAUSE (every site pinned):** async spec decode advances counts optimistically
(all drafts accepted) in `_update_states` (`gpu_model_runner.py:1319` extend `-1`s,
`:1342/:1408` set `num_computed_tokens` = scheduler's optimistic value) and *corrects
after the forward* via the GPU kernel `update_num_computed_tokens_for_batch_change`
(`:2138`). That correction is **gated on `self.valid_sampled_token_count_gpu`**
(`:2128-2132`), which is produced ONLY by the sampler in `_copy_valid_sampled_token_count`
(`:4986`) — i.e. **only on the last rank**. On the non-last rank it is `None`, so the
correction is skipped and `num_computed_tokens` is copied straight from the optimistic
CPU values (`:2147`). ⇒ the non-last rank's positions/KV-slots over-advance by the
rejected-draft count after every rejection. Single-GPU MTP (= last rank, has the
sampler) gets the correction → greedy-equiv (the s8-iso 4/5 result). **PP-specific,
exactly as s8 predicted; mechanism now fully grounded.**

**FIX (designed, analogous to B1a/C4 — distinct piece): drive the SAME correction on
the non-last rank from the BROADCAST valid counts.** The receiver already computes
per-req valid counts (`gather_valid_sampled_tokens_per_req(recv)` → `gathered`, `v =
len(gathered[i])`) purely from the broadcast. Reconstruct `valid_sampled_token_count_gpu`
(and the `prev_num_draft_tokens` / `prev_positions` the kernel reads) on the non-last
rank from those `v`, so `update_num_computed_tokens_for_batch_change` (`:2138`) fires
**identically on both ranks**. One code path, one invariant: *num_computed_tokens
advances by the valid count, the same on every rank.* (The existing C4(B) receiver
already does the analogous reconcile for `num_tokens_no_spec` + `output_token_ids`;
this adds the missing `num_computed_tokens` arm.) Oracle: PP=2 spec == `base_a.json`
modulo the tok29-class near-tie edge. TDD the count-reconstruction (CPU helper).

**Nature of the bug (for the record):** NOT a Python-async artifact and NOT careless
code — it is a *feature-interaction gap*. Async-spec-decode's optimistic-then-correct
accounting was wired for the co-located sampler (single GPU / last rank); PP splits the
sampler onto one rank only, so the correction's input doesn't exist on the others. The
hook (the GPU kernel) is even present — it is just fed `None`. Silent (wrong numbers,
no crash), so it only surfaces as non-greedy output — the hardest class to catch. This
is precisely why upstream HARD-BLOCKS MTP+PP (`SupportsPP NotImplementedError`): the
combo was never finished. Our foundation (A1c+B1a+C3+C4) enables it; this is the last
structural accounting arm. → reinforces the brick-81 thesis: the highest-value
contribution is an explicit, typed, tested **spec-decode token-accounting state
machine** (one invariant: "advance by valid count, identically per rank") — not a
pipeline rewrite. See `81-typing-and-rewrite-contribution.md` + the s9 strategy note.

### s9 FIX VERIFIED — MiMo PP=2+MTP greedy-equiv + 1.75x speedup; residual = near-tie floor

**Fix** (`8105121a9`): receiver stashes the per-req broadcast valid count
(`_pp_prev_valid_sampled_count`); `_update_states` subtracts the rejected-draft drift
from the optimistic `num_computed_tokens` (`num_computed_tokens_drift_correction`,
4 unit tests) on the non-last rank only — BEFORE the value is stored (`:1347/:1408`)
so the else-branch CPU→GPU copy (`:2147`) feeds `self.positions` (`:2160`) the right
rope/KV positions. (First placement in the receiver FAILED: it runs in `sample_tokens`
and `:1408` overwrites it — the probe showed nct unchanged. Moving to `_update_states`
fixed the timing.)

**Verification (gpu-wb, MiMo-7B PP=2+MTP async, no-offload):**
- **40 tokens × 5 seqs: 5/5 token-identical to `mimo_base.json`, deterministic ×2.**
  (Pre-fix: all 5 diverged at tok 3–8.)
- Local: 129 green (async_scheduler + pp_spec_broadcast + scheduler), ruff clean.

**Speed (clean run, NO CUDA_LAUNCH_BLOCKING / NO PPDBG, 200 tok × 5, max_num_seqs=1,
enforce_eager, ignore_eos):** baseline 22.70 tok/s vs **spec 39.66 tok/s = 1.75×**.
Correctness AND speedup together — first time for this combo.

**Longer chains (200 tok) — three-way base / PP-spec / single-GPU-spec (sg, cpu_offload):**

| seq | PP-spec vs base | SG-spec vs base | PP vs SG |
|----:|:---------------:|:---------------:|:--------:|
| 0 | identical | @25 | @25 |
| 1 | @176 | @176 | **identical** |
| 2 | identical | identical | identical |
| 3 | identical | identical | identical |
| 4 | @109 | @161 | @109 |

**Divergence-from-baseline tally @200: PP-spec 2/5, single-GPU-spec 3/5** → PP is *at
least as* greedy-equivalent as the established single-GPU MTP reference. The residual is
the **near-tie floor**, NOT a PP bug, proven by: (a) seq1@176 is bit-exact identical
between PP and SG (a shared fp near-tie: base "scatters more *than*" vs both spec "more*.
So*"); (b) single-GPU MTP itself diverges (more often); (c) seq0 PP is PERFECT while SG
diverges @25 (base/PP "Madrid is a very popular" vs SG "The city is located" — PP *more*
correct). A systematic accounting bug diverges EARLY on ALL seqs (as the s9 bug did at
tok 3–8); this is sporadic, late, fp-sensitive, and sometimes favors PP → near-tie noise.
This is exactly the "tok29-class edge" the KB anticipated. Spec decode is greedy-equiv
*up to fp near-ties* by nature (2-token vs 1-token forward → different reduction order →
argmax flips at ties); vLLM's own spec tests tolerate this.

**NET s9: C4 greedy-equiv CLOSED.** MiMo PP=2+MTP async now: runs e2e (break#2, s8) +
greedy-equiv to no-spec baseline up to the near-tie floor (s9) + 1.75× faster. The
systematic non-last-rank position bug is root-caused and fixed with one invariant
(advance `num_computed_tokens` by the valid count, identically per rank). Next: strip
PPDBG probes + throwaway run scripts, then 27B greedy-equiv (vs base.json) and the
[[CMP]] benchmark, and cut the upstream PR series. **NOTE for PR:** `e3_run.py` now has
`ignore_eos=True` (was False) + warmup + timing — revert/guard before PR; perf scripts
(run_mimo_perf.sh / run_sg_spec.sh) are throwaway.
