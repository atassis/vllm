# Brick 80 — The async-spec-PP execution pipeline (how it actually works)

Status: **RESEARCHED (session 5)** · Verified against code via 5 parallel subsystem reads.
This is the deep "what does the pipeline DO" map — engine loop · scheduler · runner ·
distributed · rejection oracle. Companion to brick 40 (the correctness gate) and
brick 81 (typing + the rewrite-contribution plan).

> Line numbers are from the session-5 working tree; treat as anchors, re-grep if drifted.
> A fact without `file:line` is a hypothesis (KB rule).

---

## 0. The shape of the whole thing (one paragraph)

Under PP>1, vLLM runs a **pipelined engine loop** (`step_with_batch_queue`) that schedules
step N, fires `execute_model` non-blocking, and only pops + applies `update_from_output`
**~`pp_size−1` steps later** (the batch_queue depth). Speculative decoding rides on top:
the **scheduler** reserves spec slots optimistically with `-1` placeholders and does
rejection accounting; the **last PP rank** runs the drafter + sampler and **broadcasts**
the sampled tokens to the other ranks; the **non-last ranks** rebuild their next input
from that broadcast + the scheduler snapshot. The **rejection sampler** is the correctness
oracle (greedy ≡ argmax-match). The whole spec-under-PP path was finished for `ngram_gpu`
and is buggy/unfinished for MTP — that's our task.

---

## 1. Engine loop + batch_queue (`vllm/v1/engine/core.py`)

- `step_fn` chosen once at init: `step` if `batch_queue is None` else `step_with_batch_queue`
  (`core.py:217-218`). `batch_queue` enabled when `max_concurrent_batches > 1`
  (`:192-198`), which is `pp_size` for PP>1 (`config/vllm.py:497-507`).
- `step_with_batch_queue` (`:484-598`): `schedule()` → `execute_model(non_block=True)` →
  `appendleft((future, scheduler_output, exec_future))` → **early-return `(None, executed)`
  while the queue isn't full**; once full, `batch_queue.pop()` **blocks** on the oldest
  future (`:555-560`) → `update_from_output` (`:570-572`). **k-step delay ≈ `pp_size−1`.**
- **`post_step`** (`:474-482`, called `:1266`): the **SYNC** spec path — pulls
  `take_draft_token_ids()` → `scheduler.update_draft_token_ids()`, **gated on
  `not async_scheduling`**. Under async it is a no-op (worker injects drafts instead).
- `deferred_scheduler_output` branch (`:574-596`): only for structured output; pulls draft
  tokens via `take_draft_token_ids` + `update_draft_token_ids_in_output`.
- Output flows out via a daemon-drained `output_queue` (`:1264`, `process_output_sockets`);
  the main loop never blocks on output.
- **болезни:** dual draft-token paths (sync `post_step` vs async worker-inject) with no
  shared abstraction; deferred-sampling gate can deadlock if the queue fills mid-defer;
  `deque(maxlen)` silently drops on overflow; futures must be awaited in exact order or the
  pipeline stalls silently.

## 2. Scheduler (`vllm/v1/core/sched/{scheduler,async_scheduler,output}.py`)

- Token-budget loop (`scheduler.py:336-950`): each request tries to make `num_computed_tokens`
  catch up to `num_tokens_with_spec` (= `len(all_token_ids) + len(spec_token_ids)`,
  `request.py:247`). `num_new_tokens` also adds `num_output_placeholders` (async in-flight).
- Spec scheduling (`:516-531`): packs `request.spec_token_ids` into
  `scheduled_spec_decode_tokens[req_id]`, trims to budget, **then clears
  `request.spec_token_ids = []`** (consumed once per step).
- `update_from_output` rejection accounting (`:1396-1417`): `num_accepted =
  len(generated)−1`; `num_rejected = num_draft − num_accepted`; decrements
  `num_computed_tokens` and (async) `num_output_placeholders` — reverting the optimistic
  advance. Uses `model_runner_output.req_id_to_index[req_id]` (the `KeyError` site when a
  worker dies, brick 40).
- **AsyncScheduler** (`async_scheduler.py:19-41`): `_update_after_schedule` reserves
  `num_output_placeholders += 1 + cur_num_spec_tokens` and sets `request.spec_token_ids =
  self._spec_token_placeholders` (a shared `[-1]*num_spec` list, `:16`). **This is where the
  `-1` placeholders originate.** `prev_step_scheduled_req_ids` tracks previous-batch
  membership — the key signal #40768 uses.
- Preemption (`:957-977`): clears `spec_token_ids`, resets `num_computed_tokens=0`.
- **болезни:** rejection-accounting off-by-one silently corrupts `num_computed_tokens`;
  `num_output_placeholders` underflow under chunked-prefill+preempt; `is_prefill_chunk`
  flag volatility; spec-token clear-after-pack loses tokens on mid-step exception; the
  placeholder list is emitted even for requests that won't get the worker-side overwrite
  (**the #40768 bug = our break #2 at the scheduler layer**).

## 3. GPU model runner async-spec-PP path (`vllm/v1/worker/gpu_model_runner.py`)

- `use_async_spec_decode = use_async_scheduling and num_spec_tokens > 0` (`:~638`).
- `_update_states` (`:1132-1500`):
  - **Optimistic extend** (`:1314`, method-agnostic for async): `output_token_ids.extend(
    [-1]*optimistic_num_accepted)`; registers a deferred `correct_spec_decode_token_counts`.
  - **`num_tokens_no_spec` advance** (`:1330`) + its correction (`:1490`) are **`is_ngram_gpu`-
    gated** → NOT applied for MTP (the asymmetry brick 40 flagged; B1b touched this).
  - Non-last-rank branch (`:1338-1358`): async → `new_token_ids=[]` (use broadcast); sync →
    `req_data.new_token_ids[i]` (scheduler ship-back).
  - Re-added request (`:1384-1399`): async recovers `output_token_ids` from
    `req_data.all_token_ids[req_id]` — but **placed without spec-offset correction** (the
    drift; #39704's fix-up targets this).
- `_prepare_input_ids` (`:1708-1832`): async path scatters
  `prev_sampled_token_ids[:, 0]` into the sample positions; draft tokens from
  `self._draft_token_ids` (**None on non-last ranks**). **Non-common early-return** when
  `prev_positions[cur] < 0` (`:1748`) → request reads raw `token_ids_cpu` → **break #2**.
- PP propagation: `_pp_broadcast_prev_sampled_token_ids` (`:4650`, last rank) /
  `_pp_receive_prev_sampled_token_ids_to_input_batch` (`:4667`, non-last) — width
  `num_spec+1` (after B1a); receiver appends `-1` placeholders + advances
  `num_tokens_no_spec` by 1; builds `prev_req_id_to_index` **after `condense()`**.
- `_update_states_after_model_execute` (`:1498`, **hybrid only**, e.g. Qwen3.5):
  `num_accepted = (sampled != -1).sum(dim=1)`.
- Drafter created **only on last rank** (`:542`), else `self.drafter = None`.
- **болезни:** two `-1` placeholder sources (scheduler emit + runner extend) backfilled
  only on the common path → leak on non-common/re-added steps (break #2); `condense()`
  reorders then `prev_req_id_to_index` rebuilt → stale-mapping risk; the GPU drift
  correction (`update_num_computed_tokens_for_batch_change`) needs exact alignment of
  prev_positions/prev_num_draft/valid_count across iterations; hybrid vs ngram vs
  method-agnostic branches are tangled in one 7000-line file.

## 4. Distributed PP (`vllm/distributed/parallel_state.py`)

- `IntermediateTensors` cross stages via `isend_tensor_dict`/`irecv_tensor_dict`
  (`:890-1069`): metadata on `cpu_group`, tensors on `device_group`; default dest
  `(rank_in_group+1)%world_size`.
- `broadcast` (`:637-650`) on `device_group`; `is_first_rank`/`is_last_rank`/
  `rank_in_group`/`world_size` (`:455-487`). **No spec-specific broadcast helper exists**
  in the coordinator — the sampled-token broadcast is hand-rolled in the runner
  (`_pp_broadcast_*`), which is exactly the seam B1a/`pp_spec_broadcast.py` owns.

## 5. Rejection sampler = the correctness oracle (`vllm/v1/sample/rejection_sampler.py`)

- Greedy path (`rejection_greedy_sample_kernel`, `:708-757`): `token = target_argmax;
  rejected = draft != target_argmax`. **Output is a pure function of (target argmax
  sequence, draft sequence)** → **weight-agnostic**: weights change only acceptance rate,
  never spec-vs-non-spec equality. → validates brick-70 A3 (dummy/MiMo are valid oracles).
- Output layout: `sampled_token_ids` = `[num_reqs, num_spec+1]`, `PLACEHOLDER_TOKEN_ID = -1`
  padding (`:30,425`), valid tokens **contiguous from position 0**, bonus at the accepted
  tail. `parse_output` filters `!= -1 & < vocab` (`:247-281`).

---

## 6. The end-to-end spec-under-PP dataflow (text diagram)

```
LAST rank (pp.last):  forward → _sample → rejection oracle → sampled[num_reqs,num_spec+1]
                       → _update_states_after_model_execute (hybrid accepted count)
                       → _pp_broadcast_prev_sampled_token_ids  ──┐ (GPU broadcast, async)
                       → propose drafts → _draft_token_ids       │
                                                                  ▼
NON-LAST ranks:        _pp_receive_… : recv[num_reqs,num_spec+1]; append -1 placeholders;
                       build prev_req_id_to_index (post-condense)
                       → next step _prepare_input_ids:
                            common (prev_pos≥0): scatter recv[:,0] → input_ids  ✓
                            non-common (prev_pos<0): EARLY RETURN → reads token_ids_cpu
                                                     which still holds -1  → EMBED OOB ✗ (break #2)
SCHEDULER (all ranks): reserves -1 placeholders (AsyncScheduler) even for requests that
                       won't get the worker overwrite → the root #40768 fixes.
```

## 7. Где наша задача внутри этого (task-specific)

- **B1a** (done): the broadcast transport (`_pp_broadcast/_pp_receive`) carried `[num_reqs,1]`
  not `[num_reqs,num_spec+1]` → fixed (brick 40 §Session-5).
- **break #2**: the `-1`-placeholder leak on non-common/re-added requests. **Root cause
  matches upstream #40768** ("stale async placeholder tokens", fixes #37159) — a
  scheduler-side fix (`num_pending_async_spec_placeholders` + only emit `-1` when the req
  was in `prev_step_scheduled_req_ids`). Complementary to B1a (scheduler-side vs worker-side).
- **MTP-vs-ngram asymmetry**: `num_tokens_no_spec` advance/correction (`:1330/:1490`) is
  ngram-gated; MTP on hybrid uses `_update_states_after_model_execute`. Need to confirm MTP
  accounting end-to-end (MiMo non-hybrid vs 27B hybrid differ here).
- Oracle is weight-agnostic → MiMo/dummy validate the cascade SHAPE; real weights needed
  only for acceptance-rate.

## Open questions spawned
- **Q16:** Does #40768 (scheduler placeholder discipline) + B1a (transport width) fully
  close break #2 for MTP+PP? (test on MiMo once integrated.)
- **Q17:** Is the MTP `num_tokens_no_spec` accounting correct without the ngram-gated
  advance, given hybrid `_update_states_after_model_execute` only runs on hybrid models
  (MiMo is non-hybrid)? (the B1b question, reframed.)
