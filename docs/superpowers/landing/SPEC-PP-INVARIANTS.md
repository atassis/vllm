# Spec-under-PP — the invariant & change map

> **Read this before you touch the speculative-decoding-under-pipeline-parallel path.**
> The other docs explain *what is there* (`PIPELINE.md`), *why it's shaped that way*
> (`PIPELINE-NARRATIVE.md`), and *what the pieces are* (`FOUNDATIONS.md`). This one is
> the **change map**: the invariants you must not break, where each can break
> (`file:line`), what a violation looks like as a bug, and the test that catches it.
> It is task-shaped — built for the moment you're editing this code, not for a tour.
>
> **Why this exists.** This seam broke for MTP precisely because nobody had written the
> contract down. Three independent bugs (a too-narrow broadcast, stale scheduler
> placeholders, an un-persisted token value) all came from violating an *unstated*
> invariant. Stating them is the cheapest way to stop the next one.
>
> Line numbers are anchors from the working tree (it carries uncommitted spec-under-PP
> changes); `check_refs.py` validates that they resolve. Re-grep the symbol if drifted.

---

## 0. The 30-second model of the seam

Under PP>1 + async scheduling, one decode step of a speculative request is split across
ranks and across time:

- **Last rank** runs the sampler + the **rejection oracle**, producing a grid
  `sampled_token_ids = [num_reqs, num_spec+1]` (valid tokens packed from column 0, `-1`
  padding after). It **broadcasts** that grid to the other ranks and proposes the next
  drafts.
- **Non-last ranks** never run the sampler. They **receive** the grid and must rebuild
  their next-step input (`input_ids`) from it — because the embedding lives on the first
  rank, and the next token has to be embedded *somewhere*.
- **The scheduler** runs ahead (async): it reserves spec slots with `-1` placeholders so
  it can schedule step N+1 before step N's output returns. The engine applies that
  output ≈ `pp_size−1` steps later (the `batch_queue` delay).

Everything that can go wrong lives in the gap between "the last rank knows the real
token" and "every other rank, k steps later, needs that real token to embed."

---

## 1. The invariants (the contract)

| # | Invariant — must always hold | Why it exists | Where it can break (`file:line`) | Violation looks like | Test / oracle |
|---|---|---|---|---|---|
| **I1** | **Spec output ≡ non-spec greedy output**, token-for-token. | The whole point: speed without changing the answer. The rejection oracle guarantees it — greedy accepts a draft *iff* it equals the target's `argmax`, else substitutes the argmax. | `vllm/v1/sample/rejection_sampler.py:708` (greedy kernel), `:452` (`target_argmax`) | Gibberish / subtly different text vs greedy (the #36872 class). Silent — no crash. | **greedy ≡ non-spec** on MiMo (fast) then 27B vs `base.json`. Weight-agnostic, so dummy/MiMo are valid. |
| **I2** | Across the **k-step `batch_queue` delay**, the counters stay coherent: `num_computed_tokens`, `num_output_placeholders`, `num_tokens_no_spec` reflect the same reality after the delayed `update_from_output`. | The `SchedulerOutput` applied at step N+k is a *snapshot* taken at N; spec/placeholder state mutated in between. | `vllm/v1/core/sched/scheduler.py:1438` (rejection accounting), `:1348` (`update_from_output`); runner `gpu_model_runner.py:1418` (branch-2 re-derive), `:1487` (num_computed correction) | Over/under-generation; off-by-one drift that corrupts `token_ids_cpu` positions. | `tests/v1/core/test_pp_spec_batch_queue.py` (count accounting, CPU); `test_async_scheduler.py`. |
| **I3** | **Non-last ranks must hold the REAL sampled-token value** in every confirmed `token_ids_cpu` position — never a `-1` placeholder. | The non-common path of `_prepare_input_ids` reads `token_ids_cpu` directly to build the embedding input. | receiver `gpu_model_runner.py:4694` (writes `-1`, not the value); consumer `:1707` (`_prepare_input_ids`); embed `vocab_parallel_embedding.py:491` | **break #2**: `-1` reaches the embedding lookup → CUDA `indexSelectSmallIndex` OOB crash on the non-last rank. | MiMo PP=2+MTP async reaches `generate` without the device assert. **This is the current open gate (C4).** |
| **I4** | A `-1` placeholder must be **filled before it's embedded, or never emitted when it can't be filled** (emit only if the req was in `prev_step_scheduled_req_ids`). | The scheduler emits `-1` optimistically; if a req won't get a worker-side overwrite, the `-1` leaks. | `vllm/v1/core/sched/async_scheduler.py:31` (placeholder origin); the `-1` constant `rejection_sampler.py:30` | A `-1` survives into the runner — feeds I3's crash, or wrong accounting. | `test_async_scheduler.py` (C3 / #40768 discipline — **green standalone**). |
| **I5** | **Rejection rollback is unified**: draft and target share `seq_lens`/block tables; rejected KV is overwritten on reuse, not explicitly cleared. | The draft has its own K/V but the same block allocation; both must roll back together. | proposer `seq_lens -= num_rejected` (brick 30); scheduler `num_computed_tokens -= num_rejected` `scheduler.py:1438` | KV positions desync between draft and target → wrong attention → I1 fails. | greedy-equiv (I1) is the end-to-end catch; brick 30 for the mechanism. |
| **I6** | The fix must **compose** with: hybrid-mamba conv state, draft-embed quant (A1c), drafter-on-last-rank-only, chunked prefill, structured output. | These are orthogonal features sharing the same 7,583-line runner; a narrow fix that ignores them regresses one of them. | hybrid path `gpu_model_runner.py:1513` (`_update_states_after_model_execute`, hybrid-only → 27B yes, MiMo no) | Works on MiMo (non-hybrid), breaks on 27B (hybrid), or vice-versa. | MiMo **and** 27B both reach greedy-equiv. MiMo isolates non-hybrid accounting cleanly. |

---

## 2. The current frontier — break #2 (= invariant I3), fully diagnosed

**State (session 6): Q16 answered NO.** B1a (width-agnostic broadcast) + C3 (#40768
scheduler placeholder discipline) are both done and green, but they do **not** close
break #2. The engine now reaches `generate`, then the non-last rank (rank0) still
crashes embedding a `-1`.

**Root cause (decisive structural finding).** Under async PP, **no code path ever writes
the real sampled-token value into the non-last rank's `token_ids_cpu`.** Traced:
- `_update_states` branch-1 (`gpu_model_runner.py:1342`): for async PP `new_token_ids ==
  []` → appends nothing.
- branch-2 (`gpu_model_runner.py:1418`): advances counts and sets `is_token_ids = True`,
  but the `token_ids_cpu` write is gated `if new_token_ids:` → False for async PP → **no
  value written**.
- receiver (`gpu_model_runner.py:4694`): writes `-1` + the count, **no value**.
- `_prepare_input_ids` (`gpu_model_runner.py:1707`): writes the GPU `input_ids` only
  (common path, column 0), never the persistent `token_ids_cpu`.

So the real value lives only transiently (GPU `input_ids` on the common path, one step in
the broadcast `recv`). `ngram_gpu` happens to populate via its own `is_ngram_gpu`-gated
lines + `update_ngram_gpu_tensors_incremental`; **MTP has no equivalent** — break #2 is
the first symptom of that missing value-population.

**The fix — C4:**
- **(A) value back-write** in the receiver (`gpu_model_runner.py:4694`): write `recv`'s
  confirmed value into `token_ids_cpu[i, pos]` instead of `-1`. (Necessary.)
- **(B) possibly** a single-site `num_tokens_no_spec` advance to `valid_count` — *or not*:
  branch-2 (`:1418`) already re-derives `num_tokens_no_spec` from the method-agnostically
  corrected `num_computed_tokens` (`:1487`), so the count may already self-correct for MTP.

**The one thing pure reading can't settle** is whether (B) is needed — the exact
`num_tokens_no_spec` trajectory across the k-step delay (branch-2's "advance to
num_computed" vs the receiver's "+1"). → **THE NEXT ACTION is one targeted instrumentation
run on MiMo** (env-gated, per non-last step log `{prev_index, num_computed_tokens,
num_tokens_no_spec pre/post branch-2 :1418 & receiver :4696, pos, recv row, valid_count}`),
which turns (A)-only vs (A+B) into data. Then implement C4 grounded, TDD a pure helper if
it factors cleanly (style: `vllm/v1/worker/pp_spec_broadcast.py:21`), MiMo greedy-equiv as
the oracle.

---

## 3. "If you touch X" — the impact map

| If you change… | …then re-check (it assumes the thing you changed) |
|---|---|
| **The broadcast width / `pp_spec_broadcast.py`** | the receiver's `recv` shape, the `valid_count` derivation, and the gloo CPU test (`tests/v1/spec_decode/test_pp_spec_broadcast.py`). Width must stay `num_spec+1`. |
| **The receiver `:4694` (the C4 fix)** | I3 (value now real), I2 (the `num_tokens_no_spec` count — does branch-2 double-count?), and `_prepare_input_ids` non-common path (it now reads a real value). |
| **The scheduler placeholder discipline (C3 / `async_scheduler.py:31`)** | I4, plus `test_async_scheduler.py` must stay green (it's a valid standalone PR — don't regress it for the worker-side fix). |
| **The accounting gates (`:1418` advance / `:1494` correct)** | I2 across MiMo (non-hybrid) **and** 27B (hybrid `:1513`) — they diverge here. The s5 lesson: editing these gate-by-gate moved the `-1` and over-advanced the count. Treat accounting **holistically**, never one gate at a time. |
| **The drafter forward flag (`qwen3_5_mtp.py`, Design C)** | only the draft side; this brick is design-independent — but confirm the standalone flag still fires under `draft_pp=1`. |
| **A1c draft-embed quant** | I1 is safe by construction (weight-agnostic oracle) — quant can only lower acceptance. Re-check the load-time peak (post-load swap OOMs), not correctness. |

---

## 4. The oracle & the green floor (don't regress these)

**The correctness oracle is always the same:** speculative output **token-identical** to
non-spec greedy of the same model. Never accept "it runs" as success. Ladder:
1. **gloo CPU** — `pp_spec_broadcast.py` round-trip (no GPU).
2. **Scheduler unit tests** — count accounting across the k-step delay (no GPU).
3. **MiMo-7B PP=2+MTP** on gpu-wb — fast real-weight oracle (non-hybrid; isolates the
   accounting cleanly). ~40 s loads.
4. **Qwen3.5-27B PP=2** vs `base.json` — final (hybrid; needs A1c int4 + `cpu_offload_gb=3`).

**Green floor — these are done and must stay green** (run before any GPU work):
```bash
VIRTUAL_ENV="$(pwd)/.venv" .venv/bin/python -m pytest \
  tests/v1/core/test_async_scheduler.py tests/v1/spec_decode/test_pp_spec_broadcast.py -q
# heavier: tests/v1/core/test_scheduler.py → expect 101 passed
```
- **A1c** (memory) — int4 draft embed, validated on 27B. Independent value.
- **B1a** (broadcast width) — `pp_spec_broadcast.py`, gloo-tested. Independent value.
- **C3** (scheduler placeholder discipline ≈ #40768) — `test_async_scheduler` green.
  Necessary-but-not-sufficient for break #2; ships standalone.

---

## 5. Why it can't "just be simpler"

The obvious simplification — *"have the non-last rank just recompute the token itself"* —
can't work: the non-last rank doesn't have the sampler or the final logits (those are on
the last rank, by the PP placement convention — `FOUNDATIONS.md` §4/§6). The token *must*
travel from the last rank, and under the pipeline it must survive the k-step delay. The
second simplification — *"make it synchronous so there's no placeholder dance"* — also
fails: sync MTP+PP **deadlocks** on current main (a rank waits on spec output that never
arrives; `PIPELINE-NARRATIVE.md` §Move 7). So the complexity is forced: the value must be
*broadcast* (not recomputed) and *persisted* (not transient), under *async* (not sync).
C4's value-back-write is the minimal honoring of that. The deeper cleanup — a typed
`SampledTokenGrid` that makes "valid-from-0, never embed a `-1`" un-representable-wrong —
is the C0/C1 contribution that would retire this whole bug class (`brick 81`).

---

**Next on the ladder:** you've now seen what must hold (here), why the system is shaped
this way (`PIPELINE-NARRATIVE.md`), where everything lives (`PIPELINE.md`), and what the
parts are (`FOUNDATIONS.md`). When you go to implement C4, start with the instrumentation
run in §2 — the docs can't tell you the count trajectory; only MiMo can.
