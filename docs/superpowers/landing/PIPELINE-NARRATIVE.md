# vLLM V1 — the pipeline as a story (why each piece exists)

> **How to read this.** `PIPELINE.md` is the **map** — what each component is and
> where it lives (`file:line`). This document is the **story** — *why* each
> component has to exist, what the obvious simpler thing would be, and exactly where
> that simpler thing breaks. Read it top to bottom; it's meant to be shown to a
> senior who has never opened the vLLM tree.
>
> The whole architecture is **one problem and a chain of forced moves**. Almost
> nothing here is gratuitous — each piece is the price of fixing the problem the
> previous piece created. The few places that *are* gratuitous are called out
> honestly at the end (Part 4), because the contrast is the point: once you can feel
> which complexity is forced and which is accreted, the codebase stops looking like
> a pile of parts.

---

## Part 0 — The one problem

> **Serve many concurrent LLM requests on scarce, expensive GPU memory — with low
> latency and high throughput — without ever corrupting the output.**

Hold those five constraints in your head: *many*, *scarce memory*, *low latency*,
*high throughput*, *exact correctness*. Every component below is a forced move from
the tension between them. We'll walk the chain: each move solves a pressure and
creates the next one.

```
autoregression → batch them → continuous batching → paged KV → chunked prefill
   → model too big → pipeline parallelism → pipeline bubble → batch_queue
   → decode is memory-bound → speculative decoding → spec × PP × async = our seam
```

---

## Part 1 — The forced-move chain

### Move 1 — From "one request at a time" to **continuous batching**

**The pressure.** A GPU is a throughput machine: it wants a big matrix multiply, not
a stream of tiny ones. But text is generated **autoregressively** — one token per
forward pass, each depending on the last. One request alone uses a sliver of the GPU.

**The naive move.** Collect a fixed batch of N requests, run them in lockstep until
all finish, then take the next N.

**Where it breaks.** Requests finish at *different* times and have *different*
lengths. A fixed batch idles every slot whose request already finished, and it can't
admit a newly arrived request until the whole batch drains. Latency and utilization
both collapse under real traffic.

**What vLLM does.** **Continuous batching**: there is no batch object at all. Every
step the scheduler re-derives the running set — finished requests drop out, waiting
ones join — from a *token budget*. The scheduler's own comment is the clearest line
in the codebase: *there is no prefill phase and no decode phase; each request just
has `num_computed_tokens` and `num_tokens_with_spec`, and each step assigns tokens so
the former catches up to the latter.* `vllm/v1/core/sched/scheduler.py:336`

**The new cost it creates.** If the batch is now fluid, the *inputs* to the model
(token ids, positions, block tables) change every step. Rebuilding those tensors from
scratch each step would burn CPU. → forces Move 5's **persistent batch**. And mixing
a 32k-token prompt with one-token decodes in the same budget would starve decode
latency → forces **chunked prefill** (next).

---

### Move 2 — From "contiguous KV buffer" to **PagedAttention**

**The pressure.** The KV cache (the attention keys/values for every past token) grows
by one slot per token per layer and quickly **dominates** GPU memory — far more than
the weights for long sequences.

**The naive move.** Give each request a contiguous KV buffer sized to `max_model_len`.

**Where it breaks.** You don't know a request's final length up front, so you reserve
the maximum — and most requests are short, so you waste enormous amounts of reserved
memory. Worse, contiguous per-request buffers can't **share** anything: two requests
with the same system prompt each pay for it. Memory fragmentation finishes the job.

**What vLLM does.** **PagedAttention** — borrow the OS virtual-memory trick. KV lives
in fixed-size **blocks**; a per-request **block table** maps logical block → physical
block id, and a per-token **slot mapping** gives each token its physical slot. Blocks
are allocated on demand and reference-counted. `vllm/v1/worker/block_table.py:70`

**The new cost it creates.** Now block allocation can *fail* mid-step (no free
blocks) — which means the scheduler needs a way to **preempt** a running request and
reclaim its blocks (the `allocate_slots → None → preempt` path). And shared blocks
need a hash index to be discoverable → **prefix caching**. `vllm/v1/core/kv_cache_manager.py:238`

---

### Move 3 — From "prefill = decode" to **chunked prefill + prefix caching**

**The pressure.** Prompts can be tens of thousands of tokens; a decode step is one
token. They share the same forward pass and the same token budget.

**The naive move.** Whenever a prompt arrives, prefill all of it in one step.

**Where it breaks.** A single 32k-token prefill monopolizes the step, and every other
request's decode stalls behind it — a latency cliff. And re-prefilling a prompt prefix
you've already computed (same system prompt, retries) wastes the most expensive part.

**What vLLM does.** Because the scheduler thinks only in "make `num_computed_tokens`
catch up," **chunked prefill** is almost free: it just clamps `num_new_tokens` to the
remaining token budget, so a long prompt is consumed across several steps while
decodes keep flowing. And **prefix caching** looks up the longest already-cached block
prefix by hash and skips recomputing it (recomputing only the last token, to produce
logits). `vllm/v1/core/sched/scheduler.py:~675`, `vllm/v1/core/kv_cache_manager.py:196`

**The new cost it creates.** None structural — this is where the "no phases" design
pays off: chunked prefill is a consequence of the budget model, not a new subsystem.
The bill comes due elsewhere: the model still has to *fit*.

---

### Move 4 — From "one GPU" to **pipeline parallelism (PP)** (not TP, and that matters)

**The pressure.** The model doesn't fit on one GPU. Our hardware is two consumer GPUs
on **PCIe without NVLink** — a slow link.

**The naive move.** Tensor parallelism (TP): split every layer's weights across both
GPUs and combine partial results.

**Where it breaks.** TP all-reduces activations **every single layer**. Over a fast
NVLink that's fine; over PCIe it's death — the link saturates and the GPUs spend their
time waiting on each other, not computing.

**What vLLM does.** **Pipeline parallelism**: split the model's *layers* into
contiguous stages (rank0 = layers 0..k, rank1 = layers k..L). What crosses a stage
boundary is tiny — a per-token `{hidden_states, residual}` dict (`IntermediateTensors`),
sent point-to-point once per boundary, not a per-layer all-reduce. That's exactly why
PP, not TP, is the right tool for the no-NVLink pair. `vllm/distributed/utils.py:109`,
`vllm/sequence.py:12`

**The new cost it creates.** A pipeline has **bubbles**: while rank0 computes stage 1
of batch B, rank1 has nothing to do until B's activations arrive — and vice versa. A
naive loop would leave each GPU idle half the time. → forces Move 5.

> **Aside — the placement convention that the spec work fights.** Under PP the
> embedding lives on the *first* rank and `norm`/`lm_head` on the *last*; absent
> layers are `PPMissingLayer` no-ops. The canonical forward is `if first: embed; elif
> not last: read intermediate tensors; ... ; if last: norm`. The MTP drafter runs on
> the *last* rank, where `is_first_rank` is False — so without a "standalone draft"
> flag it would take the read-intermediate-tensors branch and never embed. That one
> flag is the entire model-side change of Design C. (Bricks 10/20/60.)

---

### Move 5 — From "synchronous step" to **batch_queue** (and the persistent batch)

**The pressure.** PP bubbles. You're paying for N GPUs and, with a synchronous loop,
using ~1/N of them.

**The naive move.** Keep the simple loop: schedule → run → apply → repeat. (This is
exactly what `step()` does, and it's correct — `vllm/v1/engine/core.py:443`.)

**Where it breaks.** The simple loop waits for batch B to finish before scheduling the
next batch, so only one pipeline stage is ever busy. It throws away the whole point of
PP.

**What vLLM does.** **`step_with_batch_queue`**: keep `pp_size` batches in flight in a
`deque(maxlen=pp_size)`. Schedule step N, launch it non-blocking, append it, and
**return without waiting** while there's room; once full, pop the *oldest* and apply
it. Now every stage always has work. `vllm/v1/engine/core.py:484`,
`vllm/config/vllm.py:497`

**The new cost it creates — two of them, and both are load-bearing for us:**
1. **The k-step delay.** You only apply batch B's output ≈ `pp_size−1` steps after you
   scheduled it. The `SchedulerOutput` you're holding is a **stale snapshot** — spec
   tokens, placeholders, and prefill flags may have mutated in between. Any per-step
   state now has to survive that delay correctly.
2. **The persistent batch.** Because inputs change every step (Move 1) but we can't
   afford to rebuild tensors, `InputBatch` keeps CPU buffers alive across steps and
   `_update_states` reconciles them, with `condense()` compacting after removals. That
   compaction then has to rebuild index maps — and under the k-step delay, that's a
   stale-mapping hazard (weak spot W6). `vllm/v1/worker/gpu_input_batch.py:91`,
   `:683`

---

### Move 6 — From "one token per forward" to **speculative decoding**

**The pressure.** Decode is **memory-bound**: each step streams the whole KV cache and
all weights through the GPU to produce *one* token. The compute units are mostly idle —
the GPU could verify several tokens for nearly the same memory traffic.

**The naive move.** Just generate more tokens per step. But you can't — token N+1
depends on token N, which you haven't produced yet.

**Where the *obvious* speculative idea breaks.** Have a cheap "draft" model guess the
next K tokens, then have the target check them. The danger: if you simply *accept* the
draft's guesses, you've changed the output distribution — silent quality regression
(this is the gibberish-bug class). Correctness is not free.

**What vLLM does.** Speculative decoding with a **rejection sampler as the correctness
oracle**. The drafter (ngram / EAGLE / **MTP** / draft-model) proposes K tokens; the
target verifies them in **one** forward. In greedy mode the rule is exact: a draft
token is accepted *iff* it equals the target's `argmax`, otherwise it's replaced by
the target argmax and the rest are rejected. So the output is a **pure function of
(target argmax, draft sequence)** — provably identical to non-speculative greedy,
**independent of how good the draft is**. The draft only changes the *acceptance
rate*, never the result. `vllm/v1/sample/rejection_sampler.py:708`, `:452`

> **The elegant consequence (and why A1c is safe).** Because correctness is
> weight-agnostic, you're *allowed* to make the draft lossy. That's the license behind
> A1c: quantizing the draft's vocab embedding to int4 can only lower acceptance, never
> corrupt output. It also means dummy/MiMo weights are valid oracles for testing the
> machinery — you don't need the real 27B to prove the cascade is correct. (Brick 70.)

**The new cost it creates.** The sampler now emits a **variable-width grid**
`[num_reqs, num_spec+1]` (accepted drafts + a free "bonus" token if all K accept),
padded with `-1`. That `-1` sentinel, and the variable accepted-count, now have to
flow correctly through everything else — including the k-step PP delay. → Move 7.

---

### Move 7 — Composing spec × PP × async (this is our seam)

**The pressure.** All three optimizations at once: the drafter runs **only on the last
PP stage** (where the target's hidden state is resident — zero extra PCIe), but the
*earlier* stages need the sampled tokens to build their next input. And under the
batch_queue, all of this is delayed by k steps.

**The naive move.** Broadcast the last rank's sampled tokens to the other ranks; let
the scheduler optimistically reserve spec slots with `-1` placeholders so it can keep
scheduling ahead.

**Where it breaks.** Two places, precisely:
1. The broadcast was hard-coded to width `[num_reqs, 1]` — fine without spec, wrong for
   MTP's `[num_reqs, num_spec+1]`. (Fixed: **B1a**, the new width-agnostic
   `vllm/v1/worker/pp_spec_broadcast.py`.)
2. The `-1` placeholder is filled in only on the "common" code path. When a request
   goes "non-common" (no drafts scheduled that step), the non-last rank reads a `-1`
   straight into an embedding lookup → out-of-bounds crash (**break #2**). Root cause =
   upstream **PR #40768** ("stale async placeholder tokens", fixes #37159), a
   scheduler-side discipline: emit `-1` only when the request was actually scheduled
   last step.

**What this tells you about the design.** The spec-under-PP path works for `ngram_gpu`
(its accounting gates are `is_ngram_gpu`) but was never finished for MTP — the
abstraction never generalized past the one proposer it was written for (weak spot W5).
That's the whole task: make this seam correct and typed for MTP. (Bricks 40/80/81.)

---

## Part 2 — The cross-cutting moves (not on the spine, but forced)

**Why three process tiers, not one big program?** Three independent pressures:
- **Isolation** — a CUDA OOM in the model must not take down the API server. So the
  engine logic runs in its own process (`EngineCoreProc`), behind ZMQ.
- **Overlap** — serialization and socket I/O must not block the hot loop. So input and
  output each get a daemon thread, and the busy loop only ever does `put_nowait`.
- **Distribution** — data-parallel replicas need a coordinator. So there's a DP broker
  process.

The clean part: the *logic* (`EngineCore`) is separable from the *transport*.
`InprocClient` runs the exact same logic in-process for offline use; the MP clients put
it behind sockets. The only tax is that "the engine" is spread across three class names
(`EngineCore` / `EngineCoreProc` / `EngineCoreClient`). `vllm/v1/engine/core.py:95`

**Why "async" three times?** `AsyncLLM` (asyncio frontend), `async_scheduling` (engine
overlap of schedule and execute via `-1` placeholders), and `AsyncGPUModelRunnerOutput`
(overlapping the device→host copy) are three *independent* things that happen to share
the word. None of them is the others. This is a naming hazard, not a design one — but
it's the single most common way to misread the code.

**Why static types matter here.** vLLM runs mypy in CI, so `Final` / `Protocol` /
`TypedDict` / `NewType` are the only "build-time" contract the language offers. The
spec-under-PP area broke for MTP *partly because it lacked them*: the `-1` sentinel is
a bare literal reinvented in three places, the `[num_reqs, num_spec+1]` grid has its
shape and layout only in comments, and the proposer surface is `isinstance` chains
instead of a `Protocol`. (Brick 81.)

---

## Part 3 — Where the complexity is **not** forced (the honest part)

This is the contrast that makes the rest credible. Not everything is a forced move;
some of it is accreted. Naming it is how you keep the narrative honest.

- **`gpu_model_runner.py` is a 7,583-line god object (W1).** Forward, attention,
  sampling, spec, PP, multimodal, pooling, cudagraphs, profiling — all one class, with
  `is_last_rank` / `is_ngram_gpu` / `use_async_spec_decode` branches interleaved. This
  is *not* a forced move; it's what happens when every feature lands in the same file.
  A clean design extracts `SpecDecodeFlow` / `SamplingFlow` / `PPTransport` behind typed
  interfaces. `vllm/v1/worker/gpu_model_runner.py:422`
- **No owner for the request state machine (W2).** `request.status = …` and the token
  counters are mutated from many scheduler sites; the FSM is implicit, and there isn't
  even an `assert num_rejected ≤ num_computed_tokens`. A clean design has one owner with
  guarded transitions.
- **The `-1` sentinel everywhere (W3).** Emitted by both the scheduler and the runner,
  backfilled only on the common path → the break-#2 leak. This is the single
  highest-leverage cleanup, and it's the same gap that caused the typing weaknesses: a
  typed `SampledTokenGrid` would close both.
- **Two draft-token paths with no shared abstraction (W4).** Sync pulls via
  `post_step`; async injects in the worker. For MTP+PP the sync path *deadlocks* and the
  async path *crashes* — strong evidence the feature was never exercised for MTP+PP.

The pattern: the **forced** complexity (paging, continuous batching, PP, batch_queue,
the rejection oracle) is essential and elegant. The **accreted** complexity (the god
object, the implicit FSM, the scattered `-1`) is exactly where the bugs live — and
exactly what our contribution (A1c, B1a, then the typed C0/C1 contract) is hardening.

---

## Part 4 — Our change, told as part of the story

The path we touch is thin but sits on top of all six prior moves. Walk it once more,
fast: a request is **batched** (1), its KV is **paged** (2), its long prompt is
**chunked** (3), the model is **split across PP stages** (4), the engine **pipelines**
it with a k-step delay (5), and **speculative decoding** drafts K tokens that the
rejection oracle verifies (6). Our seam (7) is the place where the spec state from move
6 has to survive the delay from move 5 across the stage boundary from move 4.

Concretely: the **memory wall is solved** (A1c — load-time int4 draft embed, licensed by
the weight-agnostic oracle), the **transport is fixed** (B1a — width-agnostic
broadcast), and the **remaining gate** is break #2 = upstream #40768, with the deeper
cleanup being a typed contract for the spec-token grid (C0/C1) that would retire weak
spots W3/W4/W5 at once. The design-side change is tiny (one standalone-draft flag); the
bulk and the risk are this correctness plumbing — which is why understanding the
*forced* shape of the pipeline matters before changing it. (Bricks 40/60/70/80/81;
README + 00-map for current state.)

---

## Appendix — the understandability audit (how this doc was scored)

This narrative was written to fix a measured gap in `PIPELINE.md`. Scoring its sections
on four axes (1–5): **What/where**, **Why it exists**, **Why not simpler**, **Narrative
flow**.

| Reference §  (PIPELINE.md) | What/where | Why | Why-not-simpler | Narrative |
|---|:--:|:--:|:--:|:--:|
| §0 shape · §1 process model | 5 | 3 | 1–2 | 3 |
| §2 lifecycle / FSM | 5 | 2 | 1 | 3 |
| §3 scheduler | 5 | 3 | 2 | 3 |
| §4 engine modes / batch_queue | 5 | 3 | 2 | 3 |
| §5 executor / persistent batch | 5 | 3 | 2 | 3 |
| §6 distributed / PP | 5 | 4 | 3 | 3 |
| §7 KV / attention | 5 | 3 | 2 | 3 |
| §8 spec decode | 5 | 4 | 2 | 3 |
| §9 typing | 4 | 3 | 3 | 3 |
| §10 glossary | 5 | 4 | 2 | 2 |
| §11 weak spots | 5 | 4 | **5** | 3 |
| §12 our change | 5 | 4 | 3 | 3 |

**Finding.** The reference is excellent on *what/where* (≈4.9) and weak on
*why-not-simpler* (≈2.2) and *narrative* (≈2.8). The "weak spots" section already scored
5 on why-not-simpler — because its "clean design" column *is* the rejected-simpler
alternative — so it reads best. **This document is the why-not-simpler and narrative
layer the reference lacked**, built by turning every component into a forced move
(pressure → naive → break → fix → new cost). Use the two together: this for the first
read, `PIPELINE.md` for the precise `file:line` when you go to change something.
