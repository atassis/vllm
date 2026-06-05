# The vLLM V1 Execution Pipeline — End to End

> **Audience.** An engineer who wants to *really* understand how a request flows
> through vLLM V1: every stage, every component, the exact terminology (and where
> vLLM's naming diverges from the rest of the world), what pipeline parallelism (PP)
> is and how speculative decoding composes with it — and, honestly, where the
> implementation is architecturally weak, over-coupled, or confusingly named.
>
> **Scope.** vLLM **V1** (the default runner for our models). V2 is mentioned only
> where it differs. This report **extends** the existing code-referenced knowledge
> base under `docs/superpowers/research/pp-mtp/` (bricks 10–81) from "spec-under-PP"
> to "the whole V1 pipeline." It does not repeat the bricks; it cites them.
>
> **Provenance / how to trust this.** Every architectural claim carries a
> `file:line`. Line numbers are **anchors** from the working tree at HEAD
> `e45e5d462` (branch `feat/pp-mtp-spec-decode`) — `vllm/v1/worker/gpu_model_runner.py`
> and `vllm/v1/core/sched/scheduler.py` carry **uncommitted spec-under-PP changes**,
> so a handful of numbers there are from the modified tree. Re-grep the symbol if a
> line has drifted. Facts without a `file:line` are flagged `HYPOTHESIS`. The
> headline references in §0, §1, §4, §8, §9 were re-verified directly against the
> tree; the broader maps were produced by parallel subsystem reads and are accurate
> to within a few lines.

---

## 0. The shape of the whole thing (one screen)

A request's life, top to bottom:

```
                    ┌─────────────────── FRONTEND PROCESS ───────────────────┐
  HTTP / Python →   │  AsyncLLM (async_llm.py:70)  /  LLMEngine (47)         │
                    │    Processor  → EngineCoreRequest                       │
                    │    OutputProcessor ← EngineCoreOutputs                  │
                    │      └ IncrementalDetokenizer + LogprobsProcessor       │
                    │    EngineCoreClient  (core_client.py)  ── ZMQ ──┐       │
                    └─────────────────────────────────────────────────┼──────┘
                                                                       │ msgpack
                    ┌──────────────── ENGINECORE PROCESS ──────────────┼──────┐
                    │  EngineCoreProc (core.py:858)                     ▼      │
                    │   input thread → input_queue → run_busy_loop (1216)      │
                    │       │                                                  │
                    │       ▼  step_fn() each iteration (217)                  │
                    │   ┌── EngineCore.step / step_with_batch_queue ──┐        │
                    │   │  Scheduler.schedule()  (sched/scheduler.py)  │        │
                    │   │     → SchedulerOutput                        │        │
                    │   │  Executor.execute_model(non_block=True)      │        │
                    │   │     → Future[ModelRunnerOutput]              │        │
                    │   │  Scheduler.update_from_output()              │        │
                    │   └──────────────────────────────────────────────┘       │
                    │   post_step()  (sync spec draft-token pull, 474)         │
                    │   outputs → output_queue → output thread → ZMQ           │
                    └──────────────────────┬───────────────────────────────────┘
                                           │ collective_rpc (executor)
                    ┌──────────────────────▼─ WORKER PROCESS(es), 1 per rank ─┐
                    │  Worker (gpu_worker.py)                                  │
                    │    GPUModelRunner (gpu_model_runner.py:422)              │
                    │      _update_states → _prepare_inputs → model.forward    │
                    │      → sample → (rejection oracle) → ModelRunnerOutput   │
                    │      PP: IntermediateTensors send/recv between stages    │
                    └──────────────────────────────────────────────────────────┘
```

Three process tiers, connected by ZMQ (frontend↔core) and by the executor's RPC
(core↔workers). The engine loop is a **synchronous busy loop**, not asyncio — the
only asyncio is in the frontend `AsyncLLM`. Everything below is an expansion of this
picture.

---

## 1. Top architecture & the process/thread model

### 1.1 The component map

| Component | `file:line` | Role |
|---|---|---|
| `LLMEngine` | `vllm/v1/engine/llm_engine.py:47` | Synchronous in-process facade (`llm.generate(...)`, offline batch). |
| `AsyncLLM` | `vllm/v1/engine/async_llm.py:70` (subclasses `EngineClient`) | Asyncio facade for the API server; owns a background `output_handler` task. |
| `EngineCore` | `vllm/v1/engine/core.py:95` | The engine **logic**: scheduler + executor + KV + structured-output. No I/O. |
| `EngineCoreProc` | `vllm/v1/engine/core.py:858` (subclass) | `EngineCore` wrapped for a **separate process**: ZMQ sockets + 2 daemon I/O threads + the busy loop. |
| `DPEngineCoreProc` | `vllm/v1/engine/core.py:1676` | Data-parallel variant: wave coordination + all-reduce barriers across DP ranks. |
| `EngineCoreClient` | `vllm/v1/engine/core_client.py` | Abstract client facade. Concrete: `InprocClient` (no MP), `SyncMPClient`, `AsyncMPClient`, DP variants. |
| `Processor` | `vllm/v1/engine/input_processor.py` | Raw prompt → tokenize/multimodal → `EngineCoreRequest`. |
| `OutputProcessor` | `vllm/v1/engine/output_processor.py` | `EngineCoreOutputs` → `RequestOutput`; per-request streaming state. |
| `IncrementalDetokenizer` | `vllm/v1/engine/detokenizer.py` | Stateful token→text, stop-string detection. |
| `LogprobsProcessor` | `vllm/v1/engine/logprobs.py` | Accumulates/format logprobs. |
| `DPCoordinator` | `vllm/v1/engine/coordinator.py:23` | Frontend-side DP proxy; spawns a broker process for wave/load coordination. |

### 1.2 Process and thread boundaries

- **Frontend process.** `AsyncLLM`/`LLMEngine` + `Processor` + `OutputProcessor` +
  the `EngineCoreClient`. In `AsyncLLM` the loop is asyncio; the output side runs
  as a background asyncio task `_run_output_handler` (`async_llm.py`). In
  `SyncMPClient` the output side is a **daemon thread** polling the ZMQ socket
  (`core_client.py`).
- **EngineCore process** (`EngineCoreProc`). Spawns **two daemon threads** —
  `process_input_sockets` (ZMQ → `input_queue`) and `process_output_sockets`
  (`output_queue` → ZMQ) — and runs `run_busy_loop` on the main thread
  (`core.py:1216`). The busy loop is `while ...: _process_input_queue();
  _process_engine_step()`. Outputs are handed off with `output_queue.put_nowait(...)`
  (`core.py:~1264`), so the loop **never blocks on output serialization**.
- **Worker process(es)**, one per global rank under `MultiprocExecutor`. Each runs a
  `Worker` wrapping a `GPUModelRunner`; the executor dispatches `execute_model` via
  `collective_rpc` over message queues.

> The split between **logic** (`EngineCore`) and **transport** (`EngineCoreProc` +
> `EngineCoreClient`) is the cleanest seam in the system: `InprocClient` runs
> `EngineCore` directly in-process for offline use, while MP clients put it behind
> ZMQ. The cost is that the same concept ("the engine") is spread across three class
> names.

### 1.3 What crosses the client↔core boundary

`EngineCoreRequest` (`vllm/v1/engine/__init__.py:~83`) goes client→core;
`EngineCoreOutputs` / `EngineCoreOutput` (`__init__.py:~170`/`~215`) go core→client.
Both are `msgspec.Struct` with `array_like=True` for compact positional msgpack. The
request type is tagged by `EngineCoreRequestType` (`ADD`/`ABORT`/`UTILITY`/
`START_DP_WAVE`/...). See brick 81 for the typing analysis; §9 here for the gap that
`array_like=True` makes field-order a fragile wire contract.

---

## 2. The request lifecycle & state machine

Primary file: `vllm/v1/request.py`.

### 2.1 The `Request` object and its token accounting

`Request` (a plain class, not a `@dataclass`) carries the per-request state. The
token-accounting fields are the heart of the scheduler contract:

| Field / property | `file:line` | Meaning |
|---|---|---|
| `prompt_token_ids` / `num_prompt_tokens` | `request.py:~130` | Input tokens; immutable. |
| `_output_token_ids` → `output_token_ids` | `request.py:~133`, append at `~226` | Generated tokens (decode phase), exposed read-only via `ConstantList`. |
| `_all_token_ids` → `all_token_ids` | `request.py:~134` | prompt + output, kept in sync by `append_output_token_ids()`. |
| `num_tokens` (property) | `request.py:249` | `len(_all_token_ids)` — **confirmed** sequence length. |
| `num_tokens_with_spec` (property) | `request.py:253` | `num_tokens + len(spec_token_ids)` — **optimistic** length if all drafts are accepted. |
| `spec_token_ids` | `request.py:~148` | Candidate draft tokens (or `[-1]*k` placeholders under async). Consumed once per step. |
| `num_computed_tokens` | `request.py:~149` | How many tokens the **scheduler has decided to compute**. Rolls **back** on spec rejection; resets to 0 on preemption. |
| `num_output_placeholders` | `request.py:141` | **Async scheduling only.** Output tokens promised but not yet materialized. |

The crucial, non-obvious relationship (see the glossary, §10): `num_computed_tokens`
is a **scheduler decision counter**, not a worker fact. It can exceed `num_tokens`
(spec tokens optimistically scheduled) or fall short (after a rejection rollback).

### 2.2 `RequestStatus` and the transitions

`RequestStatus(enum.IntEnum)` at `request.py:329`. States:

- `WAITING` (332), plus blocked-waiting variants for structured-output grammar,
  remote KV load, and streaming (`WAITING_FOR_*`, 333–335 region).
- `RUNNING` (336), `PREEMPTED` (337).
- Terminal (`> PREEMPTED`, see `is_finished` at `request.py:~351`): `FINISHED_STOPPED`
  (340), `FINISHED_LENGTH_CAPPED` (341), `FINISHED_ABORTED` (342),
  `FINISHED_IGNORED` (343), `FINISHED_ERROR` (344), `FINISHED_REPETITION` (345).
- `_FINISHED_REASON_MAP` (`request.py:363`) maps each terminal status to a
  `FinishReason` (`STOP`/`LENGTH`/`ABORT`/`ERROR`/`REPETITION`).

Who drives each edge (all in `sched/scheduler.py`):

- **WAITING → RUNNING**: `schedule()` admits the request once a token budget + KV
  blocks are available (status set ~`scheduler.py:831`).
- **RUNNING → PREEMPTED**: `_preempt_request()` (~`scheduler.py:960-977`) when
  `allocate_slots()` returns `None`; frees blocks, `num_computed_tokens = 0`, clears
  `spec_token_ids`, pushes back to the front of `waiting`.
- **RUNNING → FINISHED_***: stop checks in `sched/utils.py` (`check_stop`) and
  `update_from_output`; abort via `finish_requests()`.
- **WAITING → WAITING_FOR_REMOTE_KVS → (WAITING|PREEMPTED)**: async KV-connector
  load path.

The transitions are **not** guarded by a single state-machine object — each edge is a
bare `request.status = ...` assignment scattered across the scheduler (§8, weak spot
W2).

---

## 3. The scheduler & continuous batching

Files: `vllm/v1/core/sched/{scheduler,async_scheduler,output}.py`,
`vllm/v1/core/{kv_cache_manager,block_pool,kv_cache_coordinator}.py`,
`vllm/v1/kv_cache_interface.py`. (Note: `scheduler.py` is ~2,370 lines / 109 KB —
"big" but not the runner's 7,583.)

### 3.1 There is no "prefill phase" — only a token-budget catch-up loop

The scheduler's own comment (`scheduler.py:~338`) states the model: *there is no
decode vs prefill phase; each request just has `num_computed_tokens` and
`num_tokens_with_spec`, and each step assigns tokens so `num_computed_tokens` catches
up.* `schedule()` (`scheduler.py:336`):

1. `token_budget = max_num_scheduled_tokens` (the per-step cap).
2. **Running queue first** (`~372-513`): for each running request compute
   `num_new_tokens = num_tokens_with_spec + num_output_placeholders -
   num_computed_tokens`, clamp to budget / chunked-prefill threshold, call
   `kv_cache_manager.allocate_slots(...)`. If that returns `None`, **preempt** the
   lowest-priority (or last) running request and retry.
3. **Waiting queue second** (`~558-851`): prefix-cache lookup, optional remote-KV
   load, chunked-prefill token calc, `allocate_slots`, admit → `RUNNING`.
4. Build `SchedulerOutput`; `_update_after_schedule()` advances
   `num_computed_tokens += num_scheduled_tokens` and sets `is_prefill_chunk`.

**Continuous batching** falls out of this: there is no fixed batch — every step
re-derives the running set, and finished requests drop out while waiting ones join.

**Chunked prefill** is just the budget clamp: a long prompt's `num_new_tokens` is
capped to the remaining `token_budget` (and `long_prefill_token_threshold`), so a
prompt is consumed across several steps as `num_computed_tokens` advances.

### 3.2 Prefix caching

`kv_cache_manager.get_computed_blocks()` (`kv_cache_manager.py:~196`) hashes the
request's block sequence (`Request.block_hashes`) and calls
`coordinator.find_longest_cache_hit(...)`, returning the longest cached prefix and the
token count it covers. The hash table is `BlockHashToBlockMap` in
`block_pool.py:~34`. Subtlety: the lookup caps at `num_tokens - 1` so the **last
token is always recomputed** to produce logits (`kv_cache_manager.py:~221`).

### 3.3 KV-cache manager & block pool

`BlockPool` (`block_pool.py:~130`) owns all `num_gpu_blocks` `KVCacheBlock`s, a
`FreeKVCacheBlockQueue` (eviction order), and the prefix-cache hash map. Per request,
`SingleTypeKVCacheManager.req_to_blocks[req_id]` is an append-only block list.
`allocate_slots()` (`kv_cache_manager.py:238`) is the gatekeeper: it frees
out-of-window blocks (sliding window), checks free-block availability (returns `None`
→ triggers preemption), allocates, and caches full blocks. Blocks are
reference-counted; a block returns to the free queue only at `ref_cnt == 0`. Hybrid
models use multiple `kv_cache_groups` (see §7).

### 3.4 Scheduler vs AsyncScheduler — the key difference

`AsyncScheduler(Scheduler)` (`async_scheduler.py:12`) overrides post-schedule
bookkeeping so the engine can **schedule step N+1 before step N's output is back**:

- `_update_after_schedule`: for each non-prefill request,
  `num_output_placeholders += 1 + cur_num_spec_tokens` (`async_scheduler.py:31`) and
  `request.spec_token_ids = self._spec_token_placeholders` (a shared `[-1]*num_spec`
  list, `async_scheduler.py:~16`). **This `-1` is the origin of the placeholder that
  the spec-under-PP path must backfill** (see §8).
- under PP it also sets `next_decode_eligible_step = current_step + pp_size`
  (`async_scheduler.py:45`) — throttling a request's next decode to the pipeline
  cadence.
- `_update_request_with_output`: `num_output_placeholders -= len(new_token_ids)`;
  `assert num_output_placeholders >= 0` (`async_scheduler.py:63-64`).
- `prev_step_scheduled_req_ids` tracks previous-batch membership — the exact signal
  upstream PR #40768 uses to decide when emitting a `-1` placeholder is safe.

### 3.5 The `SchedulerOutput` contract (scheduler → executor, per step)

`SchedulerOutput` (`sched/output.py:180`) ships: `scheduled_new_reqs:
list[NewRequestData]` (full state, once per request, `output.py:31`),
`scheduled_cached_reqs: CachedRequestData` (deltas every step, `output.py:111`),
`num_scheduled_tokens: dict[str,int]`, `total_num_scheduled_tokens`,
`scheduled_spec_decode_tokens: dict[str, list[int]]` (the draft tokens),
`num_common_prefix_blocks` (cascade-attention), `finished_req_ids`,
`preempted_req_ids`. Note `scheduled_spec_decode_tokens` uses **implicit "missing key =
no drafts"** (§9, typing gap).

---

## 4. The engine loop & batching modes

File: `vllm/v1/engine/core.py`. This is where PP's pipelining actually happens.

### 4.1 step_fn is chosen once

`self.step_fn = self.step if self.batch_queue is None else self.step_with_batch_queue`
(`core.py:217`). The `batch_queue` exists iff `max_concurrent_batches > 1`
(`core.py:192-198`, a `deque(maxlen=...)`).
`max_concurrent_batches` (`config/vllm.py:497-507`) returns **`pp_size`** in the base
case, `pp_size+1` for async+V2, `2` for async at pp≤1. So: **PP>1 ⇒ batch_queue is
on**, even without async. The infamous comment lives here:
`config/vllm.py:504` — *"V1 Model Runner does not fully support async scheduling with
PP."*

### 4.2 `step()` — the simple synchronous path

`core.py:443`: `schedule()` → `execute_model(non_block=True)` → **`future.result()`
blocks** (~`core.py:461`) → `update_from_output()`. One batch in flight; the engine
thread blocks on the GPU.

### 4.3 `step_with_batch_queue()` — the pipelined path

`core.py:484`. Per iteration:

1. If the deque isn't full, `schedule()` a **new** batch, launch
   `execute_model(non_block=True)`, `appendleft((future, scheduler_output,
   exec_future))`, and **early-return `(None, executed)`** — *without waiting* — as
   long as there is more work and the queue has room (`core.py:~541`).
2. Once full (or no new work), `batch_queue.pop()` the **oldest** entry and block on
   its `future.result()` (`core.py:~555`), then `update_from_output()`.

Because the queue holds up to `pp_size` in-flight batches and you only apply the
oldest, **`update_from_output` lags scheduling by ≈ `pp_size − 1` steps**. That delay
is the whole point — it keeps all PP stages busy (fills the pipeline). The
`SchedulerOutput` captured at step N is a **snapshot**; `spec_token_ids`,
`num_output_placeholders`, `is_prefill_chunk` may all mutate before the matching
output is applied k steps later. That staleness window is exactly where spec-under-PP
correctness lives (brick 40).

There is also a `deferred_scheduler_output` branch for **structured output** (defer
sampling until after applying the previous output, `core.py:~574-596`).

### 4.4 `post_step()` — the SYNC spec draft-token path

`core.py:474`:

```python
def post_step(self, model_executed):
    if not self.async_scheduling and self.use_spec_decode and model_executed:
        draft_token_ids = self.model_executor.take_draft_token_ids()
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids(draft_token_ids)
```

Called every iteration after `step_fn` (`core.py:~1264`). This is the **sync** spec
plumbing: pull the drafter's output from the executor and hand it to the scheduler.
**Under async it is a no-op** — the worker injects draft tokens directly into the
input batch instead. The two paths share no abstraction (§8, W4). Brick 40 §"Session
5" records that the sync path **deadlocks** for MTP+PP on current main (a rank waits
on spec output that never arrives), while the async path **crashes** (the `-1`
placeholder leak) — i.e. *both* modes are unfinished for MTP+PP.

### 4.5 Output path & busy loop

Outputs go to `output_queue` via `put_nowait`; a daemon `process_output_sockets`
thread serializes and ZMQ-sends them, so the loop never blocks on I/O. `run_busy_loop`
(`core.py:1216`; DP override `core.py:1844`) is the synchronous
`_process_input_queue(); _process_engine_step()` cycle.

---

## 5. Executor & worker

Files: `vllm/v1/executor/{abstract,uniproc_executor,multiproc_executor,ray_executor}.py`;
`vllm/v1/worker/{worker_base,gpu_worker,gpu_model_runner,gpu_input_batch}.py`.

### 5.1 Executors

`Executor(ABC)` (`abstract.py:~37`) exposes `collective_rpc(method, args, ...,
non_block=...)` — broadcast a call to all workers, collect results. `execute_model`
is just `collective_rpc("execute_model", ...)`. Concrete:

- `UniProcExecutor` (`uniproc_executor.py:~45`): one in-process worker, direct calls.
- `MultiprocExecutor` (`multiproc_executor.py:~103`, `supports_pp=True`): one worker
  **process per rank**; RPC via shared-memory `MessageQueue`s (broadcast in,
  per-rank responses out); a monitor thread watches worker liveness.
- `RayDistributedExecutor` (`ray_executor.py:~64`): Ray actors per rank, optionally a
  **compiled DAG** wiring TP groups within each PP stage and passing
  `IntermediateTensors` between stages.

### 5.2 Worker

`WorkerBase` (`worker_base.py:~39`) / `WorkerWrapperBase` (`~187`, lazily instantiates
the concrete worker and delegates via `__getattr__`). `gpu_worker.Worker`
(`gpu_worker.py:~112`): `init_device` (set CUDA device, init distributed),
`load_model`, **`determine_available_memory`** (the KV-profiling dummy run that
measures peak activation memory, then sizes `num_gpu_blocks`, `gpu_worker.py:~360`),
`execute_model`. The PP `IntermediateTensors` recv/send happens on the worker around
the runner forward.

### 5.3 GPUModelRunner — the god object

`GPUModelRunner` (`gpu_model_runner.py:422`, **~7,583 lines**) does *everything*:
persistent-batch reconciliation, input prep, attention metadata, forward, sampling,
spec proposing, PP broadcast, multimodal, pooling, CUDA graphs, profiling. Rough map
(line ranges are approximate, modified tree):

| Section | ~lines | Key methods |
|---|---|---|
| init / async output wrappers | 243–913 | `AsyncGPUModelRunnerOutput`, `ExecuteModelState` |
| **persistent-batch reconcile** | 1132–1559 | `_update_states`, `_update_states_after_model_execute` |
| input prep | 1560–2505 | `_prepare_inputs`, `_prepare_input_ids` (1708), `_build_attention_metadata` |
| forward orchestration | 3419–3800 | `_preprocess`, `_model_forward` |
| **execute_model** | 4010–4388 | the main forward+PP send |
| **sample_tokens** | 4389–4649 | sampler + rejection oracle + draft propose |
| **PP spec transport** | 4650–4699 | `_pp_broadcast_prev_sampled_token_ids`, `_pp_receive_*` |
| spec propose | 4701–5083 | `propose_draft_token_ids`, `take_draft_token_ids` |
| load / profile | 5096–6469 | `load_model`, `profile_run` |

The drafter is created **only on the last PP rank** (`gpu_model_runner.py:~542`),
else `self.drafter = None` — the source of session-3's five non-last-rank
`AttributeError`s (README §SESSION 3). The split of execute into
**`execute_model` (forward, all ranks) + `sample_tokens` (sample, last rank)** exists
to support Ray's compiled DAG and to overlap sampling with the next stage's prep.

### 5.4 The persistent batch (`InputBatch` / `CachedRequestState`)

`gpu_input_batch.py`: `CachedRequestState` (`:34`) is per-request worker state;
`InputBatch` (`:91`) is the **persistent batch** — CPU buffers (`token_ids_cpu`,
`is_token_ids`, `num_computed_tokens_cpu`, `block_table`) that **persist across steps**
so tensors aren't rebuilt every iteration. `_update_states`
(`gpu_model_runner.py:1132`) reconciles it with each `SchedulerOutput`: remove
finished/unscheduled, add new/resumed, update running, then **`condense()`**
(`gpu_input_batch.py:683`) compacts the dense prefix after removals (swap the
highest-index live request into the lowest free slot). After condense,
`prev_req_id_to_index` is rebuilt — a known stale-mapping hazard under the pipeline
(§8, W6; brick 80 §3).

---

## 6. Distributed parallelism (TP / PP / DP / EP / CP)

File: `vllm/distributed/parallel_state.py`. The `GroupCoordinator` (`:~290`) wraps a
torch `ProcessGroup` (a `device_group` on NCCL + a `cpu_group` on gloo). One singleton
per axis per process (`_TP/_PP/_DP/_EP/_EPLB/_PCP/_DCP`, `:~1257-1318`), each with an
accessor (`get_pp_group()` etc.). `initialize_model_parallel(...)` (`:~1522`) is
called once with the **target** model's sizes;
`world_size = ExternalDP × DP × PP × PCP × TP` (`:~1588`). Brick 10 is the deep PP
treatment; this section adds the other axes.

| Axis | Singleton | What is **sharded** | What is **replicated** | What is **communicated** |
|---|---|---|---|---|
| **TP** tensor | `_TP` | weights *within* a layer (`ColumnParallelLinear` `linear.py:~407` shards out-dim; `RowParallelLinear` `:~1389` shards in-dim; `VocabParallelEmbedding` shards vocab `vocab_parallel_embedding.py:~192`) | input ids | **all-reduce / all-gather** of activations *per layer* (the PCIe-heavy traffic we avoid by not using TP across the no-NVLink pair) |
| **PP** pipeline | `_PP` | layers *across* ranks (`make_layers` + `get_pp_indices`, `distributed/utils.py:~109`) | — | `IntermediateTensors` {hidden_states, residual} **sent stage→stage** (`send_tensor_dict`/`recv_tensor_dict` `:~852-1069`; metadata on cpu_group, tensors on device_group) |
| **DP** data | `_DP` | the **batch** (different requests per replica) | the **whole model** | wave/coordination (DP coordinator); per-step request-count sync + all-reduce barriers (`DPEngineCoreProc`) |
| **EP** expert | `_EP`/`_EPLB` | **MoE experts** across ranks (`FusedMoE`); EP group size = DP×PCP×TP, one per PP stage | non-expert weights | **all-to-all** of tokens to/from their experts |
| **CP** context | `_PCP` (prefill) / `_DCP` (decode) | the **sequence/context** dimension | model | all-gather / all-reduce (or all-to-all) of Q/K/V slices; `_DCP` reuses TP GPUs with `dcp_size ≤ tp_size` |

`broadcast` is `GroupCoordinator.broadcast` (`:~637`) on the device group.
**There is no spec-specific broadcast helper in the coordinator** — the sampled-token
broadcast is hand-rolled in the runner (`pp_spec_broadcast.py`), which is the seam our
B1a work owns (§8).

### Deep dive: pipeline parallelism

PP splits the model's **layers** into contiguous stages across ranks. With `L` layers
and `pp_size` stages, `get_pp_indices` gives each rank a `[start, end)` slice — an even
`L // pp_size` split with the remainder pushed to middle ranks (the last rank keeps the
output norm). `VLLM_PP_LAYER_PARTITION` overrides this (the Q13 memory lever, brick
40/70). **What crosses a stage boundary** is small: a per-token
`{hidden_states, residual}` dict, not the big per-layer all-reduce TP needs — that is
precisely why PP, not TP, is chosen for the no-NVLink PCIe pair.

**Placement convention** (brick 10/20): `embed_tokens` lives on the **first** rank
(or last too, if `tie_word_embeddings`); `norm` + `lm_head` on the **last**.
Non-local layers are `PPMissingLayer` (a no-op `nn.Identity`) and their weights are
skipped at load. The **canonical PP forward** (`if is_first_rank: embed; else: read
intermediate_tensors; ... ; if not is_last_rank: return IntermediateTensors; else
norm`) is the exact pattern the MTP draft mimics — and the reason Design C needs a
"standalone draft" flag so the draft (which runs on the last rank where
`is_first_rank==False`) doesn't take the read-intermediate-tensors branch (bricks
10/60).

**Bubbles.** A pipeline has fill/drain bubbles (the first/last `pp_size−1` steps where
not all stages are busy). vLLM hides steady-state bubbles by keeping `pp_size` batches
in flight via the `batch_queue` (§4.3). The V2 runner (#42187) further reduces bubbles
but is **unavailable to our quantized hybrid Qwen3.5** (brick 40: V2 requires
`not is_moe and not is_quantized` and a whitelisted arch).

**Composition.** TP×PP form a grid (`linear.py` shards within a stage, PP across
stages). PP×DP replicate the PP pipeline per DP rank. PP×spec is the hard one: the
drafter runs only on the last stage, and its sampled tokens must be broadcast back to
the earlier stages so they can build the next input (§8).

---

## 7. KV cache & attention

Files: `vllm/v1/kv_cache_interface.py`, `vllm/v1/worker/block_table.py`,
`vllm/v1/core/kv_cache_utils.py`, attention backends.

- **PagedAttention.** KV is stored in fixed-size **blocks** (OS-paging analogy). A
  per-request **block table** (`block_table.py:~70`, `[max_reqs,
  max_blocks_per_req]` int32) maps logical block index → physical block id. A
  per-token **slot mapping** (`block_table.py:~75`) gives each token's physical slot
  `block_id * block_size + offset`; padded entries use `PAD_SLOT_ID = -1`. Attention
  backends read the slot mapping to write/read KV without chasing the block table.
- **KVCacheSpec types** (`kv_cache_interface.py`): `FullAttentionSpec` (`~203`),
  `SlidingWindowSpec` (`~459`), `MLAAttentionSpec` (`~352`, DeepSeek latent),
  `ChunkedLocalAttentionSpec`, **`MambaSpec`** (`~605`, SSM/conv state — fixed shape,
  *not* token-indexed), `CrossAttentionSpec`, `EncoderOnlyAttentionSpec`.
  `KVCacheGroupSpec` (`~837`) groups layers sharing a spec; hybrid models get
  multiple groups padded to a uniform page size.
- **Hybrid (mamba/GDN + attention).** Qwen3.5 mixes attention KV with Mamba conv/SSM
  state. Attention layers use paged blocks; Mamba layers use fixed per-request state
  buffers (no block table). `_update_states_after_model_execute`
  (`gpu_model_runner.py:~1502`, **hybrid only**) computes per-request accepted counts
  `(sampled != -1).sum(dim=1)` — relevant to spec accounting (brick 80 §3, Q17). The
  `causal_conv1d` GDN kernels JIT-compile and run (brick 70 — *not* the memory wall).
- **KV profiling.** `determine_available_memory` runs a dummy forward at
  `max_num_batched_tokens`, measures peak activation memory, subtracts it (+ overhead
  + cudagraph reservations) from total VRAM, and divides by page size to get
  `num_gpu_blocks` (`gpu_model_runner.py` profile path + `kv_cache_utils.py:~1258`).
  `num_gpu_blocks_override` bypasses it.

For the draft side specifically (brick 30): the MTP draft has **its own** KV tensors
but joins the **same** `kv_cache_group` as the target, **shares block tables / slot
mapping**, and rolls back rejections via shared `seq_lens` (overwrite-on-reuse, not an
explicit clear).

---

## 8. Speculative decoding — and how it composes with PP + async

Files: `vllm/v1/spec_decode/*`, `vllm/v1/sample/rejection_sampler.py`,
`vllm/v1/worker/pp_spec_broadcast.py`. Bricks 30/40/80 are the deep treatment; this
section is the self-contained map.

### 8.1 Proposer taxonomy (the drafters)

`SpeculativeMethod` (`config/speculative.py:~59`) enumerates the methods; the runner
builds one proposer:

| Proposer | `file:line` | Draft source |
|---|---|---|
| `NgramProposer` | `spec_decode/ngram_proposer.py:~12` | CPU suffix n-gram match of the sequence (no model). |
| `NgramGPUProposer` | `spec_decode/ngram_proposer_gpu.py:~216` | Same, vectorized on GPU; supports async. |
| `EagleProposer` | `spec_decode/eagle.py:~10` | EAGLE/EAGLE3 head; `pass_hidden_states_to_model=True`. |
| **MTP** | via Eagle base; models `qwen3_5_mtp.py`, `mimo_mtp.py` | Multi-token-prediction head; **reuses the target's final hidden state** as draft input. |
| `DraftModelProposer` | `spec_decode/draft_model.py:~17` | A separate small LLM; `pass_hidden_states_to_model=False`, vocab/TP must match. |
| `MedusaProposer`, `SuffixDecodingProposer`, others | `spec_decode/{medusa,suffix_decoding}.py` | extra heads / suffix trees. |

`propose(...)` (`llm_base_proposer.py:~427`) is fed `target_hidden_states`,
`next_token_ids`, and `common_attn_metadata`, and returns `[batch, num_spec]` draft
tokens. **MTP's win**: the draft input is the target hidden state already resident on
the last rank — zero extra compute/PCIe (brick 30, Q9).

Async scheduling **auto-enables** for Eagle/MTP/ngram_gpu/draft_model
(`config/vllm.py:~935-979`); CPU ngram/medusa/suffix disable it.

### 8.2 The rejection sampler = the correctness oracle

`rejection_sampler.py`. The **greedy** kernel (`rejection_greedy_sample_kernel`,
`:708`): `target_argmax = target_logits.argmax(-1)` (`:452`); a draft token is
**accepted iff it equals the target argmax**, otherwise it is **replaced by the target
argmax** and everything after it is rejected. Therefore the greedy output is a **pure
function of (target argmax sequence, draft sequence)** → **weight-agnostic**: weights
change only the acceptance *rate*, never the spec-vs-non-spec *equality*. This is what
lets dummy/MiMo weights validate the cascade (brick 70 A3). The non-greedy path is the
true probabilistic rejection sampling (ratio test against draft probs).

- **Bonus token**: if *all* drafts are accepted, the target also emits one extra
  "free" token at the accepted tail. Hence the output grid width is
  `[num_reqs, num_spec + 1]`.
- **`PLACEHOLDER_TOKEN_ID = -1`** (`rejection_sampler.py:30`): valid tokens are
  contiguous from column 0; rejected/padding positions are `-1`. `parse_output`
  (`:247`) keeps `!= -1 & < vocab` (`:267`).

### 8.3 The spec-under-PP path — **the code we are changing** (highlighted)

This is the sub-mechanism the whole effort centers on (bricks 40/80/81). Under PP>1 +
async:

```
LAST rank:  forward → sample → rejection oracle → sampled[num_reqs, num_spec+1]
            → _pp_broadcast_prev_sampled_token_ids  (GPU broadcast)
            → propose drafts → _draft_token_ids
                    │
                    ▼
NON-LAST:   _pp_receive_… : recv[num_reqs, num_spec+1]; store; rebuild
            prev_req_id_to_index  (AFTER condense)
            → next step _prepare_input_ids:
                 common (prev_pos ≥ 0): scatter recv[:,0] → input_ids   ✓
                 non-common (prev_pos < 0): EARLY RETURN → reads token_ids_cpu
                                            which still holds -1 → EMBED OOB  ✗ break #2
SCHEDULER:  AsyncScheduler reserves -1 placeholders even for requests that
            won't get the worker overwrite  → the root #40768 fixes
```

The transport is now **width-agnostic** via the new CUDA-free
`vllm/v1/worker/pp_spec_broadcast.py` (B1a, done):
`broadcast_sampled_token_ids` (`:32`), `receive_sampled_token_ids` (`:42`),
`count_valid_sampled_tokens_per_req` (`:21`). gloo-tested
(`tests/v1/spec_decode/test_pp_spec_broadcast.py`) and validated on real MiMo-7B
PP=2+MTP (gets **past** the old `[num_reqs,1]` assert). The remaining gate is **break
#2**: a `-1` placeholder leaking into an embedding lookup on the non-last rank when a
request goes "non-common" — root-caused to upstream **PR #40768** ("stale async
placeholder tokens in spec decode", fixes #37159), a scheduler-side discipline
complementary to B1a. Reconciling the non-last-rank token accounting
(`num_tokens_no_spec` advance, the ngram-gated `:1330/:1490` vs hybrid `:1502`) is the
holistic C4 work (brick 81). See bricks 40 §"Session 5" and 80 for the full chronicle.

---

## 9. Cross-boundary data structures & typing

Files: `vllm/v1/outputs.py`, `vllm/v1/core/sched/output.py`,
`vllm/v1/worker/gpu_input_batch.py`, `vllm/v1/request.py`, `vllm/v1/engine/__init__.py`.
vLLM runs **mypy in CI**, so static types are vLLM's "build-time" contract.

**The contracts that cross a boundary:**

| Struct | `file:line` | Boundary |
|---|---|---|
| `EngineCoreRequest` (`msgspec.Struct`) | `engine/__init__.py:~83` | client → core |
| `SchedulerOutput` (`@dataclass`) | `sched/output.py:180` | scheduler → executor |
| `ModelRunnerOutput` (`@dataclass`) | `outputs.py:234` | worker → scheduler |
| `DraftTokenIds` (`@dataclass`) | `outputs.py:311` | proposer → scheduler (separate from `ModelRunnerOutput`!) |
| `EngineCoreOutputs` (`msgspec.Struct`) | `engine/__init__.py:~215` | core → client |

**Good idioms in use:** `NamedTuple` (`LogprobsLists`/`LogprobsTensors`,
`outputs.py:27/52`), `TypeAlias` (`PoolerOutput`), `IntEnum` (`RequestStatus`,
`FinishReason`), `Literal` (`PauseMode`), `msgspec.Struct` for IPC, `TYPE_CHECKING`
imports.

**Gaps where stricter types would document invariants** (brick 81 + the data-structure
read):

- The magic **`-1`** placeholder is a bare literal in the runner/input-batch even
  though `PLACEHOLDER_TOKEN_ID` exists in `rejection_sampler.py:30` → make it a shared
  `Final[int]` + a `is_placeholder()` guard, used consistently.
- `sampled_token_ids` / `prev_sampled_token_ids` carry an **implicit
  `[num_reqs, num_spec+1]` shape and `-1` layout** with no wrapper → a frozen
  `SampledTokenGrid` with a `valid_per_req()` method would encode the contract.
- `scheduled_spec_decode_tokens: dict[str, list[int]]` with **implicit "missing =
  none"** → a `TypedDict`/frozen per-request record.
- bare `req_index: int` indexing `token_ids_cpu` → `ReqIndex = NewType('ReqIndex',
  int)` validated at `add_request`.
- the proposer surface (Eagle/Ngram/MTP/Draft) uses `isinstance`/`hasattr` chains →
  a `Proposer` `Protocol` (`propose(...) -> DraftTokens`).

**Misleading names:** `ModelRunnerOutput` has **no** `spec_token_ids` field — draft
tokens travel via the separate `DraftTokenIds`. `SamplerOutput.sampled_token_ids` (a
GPU tensor) is `-1`-padded; `ModelRunnerOutput.sampled_token_ids` (a `list[list[int]]`)
is **not**. These two same-named fields mean different things on different sides of the
boundary.

---

## 10. Glossary — vLLM term ↔ common term ↔ what it actually is

| vLLM term | What people assume | What it actually is | `file:line` |
|---|---|---|---|
| **EngineCore** | "the engine" | only the **logic** (scheduler+executor+KV); transport is a separate `EngineCoreProc`/`EngineCoreClient`. | `core.py:95` |
| **client / engine split** | a network client | in-process vs ZMQ-wrapped engine; `InprocClient` has *no* sockets. | `core_client.py` |
| **batch_queue** | a FIFO queue of pending batches | a **pipelining ring** of in-flight futures (`deque(maxlen=pp_size)`); you apply the *oldest* while scheduling the newest. ⚠️ not a backlog. | `core.py:192-198, 484` |
| **async scheduling** | Python asyncio | **scheduler/execution overlap**: schedule step N+1 before N's output returns, via `-1` placeholders. ⚠️ the engine loop is a plain `while`. | `async_scheduler.py:12`, `config/vllm.py:497` |
| **AsyncLLM** vs **async_scheduling** | the same "async" | **different things**: `AsyncLLM` is the asyncio *frontend*; `async_scheduling` is the *engine* overlap mode. | `async_llm.py:70` vs `async_scheduler.py:12` |
| **continuous batching** | one big rolling batch | there is no batch object; each step re-derives the running set from a token budget. | `scheduler.py:336` |
| **chunked prefill** | a special prefill mode | just the token-budget clamp on `num_new_tokens`; prompt consumed across steps. | `scheduler.py:~675` |
| **prefix caching** | KV dedup | longest-cached-block-prefix lookup by block hash; **last token always recomputed**. | `kv_cache_manager.py:196,~221` |
| **PagedAttention** | a kernel | a memory layout: KV in fixed blocks + block table + slot mapping. | `block_table.py:70,75` |
| **num_computed_tokens** | tokens the GPU computed | a **scheduler decision counter**; rolls back on rejection, resets on preempt. | `request.py:~149` |
| **num_tokens_with_spec** | "tokens, with spec on" | optimistic length **if all drafts accepted** = `num_tokens + len(spec_token_ids)`. | `request.py:253` |
| **num_output_placeholders** | output buffer size | async-only count of **promised-but-unmaterialized** output tokens. | `request.py:141` |
| **persistent batch / InputBatch** | a batch | CPU buffers reused across steps to avoid tensor rebuilds. | `gpu_input_batch.py:91` |
| **condense** | compress data | **compact** the dense request prefix after removals (slot swap). | `gpu_input_batch.py:683` |
| **GroupCoordinator** | a scheduler/leader | a `ProcessGroup` wrapper (device+cpu groups) per parallel axis. | `parallel_state.py:290` |
| **IntermediateTensors** | generic tensors | the `{hidden_states, residual}` dict crossing a PP stage boundary. | `sequence.py:~12` |
| **drafter / proposer** | two things | one thing — the speculative draft generator. | `llm_base_proposer.py:427` |
| **bonus token** | a reward | the free target token emitted when *all* drafts are accepted. | `rejection_sampler.py:~47` |
| **rejection sampling (greedy)** | probabilistic accept/reject | **deterministic argmax match** — weight-agnostic; "rejection sampling" only literally applies to the non-greedy path. | `rejection_sampler.py:708,452` |
| **PLACEHOLDER_TOKEN_ID / -1** | a real token | a sentinel for rejected/padded positions; OOB if it reaches an embedding lookup (break #2). | `rejection_sampler.py:30` |
| **uniproc / multiproc executor** | thread pools | in-process worker vs one OS process per rank (ZMQ/MQ). | `uniproc_executor.py:45`, `multiproc_executor.py:103` |
| **V1 vs V2 runner** | versions of vLLM | two model-runner implementations; V2 reduces PP bubbles but is gated to non-MoE/non-quantized whitelisted archs (so *not* our Qwen3.5). | brick 40 |
| **EP / EPLB** | "everything parallel" | MoE **expert** parallel (all-to-all); EPLB = a separate group for load-balancing collectives. | `parallel_state.py:~1289,1301` |
| **PCP / DCP** | one "context parallel" | **prefill** vs **decode** context (sequence) parallel; DCP reuses TP GPUs (`dcp_size ≤ tp_size`). | `parallel_state.py:~1313,1265` |

---

## 11. Weak spots & "not the strongest solutions"

Honest, code-referenced. Severity = correctness/maintenance impact; Entrenchment =
how hard to change (how many call sites / how central). Each notes whether upstream is
already fixing it.

| # | Weak spot | `file:line` | Severity × Entrenchment | What a clean design looks like | Upstream? |
|---|---|---|---|---|---|
| **W1** | **`gpu_model_runner.py` god object** (~7,583 lines): forward, attention, sampling, spec, PP, multimodal, pooling, cudagraphs, profiling in one class with tangled `is_last_rank`/`is_ngram_gpu`/`use_async_spec_decode` branches. | `gpu_model_runner.py:422` (whole file); branch knot `~1286-1500` | **High × Very high** | Extract `SpecDecodeFlow`, `SamplingFlow`, `PPTransport` collaborators behind typed interfaces; the runner orchestrates, doesn't implement. (brick 81 scope verdict: own the *spec-flow sub-mechanism*, not a rewrite.) | partial — V2 runner refactor (#42187) but unavailable to us |
| **W2** | **No owner for the request state machine**: `request.status = ...` and the token counters (`num_computed_tokens`, `num_output_placeholders`, `spec_token_ids`) are mutated from many scheduler sites; the FSM is implicit and fragile. | `scheduler.py:831, ~960-977, 1446, 1450`; `async_scheduler.py:31,63` | **High × High** | A `RequestState` owner with guarded transitions + a single accounting method; assert invariants (`num_rejected ≤ num_computed_tokens`). (brick 80 §2 "болезни".) | no |
| **W3** | **`-1` magic placeholders** + optimistic-extend/correct: `-1` is emitted by both the scheduler (`_spec_token_placeholders`) and the runner (optimistic extend), backfilled only on the common path → leaks on non-common/re-added requests → **embed OOB (break #2)**. | `async_scheduler.py:~16`; `rejection_sampler.py:30`; `_prepare_input_ids` `gpu_model_runner.py:1708`; brick 40 §"Session 5" | **High × High** | A typed `SampledTokenGrid` + "emit `-1` only when fillable" discipline (req ∈ `prev_step_scheduled_req_ids`). | **yes — PR #40768** (fixes #37159) |
| **W4** | **Dual sync/async draft-token paths with no shared abstraction**: sync pulls via `post_step`→`take_draft_token_ids`; async injects in the worker. Sync **deadlocks** for MTP+PP; async **crashes**. | `core.py:474` (sync); `gpu_model_runner.py` worker-inject (async); brick 40 §"Session 5" | **High × Medium** | One `DraftTokenChannel` abstraction with two backends behind a typed contract (brick 81 C2). | partial (#39704 sync, #40768 async) |
| **W5** | **`is_ngram_gpu`-gated accounting left MTP behind**: the `num_tokens_no_spec` advance/correction (`:1330/:1490`) is gated to ngram_gpu; hybrid MTP uses `_update_states_after_model_execute` (`:1502`) — the two don't compose, so MTP accounting is inconsistent (Q17). | `gpu_model_runner.py:~1330,~1490,~1502` (modified tree) | **Medium × Medium** | Method-agnostic accepted-count accounting (C4); resolve the ngram/hybrid asymmetry holistically. | no (our C4) |
| **W6** | **`prev_req_id_to_index` rebuilt after `condense()`**: the persistent batch is reordered, then the prev-step index map is rebuilt — a stale-mapping risk under the k-step pipeline delay. | `gpu_input_batch.py:683` (condense); map rebuild `gpu_model_runner.py:~4686` | **Medium × Medium** | `condense()` returns the index remap; consumers use it, no post-hoc reconstruction. | no |
| **W7** | **`deque(maxlen=…)` silently drops on overflow**: the batch_queue relies on the pop-before-append invariant; a logic change that appends without popping would *silently* drop a future and stall the pipeline with no error. | `core.py:198` | **Low × Low** (currently safe by invariant) | A bounded queue that *raises* on overflow, or an explicit assert at append. | no |
| **W8** | **No spec broadcast helper in the coordinator**: the sampled-token broadcast is hand-rolled in the runner (now extracted to `pp_spec_broadcast.py`), separate from `GroupCoordinator`. Fine for testability, but it means PP spec transport lives outside the distributed abstraction. | `parallel_state.py:~637`; `pp_spec_broadcast.py:32` | **Low × Low** | Keep the CUDA-free helper (good for gloo tests) but register it as a coordinator method so the contract is discoverable. | our B1a (done) |
| **W9** | **`IntermediateTensors` is an untyped `dict[str, Tensor]`**: PP stages must agree on keys (`hidden_states`/`residual`) by convention; a key mismatch is a runtime `KeyError`. | `sequence.py:~12-62` | **Low × Medium** | A frozen dataclass with typed fields. | no |
| **W10** | **`array_like=True` msgspec structs make field order the wire contract**: adding/reordering a field on `EngineCoreRequest`/`EngineCoreOutputs` silently breaks cross-version compatibility. | `engine/__init__.py:~84,~215` | **Low × Medium** | Versioned schema or named (map) encoding for evolution-critical structs. | no |

**Top 5** (by severity×entrenchment): **W1** (god object), **W2** (no FSM owner),
**W3** (`-1` leak / break #2), **W4** (dual draft paths), **W5** (ngram-gated MTP
accounting). W3 and W4 are exactly where the spec-under-PP effort lives; W3 is being
fixed upstream by **#40768** and our **B1a** is the complementary worker-side half.

---

## 12. Where *our* change sits in the whole picture

The path we are changing — **speculative-token flow under PP** — is a thin but
load-bearing seam: the last PP rank samples + runs the rejection oracle, broadcasts the
`[num_reqs, num_spec+1]` grid to the earlier ranks, and those ranks reconstruct their
next input from the broadcast + the scheduler snapshot. It touches §3 (scheduler
placeholder accounting), §4 (the k-step batch_queue delay), §5 (the runner's
`_update_states`/`_prepare_input_ids`/PP transport), §6 (the hand-rolled broadcast),
and §8 (the rejection oracle as the correctness gate). The design-side change is tiny
(Design C: one "standalone draft" forward flag, brick 60); the **bulk and the risk** is
this correctness plumbing (bricks 40/80), which is *design-independent*. Our completed
standalone deliverables — **A1c** (load-time int4 draft-embed quant, memory wall) and
**B1a** (width-agnostic broadcast transport) — and the planned **C0/C1** (executable
spec + typed state model) directly harden W3/W4/W5 and the §9 typing gaps. See brick 81
for the chunked C0–C5 contribution arc and the #40768 alignment.

---

## Appendix — brick cross-reference

| This report § | Existing brick (deeper / source) |
|---|---|
| §1 process model | — (new); brick 80 §1 engine loop |
| §2 request lifecycle | — (new) |
| §3 scheduler / KV | brick 80 §2; brick 30 (draft KV); brick 40 (batch_queue) |
| §4 engine loop modes | brick 80 §1; brick 40 (two plumbing paths) |
| §5 executor / runner | brick 80 §3; brick 81 (scope) |
| §6 distributed / PP | **brick 10** (PP & groups), brick 20 (embeddings) |
| §7 KV & attention | brick 30, brick 60 (draft attn PP), brick 70 (hybrid/memory) |
| §8 spec decode | **bricks 30/40/80**; brick 70 A3 (weight-agnostic oracle) |
| §9 typing | **brick 81** |
| §11 weak spots | bricks 80 §"болезни", 81 §3 |
| §12 our change | bricks 40/60/70/80/81; README + 00-map |
