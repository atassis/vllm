# Ready-to-run prompt — "Understand the vLLM V1 engine pipeline" (research → interactive landing)

> Paste the block below into a fresh parallel session running **in this repo**
> (`/home/atassis/repositories/ns/ai/vllm`, branch `feat/pp-mtp-spec-decode`).
> It is self-contained. After it finishes, we reconcile its output with the MTP+PP work.

---

You are a senior systems engineer + technical educator. Your job in THIS session is to
**deeply understand and then explain vLLM's V1 inference engine execution pipeline,
end to end**, and produce an **interactive landing page** a human can click through to
learn the architecture. Work in the repo at the current directory; do NOT modify any
production code (read-only research + generate docs/landing artifacts only).

## Audience & goal
The reader is an engineer who wants to *truly* understand: how a request flows through
vLLM, every stage of the engine pipeline, what each component does, the exact
terminology (and where vLLM's naming diverges from common industry usage), what
pipeline parallelism (PP) is and where it's used, how speculative decoding composes
with it, and — explicitly — **where the implementation is architecturally weak,
over-coupled, confusingly named, or "not the strongest decision."** The reader is about
to modify the speculative-decoding-under-PP path, so situate that path within the whole.

## Build on existing knowledge (do NOT re-derive from zero)
A living, code-referenced knowledge base already exists at
`docs/superpowers/research/pp-mtp/`. **Read it first** and cite/extend it rather than
repeat it:
- `00-map.md` (index + open questions), bricks `10` (PP & process groups), `20`
  (embeddings), `30` (spec dataflow & KV), `40` (PP × spec batch_queue — the correctness
  gate), `60` (draft attn/KV PP deps), `70` (memory/A1c), **`80` (the async-spec-PP
  execution pipeline mechanism)**, **`81` (typing + rewrite decomposition)**.
- Treat brick 80/81 as the current best map; your report should broaden it from
  "spec-under-PP" to "the whole V1 engine pipeline," and deepen the terminology +
  weak-spots dimensions.

## Method
- Use **parallel Explore subagents**, one per subsystem (below), each returning a
  code-referenced map (every claim needs `file:line`). Then synthesize.
- **Accuracy over breadth.** A fact without `file:line` is a hypothesis — mark it as such.
- Verify class/function names against the code; do not trust memory.

## Scope — the parts of the pipeline to cover (with real entry points)
1. **Top-level architecture & process model.** `LLMEngine` (`vllm/v1/engine/llm_engine.py:47`),
   `AsyncLLM` (`async_llm.py:70`), `EngineCore`/`EngineCoreProc` (`core.py:95/858`),
   the client/engine split + ZMQ (`core_client.py`, `coordinator.py`), input path
   (`input_processor.py`), output path (`output_processor.py:417`, `detokenizer.py`,
   `logprobs.py`). Draw the process/thread boundaries.
2. **Request lifecycle & state machine.** `Request` + `RequestStatus`
   (`vllm/v1/request.py`): the WAITING→RUNNING→PREEMPTED→FINISHED_* transitions, who
   drives them, and the token bookkeeping (`num_computed_tokens`, `num_tokens_with_spec`,
   `num_output_placeholders`, `output_token_ids`, `spec_token_ids`).
3. **Scheduler & continuous batching.** `vllm/v1/core/sched/{scheduler,async_scheduler,
   output}.py`: token-budget loop, waiting/running queues, **chunked prefill**, **prefix
   caching**, preemption, the KV-cache manager / block pool (`vllm/v1/core/`,
   `vllm/v1/kv_cache_interface.py`). `Scheduler` vs `AsyncScheduler`.
4. **Engine loop & batching modes.** `core.py`: `step` vs `step_with_batch_queue`, the
   `batch_queue` pipelining + the k-step (~pp_size−1) delay, `post_step`, async
   scheduling, `output_queue` + the output daemon thread, `max_concurrent_batches`
   (`vllm/config/vllm.py`).
5. **Executor & worker.** `vllm/v1/executor/{abstract,multiproc_executor,uniproc_executor,
   ray_executor*}.py`; `vllm/v1/worker/{worker_base,gpu_worker,gpu_model_runner,
   gpu_input_batch}.py`. The **persistent batch** (`InputBatch`, `CachedRequestState`),
   `_update_states`, `_prepare_inputs`, forward, sampling, `condense`,
   `discard_request_mask`. (Note `gpu_model_runner.py` is ~7000 lines — flag that.)
6. **Distributed parallelism.** `vllm/distributed/parallel_state.py` (`GroupCoordinator`):
   **TP / PP / DP / EP / CP (PCP/DCP)** — define each, what is sharded vs replicated vs
   communicated, the singletons (`_TP/_PP/_DP/_EP`), `IntermediateTensors` send/recv
   across PP stages, broadcast (device vs cpu group). **Explain PP in depth** + where it's
   used across the codebase.
7. **KV cache & attention.** PagedAttention / block tables / slot mapping; hybrid
   (mamba/GDN) vs attention KV; how KV is allocated/profiled.
8. **Speculative decoding.** `vllm/v1/spec_decode/`, `vllm/v1/sample/rejection_sampler.py`,
   `vllm/v1/worker/gpu/spec_decode/`: draft vs target, proposer types (ngram/ngram_gpu/
   eagle/MTP/draft-model), the **rejection sampler** (greedy = argmax-match, weight-
   agnostic), bonus token, the `-1` `PLACEHOLDER_TOKEN_ID`, and **how spec composes with
   PP + async** (this is the path under modification — bricks 40/80/81).
9. **Data structures + typing.** `vllm/v1/outputs.py`, `vllm/v1/core/sched/output.py`,
   `gpu_input_batch.py`, `request.py`: the cross-boundary contracts and the current
   static-typing idioms (dataclass/NamedTuple/TypeAlias/IntEnum) vs gaps (no Literal/
   Protocol/TypedDict/NewType/Final where they'd help). Note vLLM runs mypy in CI.

## Required deliverables

### Deliverable 1 — Research report (markdown)
`docs/superpowers/landing/PIPELINE.md`: structured, code-referenced, covering all 9 scope
areas, PLUS these cross-cutting sections:
- **Terminology glossary** — a table: *vLLM term* ↔ *common industry term* ↔ *what it
  actually is* ↔ `file:line`. Include at least: continuous batching, chunked prefill,
  prefix caching, PagedAttention, EngineCore, the client/engine split, `batch_queue`
  (pipelining — NOT a literal queue of batches in the usual sense), async scheduling
  (scheduler/execute overlap — NOT asyncio), PP/TP/DP/EP/CP, V1 vs V2 model runner,
  uniproc/multiproc executor, GroupCoordinator, IntermediateTensors, drafter/proposer,
  bonus token, `PLACEHOLDER_TOKEN_ID`/`-1`, persistent batch, `condense`,
  `num_tokens_no_spec`, `num_output_placeholders`. Flag every case where vLLM's name is
  non-obvious or differs from what the wider ML-systems world calls it.
- **Pipeline parallelism deep-dive** — what PP is, the math of layer-splitting, what
  crosses stage boundaries, bubbles, and how PP interacts with TP/DP/spec/KV.
- **Weak spots & "not-the-strongest decisions"** — a candid, code-referenced section.
  Seed candidates (verify + expand, and for each propose what a cleaner design looks
  like): the spec-under-PP `-1`-placeholder + optimistic-extend/correct dance spread
  across scheduler + runner with no single owner (a fragile implicit state machine; cf.
  brick 80); the ~7000-line `gpu_model_runner.py` god-object; dual sync/async draft-token
  paths with no shared abstraction; `deque(maxlen=...)` silently dropping futures;
  `is_ngram_gpu`-gated accounting that left MTP behind; magic `-1` sentinels instead of
  typed placeholders; `prev_req_id_to_index` rebuilt post-`condense` (stale-mapping risk).
  Rate each (severity × how-entrenched) and note if upstream is already addressing it
  (e.g. PRs #40768, #39704, #38104).

### Deliverable 2 — Interactive landing page
A **self-contained, single-file** `docs/superpowers/landing/index.html` (embedded CSS+JS;
external libs only via CDN, e.g. mermaid or hand-rolled SVG; **no build step, openable
directly in a browser** — if you must split assets, ensure it runs via
`python -m http.server` with a one-line README). It should let the reader:
- See a **clickable top-level architecture diagram** (Client → EngineCore → Scheduler →
  Executor → Worker → ModelRunner → Sampler/Spec), each box expanding to its role +
  key `file:line` (link to the local path and/or the GitHub blob URL for vllm-project/vllm).
- **Step through the request lifecycle** (a stepper/animation: add_request → schedule →
  execute → sample → (spec verify) → update_from_output → detokenize → output).
- **Toggle engine modes** (sync `step` vs async `step_with_batch_queue`) and visualize the
  PP pipeline + the k-step delay (a small timeline/animation of microbatches across stages).
- Browse a **searchable terminology glossary** (from Deliverable 1).
- Show **PP/TP/DP** layout diagrams (what's sharded/replicated/communicated).
- A dedicated **"weak spots" panel** with the candid findings, severity-tagged, each
  linking to `file:line`.
- Visually mark **the path we are modifying** (spec-token-flow under PP) within the whole.
Keep it clean and not over-engineered: legible, fast, works offline. Prioritize correctness
of content (code-referenced) over visual flourish.

## Constraints
- Read-only on production code. Outputs go ONLY under `docs/superpowers/landing/`
  (these are untracked work artifacts by convention — do NOT `git add` them).
- Every architectural claim: `file:line`. Mark hypotheses explicitly.
- Reuse + cite the existing bricks (10–81); broaden, don't duplicate.
- Target the **V1** engine (the default for our models); mention V2 only where it differs.

## Output when done
A short summary listing: the report path, the landing path (how to open it), the top
5 weak-spots found, and the top 5 terminology "gotchas" (vLLM-name vs world-name).
