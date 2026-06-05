# PP + MTP Speculative Decoding — Working README (session continuity)

**Read this first — then read EVERYTHING.** This is the living entry point for the
PP>1 + MTP speculative-decoding effort in vLLM. It is NOT a handoff doc; it (and
every doc under `docs/superpowers/`) is kept current to reflect the *actual present
state*. **Before doing any work, read all of it** — this README, the map
`research/pp-mtp/00-map.md`, every brick (10/20/30/40/60), and every spec
(`design-c-phase2.md`, `e3-execution-log.md`, `pp-mtp-spike-plan.md`). On this task,
knowing everything already discovered is worth it — the docs exist precisely so you
don't re-walk dead ends. Each session updates these docs in place (state, next
step, findings with `file:line`).

> Convention (stakeholder): everything under `docs/superpowers/` is a **work
> artifact** — kept on disk, **NOT committed and NOT gitignored** (untracked).
> Only real product/test code gets committed. Don't `git add docs/superpowers`.

Last updated: 2026-06-05 (session 6). **Memory wall SOLVED (A1c).** Execution
cascade (B1) is the gate. **B1a (broadcast width) DONE.** **C3 (#40768 scheduler
placeholder discipline) ported + local-green.** **Q16 ANSWERED — NO (s6, MiMo
gpu-wb): C3+B1a do NOT close break #2.** Engine reaches `generate`, then break #2
(`indexSelectSmallIndex`) STILL fires on rank0 (non-last) at `embed_tokens(input_ids)`
— full traceback pinned (`mimo.py:73` → `vocab_parallel_embedding.py:491`). The
proximate `-1` is **worker-side** (receiver local buffer on the non-common path), NOT
scheduler emission → **C3 is necessary-but-not-sufficient** (stays a valid green
standalone). **Next gate = F2/C4: holistic non-last-rank input reconstruction** (the
non-last rank must hold the REAL broadcast sampled-token value in ALL confirmed local
buffer positions, never a `-1`). Q15 closed. **Option map:
`specs/2026-06-05-work-branches.md`.** Cascade detail + the exact Q16 frame:
`research/pp-mtp/40-pp-x-spec-decode.md` (Session 6 section).

Earlier (session 3): Design-C runs on real 27B PP=2; 5 spec+PP drafter bugs fixed;
the spec+PP+async cascade + the Q13 memory wall both blocked greedy-equivalence —
chronicle in `specs/2026-06-04-e3-execution-log.md`. (Session 4 solved the memory
wall and re-scoped "embed-sharing" → A1c quantize; greedy-equivalence is
weight-agnostic so MiMo/dummy validate the cascade.)

## CURRENT STATE / NEXT STEP (one-paragraph truth — post session 4)
Design C is the chosen design (NOT B). **The memory wall (Q13) is SOLVED** by **A1c**:
quantize the draft's vocab embedding to int4 **at load time** (`QuantizedVocabEmbedding`,
int storage allocated up front + a quantizing weight_loader — a post-load swap OOMs at
the load peak) **and skip allocating the draft's own fp16 lm_head** (it's always shared
with the target's; `qwen3_5_mtp.py` builds `PPMissingLayer()` under the A1c flag). On the
real 27B PP=2 this **fits + reaches `generate`** (with a SMALL `cpu_offload_gb=3`; only
~144 MiB short without it — A1c did the heavy lift from session-3's hard "one layer too
many" deadlock). GDN/mamba `causal_conv1d` kernels JIT-compiled and **ran** (not the wall).
**The remaining gate is the execution cascade (B1), now advancing on MiMo-7B.**
**B1a DONE (session 5):** the PP sampled-token broadcast (`gpu_model_runner.py:4651`)
no longer asserts `[num_reqs,1]` — a new `vllm/v1/worker/pp_spec_broadcast.py` makes
the transport width-agnostic (sender broadcasts `[num_reqs,num_spec+1]`, receiver
allocs the matching width); gloo-CPU unit-tested (`tests/v1/spec_decode/
test_pp_spec_broadcast.py`, 3 green) and validated on real MiMo PP=2+MTP — it gets
PAST `:4653`. **Next gate = B1c break #2:** a CUDA `indexSelectSmallIndex` device
assert on the NON-last rank (rank0) in the forward — its next-step `input_ids` carry
an invalid embedding index under MTP+PP+async (`_prepare_input_ids` scatters only
`prev_sampled_token_ids[:,0]`; draft tokens are `None` on non-last ranks). Needs
input_ids instrumentation on MiMo. (The downstream `scheduler.py:1388` KeyError is
fallout of the crashed worker, not an independent bug.) **MiMo-7B**
(`/models/MiMo-7B-Base`, MiMoMTP already standalone) is the cheap vehicle (~40 s loads).
**The full option map + per-stage preferences is in `specs/2026-06-05-work-branches.md`
(read it for "where can we go next").** A1c details + the run chronicle:
`research/pp-mtp/70-memory-and-validation.md`.

## SESSION 3 RESULT (read with the e3-execution-log)
- **Design C (Slice B) IMPLEMENTED + unit-tested:** standalone-draft forward flag
  in `qwen3_5_mtp.py` (`self.standalone_draft = spec.draft_pipeline_parallel_size
  == 1`; forward treats first==last when set). Test
  `tests/v1/spec_decode/test_qwen3_5_mtp_standalone.py` (red→green). Slice A
  (config) was already done by committed code (MTP flows through the else-branch
  `create_draft_parallel_config`, draft_pp defaults to 1).
- **Built + ran on gpu-wb** (isolated `/root/vllm-dev`, precompiled editable).
  Baseline (no-spec) PP=2 generates fine = oracle `base.json`. **Design C LOADS on
  the real 27B**: draft constructs standalone on the last rank, shares target
  lm_head, loads its own embed — our flag path is exercised, no crash from our code.
- **Found + fixed 5 real spec+PP V1-runner bugs** (uncommitted, in
  `vllm/v1/worker/gpu_model_runner.py`): non-last ranks have no `self.drafter`, but
  `_dummy_run`, `initialize_attn_backend`, `_check_and_update_cudagraph_mode`,
  `validate_same_kv_cache_group`, and `_build_attention_metadata` all accessed it.
  Fix: `get_pp_group().is_last_rank` guards + `self.drafter = None` on non-last
  ranks. These got the run from "crash at profiling" → "executing the forward pass".
  **This is concrete content of "V1 not fully support spec+PP".**
- **HARD BLOCKER — Q13 confirmed numerically:** 27B-AWQ (20.35 GiB) + MTP draft
  (own 2.37 GiB vocab embed; lm_head IS shared) + ~2 GiB/rank CUDA context does NOT
  fit on 2×16 GiB. Empirically: rank0 fits ≤45 layers, rank1 (with draft) fits ≤18
  → 45+18=63 < 64. **One layer too many.** `VLLM_PP_LAYER_PARTITION` can't solve it.
  `cpu_offload_gb=8` (UVA, ~7 GiB/rank to RAM) makes it FIT (KV profiled 7.96 GiB) —
  this is the practical Q13 mitigation (PP=2 + weights in RAM, KV/draft in VRAM).
- **Execution cascade beyond the 5 drafter fixes:** with cpu_offload the run reached
  the forward. Qwen3.5 is a **hybrid GDN/mamba** model; one run hit a
  `causal_conv1d_update` device-side assert, but under `CUDA_LAUNCH_BLOCKING=1` the
  GDN kernel **JIT-compiled and ran** — mamba is NOT the wall (the −1
  `conv_state_token_offset` when `num_accepted_tokens==0` is a latent hazard to
  re-check). The **current front** is a NOT-fixed spec+PP+async bug:
  `_pp_broadcast_prev_sampled_token_ids` (`gpu_model_runner.py:4653`) asserts
  `sampled_token_ids` shape `[num_reqs, 1]`, but MTP spec emits `[num_reqs,
  num_spec+1]` (accepted drafts + bonus) → this is brick-40 **lead #3**, genuinely
  unhandled (earlier "already plumbed" read was too optimistic).
- **To actually validate greedy-equivalence, need ONE of:** (a) the efficiency-lens
  **embed-sharing** optimization (remove the draft's duplicate 2.37 GiB embed —
  code work; the real fix for huge-vocab MTP+PP), (b) more VRAM / a smaller MTP
  model, or (c) fix GDN+cpu_offload compatibility. Tools added:
  `docs/superpowers/tools/plan-pp-memory.py` (memory planner; under-estimates the
  last-rank overhead ~4 GiB — needs recalibration), `sync-gpu-wb.sh`,
  `deploy-gpu-wb.md`.
- Local env note: gpu-wb has NO nvcc → run with `VLLM_USE_FLASHINFER_SAMPLER=0`
  (attention auto-selects a precompiled backend on sm89+sm120). Multiproc spawn
  needs `if __name__=="__main__"` in the runner. Crashed runs leak GPU memory →
  kill stale `--query-compute-apps` PIDs before re-running.

---

## 1. Goal

Enable pipeline parallelism (PP>1) together with MTP speculative decoding for
Qwen3.5 in vLLM, upstream-acceptable, **output token-identical to non-spec
greedy** (no silent quality regression — cf. gibberish #36872). Motivation: the
same MTP on llama.cpp gives ~71% draft acceptance; we want that on the user's
PP=2 setup (TP across the no-NVLink PCIe pair is slow).

## 2. What's available to us

- **`ssh gpu-wb`** — LXC on `ru-oset-nas`. GPU0 = RTX 4060 Ti 16GB (Ada sm89),
  GPU1 = RTX 5060 Ti 16GB (Blackwell sm120), PCIe `NODE`, **no NVLink**.
  **Prod runs there** (vllm 0.22 serving `/models/Qwen3.5-27B-AWQ` at pp=2 tp=1
  enforce-eager) and **saturates both GPUs** → any real PP=2 run (E3) needs a
  **maintenance window** (stop the qwen3.5-27b service first). Don't disrupt prod.
  Models on server: 27B-AWQ (dense, has `mtp.*` tensors), 35B-A3B-GPTQ-Int4 (MoE),
  122B-A10B(-MTP). Note: the 122B MTP experiments in `/models/*.log` are
  **llama.cpp**, not vLLM.
- **Local box** — this repo + RTX 3080 Ti (1 GPU). vLLM is **built** in `.venv`
  (precompiled, vllm 0.22.1rc1.dev…+cu130, torch 2.11). Test deps installed:
  `pytest`, `ruff`, `tblib`. Build cmd if rebuilding:
  `VLLM_USE_PRECOMPILED=1 VIRTUAL_ENV="$(pwd)/.venv" uv pip install -e . --torch-backend=auto`.
  Run CPU unit tests; full `tests/conftest.py` works now (tblib installed).
- Memory file (cross-session, in `~/.claude/.../memory/pp-mtp-spec-decode.md`)
  mirrors this at a glance.

## 3. Decision so far

**Design C (leading)** — "standalone draft on the last rank":
- The drafter already runs only on the last PP rank, where the target's final
  hidden state is resident (zero cross-stage traffic) and the draft's
  `embed_tokens` is already weight-loaded (MTP creates embed unconditionally).
- The ONE draft-side change: a flag making `Qwen3_5MultiTokenPredictor.forward`
  behave as first==last (embed→fc→layer→norm) instead of branching on the global
  PP group. `draft_pp=1` makes the SupportsPP guard not fire.
- Memory risk (whole draft on the last/fuller GPU) is **mitigated** by
  `VLLM_PP_LAYER_PARTITION` (shift target layers off the last rank). Stakeholder
  idea; built-in knob.

**Design B (fallback)** — draft sharded across PP stages; model is shaped for it
(we declared SupportsPP) but it has a likely backward rank1→rank0 dependency and
adds an inter-stage hop. Use only if C fails. **Design D** — parked, not worth it.

**The real remaining work is design-INDEPENDENT:** correctness of spec output
under PP `batch_queue` (pipelined microbatches delay update_from_output). Three
plumbing leads (per PR #39704): draft-token retrieval in the batch_queue path;
stale-snapshot guard; non-last-rank accepted-draft token accounting. This is the
bulk + the gibberish risk. **Good news:** it's locally TDD-able (see §6).

> **Phase-2 design doc (session 2):** `specs/2026-06-04-design-c-phase2.md` — the
> concrete Design-C implementation plan (2 slices) + E3 checklist + open decisions,
> grounded in the session-2 findings below.

## 4. Knowledge base (how the system works, code-referenced)

`docs/superpowers/research/pp-mtp/00-map.md` — the mindmap/index: problem ↔
directions ↔ knowledge bricks ↔ open-questions register. **Start there.**
Bricks (all code-referenced): `10` PP & process groups · `20` embeddings (key:
draft embed loaded on every rank) · `30` spec dataflow + KV (draft input resident
on last rank; KV own-but-shared block tables; cross-model KV sharing exists for
Gemma4 but not Qwen) · `40` PP × spec batch_queue (the cascade B1 + testing
strategy) · `60` draft attn/KV PP deps (Q4 closed) · `70` memory fix (A1c) +
cheap validation (A3): the int4/lm_head fixes + the real-27B run chronicle ·
**`80` deep end-to-end map of the async-spec-PP execution pipeline (engine loop ·
scheduler · runner · distributed · rejection oracle; where break #2 lives)** ·
**`81` typing/idiomatic-code + the rewrite-as-contribution plan (scope verdict, the
invariant contract, chunks C0–C5, the #40768 alignment)**.
Experiment logs: `specs/2026-06-04-e3-execution-log.md` (session-3 runs),
`specs/2026-06-04-pp-mtp-spike-plan.md` (A/B spike, superseded). **Option map /
decision tree: `specs/2026-06-05-work-branches.md`.**

Open questions Q1–Q13 with statuses live in `00-map.md`. Most are answered;
the live ones: **Q8** (the 3 batch_queue plumbing leads — confirm via tests),
**Q13** (memory headroom — confirm numerically at E3).

## 5. Completed work, step by step

**Branch topology (session 4):** the foundational, upstreamable fixes live on
**`feat/pp-mtp-foundation`** (pushed to remote **`fork`** =
`github.com/atassis/vllm`); the working branch **`feat/pp-mtp-spec-decode`** is
re-branched **off** foundation (the ongoing work depends on these fixes directly).
Commits carry only `Signed-off-by` (DCO); AI assistance is disclosed in the PR
description, not via a commit trailer (stakeholder decision).

**`feat/pp-mtp-foundation`** (pushed to fork, tip `16503aada`) = `b93190138` +
`e6b5ea6ad` drafter guards + `f99c82ad1` standalone forward flag + `16503aada`
low-bit embed (the EARLY from-weight-only `QuantizedVocabEmbedding`). ⚠️ **Foundation
is BEHIND the working branch** — its embed module predates the load-time redesign;
when cutting the real A1c PR, **re-cut from the working branch's validated A1c**.

**`feat/pp-mtp-spec-decode`** (working, tip `e45e5d462`) = foundation + the A1c
slice, all validated:
- `1f6d2ff31` wire `draft_embed_quant_bits` into SpeculativeConfig + load path.
- `67f677c63` quantize draft embed at LOAD time (the OOM-peak fix).
- `e45e5d462` skip draft lm_head alloc in A1c memory mode.
Uncommitted WIP (brick-40): `tests/v1/core/utils.py` (M), `test_pp_spec_batch_queue.py`.
Plan: finish B1 on the working branch, then ship as a series of PRs.

**Committed (tracked) — base of `feat/pp-mtp-spec-decode` (via foundation):**
- `b93190138` `[WIP][Spec][PP] groundwork…`:
  - `vllm/config/speculative.py`: add `draft_pipeline_parallel_size` (default 1)
    + `_verify_and_get_draft_pp` + thread through `create_draft_parallel_config`
    & `__post_init__`. (Revives #16568's idea; lets draft_pp=1 bypass the guard.)
  - `vllm/model_executor/models/qwen3_5_mtp.py`: `Qwen3_5MTP`/`Qwen3_5MoeMTP`
    declare `SupportsPP` + expose `make_empty_intermediate_tensors`. (Needed for
    Design B; not strictly needed for C.)
  - `tests/v1/spec_decode/test_pp_draft_config.py`: 5 CPU unit tests (green).
  - History was rewritten to drop earlier doc commits (docs now untracked).

**Uncommitted working changes (tracked files, NOT yet committed):**
- `tests/v1/core/utils.py` — two test-infra tweaks: (a) `create_scheduler` builds
  `ParallelConfig` with `distributed_executor_backend="mp"` for pp>1, so PP>1
  scheduler-logic tests construct on a 1-GPU/CPU host; (b) under `async_scheduling`
  it builds the spec config with `method="ngram_gpu"` (the async-path validator
  rejects CPU `ngram`; scheduler logic is proposer-agnostic). Legit test-infra
  fixes; decide later whether to commit. Full `test_scheduler.py` = 101/101 green
  with these.
- `tests/v1/core/test_pp_spec_batch_queue.py` — NEW, brick-40 scheduler probe.
  **Status: GREEN and faithful (59 passed / 15 skipped).** Drives the
  `AsyncScheduler` through engine-core's `step_with_batch_queue` ordering with a
  prompt-aware synthetic worker; asserts "stops at EXACTLY max_tokens" across
  `num_spec∈{1,2,3} × accept∈{0…num_spec} × max_tokens`, chunked prefill over 3
  in-flight batches, and a mid-pipeline stop token; asserts queue depth ≥2 so it
  can't degrade to lockstep. **→ Lead #2 (stale-snapshot accounting) does NOT
  reproduce as a scheduler bug on current main** (see brick-40 "Verified" section).

**Local validation done:**
- Built vLLM locally; `tests/v1/spec_decode/test_pp_draft_config.py` 5/5 green.
- Confirmed the scheduler test harness works on CPU; `create_scheduler` exposes
  `async_scheduling`, `pipeline_parallel_size`, `num_speculative_tokens`.
- Confirmed `async_scheduling` enables `batch_queue` even at pp=1 (the delay is
  reproducible without 2 GPUs); `load_format="dummy"` gives tiny random models.
- **Layer-2 greedy-equivalence (1 GPU, real execution): PASS.** `ngram_gpu` spec
  + `async_scheduling` produces **token-identical** output to non-spec greedy on
  `opt-125m` (4 prompts × 48 tok). Empirically confirms the async+spec V1 path is
  correct at pp=1. **Local env workaround (no CUDA toolkit / nvcc here):** run with
  `VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0` — otherwise
  flashinfer JIT-compiles kernels and dies on missing `nvcc`. So **single-GPU real
  runs DO work locally**; only PP=2 needs gpu-wb.

**Session-3 uncommitted code (the real deliverables, user is PR submitter):**
- `vllm/model_executor/models/qwen3_5_mtp.py` — Design C standalone-draft forward
  flag (`self.standalone_draft = vllm_config.speculative_config.
  draft_pipeline_parallel_size == 1`; forward acts first==last when set).
- `vllm/v1/worker/gpu_model_runner.py` — 5 non-last-rank `self.drafter` guards
  (`is_last_rank`) + `self.drafter = None` on non-last ranks.
- `tests/v1/spec_decode/test_qwen3_5_mtp_standalone.py` — forward-flag unit test.

**On gpu-wb (session 3): E3 WAS RUN.** Dev tree built at `/root/vllm-dev`
(precompiled editable). Baseline PP=2 PASS (oracle `base.json`). Design C loads +
executes; greedy-equivalence not reached (execution cascade + Q13). Full chronicle:
`specs/2026-06-04-e3-execution-log.md`. Deploy tooling:
`docs/superpowers/tools/sync-gpu-wb.sh` (set `REMOTE_DIR=/root/vllm-dev`) +
`deploy-gpu-wb.md`. GPUs are freed after each session (kill stale compute-apps PIDs).

## 6. Local testing strategy for the hard part (no GPU window)

Brick-40 correctness is scheduler/engine-core logic → unit-testable with
synthetic `ModelRunnerOutput`s + hand-computed expectations. Layers:
1. **Scheduler unit tests** (`tests/v1/core/`, the harness above): pin the 3
   leads with `create_scheduler(async_scheduling=True, pipeline_parallel_size=2,
   num_speculative_tokens=k)` + delayed synthetic outputs. No GPU.
2. **Tiny target + ngram spec + async on 1 GPU** (dummy weights): end-to-end
   delay path; oracle = greedy ≡ non-spec. Ngram needs no draft model.
3. **2 ranks (CPU/2 GPU)** for the non-last-rank lead.
4. **Real Qwen3.5 PP=2 on gpu-wb (E3)** — final greedy-equivalence + memory.

## 7. NEXT STEP (exact) — session 7 entry, the "продолжаем работу" target

**State (session 6 end):** A1c done · B1a done · C3 done (local-green) · **Q16 = NO**
(C3+B1a do NOT close break #2 — see top headline + brick 40 §Session 6). break #2 is
fully diagnosed (brick 40 §"F2/C4 grounded diagnosis" + §"full accounting trace"):
**no site writes real sampled-token values into the non-last-rank `token_ids_cpu`
under async PP** (ngram has its own population path; MTP doesn't) → `-1` → embed OOB on
the non-common step. The fix is **C4 = (A) value back-write** in the receiver
(`_pp_receive…:4694`), possibly + **(B)** a single-site `num_tokens_no_spec` advance.

**THE NEXT ACTION (decided, grounded — do this first):** ONE targeted **instrumentation**
run on MiMo to capture the per-step non-last-rank accounting trajectory (env-gated, like
session-5's `VLLM_PP_SPEC_DEBUG`): log per req `{prev_index, num_computed_tokens,
num_tokens_no_spec pre/post branch-2 (`:1418`) and pre/post receiver (`:4696`), pos, recv
row, valid_count}`. This resolves **(A)-only vs (A+B)** with DATA, not a guess (the one
remaining blind spot — the exact count trajectory across the k-step delay). MiMo is
non-hybrid → `_update_states_after_model_execute:1513` doesn't fire → isolates the
non-hybrid accounting cleanly. Then implement C4 fully grounded (TDD where it factors
into a pure helper, MiMo greedy-equiv as the integration oracle) → 27B vs `base.json`.

**Deploy reminders (grabли):** manual rsync **without `--delete`** (only changed files;
`sync-gpu-wb.sh`'s `--delete` wipes the remote-only harness). **NEVER** put
`pkill -f e3_run.py` in an inline ssh string (kills the ssh shell → exit 255; I hit this
in s6) — cleanup via compute-app PIDs only. MiMo launcher = `run_mimo.sh` (no QUANT_BITS).
gpu-wb = standing auth, both GPUs were free. Free GPUs after (compute-app PIDs).

**Then (contribution track, can parallel):** ship A1c + B1a + C3 as standalone PRs
(re-cut foundation, which is BEHIND), C0/C1 typed contract (brick 81 §4), help land
#40768 for PP+MTP. **Reframed headline (worth making explicit):** the non-last-rank
receiver path looks broken for *every* method under PP+async+spec, not just MTP — so the
real contribution is "complete + test vLLM's spec-under-PP non-last-rank reconstruction,"
bigger than "fix Qwen MTP" (brick 81).

## 8. Guardrails / conventions

- **gpu-wb = STANDING AUTHORIZATION (2026-06-04):** personal box; full access,
  may stop prod (`qwen3.5-27b`) to free both GPUs anytime — **do NOT wait for a
  window**, just run; the user says explicitly if a given run should wait. (As of
  this note prod is idle-unloaded, both GPUs free.) Always free the GPUs (kill
  stale `--query-compute-apps` PIDs) when done. Note: `vllm-gateway.service`
  (idle-unload proxy) may spin prod up on an incoming request — stop it for the
  duration of a long run if contention appears.
- Don't commit `docs/superpowers/`; commit only product/test code.
- Correctness oracle = greedy ≡ non-spec; never accept "it runs" as success.
- AGENTS.md: any PR must be human-defended, list tests, disclose AI assistance.
- Use `uv` via `VIRTUAL_ENV=$(pwd)/.venv`, not `.venv/bin/python -m uv`.
