# Spike Plan: Pipeline Parallelism (PP>1) + MTP Speculative Decoding for Qwen3.5

> ⚠️ **SUPERSEDED DECISION — read this first.** This is the original A-vs-B spike.
> Its §6–§7 conclusion "**Design B**" is **STALE**: after this spike, bricks
> 20/30/60 found that the MTP draft already has a weight-loaded embedding on the
> last rank, which spawned **Design C (standalone draft on the last rank)** —
> **C is the chosen, leading design; B is only the fallback.** Keep this doc as the
> historical A/B exploration + the E0–E3-prep experiment log. For the CURRENT
> decision and plan, read `2026-06-04-design-c-phase2.md`; for what happened when C
> was run on real hardware, read `2026-06-04-e3-execution-log.md`. Do NOT act on the
> "Design B" wording below.

**Status:** Phase 1 of 2 (spike → data → full design) — **decision now = Design C**
**Date:** 2026-06-04
**Owner:** Taimuraz Kaitmazov
**Related upstream:** issues #14044 (RFC), #36643 (Qwen3.5 PP bug), #36872 (gibberish), #14117 (roadmap); PRs #16568 (closed, Eagle+PP), #39704 (open, DeepSeek MTP+PP), #15173 (closed, V0)

> This is **not** the feature spec. It is a deliberately small plan to run experiments
> that collapse the single biggest unknown (Design A vs B, below) **before** writing the
> full design. Phase 2 (the real feature spec) is written from the data this spike produces.

---

## 1. Why a spike first

The guard `NotImplementedError: Pipeline parallelism is not supported for this model` is
**architectural, not a hard ban**: it fires because the MTP *draft* model trips the
`SupportsPP` check. The hard part is **not** removing the guard — it is making the draft
draft+verify loop produce output **identical** to non-spec greedy under PP's pipelined
(`batch_queue`) execution.

Static reading took us to a fork that cannot be resolved by reading alone:

- **Design A** — `draft_pp=1`, the whole (1-layer) MTP draft lives on the **last** PP
  stage. This is the approach of closed PR #16568 (which its author confirmed was
  *benchmarked working on H200* for Eagle/Llama — closed due to V0→V1 migration + rebase
  abandonment, **not** a technical reject).
- **Design B** — `draft_pp = target_pp`, the MTP draft is **sharded across the same PP
  stages** as the target, threading `IntermediateTensors` between stages.

For **Qwen3.5 specifically**, `Qwen3_5MultiTokenPredictor.forward`
(`vllm/model_executor/models/qwen3_5_mtp.py:132-159`) is wired to the **global** PP group:
`if get_pp_group().is_first_rank:` embeds + runs `fc`; `else:` it expects
`intermediate_tensors`; `if not is_last_rank:` it returns `IntermediateTensors`. That
shape is written for **Design B**. On the last target rank `is_first_rank == False`, so a
naive Design-A placement would skip the embedding path entirely and demand intermediate
tensors that do not exist. Whether Design A can be made to work (by detaching the draft
forward from the global group + replicating `embed_tokens` onto the last rank) — or whether
we must commit to Design B — is an **empirical** question. The spike answers it.

---

## 2. Target configuration (fixed)

Real hardware (`ssh gpu-wb`, LXC on `ru-oset-nas`):

- GPU0: **RTX 4060 Ti 16GB** (Ada, sm_8.9) + GPU1: **RTX 5060 Ti 16GB** (Blackwell, sm_12.0)
  — heterogeneous pair, PCIe topology `NODE` (**no NVLink**). TP across them is slow +
  mixed-arch; **PP is the natural choice** — hence PP+spec, not TP.
- Prod already runs (vLLM 0.22): `vllm serve /models/Qwen3.5-27B-AWQ --tensor-parallel-size 1
  --pipeline-parallel-size 2 --enforce-eager ...` **without** spec-decode. PP=2 already works;
  MTP speculation on top is the missing piece.
- Motivation is proven: the same MTP on **llama.cpp** gives **~71% draft acceptance**
  (`/models/122b-server-mtp.log`: `draft acceptance = 0.71429, 135/189`). We want that win
  inside vLLM, where PP lives.

Models in scope (both required by stakeholder):

| Variant | Checkpoint | Target arch | MTP draft class |
|---|---|---|---|
| Dense 27B | `/models/Qwen3.5-27B-AWQ` | `Qwen3_5ForConditionalGeneration` | `Qwen3_5MTP` |
| MoE 35B-A3B | `/models/Qwen3.5-35B-A3B-GPTQ-Int4` | `Qwen3_5MoeForCausalLM` | `Qwen3_5MoeMTP` |

Confirmed checkpoint facts (27B-AWQ): 15 `mtp.*` tensors present (`mtp.fc`,
`mtp.layers.0.*`, `mtp.norm`, `mtp.pre_fc_norm_embedding/hidden`); **no `embed_tokens`
inside `mtp.*`** (draft reuses base embedding); `tie_word_embeddings: False`;
`mtp_num_hidden_layers = 1` (single draft layer).

Spike scope fixes: **1 node, PP=2, TP=1, `num_speculative_tokens=1`**, `--enforce-eager`
(matches prod; sidesteps CUDA-graph interactions for now).

---

## 3. Code anchors (verified)

| Concern | Location |
|---|---|
| The guard | `vllm/config/model.py:1199-1206` (`if pipeline_parallel_size > 1 and not is_pp_supported_model`) |
| Draft inherits target PP | `vllm/config/speculative.py:945-964` (line 955 hardcodes `pipeline_parallel_size=target...`) |
| Guard trigger for draft | `vllm/config/speculative.py:1013-1016` (`_verify_args` → `verify_with_parallel_config(draft_parallel_config)`) |
| Qwen MTP inner (global pp group, own embed) | `vllm/model_executor/models/qwen3_5_mtp.py:59-159` (embed `:75`, forward `:132-159`) |
| Qwen MTP wrappers | `qwen3_5_mtp.py:349` `Qwen3_5MTP(nn.Module, SupportsMultiModal)`; `:455` `Qwen3_5MoeMTP` |
| Drafter gated to last rank (V1) | `vllm/v1/worker/gpu_model_runner.py:~542` (`if self.speculative_config and get_pp_group().is_last_rank:`) |
| Eagle/MTP loader skips embed share under PP | `vllm/v1/worker/gpu/spec_decode/eagle/utils.py:46-57` |
| Runner selection (Qwen3.5 → **V1** runner) | `vllm/config/vllm.py:69-75` (`DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` lacks Qwen3.5 arches), `:519-554` |
| `SupportsPP` template MTP classes | `vllm/model_executor/models/nemotron_h_mtp.py`, `glm4_moe_lite_mtp.py` |
| Tests: PP-parametrised proposer | `tests/v1/spec_decode/test_eagle.py` (`pp_size=[1,2]`, `_create_proposer`) |
| Tests: e2e equivalence (greedy/GSM8K) | `tests/v1/e2e/spec_decode/test_spec_decode.py` |

**Runner decision:** Qwen3.5 arches are **not** in `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`,
so by default they use the **V1 runner** (`vllm/v1/worker/gpu_model_runner.py`) — the same
file PR #39704 patches. The V2 runner (`vllm/v1/worker/gpu/model_runner.py`) is a secondary
concern, out of scope for the spike.

---

## 4. Environments

- **Local** (workstation): RTX 3080 Ti (1 GPU, ~12GB). vLLM **not yet built** here.
  Used for cheap static / unit / single-GPU spikes (E0–E2). Cannot do real PP=2 (1 GPU).
- **gpu-wb** (2 GPUs): the only place a real PP=2 Qwen3.5 run is possible. GPUs are
  **saturated by prod** (GPU0 ~13.5/16GB, GPU1 ~15.5/16GB) → E3/E4 require a **maintenance
  window** (stop the `qwen3.5-27b` service first). Coordinate before running.

Local build (E0):
```bash
cd /home/atassis/repositories/ns/ai/vllm
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements/lint.txt && pre-commit install
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/test/cuda.in
```

---

## 5. Experiments

Each experiment states: **where**, **steps**, **decision criterion**, **data to record**.
Record raw stack traces and exact diffs in §7 as they run.

### E0 — Local build + green baseline *(local)*
- **Steps:** build per §4; run an existing spec-decode unit test that needs no large model
  (e.g. selected cases in `tests/v1/spec_decode/test_eagle.py::test_load_model`).
- **Criterion:** suite imports and a baseline (PP=1) spec test passes — toolchain is sound.
- **Data:** pass/fail, env (torch/cuda versions).

### E1 — Decouple draft PP, confirm the guard is bypassable *(local, static + unit)*
- **Steps:** add a `draft_pipeline_parallel_size` field to `SpeculativeConfig` (default 1)
  and thread it into `create_draft_parallel_config` (`speculative.py:945-964`) instead of the
  hardcoded inherit (line 955) — minimal port of the #16568 idea. Write a unit test:
  construct `SpeculativeConfig` with target `pipeline_parallel_size=2`,
  `draft_pipeline_parallel_size=1`, MTP method, and assert `_verify_args` →
  `verify_with_parallel_config` does **not** raise `NotImplementedError`.
- **Criterion:** guard no longer fires when `draft_pp=1`, while it still fires for
  `draft_pp=2` with a non-`SupportsPP` MTP class (control).
- **Data:** does the config construct; does the control still raise.

### E2 — What PP group does the draft forward see under `draft_pp=1`? *(local, decisive for A/B)*
- **Steps:** trace/instrument how the draft worker's forward context is set when
  `draft_pp=1` but `target_pp=2`. Determine whether `get_pp_group()` inside
  `Qwen3_5MultiTokenPredictor.forward` returns a **draft-local** group (size 1 → draft is
  both first & last rank) or the **global** target group (size 2 → on last rank
  `is_first_rank == False`). Use a focused unit test with mocked process groups if a full
  model run isn't needed.
- **Criterion (the A/B fork):**
  - If draft sees a **size-1** group → **Design A is viable with small model changes**
    (embed runs, no intermediate tensors needed). Proceed to E3 as Design A.
  - If draft sees the **global size-2** group → Design A requires **forward detachment from
    the global group** (treat draft as always first=last) **and** embed replication onto the
    last rank. Note the exact edits needed; still proceed to E3 to test that minimal A-patch,
    but flag Design B as the likely fallback.
- **Data:** the concrete group the draft sees; the minimal model edits Design A would need.

### E3 — Repro + minimal Design-A patch on real hardware *(gpu-wb, window)*
- **Steps:**
  1. **Repro** the guard: `vllm serve /models/Qwen3.5-27B-AWQ --tensor-parallel-size 1
     --pipeline-parallel-size 2 --enforce-eager --speculative-config
     '{"method":"mtp","num_speculative_tokens":1}'` → capture exact stack (the
     `model.py:1199` guard via the draft `_verify_args`).
  2. Apply the **minimal Design-A patch**: E1 config decouple + add `SupportsPP` to
     `Qwen3_5MTP` (template: `nemotron_h_mtp.py`) + the forward/embed fixes E2 identified +
     ensure `embed_tokens` weights are present on the last rank.
  3. Run again; advance through failures (load → forward → sample → accept) recording where
     it breaks.
- **Criterion (the real bar):** with greedy sampling, **PP=2 + MTP output is token-identical
  to PP=2 without spec** on a fixed prompt set (rejection/greedy equivalence; cf. gibberish
  bug #36872). Plus a non-trivial acceptance rate (sanity vs llama.cpp's ~71%).
- **Data:** load OK?; runtime errors; **equivalence diff** vs baseline; acceptance %; notes on
  the `batch_queue` plumbing (does `spec_token_ids` reach `update_from_output` correctly under
  PP — the zone PR #39704 had to patch in `scheduler.py`/`outputs.py`).

### E4 — Design-B probe *(gpu-wb, window; only if E3 shows A is blocked)*
- **Steps:** prototype running the draft across **both** stages: instantiate the drafter on
  non-last ranks too, thread `IntermediateTensors` through the proposer draft forward (the
  Qwen MTP inner model already supports this shape). Measure how invasive the proposer /
  `gpu_model_runner` changes are.
- **Criterion:** is Design B tractable as a reviewable slice, or genuinely "core-weeks"?
- **Data:** scope estimate, blockers, whether equivalence is reachable.

---

## 6. Decision gate (exit of spike)

After E1–E3 (and E4 if reached), record a one-paragraph decision:

> **Chosen design = A or B**, because it produces correct (equivalent) greedy output under
> PP=2 with the least invasive, most upstream-reviewable change. The full spec (Phase 2) is
> written against this choice, with the `batch_queue`/scheduler plumbing scoped from the E3
> notes and PR #39704 as prior art.

If **both** A and B prove to be multi-week core work → the deliverable becomes a **design +
WIP branch + RFC** in #14044 (referencing #16568/#39704 and this spike's data), per the
original fallback, rather than a merge-ready PR.

---

## 7. Results log *(fill during execution)*

| Exp | Date | Where | Outcome | Key data / stack / diff |
|---|---|---|---|---|
| E0 | 2026-06-04 | local | **PASS** | Built precompiled into `.venv` (py3.12.13). `vllm 0.22.1rc1.dev173+g59478f369`, `torch 2.11.0+cu130`, CUDA True (RTX 3080 Ti). Note: invoke uv as system binary with `VIRTUAL_ENV=$(pwd)/.venv`, not `.venv/bin/python -m uv`. CPU unit tests run with `--noconftest` (full `tests/conftest.py` needs heavy test deps). |
| E1 | 2026-06-04 | local | **PASS** | Added `draft_pipeline_parallel_size` (default→1) + `_verify_and_get_draft_pp` + extended `create_draft_parallel_config`, wired in `__post_init__` (`vllm/config/speculative.py`). 4 CPU unit tests green (`tests/v1/spec_decode/test_pp_draft_config.py`). Mechanism proven: `draft_pp=1` → draft `ParallelConfig.pipeline_parallel_size == 1`. Since the guard (`model.py:1199`) is `if pipeline_parallel_size > 1 ...`, pp=1 makes its condition False → guard cannot fire (short-circuits before `is_pp_supported_model`). Validator rejects draft_pp∉{1,target_pp}. **Caveat:** end-to-end guard bypass through `verify_with_parallel_config` is exercised by E3 (needs a real draft `ModelConfig`). |
| E2 | 2026-06-04 | local | **DECISIVE → lean B** | **No separate draft PP group exists.** `_PP` is a global singleton (`parallel_state.py` `get_pp_group`), initialized once from the **target** config (`gpu_worker.py:1171` `ensure_model_parallel_initialized(... parallel_config.pipeline_parallel_size ...)`). `draft_parallel_config.pipeline_parallel_size` is consumed only for draft model instantiation (`draft_model.py:45,62`), never to build a process group; `set_forward_context` does not swap groups. → the draft forward sees the **global size-2 group**; on the last rank `is_first_rank == False`. Existing PP-specific code is gated behind `get_pp_group().world_size == 1` (e.g. `eagle/utils.py:47`, `llm_base_proposer.py:1282`), confirming draft PP handling is absent. |
| E3 | 2026-06-04 (prep) | local→gpu-wb | **step 1 done locally** | Design-B step 1 implemented + unit-tested locally ahead of the window: `Qwen3_5MTP`/`Qwen3_5MoeMTP` now declare `SupportsPP` and expose `make_empty_intermediate_tensors` (`qwen3_5_mtp.py`). `supports_pp(Qwen3_5MTP) is True` with no interface warnings → with `draft_pp == target_pp` the `model.py:1199` guard passes via the interface. **Remaining (needs 2-GPU window):** run drafter on all PP ranks (relax `gpu_model_runner.py:~542` `is_last_rank` gate) + thread `IntermediateTensors` through the proposer draft loop + `batch_queue`/scheduler plumbing; then the equivalence run (greedy ≡ PP=2-no-spec). |
| E4 | | gpu-wb | _pending_ | |

**Decision (provisional, pending E3): Design B.** — ❌ **SUPERSEDED. The chosen
design is C** (standalone draft on the last rank). This B rationale below was
written before bricks 20/30/60 showed the draft already has a weight-loaded embed
on the last rank, which makes C the clean choice (a forward flag, no
`IntermediateTensors` threading, no B backward dependency). See
`2026-06-04-design-c-phase2.md`. The text below is kept only as the historical B
rationale; it is NOT the current plan.

Rationale: E2 shows the draft cannot get its own size-1 PP group without new
distributed plumbing, and the Qwen MTP model is **already written for B** —
`Qwen3_5MultiTokenPredictor.forward` (`qwen3_5_mtp.py:132-159`) embeds on the
global first rank, threads `IntermediateTensors` between stages, norms on the
last rank, and exposes `make_empty_intermediate_tensors` (`:108`). Naive Design A
(`draft_pp=1`, draft on last rank only) would require ugly surgery (fake a size-1
group around the draft forward, or detach the forward from the global group +
replicate `embed_tokens` onto the last rank) and is rejected unless B proves
intractable.

Minimal Design-B slice to validate in E3:
1. Declare `SupportsPP` on the wrapper classes `Qwen3_5MTP` / `Qwen3_5MoeMTP`
   (template: `nemotron_h_mtp.py`) + expose `make_empty_intermediate_tensors`,
   so `draft_pp = target_pp` passes `verify_with_parallel_config`.
2. Run the drafter on **all** PP ranks (remove/relax the `is_last_rank` gate at
   `gpu_model_runner.py:~542`) and thread `IntermediateTensors` through the
   proposer draft forward + the autoregressive draft loop.
3. `batch_queue`/scheduler spec plumbing under PP (the zone PR #39704 patches:
   `scheduler.py`, `outputs.py`).

> **⚠ Branch caveat:** the E1 commit (`draft_pipeline_parallel_size`) is a spike
> increment, **not** shippable alone. With only E1, the guard stops firing but no
> A/B implementation exists, so PP+MTP would reach a broken draft forward. E1 must
> stay gated behind the full B implementation (or be re-defaulted) before any PR.

**Open question carried to E3:** does the minimal B slice produce greedy output
**token-identical** to PP=2 without spec? That is the correctness bar and cannot
be answered without 2 GPUs.

---

## 8. Upstream coordination

- This is wanted, not rejected: #14117 "roadmap H1", #22794 maintainer "feel free to
  contribution", #36643 acknowledged limitation.
- Before any large PR: post an RFC comment in **#14044** (and/or #36643) summarizing the A/B
  data, explicitly referencing #16568 (revive its `draft_pipeline_parallel_size` idea) and
  #39704 (reuse its scheduler/outputs plumbing, drop its `post_step`-commented hack), and
  agree the minimal slice with maintainers. **Do not** compete with #39704 as a monolith.
- Accountability (AGENTS.md): any eventual PR must be human-defended, list tests run, and
  disclose AI assistance.
