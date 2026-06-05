# Phase-2 spec: Design C (standalone-draft on the last rank) — implementation plan

**Status:** design locked (C), grounded in verified facts. Implementation is
two small slices; full validation is the E3 window run.
**Date:** 2026-06-04 (session 2).
**Supersedes** the spike-plan's provisional "Design B" decision (that log predates
bricks 20/30/60 and the session-2 findings).

> Read with: `../research/pp-mtp/00-map.md` (index), brick 40 (the verified
> correctness picture), brick 60 (why C is safe). This doc is the *how to build &
> validate it* plan.

---

## 1. What changed our understanding (session 2, all code-referenced)

1. **The scheduler layer is already correct.** Local 2-deep batch_queue tests
   (`tests/v1/core/test_pp_spec_batch_queue.py`) prove the AsyncScheduler stops at
   exactly `max_tokens` across acceptance rates / num_spec / chunked prefill /
   mid-pipeline stop. **Lead #2 refuted** as a scheduler bug.
2. **Our config auto-enables async scheduling.** `method=mtp ∈ EagleModelTypes`
   (`speculative.py:56`) + `MultiprocExecutor.supports_async_scheduling()==True`
   → `vllm/config/vllm.py:969-997` auto-enables async for MTP+PP. So the relevant
   spec plumbing is the **async** path (worker-side `update_async_spec_token_ids`
   + AsyncScheduler placeholders), which the local tests cover.
3. **V2 runner is unavailable to us.** `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` =
   {Llama, Mistral, Qwen3ForCausalLM} and requires `not is_moe and not
   is_quantized` (`vllm/config/vllm.py:519-555`). Our checkpoints are `Qwen3_5…` +
   AWQ/GPTQ (one MoE) → **V1 runner**. The "V1 not fully support async+PP"
   (`vllm.py:504`, PR #42187) is about **pipeline bubbles (perf)**, not a spec
   correctness gap.
4. **V1 already plumbs async+PP+spec, method-agnostically.**
   `use_async_spec_decode = use_async_scheduling and num_spec_tokens>0`
   (`gpu_model_runner.py:628`); non-last-rank tokens propagate by GPU broadcast
   (`:1332-1340`); accepted-draft drift corrected via `prev_num_draft_len` +
   optimistic-extend-then-correct (`:1287-1328`, `:1353-1363`). **Leads #1/#3 are
   largely already handled on main** — the #39704-era "missing fixes" framing is
   stale.
5. **Local async+spec is greedy-equivalent (1 GPU).** ngram_gpu + async ≡ non-spec
   greedy on opt-125m (token-identical). Confirms the async-spec V1 path is
   correct at pp=1.

**Net:** the remaining work is NOT a plumbing rewrite. It is **(a)** the Design-C
draft-side change (below), then **(b)** the E3 PP=2 greedy-equivalence run to
confirm the existing async+PP+spec infra carries Qwen3.5 MTP correctly.

## 2. The Design-C change — two connected slices

### Slice A — config: let the MTP draft run pp=1 — ✅ ALREADY DONE (session-3 correction)
**Session-2 claim was WRONG.** I thought `method=="mtp"` inherited the target
parallel config verbatim (`speculative.py:631-632`) and never honored
`draft_pipeline_parallel_size`. **That was a misread** — lines 631-632 are inside
the **ngram** block, not MTP. For MTP, `self.model` is set to the target model path
(`:571`), so it flows through the **`else` branch** (`:675` → `:679 if self.model
is not None`), where the method is detected as `"mtp"` (`:719-722`) and then
`_verify_and_get_draft_pp` (default None→**1**) + `create_draft_parallel_config`
run (`:802-823`). **So the committed code already gives MTP `draft_pp = 1` by
default** → Design C config is in place; no change needed here. Confirmed on the
real run: arch resolves to `Qwen3_5MTP`, draft builds standalone on the last rank.

(Open decision still stands per §4: whether draft_pp should default to 1 for MTP —
it currently does — vs require explicit opt-in. Default-1 = Design C, which is what
we want; leave as is.)

**Test (CPU):** the committed `tests/v1/spec_decode/test_pp_draft_config.py` already
covers `_verify_and_get_draft_pp`/`create_draft_parallel_config`. Optionally assert
that a full MTP `SpeculativeConfig` with `draft_pipeline_parallel_size=1` yields
`draft_parallel_config.pipeline_parallel_size == 1` (and the chosen default).
*Construction note:* full `SpeculativeConfig.__post_init__` on the mtp path needs a
target `ModelConfig` whose model exposes MTP — may require a small real config;
if that's heavy on CPU, test the branch via the static helpers as the committed
tests do, and cover the full path at E3.

### Slice B — model: the standalone-draft forward flag (E3-validated)
**Signal source (session 2 finding):** `_create_draft_vllm_config`
(`llm_base_proposer.py:1144`) does `base = self.vllm_config` and overrides only
kernel/attention — it does **NOT** swap in `draft_parallel_config`. So the draft
model's `__init__` sees the **global** `parallel_config` (pp=2); the flag must read
the **speculative** signal instead:
```python
# qwen3_5_mtp.py, Qwen3_5MultiTokenPredictor.__init__
spec = vllm_config.speculative_config
self.standalone_draft = (
    spec is not None and spec.draft_pipeline_parallel_size == 1
)
```
**Forward change** (`qwen3_5_mtp.py:133,154`): treat as first==last when standalone,
so it always embed→fc→layer→norm and never reads/returns `IntermediateTensors`:
```python
acts_as_first = self.standalone_draft or get_pp_group().is_first_rank
acts_as_last  = self.standalone_draft or get_pp_group().is_last_rank
# ... use acts_as_first in place of get_pp_group().is_first_rank (line 133)
# ... use `not acts_as_last` in place of `not get_pp_group().is_last_rank` (154)
```
Rationale (brick 60): on the last global rank `is_first_rank==False` today → the
draft wrongly takes the intermediate-tensors path; the flag fixes exactly that.
Embed is already weight-loaded on the last rank (brick 20 / Q3), input hidden
state is resident (brick 30 / Q9), no other PP dep (brick 60 / Q4).

**Why E3-gated:** the branch logic is trivial, but "standalone behavior is
correct" can only be proven by the real draft producing greedy-equivalent tokens
on the last rank under PP=2. A CPU branch-logic unit test (mock `get_pp_group()`)
can guard the wiring; it cannot prove correctness.

## 3. E3 checklist (window run on gpu-wb)
Pre: stop prod `qwen3.5-27b`; confirm both GPUs free; `APPLY=1
docs/superpowers/tools/sync-gpu-wb.sh`.
1. **Repro guard pre-change** (sanity): MTP + pp=2 raises the expected guard (or,
   with the committed SupportsPP, passes via Design-B shape).
2. **Apply slices A+B**, `draft_pipeline_parallel_size=1`.
3. **Load**: draft builds on the last rank; confirm `mtp.*` + base `embed_tokens`
   weights present on rank 1 (brick 20 item-3).
4. **Run greedy-equivalence**: PP=2 + MTP vs PP=2 no-spec, fixed prompts, greedy →
   **token-identical** (the bar; cf. gibberish #36872). Reuse the layer-2 harness
   shape (`/tmp/layer2_equiv.py`) adapted to the real model + `method=mtp`.
5. **Acceptance rate** sanity (vs llama.cpp ~71%).
6. **Memory (Q13)**: watch last-rank VRAM; if tight, `VLLM_PP_LAYER_PARTITION` to
   shift target layers off rank 1.
7. Verify the **non-last-rank GPU-broadcast** path (`gpu_model_runner.py:1332-1340`)
   actually carries MTP accepted-drafts (the one untested-for-MTP infra point).

## 4. Open decisions (for the human / the window)
- **MTP draft_pp default:** flip to 1 (C as default) vs opt-in (B default kept).
  Recommend **opt-in until E3 green**, then flip.
- **async on/off:** auto-enabled for us; confirm we *want* it (it's the path with
  the most existing infra). If we ever disable async, the PP-sync `post_step`
  draft-token timing (lead #1) becomes live again — but that path is not our
  default.
- **Upstream framing:** with leads #1–#3 mostly on main, the PR is small (Slice
  A+B) → easier RFC in #14044/#36643; coordinate, don't monolith.
