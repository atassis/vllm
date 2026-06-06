<!--
  ОПИСАНИЕ ОДНОГО feature-PR: A1c + B1a + C4 + s9 (БЕЗ C3 — он в #40768).
  Заголовок PR (предложение):
    [Spec][PP] MTP + pipeline-parallel (PP>1) speculative decoding: non-last-rank spec accounting
  Плейсхолдеры ‹…› — подтверди руками перед публикацией.
-->

## Purpose

Enable **MTP speculative decoding under pipeline parallelism (PP > 1)** and make it
**greedy-equivalent** to the single-GPU / no-spec baseline. This path was previously
unsupported: the speculative-decode token *accounting* is computed on the last PP rank
(where the sampler lives) and was never propagated to the non-last ranks, so under PP the
non-last ranks ran on stale/optimistic state — crashing in some configs and silently
diverging from greedy in others.

The unifying invariant this PR enforces:

> **Every non-last PP rank applies the sampler's broadcast per-request valid/accepted
> count to its local spec-decode accounting** — token values (to embed the next input),
> `num_computed_tokens` (positions / KV / rope), and `num_accepted_tokens` (hybrid
> conv1d/SSM state rollback) — identically to the last rank.

Closes #36643 (Qwen3.5 does not work with pipeline parallelism).
Closes #36872 (gibberish output under Qwen3.5 + spec decode).
Complements #40768 (async scheduler `-1`-placeholder discipline). Not a hard dependency:
with that scheduler change reverted, the validated MiMo PP=2 + MTP greedy-equiv result
still holds (5/5 token-identical, no crash) for the steady-state decode path. #40768
additionally hardens the preemption / re-added-request edge, so it is recommended
alongside this PR but not required by it. Coordinated with @z1ying, not duplicated here.

### How this differs from existing PRs (duplicate-work check)
- **#40768** — fixes the scheduler placeholder *crash*; it does **not** fix non-last-rank
  correctness or PP greedy-equivalence. Complementary, not a dependency (see above) —
  not re-implemented here.
- **#39704 / #38104** — earlier PP+MTP attempts; both untested and currently CONFLICTING.
  This PR adds verified greedy-equivalence, hybrid (GDN/mamba) support, the memory
  enablement for the draft on the last rank, and a full test + benchmark suite.

## Changes (reviewable as separate commits)

1. **A1c — draft memory enablement.** Load-time int4 quantization of the draft embedding +
   skip the draft `lm_head` allocation, so the draft fits on the last PP rank for large
   models (e.g. Qwen3.5-27B at PP=2). Draft-only — cannot affect output (the target is the
   verification oracle); worst case is acceptance rate, and int4 measured == int8.
2. **B1a — width-agnostic broadcast transport** (`vllm/v1/worker/pp_spec_broadcast.py`).
   Typed transport of the sampler's per-request tokens/counts from the last rank to the
   non-last ranks (gloo-tested CPU round-trip).
3. **C4 — non-last-rank input reconstruction.** Back-write the real broadcast token values
   into `token_ids_cpu` (never leave `-1`), scatter the broadcast draft tokens into the
   spec positions on non-last ranks, and pad the sender width. Closes the crash; MiMo PP=2
   + MTP now runs end-to-end.
4. **s9 positions — `num_computed_tokens` drift correction.** The GPU kernel that corrects
   the optimistic `num_computed_tokens` runs only on the sampler/last rank; the non-last
   rank reconstructs the same correction from the broadcast valid count in `_update_states`,
   so rope/KV positions are not off-by-one after a draft rejection.
5. **s9 hybrid — `num_accepted_tokens` for GDN/mamba.** The non-last rank sources the
   per-request accepted count from the broadcast so its GDN layers roll back conv1d/SSM
   state identically to the last rank. Gated on `is_hybrid` (pure-attention models
   unaffected).

## Test Plan

```bash
# Unit (CPU, no GPU):
.venv/bin/python -m pytest \
  tests/v1/spec_decode/test_pp_spec_broadcast.py \
  tests/v1/spec_decode/test_quantized_draft_embedding.py \
  tests/v1/spec_decode/test_pp_draft_config.py \
  tests/v1/spec_decode/test_qwen3_5_mtp_standalone.py \
  tests/v1/spec_decode/test_draft_embed_quant_integration.py -q

# Lint as CI:
pre-commit run --all-files

# Integration (2 GPU), greedy-equivalence oracle + speedup:
#   MiMo (pure attention) PP=2 + MTP vs no-spec baseline, 40 tokens, deterministic.
#   Qwen3.5-27B-AWQ (hybrid GDN) PP=2 + MTP vs baseline (cpu_offload_gb=3, int4 draft).
#   Oracle: spec output token-identical to the same-config no-spec baseline.
```

## Test Result

- **Unit: 31 passed** (CPU, no GPU) — includes the 2-rank gloo broadcast round-trip
  (`test_pp_spec_broadcast.py`) and int4 draft-embed quant (`test_quantized_draft_embedding.py`).
- **Lint:** `ruff check` + `ruff format` clean on all changed files. ‹run `pre-commit run
  --all-files` once before opening›
- **MiMo PP=2 + MTP (pure attention):** **5/5 token-identical** to the no-spec baseline at
  `num_speculative_tokens` = **1, 2, and 3** — the fix generalizes past k=1. Speedup
  **1.68×** at k=1 (spec 38.16 vs baseline 22.75 tok/s). _(re-validated on current `main`,
  base c73b0d0db; this branch carries no scheduler change, so this also confirms the
  greedy-equiv result does not depend on #40768.)_
- **Qwen3.5-27B-AWQ PP=2 + MTP (hybrid GDN):** **5/5 token-identical** to baseline at
  **k = 1, 2, and 3**, speedup **1.89×** at k=1 (offload-bound absolute tok/s; ratio is the
  signal). Also **5/5 with `mamba_cache_mode=align`** (the production-gateway regime).
  _(re-validated on current `main`.)_
- **Sampling (current `main`):** MiMo PP=2 + MTP at temperature 0.8 with a fixed seed is
  **deterministic** (two runs token-identical) and produces exactly the requested length —
  the PP spec path has no race under stochastic rejection sampling.
- **Acceptance (current `main`):** Qwen3.5-27B **94.7%** per-position (mean acceptance
  length 1.95 at k=1, 2.58 at k=2, 3.22 at k=3); MiMo 82.5% at k=1. MiMo per-position
  acceptance falls off fast (k=3: 0.81 / 0.14 / 0.01), so mean acceptance length saturates
  ~2 and **k=2 is the sweet spot** — the third draft position almost never accepts.
- **Isolation (pre-rebase):** single-GPU 27B + MTP == baseline (bugs are PP-specific);
  int4 draft == int8 draft (quantization ruled out as a divergence source).
- **Longer runs, 256 tokens, 3-way (current `main`):** under PP=2, 3/5 sequences are
  token-identical to 256; 2/5 diverge late (@109, @176). A single-GPU MTP run of the same
  prompts also diverges late, on a different but comparable set (seq0@25, seq4@161) — PP is
  not systematically worse (it is *perfect* on the one sequence where single-GPU diverges
  earliest), and both diverge at the identical point on the shared near-tie (seq1@176). So
  the residual is the fp near-tie floor inherent to MTP spec decode, not PP-specific.
- **Concurrency under load (current `main`, batch=16, no scheduler change):** the
  fp-independent invariants hold — no crash and **exactly the requested length on all 20
  sequences**. Token-level near-tie divergences are comparable to single-GPU (PP diverges on
  7/20 seqs vs single-GPU's 8/20, overlapping-but-different sets). This is the regime where
  the #40768 placeholder discipline would matter if it were load-bearing here; it is not
  (no leak/crash), reinforcing that this PR complements #40768 without depending on it.
- **Composition with production knobs (current `main`):** greedy-equivalent with **fp8 KV
  cache** on Qwen3.5-27B (the production-gateway regime) and with **chunked prefill** (long
  prompt, `max_num_batched_tokens=64`). All such configs ran without crashes. (fp8 KV on the
  smaller MiMo shows more near-tie divergence — expected from fp8's coarser precision flipping
  more ties, not a correctness regression. EOS early-stop was not exercised: the MiMo base
  model does not emit EOS within 128 tokens, where it stays greedy-equivalent 4/5 with the one
  residual being the same reproducible near-tie as the long run.)

## Relationship to MRV2 (V2 model runner)

I'm aware the V2 model runner is the longer-term home for spec decode under PP (e.g.
#42538 sharing identical MTP weights, #43732 cleaning up KV-connector + PP), so to check
whether this V1 work is redundant I tried the V2 path directly: with
`VLLM_USE_V2_MODEL_RUNNER=1`, MiMo PP=2 + MTP loads both ranks and sizes the KV cache but
**currently deadlocks during engine construction** (EngineCore `shm_broadcast` stuck,
`Worker_PP1` in `futex_wait`, never reaches generate). So V2 doesn't yet cover this
config, and these open bugs (#36643 / #36872) have no working path today without the V1
fix.

The invariant this PR establishes - the non-last PP rank applies the sampler's broadcast
per-request valid/accepted count to its local accounting - is intentionally small and
should port cleanly to V2's `PPHandler` when that path is ready. Happy to align the V1
shape here with whatever the V2 design prefers.

## Notes
- AI assistance was used in diagnosing, implementing, and drafting this PR. The changes
  have been reviewed line-by-line and the tests above were run by the human submitter.
- Untested / out of scope for this PR (flagged for reviewers): `mamba_cache_mode=all`
  (`none` and `align` both verified greedy-equiv); draft PP > 1 (draft stays on one stage);
  EOS-triggered early stop (the base model used for validation does not emit EOS, so the
  variable-length stop path is not exercised — the exact-length invariant under `ignore_eos`
  is, though).

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
</details>

<!-- Commit trailers (DCO sign-off + Co-authored-by: Claude) are already on each commit;
     they are NOT part of the PR description body. The "AI assistance was used" statement
     required by AGENTS.md is in the Notes section above. -->
<!-- The `## How this differs` / `## Changes` / `## Relationship to MRV2` / `## Notes`
     sections are supplementary context for reviewers; the three template sections
     (Purpose / Test Plan / Test Result) are the backbone. Keep all of this ABOVE the
     "BEFORE SUBMITTING" line that GitHub injects (anything below it is stripped). -->
