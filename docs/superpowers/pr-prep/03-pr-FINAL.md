## Purpose

Enable MTP speculative decoding under pipeline parallelism (PP > 1), greedy-equivalent to the no-spec baseline. This path used to crash or silently diverge: the speculative-decode token accounting is computed on the last PP rank (where the sampler lives) and never reached the non-last ranks, so they ran on stale, optimistic state.

I validated it on current `main`: greedy-equivalent across MiMo (pure attention) and Qwen3.5-27B (hybrid GDN) at `num_speculative_tokens` 1-3, including `mamba_cache_mode=align` and fp8 KV cache, at 1.68-1.89x. Implements the design in #44697; closes #36643, #36872.

## Root cause

Spec decode advances `num_computed_tokens` optimistically (assuming all drafts are accepted) and corrects it after the forward via the GPU kernel `update_num_computed_tokens_for_batch_change` (`gpu_model_runner.py:2115`), gated on `valid_sampled_token_count_gpu` (`:2107`). That tensor is only produced by the sampler, i.e. the last PP rank. On non-last ranks it's `None`, so the correction is skipped, positions over-advance by the rejected-draft count after every rejection, rope/KV goes off by one, and verification produces non-greedy output.

Two more pieces have the same shape: hybrid (GDN/mamba) models roll back conv1d/SSM state by `num_accepted_tokens` (set in `_update_states_after_model_execute` `:1529`, last-rank only), and the sampled-token / draft values the non-last ranks need to embed the next input are also only resident on the last rank. One invariant fixes all three: the non-last rank applies the sampler's broadcast per-request valid/accepted count to its own accounting.

## Why this isn't a duplicate

- #40768 fixes the scheduler placeholder crash; it doesn't fix non-last-rank correctness or PP greedy-equivalence. It's complementary, not a dependency: I checked by reverting it and greedy-equivalence still holds, including at batch=16. I don't re-implement it here.
- #39704 / #38104 are earlier PP+MTP attempts, both untested and currently conflicting. This adds verified greedy-equivalence, hybrid (GDN/mamba) support, the draft-memory enablement, and a full test + benchmark suite.

## Changes (6 commits)

1. Experimental gate: a one-time `logger.warning_once` when MTP runs with PP > 1, so users know the path is new while V2 catches up.
2. Draft memory enablement: int4 draft-embed load-time quant + skipping the draft `lm_head`, so the draft fits on the last PP rank for large models. Draft-only, so it can't change output; int4 measured == int8.
3. Width-agnostic broadcast (`vllm/v1/worker/pp_spec_broadcast.py`): typed transport of the sampler's per-request tokens/counts to the non-last ranks (gloo-tested).
4. Non-last-rank input reconstruction: write the real broadcast values into `token_ids_cpu` (never leaving `-1`), scatter draft tokens into spec positions, pad the sender width. Fixes the crash.
5. `num_computed_tokens` drift correction: reconstruct the skipped correction on the non-last rank from the broadcast valid count, in `_update_states`.
6. `num_accepted_tokens` for GDN/mamba: source the accepted count from the broadcast so non-last GDN layers roll back state identically. Gated on `is_hybrid`.

## Test Plan

```bash
# Unit (CPU, no GPU):
.venv/bin/python -m pytest \
  tests/v1/spec_decode/test_pp_spec_broadcast.py \
  tests/v1/spec_decode/test_quantized_draft_embedding.py \
  tests/v1/spec_decode/test_pp_draft_config.py \
  tests/v1/spec_decode/test_qwen3_5_mtp_standalone.py \
  tests/v1/spec_decode/test_draft_embed_quant_integration.py -q
pre-commit run --all-files

# Integration (2 GPU): spec output token-identical to the same-config no-spec baseline,
# on MiMo (pure attention) and Qwen3.5-27B-AWQ (hybrid GDN, cpu_offload_gb=3, int4 draft).
```

## Test Result

- Unit: 31 passed (CPU; includes the 2-rank gloo broadcast round-trip and int4 draft-embed quant). `ruff check` and `ruff format` pass on all changed files.
- Integration (current `main`, 2x 16 GB GPUs): greedy-equivalent on every config I tried, 1.68-1.89x, ~94% acceptance on 27B.

<details>
<summary>Full validation matrix</summary>

- MiMo PP=2 + MTP: 5/5 token-identical at k = 1, 2, 3 (so the fix generalizes past k=1); 1.68x at k=1 (38.16 vs 22.75 tok/s).
- Qwen3.5-27B-AWQ PP=2 + MTP (hybrid GDN): 5/5 at k = 1, 2, 3; 1.89x at k=1 (offload-bound absolute tok/s, the ratio is the signal). Also 5/5 with `mamba_cache_mode=align` and 5/5 with fp8 KV cache (the production-gateway regime).
- Sampling: temperature 0.8 with a fixed seed is deterministic (two runs identical) and exact length, so there's no race under stochastic rejection sampling.
- Chunked prefill: greedy-equivalent (long prompt, `max_num_batched_tokens=64`).
- Acceptance: 27B 94.7% per-position (mean accept length 1.95 / 2.58 / 3.22 at k=1/2/3); MiMo 82.5% at k=1, per-position falls off fast (k=3: 0.81 / 0.14 / 0.01), so length saturates around 2 and k=2 is the sweet spot.
- 256-token runs (3-way): PP=2 has 3/5 sequences identical to 256, 2/5 diverge late (@109, @176). A single-GPU MTP run of the same prompts also diverges late, on a comparable set (seq0@25, seq4@161), so PP isn't systematically worse (it's perfect on the sequence where single-GPU diverges earliest), and both share the same near-tie (seq1@176). The residual is the floating-point near-tie floor of MTP spec decode, not something PP-specific.
- Concurrency (batch=16, no scheduler change): no crash, and exactly the requested length on all 20 sequences. Near-tie divergences are comparable to single-GPU (PP 7/20 seqs vs single-GPU 8/20). This is the regime where #40768's placeholder discipline would matter if it were load-bearing here, and it isn't, which is why this complements #40768 without depending on it.

</details>

<details>
<summary>Relationship to MRV2 (V2 model runner)</summary>

V2 is the longer-term home for spec-under-PP (#42538, #43732). I checked it directly: with `VLLM_USE_V2_MODEL_RUNNER=1`, MiMo PP=2 + MTP loads both ranks and sizes the KV cache but then deadlocks during engine construction (EngineCore `shm_broadcast` stuck, `Worker_PP1` in `futex_wait`, never reaches generate). So V2 doesn't cover this config yet, and #36643 / #36872 have no working path today without this V1 fix. The invariant here is small and should port cleanly to V2's `PPHandler`, and I'm happy to align with whatever V2 prefers.

</details>

## Notes

- AI assistance was used in diagnosing, implementing, and drafting this PR. I reviewed every changed line and ran the tests above myself.
- Out of scope: `mamba_cache_mode=all` (`none` and `align` are verified); draft PP > 1 (the draft stays on one stage); EOS-triggered early stop (the validation base model emits no EOS, so only the `ignore_eos` exact-length path is exercised).

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
</details>
