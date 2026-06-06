<!--
  Open via "New issue" -> choose the "💬 Request for comments (RFC)" template (750-RFC).
  The template auto-adds the "[RFC]: " title prefix and the "RFC" label.
  Title (after the auto prefix): MTP speculative decoding under pipeline parallelism (PP>1): non-last-rank spec accounting
  Paste each block below into the matching template field. Tick the "I already searched" checkbox.
-->

# ── Field: Motivation. ──

MTP speculative decoding doesn't work under pipeline parallelism (PP > 1). In some configs it crashes; in others it silently diverges from the no-spec greedy baseline. People are hitting this: #36643 (Qwen3.5 + PP) and #36872 (gibberish under Qwen3.5 + spec), and the general request goes back to #14044 (closed).

The reason is that the speculative-decode token accounting is computed on the last PP rank only (where the sampler lives) and never makes it to the non-last ranks:

- `num_computed_tokens` is advanced optimistically (assuming every draft is accepted) and then corrected after the forward by a GPU kernel that's gated on `valid_sampled_token_count_gpu`, which only the sampler produces. On the non-last ranks that correction is skipped, so positions over-advance after every rejection. Rope/KV ends up off by one and verification goes wrong.
- Hybrid (GDN/mamba) models have the same problem on a second tensor, `num_accepted_tokens`, which drives the conv1d/SSM state rollback and is also set last-rank only.
- The sampled-token and draft values the non-last ranks need to embed the next input are likewise only resident on the last rank.

# ── Field: Proposed Change. ──

There's one invariant behind all of it: the non-last PP rank should apply the sampler's broadcast per-request valid/accepted count to its own accounting, exactly like the last rank does. In practice that's five small pieces plus an experimental gate:

1. Draft memory enablement (int4 draft-embed quant + skipping the draft `lm_head`) so the draft fits on the last rank for large models.
2. A typed, width-agnostic broadcast of the sampler's per-request tokens/counts to the non-last ranks (`pp_spec_broadcast.py`).
3. Non-last-rank input reconstruction (write the real broadcast values into `token_ids_cpu`, scatter the draft tokens). This is what fixes the crash.
4. `num_computed_tokens` drift correction on the non-last rank.
5. `num_accepted_tokens` rollback for hybrid GDN, gated on `is_hybrid`.
6. A one-time `warning_once` that flags MTP + PP > 1 as experimental while it matures.

I kept these small and contained (a few pure functions and some typed state) rather than rewriting the pipeline. The area broke for MTP in the first place because this accounting had no explicit contract.

On V1 vs V2: I think V2 (MRV2) is the right long-term home for spec-under-PP, and I'd be glad to help get it there (#42538 and #43732 are clearly the direction). But forcing V2 today (`VLLM_USE_V2_MODEL_RUNNER=1`) on MiMo PP=2 + MTP deadlocks at engine construction (EngineCore `shm_broadcast` gets stuck and it never reaches generate), so V2 doesn't cover this config yet and the people on #36643 / #36872 have no working path. The invariant here is small and should port cleanly to V2's `PPHandler` later, and the tests should make that port safer. So my suggestion is to land this on V1 now and follow up on V2 once that path works.

Validation, on current `main`: greedy-equivalent across MiMo (pure attention) and Qwen3.5-27B (hybrid GDN) at `num_speculative_tokens` 1-3, including `mamba_cache_mode=align` and fp8 KV cache; 1.68-1.89x speedup; ~94% acceptance on 27B. batch=16 and 256-token runs hold the fp-independent invariants (no crash, exact length), and the residual token divergence is the floating-point near-tie floor of MTP spec decode, not something PP-specific.

A couple of things I'd genuinely like a steer on before review:

- Scope: it's ~544 non-test LOC, over the 500-line guideline. Keep it as one PR, or split the draft-memory piece out (~190 LOC, which puts the rest under 500)?
- #40768: this complements it (the async placeholder discipline) but isn't a hard dependency. I checked by reverting #40768, and greedy-equivalence still holds, including at batch=16. I don't re-implement its scheduler change, and I'm coordinating with @z1ying.

A PR implementing this is ready and references this RFC. Happy to adjust the approach, split it, or align with V2 if that's what you'd prefer.

# ── Field: Feedback Period. ──

1 week.

# ── Field: CC List. ──

@njhill (codeowner of `/vllm/v1/worker` and `/vllm/v1/core`, MRV2), the spec-decode owners @benchislett @luccafong @MatthewBonanni, and @z1ying (#40768).

# ── Field: Any Other Things. ──

There are two earlier PP+MTP attempts (#39704, #38104), but both are untested and currently conflicting. This work adds the verified greedy-equivalence, hybrid (GDN/mamba) support, the draft-memory enablement, and a full test and benchmark suite.

AI assistance was used in diagnosing and drafting this. I reviewed every changed line and ran the tests myself.
