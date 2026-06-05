# Work-branches tree (refreshed after session 4) — where we can go, what's preferred

**Purpose.** The stakeholder wants, at each stage, the *map of options* and which is
preferable — like the 5-axis branch map from session 4's start, but updated for what
we now know: **memory (A1c) is solved; the front has moved to the execution cascade
(B1).** Read with: `../README.md` (state), `../research/pp-mtp/70-memory-and-validation.md`
(A1c + runs), `../research/pp-mtp/40-pp-x-spec-decode.md` (cascade), `../archive/2026-06-04-e3-execution-log.md`.

Legend: ⭐ preferred · ✅ done · 🔓 unblocked/ready · ⛔ blocked · ⏸ parked.

---

> **Live state moved out.** "Where we are now / what to do right now" is derived state —
> see `../../state.yml` (NEXT ACTION + deliverable statuses) via `../../tools/build_status.py`.
> This file keeps only the **durable** option map + preference rationale (changes rarely).

**Standing frame:** Axis 1 (memory) is closed by **A1c** → **Design B is demoted** (its only
advantage was distributing last-rank memory, which A1c removed). The dominant remaining axis
is **B1 (execution cascade)**; cheap vehicle = **MiMo-7B**. Greedy-equivalence is gated on B1.

---

## STAGE 1 — B1: the execution cascade (the gate to correctness)

First action: **run MiMo-7B PP=2 + MTP to map the WHOLE cascade** (cheap), instead
of fixing blind. MiMoMTP is already standalone (no Design-C fix) and fits with margin.

Known sub-branches (updated after session 5 MiMo mapping):
| Sub | What | How | Status |
|---|---|---|---|
| **B1a** | broadcast transport width: sender drops the `[num_reqs,1]` assert + broadcasts full width; receiver allocs `[num_reqs,num_spec+1]` (`gpu_model_runner.py:4651/4667` via new `pp_spec_broadcast.py`) | gloo-CPU TDD (3 green) → MiMo | ✅ **DONE** — MiMo past `:4653` |
| **B1c-#2** | **non-last-rank embedding index OOB** in the forward (`indexSelectSmallIndex`): next-step `input_ids` on rank0 carry an invalid id under MTP+PP+async; `_prepare_input_ids` only scatters `prev_sampled_token_ids[:,0]`, draft tokens are `None` on non-last ranks | **instrument input_ids on MiMo non-last rank** → fix consumption | 🔓 **NEXT GATE** |
| **B1b** | `num_tokens_no_spec` accounting for MTP (the `is_ngram_gpu` gate at `:1330` + matched correction `:1490`; hybrid uses `_update_states_after_model_execute:1498`) | likely folds into B1c | ⏸ pending B1c-#2 |
| **B1c-rest** | further cascade after #2 (count unknown — only reached once #2 green) | map on MiMo | ⏸ |

> Note: the session-4 framing of B1a as "per-req accepted-count advance" was **partly
> speculative** — B1a needed only the **transport width**; the accepted-count
> *accounting* belongs to B1c-#2 (break #2 reproduces with the original `+1` advance).

**Fork inside B1: async vs sync.**
- **async (current, auto-enabled for MTP+PP)** — most infra already present (optimistic-extend-correct), broadcast is the first break. ⭐ default.
- **sync (`post_step` path)** — would avoid the GPU broadcast but hits lead #1 (draft-token timing) and is the prod-default-without-spec path. Investigate ONLY if the async cascade (B1c) turns out large.

**Decision rule:** map the async cascade on MiMo first. If it's a handful of bugs →
finish async. If it balloons → spend a day evaluating sync as an escape hatch.

---

## Alternative direction (s7) — pivot to the V2 runner instead of fixing V1

Stakeholder reframe: don't fix V1's non-last-rank reconstruction (C4) — write/run on the **V2
runner** (`vllm/v1/worker/gpu/`), which already does non-last-rank reconstruction correctly
(`PPHandler`, `pp_utils.py`) **and** has a full spec-decode subpackage (`gpu/spec_decode/`: eagle,
`DraftModelSpeculator`, rejection sampler, `init_speculator`). So "target V2" is NOT a from-scratch
spec rewrite — much narrower.

**Barrier:** our 27B is **AWQ-quantized** + arch `Qwen3_5…`; the V2 gate
(`vllm/config/vllm.py:558` `not is_quantized and not is_moe`, allowlist `{Llama,Mistral,Qwen3}`)
locks quantized/MoE/Qwen3.5 OUT. The quant gate is the crux (likely "V2 quant unvalidated", upstream-owned).

**Verdict:** for the *speedup* goal, C4-on-V1 is still the shorter path (our model is quant-locked to
V1; C4 is now a small data-grounded change). For the *contribution* goal, V2 may be the longer-lived
result. **Decisive (cheap, ~1h, like the s7 de-risk) before committing either way — Q18:**
(a) does V2 spec support **MTP** specifically (not just eagle/draft-model)?
(b) is the quant gate "not implemented" or merely "not validated" (can it be lifted for AWQ)?
(c) is Qwen3.5 arch supportable on V2?
**Recommend: next session OPENS with this V2-viability investigation, then decide C4-finish vs V2-pivot.**

## STAGE 2 — greedy-equivalence (the correctness bar)

Reached once B1 is green. Validation ladder (cheap → final):
| Vehicle | Use | Note |
|---|---|---|
| tiny synthetic Qwen3.5 + `load_format=dummy` | shape/plumbing only | accept≈0 (dummy) — proves the cascade SHAPE, not accepted-accounting |
| **MiMo-7B (real)** ⭐ | end-to-end greedy-equiv + accepted-accounting | acceptance > 0; fast; primary oracle |
| **Qwen3.5-27B (real)** | final greedy-equiv vs `base.json` | the actual target; needs A1c (+ small offload) to fit |

Bar: spec output **token-identical** to non-spec greedy (cf. gibberish #36872).

---

## STAGE 3 — tuning, memory polish, upstream

| Branch | What | Pref |
|---|---|---|
| A1c acceptance sweep | measure acceptance vs bits {int8, int4, nf4}; pick lowest that holds ~71% | ⭐ (correctness-safe; pure speed knob) |
| A1c no-offload finish | close the last ~144 MiB (reduce profiling/activation footprint) so 27B fits with ZERO offload | optional polish |
| nf4 draft embed | add nf4 cycle if int4 acceptance too low | conditional |
| **Upstream series of PRs** | (1) drafter guards, (2) forward flag, (3) A1c memory, (4) B1 plumbing — RFC in #14044/#36643 first | ⭐ end-game |

---

## Parked / demoted

- **Design B** (draft sharded across PP stages) — ⏸ **demoted**: its advantage was
  distributing last-rank memory, which A1c solved directly. Revisit only if Design C
  hits an unfixable execution wall. (`SupportsPP` on the wrappers is harmless to keep.)
- **Design D** — ⏸ parked.
- **Cross-model KV sharing** (Q12) — ⏸ logged as a lever for other models, not Qwen.

---

## Preference summary (one line)

**Now → B1 on MiMo (map cascade, fix B1a/B1b/B1c via TDD+MiMo) → greedy-equiv on
MiMo → greedy-equiv on 27B → acceptance sweep + upstream series.** Memory (A1c) and
draft placement (Design C) are done; the whole remaining critical path is B1.
