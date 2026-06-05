# Brick 81 — Typing/idiomatic code + the rewrite-as-contribution plan

Status: **RESEARCHED (session 5)** · Companion to brick 80 (mechanism). Answers the
stakeholder's strategic questions: can we write our own clean solution instead of
debugging the old state machine, at what scope, with build-time types, and how to
decompose it into AI-solvable chunks that become a bigger vLLM contribution.

---

## 1. Scope verdict (the key strategic finding)

**The async *pipeline* (engine loop + scheduler core + 7000-line runner) is NOT a module —
it is cross-cutting glue in the hottest, most-shared, fastest-churning files** (scheduler,
core, gpu_model_runner, parallel_state), entangled with LoRA / structured output /
multimodal / KV-transfer / pooling / chunked-prefill / cudagraphs / **hybrid mamba** /
quantization / every attention backend / TP / DP / EP. Evidence: #39704's base is 1547
commits behind ours, #40768's 1170 — i.e. **months of >1000-commit churn** in these files.

→ A from-scratch "rewrite the async pipeline and swap it in" is **not viable** (upstream
won't take a monolith — AGENTS.md; a fork diverges in days; it must re-support every
orthogonal feature). **But the stakeholder's instinct is right at a narrower scope:** the
**spec-token-flow-under-PP** sub-mechanism CAN be studied from requirements, closed with an
executable spec + tests, and owned as clean, typed, reusable components. `pp_spec_broadcast.py`
(B1a) is the proof-of-concept micro-version.

## 2. The contract to honor (the requirements, from brick 80)

Any implementation — patch or rewrite — must satisfy these invariants:
1. **Correctness:** spec output ≡ non-spec greedy (rejection oracle is weight-agnostic, brick 80 §5).
2. **k-step delay:** under batch_queue, `update_from_output` lags ~`pp_size−1`; accounting
   (`num_computed_tokens`/`num_output_placeholders`/`num_tokens_no_spec`) stays consistent.
3. **Non-last ranks need token VALUES** (to embed next input) + advance positions by the
   **variable accepted count**, for BOTH common and non-common/re-added requests.
4. **`-1` placeholders** must be filled before embedding, **or never emitted when
   unfillable** (#40768's discipline: emit only if req ∈ `prev_step_scheduled_req_ids`).
5. **Rejection rollback** via shared seq_lens / overwrite-on-reuse.
6. **Compose** with hybrid-mamba conv-state, quantization, drafter-on-last-rank-only,
   chunked prefill, structured output. (← the reason rewrite must stay narrow-scoped.)

## 3. Typing / idiomatic-code angle (Python "build-time" = mypy static)

Python has no compile-time types, but vLLM **runs mypy in CI** (AGENTS.md) → `Literal`,
`Protocol`, `TypedDict`, `NewType`, `Final`, frozen `@dataclass`, `enum` are our
"build-time" contracts. **Current idioms** (well-used): `@dataclass`, `NamedTuple`
(`LogprobsLists`…), `TypeAlias`, `IntEnum` (`RequestStatus`), `TYPE_CHECKING` imports.
**Absent where it would help spec/PP:** `Literal`, `Protocol`, `TypedDict`, `NewType`, `Final`.

**Weak spots that a typed rewrite should fix** (file:line in brick-80 §data-structures read):
- Magic `-1` sentinel everywhere → `PLACEHOLDER_TOKEN_ID: Final[int] = -1` + a
  `TypeGuard`-based `is_placeholder()`; it already exists as `PLACEHOLDER_TOKEN_ID`
  (`rejection_sampler.py:30`) but isn't used consistently in the runner/input-batch.
- Bare `req_index: int` indexing `token_ids_cpu`/`is_token_ids` → `ReqIndex = NewType(...)`
  validated at `add_request`.
- `sampled_token_ids: list[list[int]]` / `prev_sampled_token_ids` raw tensor with an
  implicit `[num_reqs, num_spec+1]` shape + `-1` layout → a small typed wrapper /
  `Protocol` documenting the contract (valid-contiguous-from-0).
- `scheduled_spec_decode_tokens: dict[str, list[int]]` with implicit "missing = none" →
  explicit, or a `TypedDict`/frozen dataclass per-request record.
- Proposer surface (Eagle/Ngram/MTP/Draft) → a `Protocol` (`propose(...) -> DraftTokens`)
  so the runner depends on an interface, not `isinstance` chains.

**Idiomatic-code target the stakeholder asked for:** not too clever, not too thin —
small **pure functions + frozen dataclasses + Protocols**, each unit-testable on CPU,
mypy-clean, reusable. `pp_spec_broadcast.py` (3 functions, gloo-tested) is the template.

## 4. The rewrite-as-contribution, decomposed into AI-solvable chunks

Frame for an AI agent: **"Own the spec-token-flow-under-PP behind a typed, tested
contract"** — NOT "rewrite the pipeline." Each chunk = one focused, independently
testable PR with a crisp spec. Chunks 0–1 are pure/local (no GPU); 2–4 componentized;
5 integration.

| Chunk | Task (poseable to an AI) | Test vehicle | Dep |
|---|---|---|---|
| **C0 — Executable spec** | Write the invariant suite (brick-81 §2) as CPU tests: 2-rank gloo for transport + synthetic `ModelRunnerOutput`/scheduler-step harness asserting greedy-count accounting across accept∈{0..k}, re-added/preempted, chunked-prefill. (Extends brick-40 `test_pp_spec_batch_queue.py` + B1a `test_pp_spec_broadcast.py`.) | CPU/gloo | — |
| **C1 — Typed state model** | Introduce `PLACEHOLDER_TOKEN_ID: Final`, `ReqIndex = NewType`, a frozen `SampledTokenGrid` wrapper (shape+layout contract + `valid_per_req()`), a `Proposer` `Protocol`. mypy-clean, no behavior change. | mypy + unit | — |
| **C2 — Transport component** | Finish/own `pp_spec_broadcast.py`: width-agnostic, typed sampled-token transport to non-last ranks. (B1a done; add typing from C1.) | gloo (have it) | C1 |
| **C3 — Scheduler placeholder discipline** | Port/adapt #40768: `num_pending_async_spec_placeholders` + `_consume_spec_decode_tokens_for_step` — emit `-1` only when req ∈ `prev_step_scheduled_req_ids`. | async_scheduler unit tests (#40768 ships them) | C0 |

> **C3 STATUS (session 5): IMPLEMENTED + TDD-green LOCALLY.** Ported to current main
> (its scheduler/async_scheduler spec code matches #40768's base near-verbatim despite the
> 1170-commit distance): `Request.num_pending_async_spec_placeholders` (`request.py`, folded
> into `num_tokens_with_spec`); `AsyncScheduler._update_after_schedule` sets the intent count
> instead of `request.spec_token_ids = self._spec_token_placeholders` (removed);
> `Scheduler._consume_spec_decode_tokens_for_step` (`scheduler.py`) emits `-1` only when
> `req ∈ prev_step_scheduled_req_ids`; cleared on `_preempt_request` + `update_draft_token_ids`.
> 5 ported unit tests green (`tests/v1/core/test_async_scheduler.py`), `test_scheduler.py`
> **101/101 no regression**, ruff clean. One integration-test assertion adjusted: current main
> calls `_update_after_schedule` at the END of `schedule()` (`:943`) → it re-reserves the
> intent for the next step right after `_consume` clears it (#40768's base called it at
> schedule-start). **NOT yet validated end-to-end → Q16** (MiMo on gpu-wb).
| **C4 — Non-last-rank input reconstruction** | A typed pure function: build non-last input_ids + position accounting from (`SampledTokenGrid`, scheduler tokens), handling common AND re-added; fix the `num_tokens_no_spec` accounting method-agnostically (resolve the ngram-gated `:1330/:1490` vs hybrid `:1498` asymmetry). | unit + MiMo | C1,C3 |
| **C5 — Integration + greedy-equiv** | Wire C2–C4, validate MiMo (non-hybrid) → 27B (hybrid) greedy-equivalence vs `base.json`. | gpu-wb | C2-4,A1c |

**Dependency note:** C3 ≈ upstream #40768; the highest-leverage, least-duplicative move is
to **co-author/validate #40768 for PP+MTP** rather than re-derive (AGENTS.md: coordinate).
C1+C0 (typed contract + test suite for an under-tested area) are a **genuinely valuable
standalone vLLM contribution** regardless of the rest — the area broke for MTP precisely
because it lacked this.

## 5. Recommended contribution arc (bigger than just "fix MTP+PP")
1. **Now (standalone PRs):** A1c (memory), B1a (broadcast width). Independent value.
2. **C0 + C1:** executable spec + typed state model for spec-under-PP — fills the test/typing
   gap that caused these bugs. Upstreamable on its own; makes everything after safer.
3. **C3:** help land #40768 (PP+MTP validation / co-author).
4. **C4 + C5:** the MTP+PP completion, on top of the typed contract.
5. **RFC** in #14044/#36643 tying it together; reference #38104/#39704/#40768 prior art.

→ This turns "fix one model's PP+MTP" into "harden + type + test vLLM's spec-under-PP
path" — the bigger contribution the stakeholder wants.

## Session 9 — stakeholder's recurring "rewrite from scratch with AI" question (recorded)

The stakeholder asked again (s9): *given how subtle these bugs are, is it viable to
have an AI write a clean spec from scratch and use it across the project — fully, or at
least some part?* Recorded with the s9 root-cause as concrete evidence.

**Assessment (honest, evidence-based):**
- **Full pipeline rewrite — NOT viable** (unchanged from s5): the async scheduler +
  batch_queue + PP + sampler glue is >1000-commit cross-cutting churn that must compose
  with every model/backend. A from-scratch parallel implementation can't track upstream
  and won't land.
- **BUT the s9 bug is the strongest argument yet FOR the surgical version of the idea.**
  Both break#2 (s8) and the s9 position bug are the SAME shape: the spec-decode token
  *accounting* (num_computed_tokens / num_tokens_no_spec / output_token_ids / valid
  counts, replicated across ranks) has **no explicit contract** — it's implicit
  invariants smeared across `_update_states`, `_prepare_inputs`, the receiver, and the
  sampler-rank correction. The async "optimistic-then-correct" pattern makes it worse:
  the truth is reconstructed late, on one rank, and the others were simply never wired.
- **So the live, high-value path is NOT "rewrite the forward" but "own + spec the STATE
  sub-mechanism":** extract the spec-decode token-accounting into a typed, unit-tested
  module with ONE enforced invariant — *num_computed_tokens (and the dependent
  positions/seq_lens/output) advances by the per-request valid count, identically on
  every PP rank* — that is impossible to drive into an inconsistent state. That is the
  C0/C1 work (brick 81), it directly *closes the class* that produced these bugs, and it
  is a real upstream contribution (fills the gap that made MTP+PP a `NotImplementedError`).
  Partial, contract-first, surgical — yes. Whole-cloth AI rewrite of the pipeline — no.

**Sequencing:** finish the s9 fix (small, mechanical, closes the immediate greedy-equiv
gap) → that gives a *worked example* of the invariant → then the C0/C1 typed state model
is "extract the invariant we just hand-proved," not a speculative redesign. The bug pays
for the spec.
