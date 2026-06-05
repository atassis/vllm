# Archived session log — sessions 1–4

> Moved out of `research/pp-mtp/00-map.md` in the session-7 KB restructure (state/narrative was duplicated across 6 docs).
> Sessions 5–6 stay inline in 00-map. Full lossless pre-image also in the Obsidian snapshot (pp-mtp-kb/).

- 2026-06-04 — KB created. Brick 10 (PP & process groups) researched & written
  from code. Confirmed: single global `_PP` per process; embed only on first
  rank, norm/lm_head only on last; layers split by `get_pp_indices`;
  IntermediateTensors cross PP via send/recv. A-vs-B reopened; C/D to explore.
- 2026-06-04 — Brick 20 (embeddings) done. **Key result:** the MTP draft creates
  `embed_tokens` unconditionally (no PP guard) → it is weight-loaded on every
  rank incl. the last (Q3 = YES). This removes the embedding obstacle from a
  draft-on-last-rank design and spawns **Design C** (refined A: a standalone
  flag in the draft forward, no separate group, no embed replication, no B
  backward-dependency). C is now the leading candidate. Next: brick 30
  (spec-decode arch) → Q5/Q9 (what feeds the draft, on which rank) + brick 60
  (does the draft layer/attention consult the PP group?) → Q4.
- 2026-06-04 — **Efficiency lens** added (stakeholder directive: hunt
  compute/memory/KV savings). Brick 30 (spec dataflow + KV) done. **Q9 = YES:**
  the last-rank drafter already holds the target hidden state (zero cross-rank
  fetch) → Design C input is free. Q5/Q10/Q11 answered: draft has own KV but
  shares block tables/slot mapping with unified rollback. **Stakeholder KV-share
  idea:** precedented (Gemma4/Step3.5 cross-model KV sharing) but inapplicable to
  Qwen3.5 (own k/v projections) and low-value (1-layer draft) — logged as a lever
  for other models. Design C feasibility now strongly supported; only Q4 (brick
  60) remains before E3. Next: brick 60 (attention/KV under PP) → close Q4, or
  jump to drafting the Design-C forward flag under TDD.
- 2026-06-04 — Brick 60 (draft attn/KV PP deps) done. **Q4 CLOSED:** the only PP
  dependency that matters in the draft path is the MTP forward branching; no
  other PP dep, no deadlock (non-last ranks have no drafter). A standalone-draft
  flag fixes it. Noted: SupportsPP (E3-step1) is **not needed for C** (only B).
  New risk **Q13**: Design C loads the whole draft (embed≈1GB + layer + lm_head)
  onto the last-rank GPU, already the fuller one — assess at E3. **Design C
  feasibility is now fully confirmed; the remaining gate is correctness, not
  feasibility:** brick 40 / Q8 (spec output through PP `batch_queue`, the #39704
  zone), which applies to any design. Next: brick 40, or draft C + go to E3;
  D likely not worth exploring unless Q13 bites.
- 2026-06-04 — Brick 40 (spec output under PP batch_queue) done. **The real
  remaining work is here and it's design-independent:** under PP the
  `batch_queue` delays `update_from_output` by k steps; three plumbing areas need
  fixing (draft-token retrieval in the normal batch_queue path; stale-snapshot
  guard for is_prefill_chunk; non-last-rank accepted-draft token accounting) —
  exactly what PR #39704 patched. Current main has the config/model groundwork
  (our commit) but NOT this plumbing; #39704's `ModelRunnerOutput.spec_token_ids`
  approach differs from main's `DraftTokenIds` path, so it's a hazard-map not a
  cherry-pick. **Q13 MITIGATED** via `VLLM_PP_LAYER_PARTITION` (stakeholder
  layer-rebalance idea — built-in knob, removes C's only downside). Net picture:
  C makes the draft side trivial; brick-40 plumbing is the bulk and the
  correctness/gibberish risk. Git: docs/superpowers now kept UNTRACKED on disk
  (per stakeholder) — only code is committed (b93190138). Next: confirm brick-40
  leads + memory headroom at E3, and/or draft the C forward flag locally (TDD).
- 2026-06-04 — **Testing-strategy breakthrough (stakeholder idea).** The brick-40
  correctness leads are scheduler/engine-core state logic → unit-testable with
  synthetic ModelRunnerOutputs + hand-computed expectations, NO GPU window for the
  bulk. Confirmed: `async_scheduling` enables `batch_queue` at pp=1 (vllm.py:497);
  `load_format=dummy` gives tiny models; **existing harness** (test_scheduler.py:131
  PP+async, :337+ spec, _make_model_runner_output) already covers the pieces
  separately — the gap is their combination. Layered plan: (1) scheduler unit
  tests pin the 3 leads locally, (2) tiny target + ngram + async on 1 GPU for
  end-to-end delay, (3) 2 ranks for the non-last-rank lead, (4) E3 only for the
  real model. **This converts "implement + iterate fixes" into "failing tests
  pin the bugs → fix to green," and shrinks the window to step 4.** Confidence in
  reaching correct batch_queue behavior rises materially. Research front is now
  effectively closed; next is design + local TDD.
- 2026-06-04 — Validated the scheduler harness locally (pp=1 spec 8/8 green; the
  one pp=2 test failed only on the world-size>GPU check → fixed via a
  `create_scheduler` mp-backend tweak in tests/v1/core/utils.py; then 20/20).
  Wrote the first brick-40 probe (tests/v1/core/test_pp_spec_batch_queue.py) but
  it does **not yet give a real red/green** — it fails on config construction
  (async scheduling requires a concrete spec method EAGLE/MTP/Draft/NGram; the
  helper sets none). **Next session: give create_scheduler a method (ngram) so
  the async+spec config constructs, then get the real signal.** Session wrapped;
  created `docs/superpowers/README.md` as the living session-continuity entry
  point (read it first). Git: only `b93190138` committed; utils.py + the new
  probe test are uncommitted; docs/superpowers untracked.
- 2026-06-04 (session 2) — **Brick-40 scheduler layer VERIFIED.** Fixed config
  construction (`create_scheduler` uses `method="ngram_gpu"` under async — CPU
  `ngram` is rejected by the async validator, vllm.py:943). Then, instead of a
  lockstep probe, built a **faithful** driver mirroring `step_with_batch_queue`
  (core.py:484): prompt-aware synthetic worker, queue genuinely 2-deep (asserted),
  invariant = "stops at EXACTLY max_tokens". **Green across** num_spec{1,2,3} ×
  accept{0..num_spec} × max_tokens, + chunked prefill (3 in-flight batches) +
  mid-pipeline stop token → **lead #2 (stale-snapshot accounting) does NOT
  reproduce as a scheduler bug.** Full test_scheduler.py 101/101 still green.
  **Corrected lead #1:** there are **two** spec-under-PP plumbing paths —
  PP-sync uses `post_step` (`take_draft_token_ids`→`update_draft_token_ids`,
  gated `not async`); PP-async uses worker-side `update_async_spec_token_ids` +
  AsyncScheduler placeholders. **Strategic fork surfaced:** prod runs PP-sync and
  the V1 runner "does not fully support async+PP" (vllm.py:504) → lead #1 *timing*
  on the sync path is the live risk; async-pp=1 tests a different path. Calibration:
  this verifies COUNT accounting only (not greedy token-VALUE equivalence — needs
  E3). Q8 → PARTIALLY CLOSED. Files: utils.py (+ngram_gpu), test_pp_spec_batch_queue.py
  rewritten (59 pass/15 skip), all uncommitted; docs updated. Next: resolve the
  async/sync fork, then lead #1 (engine-core unit test w/ mock executor, or local
  1-GPU tiny+ngram run), then the Design-C forward flag under strict TDD.
- 2026-06-04 (session 3) — **Design C BUILT + RUN on the real 27B (gpu-wb, E3).**
  Implemented the standalone-draft forward flag (Slice B, `qwen3_5_mtp.py`) + unit
  test (red→green). Slice A was already done by committed code (MTP flows through
  the `else`-branch `create_draft_parallel_config`, draft_pp defaults to 1 — earlier
  "MTP ignores draft_pp" was a misread). Local layer-2 greedy-equivalence PASS
  (ngram_gpu+async ≡ non-spec on opt-125m, 1 GPU). Built dev tree on gpu-wb
  (`/root/vllm-dev`, precompiled editable); baseline PP=2 PASS (oracle `base.json`).
  **Design C loads + executes on the real model** (draft standalone on last rank,
  shares lm_head, loads own embed). **Two walls hit:** (1) **Q13 HARD** — 27B+draft
  is one layer too big for 2×16GiB (no PP split fits; cpu_offload fits but fragile on
  the hybrid **GDN/mamba** model); (2) **spec+PP+async execution cascade** — never
  run on V1: fixed 5 non-last-rank `self.drafter` AttributeErrors
  (`is_last_rank` guards + `drafter=None`), then hit `_pp_broadcast_prev_sampled_
  token_ids` asserting `[num_reqs,1]` vs spec `[num_reqs,num_spec+1]` (lead #3, NOT
  handled — disproves the session-2 "already plumbed" optimism). GDN conv1d actually
  ran under CUDA_LAUNCH_BLOCKING (not the wall). **Greedy-equivalence NOT reached.**
  Full chronicle: `../../specs/2026-06-04-e3-execution-log.md`. **Re-scope: this is
  multi-session engineering on two axes (execution cascade + memory duplication),
  not "small flag + done".** Recommended order: solve memory (embed-sharing) first,
  then fix the cascade with real-run feedback. Fixed the stale spike-plan "Design B"
  decision (→ marked SUPERSEDED; C is chosen). All session-3 code uncommitted.
- 2026-06-04 (session 4) — **Branch-evaluation + grounding of the chosen attack.**
  Mapped all work branches across 5 axes (memory / execution / validation / design
  alts / upstream). **Stakeholder chose A3 + A1c** as the first connected pair.
  Grounded both into **brick 70** (code-referenced), with two corrections to the
  research: (1) **A3** — greedy-equiv is weight-agnostic (`rejection_sampler*`) so
  `load_format=dummy` validates the cascade WITHOUT solving Q13 (decouples the two
  walls); but dummy → accept≈0, so it reproduces the broadcast shape bug yet needs
  **MiMo-7B real weights** to cover accepted-token accounting. (2) **A1c** — the
  int8-embed hook is the **PP separate-load path** (`llm_base_proposer.py:1338`
  else), NOT `_maybe_share_embeddings` (that's `world_size==1` only — a hook there
  is a no-op under PP); TP=1 makes a lookup+dequant int8 embed trivial; no
  off-the-shelf quantized embed exists (only GGUF). Insight: the **embed-sharing**
  framing in the README is superseded by **A1c (quantize, don't share)** — sharing
  is physically impossible across the no-NVLink PP pair; the real lever is "the
  draft is allowed to be lossy." Upstream plan (stakeholder): stage clean slices in
  a fork branch, finish the task, then ship as a series of PRs. New Qs: Q14 (tiny
  config → GDN?), Q15 (MiMo-7B PP=2+MTP without SupportsPP?). Next: code A1c (int8
  draft embed, local unit test) + A3.5 (2-rank gloo broadcast TDD) — both local, no
  prod window; then tiny-config / MiMo repro on gpu-wb.
- 2026-06-05 (session 4, end) — **A1c IMPLEMENTED + VALIDATED on the real 27B;
  memory wall SOLVED; execution cascade reached.** `QuantizedVocabEmbedding`
  (int8/int4, **load-time** quant — post-load swap OOMs at the peak) +
  `draft_embed_quant_bits` config knob + qwen3_5_mtp constructs it + **skip draft
  lm_head alloc** (PPMissingLayer; it's shared with target). 17 unit tests green.
  gpu-wb runs v1→v7: OOM marched downstream each fix → **v7 (int4 + lm_head skip +
  small `cpu_offload_gb=3`) FITS + reaches `generate`** (gpu0 9.30/gpu1 12.67 GiB;
  GDN conv1d ran — not the wall), then hit **brick-40 lead #3** broadcast `:4653`
  (`[FAIL@GENERATE]`, past memory). Finding: the async-spec-PP state machine is
  wired for **ngram_gpu only**, not MTP (e.g. the `is_ngram_gpu` gate at
  gpu_model_runner.py:1330). Working-branch commits `1f6d2ff31`/`67f677c63`/
  `e45e5d462` on top of foundation (foundation is BEHIND — re-cut at PR time).
  Downloaded **MiMo-7B** as the cheap cascade vehicle (MiMoMTP already standalone).
  Q13 → **SOLVED**. **Next gate = B1 (execution cascade); option map in
  `../../specs/2026-06-05-work-branches.md`.** Recorded gpu-wb standing
  authorization. Brick 70 holds the full A1c chronicle.
