# Session-7 start — what "продолжаем работу" expands to

> The user may now just type **«продолжаем работу»** with NO prompt. This file (and the
> auto-loaded memory) is the bootstrap. State is a clean boundary: break #2 fully
> diagnosed; the next action is a single instrumentation run on MiMo.

---

## On "продолжаем работу" — do this, in order
1. **Read ALL of `docs/superpowers/` before any work** (living code-referenced KB — exists
   so you don't re-walk dead ends). Order: `README.md` → `specs/2026-06-05-work-branches.md`
   → `research/pp-mtp/00-map.md` (Q1–Q17 + session log) → bricks `10/20/30/40/60/70/80/81`
   → specs (`design-c-phase2`, `e3-execution-log`). **Brick 40 §"Session 6" + §"F2/C4
   grounded diagnosis" + §"F2/C4 full accounting trace" are the live front — read them
   carefully.** NOTE: `docs/superpowers/landing/` is now a **separate teaching ladder**
   (FOUNDATIONS→NARRATIVE→PIPELINE→INVARIANTS, EN+RU, generated HTML, ~700 KB) — a teaching
   surface, NOT part of the mandatory technical read; extend it deliberately, don't re-read
   it to do engineering ([[landing-learning-ladder]]).
2. **Verify the working tree still matches the docs** (honest calibration): C3 symbols
   (`grep num_pending_async_spec_placeholders vllm/v1/request.py`), B1a wiring
   (`grep pp_spec_broadcast vllm/v1/worker/gpu_model_runner.py`). Optionally re-run the
   green check (below).
3. **OPENING FORK — raise this BEFORE the NEXT ACTION (the user explicitly asked s6 that
   you proactively propose it).** Offer the **Level-1 KB redesign** and ask the order:
   - **(a) Redesign first (~20–30 min), then instrumentation** — create `STATUS.md` (the
     ONLY mutable state: deliverables A1c/B1a/C3/C4 with status + verify-command + result +
     PR-status; Q1–Q17 table; one-line NEXT ACTION), collapse `README.md` to a ~15-line
     pointer, add `runs.md` (one row per gpu-wb run), archive sessions 2–4 narrative to
     `ARCHIVE.md`, keep last 2 session-log entries inline. Payoff: stop duplicating state
     across ~6 docs + stop reading ~5000 lines each session.
   - **(b) Straight to the NEXT ACTION** (instrumentation on MiMo), redesign later.
   Recommend (a) — it pays back the same session. This is a genuine fork → ask, don't assume.
   **Full rationale + all weak spots + Level 1–4 options: `specs/2026-06-05-kb-redesign.md`.**
4. **Continue from THE NEXT ACTION** (below). Confirm you read everything + understood
   state, then proceed (the user trusts you to start; ask only at genuine forks).

## State (session 6 end — don't trust this summary over reading the docs)
- **A1c** (memory, Q13) — DONE, validated on 27B.
- **B1a** (broadcast width) — DONE, gloo-tested + MiMo past `:4653`.
- **C3** (#40768 scheduler placeholder discipline) — ported, **local-green**
  (`test_async_scheduler` + `test_pp_spec_broadcast` 17 passed, `test_scheduler` 101/101,
  ruff clean — re-verified s6).
- **Q16 = NO (s6, MiMo gpu-wb):** C3+B1a do NOT close break #2. Engine reaches `generate`,
  break #2 (`indexSelectSmallIndex`) STILL fires on rank0 (non-last) at
  `embed_tokens(input_ids)` (`mimo.py:73` → `vocab_parallel_embedding.py:491`). Root `-1`
  is **worker-side** (non-last `token_ids_cpu` never gets the real value under async PP),
  not scheduler emission → C3 = necessary-but-not-sufficient (stays a valid green standalone).
- **break #2 fully diagnosed** (brick 40): no value-back-write site exists for the non-last
  rank under async PP; ngram populates via its gated lines + `update_ngram_gpu_tensors_
  incremental`, MTP has no equivalent. Fix = **C4 (A)** value back-write in the receiver
  `_pp_receive…:4694` (write `recv`'s confirmed value into `token_ids_cpu[i,pos]`), possibly
  + **(B)** a single-site `num_tokens_no_spec` advance to `valid_count`.

## THE NEXT ACTION (decided, grounded)
**ONE targeted instrumentation run on MiMo** (env-gated, like s5 `VLLM_PP_SPEC_DEBUG`): per
non-last-rank step, log per req `{prev_index, num_computed_tokens, num_tokens_no_spec
pre/post branch-2 (`:1418`) and pre/post receiver (`:4696`), pos, recv row, valid_count}`.
This turns **(A)-only vs (A+B)** into DATA (the one blind spot pure reading can't close: the
exact count trajectory across the k-step pipeline delay). Then implement C4 grounded — TDD a
pure helper if it factors cleanly (style: `pp_spec_broadcast.py`), MiMo greedy-equiv as the
integration oracle → then 27B vs `base.json` (needs A1c int4 + `cpu_offload_gb=3`).

Uncommitted coherent set unchanged: B1a (`gpu_model_runner` + `pp_spec_broadcast.py` + test)
+ C3 (`request`/`scheduler`/`async_scheduler` + `test_async_scheduler`) + test infra
(`utils.py`). `docs/superpowers/` untracked.

## Green check (local, ~2 min, before touching GPU)
```bash
VIRTUAL_ENV="$(pwd)/.venv" .venv/bin/python -m pytest \
  tests/v1/core/test_async_scheduler.py tests/v1/spec_decode/test_pp_spec_broadcast.py -q
# (heavier) tests/v1/core/test_scheduler.py  → expect 101 passed
VIRTUAL_ENV="$(pwd)/.venv" .venv/bin/python -m ruff check vllm/v1/core/sched/ vllm/v1/request.py vllm/v1/worker/pp_spec_broadcast.py
```

## Deploy to gpu-wb (Q16 mechanics, reuse for the instrumentation run)
- Both GPUs were free (prod idle); gpu-wb = standing auth, no window.
- **Manual rsync WITHOUT `--delete`** (only changed files), e.g.:
  ```bash
  rsync -azR --no-perms vllm/v1/worker/gpu_model_runner.py vllm/v1/worker/pp_spec_broadcast.py \
    vllm/v1/core/sched/async_scheduler.py vllm/v1/core/sched/scheduler.py vllm/v1/request.py \
    gpu-wb:/root/vllm-dev/
  ```
- Run: `ssh gpu-wb 'cd /root/vllm-dev && CUDA_LAUNCH_BLOCKING=1 bash run_mimo.sh'` (writes
  `mimo_run.log`, `mimo_spec.json`). Poll the log for `[OK] engine`/`FAIL@`/assert.
- **Grabли:** never `pkill -f e3_run.py` in an inline ssh string (kills the ssh shell, exit
  255 — hit in s6); cleanup via compute-app PIDs only. Free GPUs after.

## Conventions (working style)
Living code-referenced KB, brick by brick (a fact without `file:line` = hypothesis).
Empirics before theory. invent>copy. Honest calibration + pushback with evidence (engineer
co-author, not yes-man). Everything researched → docs (bricks + 00-map + session log). TDD
red→green. Python via `uv`/`.venv` (`VIRTUAL_ENV=$(pwd)/.venv`). `docs/superpowers/` NOT
committed, NOT gitignored. Commits: `Signed-off-by` only, no `Co-authored-by` — the human
discloses AI in the PR. I (the user) am the human submitter, defend every line (AGENTS.md).
Foundation branch is BEHIND — re-cut from the working branch at PR time.

> ⚠️ Meta-note: the KB/workflow redesign is the **OPENING FORK (step 3 above)** — the user
> asked you to proactively offer it before the NEXT ACTION, not bury it. If `STATUS.md`
> already exists, option (a) was chosen in a prior session and the structure changed —
> read `STATUS.md` first and treat it as the single source of state.
