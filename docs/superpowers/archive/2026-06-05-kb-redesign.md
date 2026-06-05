# KB / workflow redesign — weak spots + options (session 6 meta-analysis)

> Written at the end of session 6, when the stakeholder asked for a candid critique of the
> `docs/superpowers/` knowledge base + the way we work, and "interesting options, thinking
> broadly, including changing the work format." This is the **OPENING FORK** the next
> session must offer before the engineering NEXT ACTION (see `2026-06-05-next-session-prompt.md`
> step 3). Recommendation: do **Level-1** first (pays back the same session), then continue.

---

## Weak spots (lived this session, not abstract)

1. **State duplicated in ~6 places.** One finding (Q16=NO) had to be written into: README
   headline, README §7, 00-map Q16 row, 00-map session-log, brick 40 (×3 sections), the
   memory file, MEMORY.md index, next-session-prompt. → every session spends a large slice
   of budget re-syncing state instead of doing engineering; high drift risk.
2. **"Read everything" grows unbounded.** Bricks + specs + memory + landing ≈ 5000+ lines,
   +more each session. The start-of-session ritual eats context before any real work. The
   00-map session-log is append-only and never pruned; README keeps SESSION 2/3/4 narrative
   forever.
3. **Same fact in 6 docs.** break #2 now lives in bricks 40/80/81 + landing + README +
   memory. Consistent updates are expensive → copies go stale.
4. **Truth is prose-asserted, not test-anchored.** "C3 is green, trust me" instead of "here
   is the command; it passes." Line numbers drift (uncommitted edits shift them) → re-verify
   cost on every read.
5. **Fuzzy boundaries** brick / spec / session-log / memory → unclear where a finding goes,
   so it goes everywhere (feeds #1/#3).
6. **Contribution under-instrumented.** A1c/B1a/C3 are "done"/green but sit as one
   uncommitted pile; the foundation branch is BEHIND. Real risk the validated value never
   ships while we crack C4.
7. **Empirical runs are ephemeral.** gpu-wb logs are remote + overwritten; findings are
   hand-transcribed into prose. No structured run journal.
8. **Two distinct surfaces, not separated in the read ritual.** `landing/` has grown into a
   full **teaching ladder** (FOUNDATIONS → PIPELINE-NARRATIVE → PIPELINE → SPEC-PP-INVARIANTS,
   EN+RU, `index.html`/`landing.html`, `build_landing.py` + `check_refs.py`, Docker/DEPLOY —
   ~700 KB) — a *separate teaching surface* from the technical KB (`research/pp-mtp/` bricks).
   The "read ALL of `docs/superpowers/`" ritual conflates them: the generated HTML + the
   learning ladder should be **excluded from the per-session technical read** (extend it
   deliberately, don't re-read it to do engineering). [[landing-learning-ladder]]

---

## Options, cheap → bold

### Level 1 — minimal redesign (high ROI, ~20–30 min)
- **`STATUS.md` = the ONLY mutable state.** A tiny dashboard: deliverables (A1c/B1a/C3/C4)
  with `status + verify-command + last-result + PR-status`; Q1–Q17 as one table; a one-line
  **NEXT ACTION**. README collapses to a ~15-line pointer. Bricks become **append-mostly
  knowledge** (they stop holding state). → state sync 6 places → **1**.
- **Tiered reading.** Always: STATUS + the latest session-log entry. On demand: only the
  brick the NEXT ACTION needs. Archive: never, unless digging. → start-of-session cost drops
  sharply.
- **Prune/archive.** Session 2–4 narrative → `ARCHIVE.md`; keep the last 2 session-log
  entries inline.
- **`runs.md`** — one row per gpu-wb run (date / model / config / marker / conclusion),
  replacing prose run reports.

### Level 2 — verification-anchored truth
- Each deliverable carries its **verify command**; "done" = the command passes. Re-run, don't
  re-read claims (honest + self-checking; matches verification-before-completion). Treat
  line numbers strictly as "anchors, re-grep" — never as fact.
- **Structured state** (a YAML/table for Qs + deliverables) from which prose can be
  regenerated → drift becomes impossible by construction.

### Level 3 — change the work format (the interesting part)
- **Ship-as-you-go, not big-bang.** Cut A1c/B1a/C3 into PR branches **now** (they're green),
  decoupling "land standalone value" from "crack C4." Removes weak-spot #6 + the
  foundation-behind problem; also gives grounded progress while C4 resists.
- **Instrument-then-fix as the standing method** for every cascade bug (it is what worked on
  break #2: instrument → data → grounded fix, vs read-infer-guess). Make it the default loop.
- **Subagents/workflows for the rituals.** The start-of-session "read everything + reconcile
  drift + return state + next action" is a perfect Explore-subagent job: it consumes the
  ~5000 lines in *its* context and hands back a ~20-line digest, keeping the main context for
  engineering. Same for "run + parse log + conclude."
- **Contract-first C4** (brick 81 C0/C1): make the deliverable a typed, tested non-last-rank
  reconstruction component + an executable invariant spec, with the MiMo run as the
  integration check. Upstreamable on its own; turns hard debugging into a component.
- **Time-box + decision-gate on F2/C4.** The docs flag "how deep to invest" three times. Set
  an explicit gate: N sessions on C4; meanwhile the contribution track (A1c/B1a/C3/C0/C1)
  proceeds. If C4 balloons, standalone value still lands.

### Level 4 — reframe the goal (one shift, changes everything)
We found empirically that the **non-last-rank receiver path looks broken for *every* method
under PP+async+spec, not just MTP** (local greedy-equiv was pp=1, no broadcast; ngram-under-PP
is also unproven). That reframes the contribution from "fix Qwen MTP" to **"complete + test
vLLM's spec-under-PP non-last-rank reconstruction"** — bigger, more upstream-attractive, the
"bigger contribution" the stakeholder wanted (brick 81 already gestures at this). Making it
the explicit headline also justifies C0/C1 (contract + tests) as valuable regardless of
whether we finish the 27B.

---

## Recommended sequencing
1. **Level 1 first** (the same session it's proposed) — it pays back immediately.
2. Then the engineering **NEXT ACTION** (instrumentation on MiMo → C4).
3. **Level 3 "ship-as-you-go"** as the next format change once C4 has a direction (or sooner,
   if C4 stalls — land A1c/B1a/C3 to de-risk).
4. Keep **Level 4** as the framing for any RFC / PR descriptions.

> Note on what NOT to throw away: the "read everything each session" ritual the stakeholder
> instituted **did** pay off (it stopped us re-walking B1b and the sync-deadlock dead ends).
> The fix is not to abandon it but to make "everything" **small + tiered** (STATUS + recent +
> on-demand bricks) and to stop duplicating state.
