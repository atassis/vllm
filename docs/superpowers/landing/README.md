# vLLM V1 pipeline — research artifacts (the "ladder")

Read-only research + teaching artifacts. **Untracked work** — do **not** `git add`
(later they can move to `~/Nextcloud/Obsidian`). Built as a four-floor learning ladder:
each floor is a different altitude, and each links down to the one below.

## The ladder (read in this order)

| Floor | File (EN / RU) | What it answers |
|---|---|---|
| **0 — Foundations** | `FOUNDATIONS.md` / `.ru.md` | *What an LLM is made of* — token→embedding→layers→attention→**KV cache**→head→logits→sampling, how MTP works, and the math. Built from Markov chains / state machines. |
| **1 — Why** | `PIPELINE-NARRATIVE.md` / `.ru.md` | *Why the serving system is shaped this way* — every component as a forced move (pressure → naive → where it breaks → the fix → new cost). |
| **2 — The map** | `PIPELINE.md` / `.ru.md` | *Where everything lives* — precise `file:line` reference for all 9 subsystems + glossary + weak spots. |
| **3 — Change map** | `SPEC-PP-INVARIANTS.md` / `.ru.md` | *What you must not break* when editing spec-under-PP — invariants, the impact map, the test that catches each violation. Task-shaped for the F2/C4 fix. |

## The reading surfaces

- **`landing.html` / `landing.ru.html`** — the **book**: all four floors in ONE
  self-contained, offline, mobile-first HTML file. **This is what you deploy / read on a
  phone.** Generated from the floor `.md` files by `build_landing.py`.
- **`index.html` / `index.ru.html`** — the **interactive** landing (clickable
  architecture diagram, lifecycle stepper, engine-mode timeline, searchable glossary,
  severity-tagged weak spots). Best on a laptop.

## Tooling

- **`build_landing.py`** — assembles the floor `.md` files into the single-file book.
  Run after editing any floor: `python3 build_landing.py` (EN) / `--ru` (RU).
- **`check_refs.py`** — validates that every `file:line` reference still resolves in the
  tree (turns "code-referenced" into a test; anti-rot). `python3 check_refs.py`.
  It already caught one wrong path (`processor.py` → `input_processor.py`).
- **`Dockerfile` / `default.conf` / `DEPLOY.md`** — serve the folder over nginx (e.g. on
  gpu-wb via headscale) so you can read it from your phone. See `DEPLOY.md` for all
  options (Docker, `python -m http.server`, or just open the file directly).

## Provenance

Line numbers are anchors from the working tree at HEAD `e45e5d462` (branch
`feat/pp-mtp-spec-decode`, which carries uncommitted spec-under-PP changes). `check_refs.py`
guards them. This landing does **not** touch the technical KB under
`docs/superpowers/research/` or the session specs — those are the live work surface.
