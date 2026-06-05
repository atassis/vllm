# PP + MTP spec-decode — working KB (entry point)

Living knowledge base for enabling pipeline parallelism (PP>1) **with** MTP speculative
decoding for Qwen3.5 in vLLM, **output token-identical to non-spec greedy** (no quality
regression — cf. gibberish #36872). Untracked work artifact: **NOT committed, NOT
gitignored.** Don't `git add docs/superpowers`. Full off-box pre-image:
`~/Nextcloud/Obsidian/Projects/local-llm-oset-nas/pp-mtp-kb/`.

## On «продолжаем работу» — the read ritual (small + tiered)
1. **State** — render the dashboard (single source = `state.yml`):
   ```bash
   VIRTUAL_ENV="$(pwd)/.venv" .venv/bin/python docs/superpowers/tools/build_status.py
   ```
   It prints: NEXT ACTION, deliverables (A1c/B1a/C3/C4 + verify cmd + result + PR), Q1–Q17,
   and which brick the live front needs.
2. **Latest narrative** — the last 1–2 entries of the session log in `research/pp-mtp/00-map.md`.
3. **On demand only** — the brick the NEXT ACTION points at (don't re-read all ~5000 lines).
4. **Verify, don't trust** — `file:line` are anchors, re-grep them; a claim without one is a
   hypothesis. Re-run a deliverable's `verify` command instead of trusting "it's green".

## Map of the KB
- `state.yml` + `tools/build_status.py` — **mutable state** (the only place it lives).
- `research/pp-mtp/00-map.md` — KB index: problem · solution space (Design C) · brick status · session log.
- `research/pp-mtp/{10,20,30,60}` — foundations (groups · embeddings · spec dataflow+KV · draft attn/KV PP deps).
- `research/pp-mtp/40` — **the live front**: spec under PP `batch_queue`; break #2 mechanism + the C4 fix plan.
- `research/pp-mtp/{70,80,81}` — A1c memory fix + run chronicle · async-spec-PP pipeline map · typing/rewrite chunks C0–C5.
- `specs/2026-06-05-work-branches.md` — option map / decision tree (where can we go, what's preferred).
- `runs.md` — gpu-wb run journal. `archive/` — cold (superseded specs + pre-restructure README).
- `landing/` — **separate teaching ladder** (HTML), NOT part of the engineering read.

## Decision & hardware (durable)
- **Design C** (standalone draft on the last rank: one forward flag, `draft_pp=1` bypasses the
  SupportsPP guard). B is a demoted fallback; A superseded; D parked. Rationale: brick 60 / 00-map.
- **Async is the path** (sync deadlocks on current main); V1 runner (Qwen3.5 quantized → not V2).
- **`ssh gpu-wb`** = personal LXC on `ru-oset-nas`: GPU0 RTX 4060 Ti 16G (sm89) + GPU1 RTX 5060 Ti
  16G (sm120), PCIe, **no NVLink**. **Standing auth** — may stop prod (`qwen3.5-27b`) to free both
  GPUs, no window needed. Always free GPUs after (kill stale `--query-compute-apps` PIDs).
  Models: 27B-AWQ (hybrid GDN/mamba, has `mtp.*`), MiMo-7B (cheap cascade vehicle, ~40s loads).
- **Local box** — RTX 3080 Ti (1 GPU); vLLM built in `.venv`. Single-GPU real runs work
  (`VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0`, no nvcc here); PP=2 needs gpu-wb.

## Conventions (working style)
Living code-referenced KB, brick by brick. Empirics before theory. invent>copy. Honest
calibration + pushback with evidence (engineer co-author, not yes-man). TDD red→green.
Python via `uv`/`.venv` (`VIRTUAL_ENV=$(pwd)/.venv`). Commits: `Signed-off-by` only; AI
disclosed in the PR by the human (user = submitter, defends every line — AGENTS.md).
Foundation branch is BEHIND — re-cut from the working branch at PR time. **Grab:** never
`pkill -f e3_run.py` inline in an ssh string (kills the ssh shell, exit 255); rsync WITHOUT
`--delete`.
