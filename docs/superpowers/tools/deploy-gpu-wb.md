# Deploying our branch to gpu-wb (status: DONE in session 3)

Work artifact (untracked). **Session 3 update:** the dev tree is already built at
`/root/vllm-dev` and E3 was run (see `../archive/2026-06-04-e3-execution-log.md`).
gpu-wb prod is the **user's personal box — stopping it for runs is OK** (no formal
"window" needed; just stop prod, free the GPUs, run, restart). The "least-invasive"
notes below remain valid for rebuilding/iterating. To re-run: free the GPUs (kill
stale `nvidia-smi --query-compute-apps` PIDs), then use `sync-gpu-wb.sh`
(`REMOTE_DIR=/root/vllm-dev`) to push Python edits (editable install → no rebuild).

## Key facts driving the approach
- Our changes are **Python-only** (config + the Qwen MTP forward flag + tests).
  No C++/CUDA kernel changes → **no full rebuild needed** on the server.
- Prod runs vLLM 0.22 (its own install). We must **not** touch it. Use a
  **separate, isolated dev checkout + venv**.
- Build/install (precompiled download, editable install) uses **CPU/disk/network
  only** — it does **not** need the GPUs free, so the one-time setup below can be
  done **without a window**. Only the actual PP=2 *run* needs the window.

## One-time setup on gpu-wb (no window required)
Run these yourself on the server (e.g. via `! ssh gpu-wb` or an interactive
shell). Pick a dir isolated from prod, e.g. `~/vllm-dev`:
```bash
mkdir -p ~/vllm-dev && cd ~/vllm-dev
# Option A (preferred): clone, then we rsync over it for iteration.
git clone <this-repo-url> . && git checkout feat/pp-mtp-spec-decode
# Isolated venv; editable + precompiled so later source rsync needs NO rebuild:
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 VIRTUAL_ENV="$(pwd)/.venv" uv pip install -e . --torch-backend=auto
```
Editable install means: once set up, syncing changed `.py` files is enough — no
reinstall. (If a precompiled wheel matching the server's CUDA/arch isn't
available, fall back to a full `uv pip install -e .` build once — heavier, still
CPU-only, still windowless.)

## Iteration (no window)
From the local repo root:
```bash
# dry-run first (default), then APPLY:
docs/superpowers/tools/sync-gpu-wb.sh
APPLY=1 docs/superpowers/tools/sync-gpu-wb.sh
# CPU-only checks can even run on the server without a window, e.g.:
APPLY=1 docs/superpowers/tools/sync-gpu-wb.sh -- \
  .venv/bin/python -m pytest tests/v1/core/test_pp_spec_batch_queue.py -q
```
Set `REMOTE_DIR`/`REMOTE_HOST` env if they differ from `~/vllm-dev` / `gpu-wb`.

## E3 run (NEEDS a window — explicit, separate)
Only when a maintenance window is agreed:
1. Stop prod: `sudo systemctl stop qwen3.5-27b` (or however it's managed) —
   **user runs this**; confirm both GPUs freed via `nvidia-smi`.
2. Sync latest: `APPLY=1 docs/superpowers/tools/sync-gpu-wb.sh`.
3. Run the greedy-equivalence harness at PP=2 (see E3 checklist in the spec).
4. Restart prod afterwards.

## What I still need from you to finalize the script target
- The remote dir/host if not `~/vllm-dev` / `gpu-wb`.
- Whether a dev checkout/venv already exists there (a quick read-only
  `ls ~/vllm* ; which uv` tells us) — so we skip or do the one-time setup.
