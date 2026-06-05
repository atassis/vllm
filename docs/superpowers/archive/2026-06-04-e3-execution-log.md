# E3 execution log — Design C MTP+PP on real Qwen3.5-27B (gpu-wb), 2026-06-04

**What this is.** The chronological record of the first real attempt to run Design
C (MTP speculative decoding under PP=2) on the production model, on gpu-wb. It
captures every error hit, the fixes applied, the memory math, and the conclusion.
Read it with `../research/pp-mtp/40-pp-x-spec-decode.md` (the design-independent
correctness brick) and `2026-06-04-design-c-phase2.md` (the implementation plan).

**Headline.** Design C *loads and executes* on the real model (our forward-flag
path works). But MTP+PP(+async) spec on the **V1 runner is fundamentally
unfinished** — a cascade of real bugs in never-exercised code paths — **and** the
27B+draft does **not fit** on 2×16 GiB (Q13, hard). Greedy-equivalence NOT reached.

---

## 1. Environment setup (reproducible)

- gpu-wb: 2 GPUs, **both free when prod is stopped** (GPU0 RTX 4060 Ti sm89 ~16 GiB,
  GPU1 RTX 5060 Ti sm120 ~16 GiB). Prod is the user's personal box; stopping it is OK.
- **No `uv`, no dev checkout, no nvcc** on gpu-wb initially. Setup done this session:
  - install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` → `/root/.local/bin/uv`.
  - dev tree at **`/root/vllm-dev`** (rsync of local working tree incl. uncommitted
    edits; `.git` shipped too so setuptools-scm resolves the version).
  - build: `uv venv --clear --python 3.12` then
    `VLLM_USE_PRECOMPILED=1 VIRTUAL_ENV=/root/vllm-dev/.venv uv pip install -e . --torch-backend=auto`
    → `vllm 0.22.1rc1.dev173+gb93190138.d20260604.precompiled`, torch 2.11+cu130.
  - `git config --global --add safe.directory /root/vllm-dev` (files owned by root).
- Sync tooling: `docs/superpowers/tools/sync-gpu-wb.sh` (set REMOTE_DIR=/root/vllm-dev).

### Runtime gotchas (each cost a failed run)
1. **No nvcc** → flashinfer JIT dies. Run with `VLLM_USE_FLASHINFER_SAMPLER=0`. The
   attention backend auto-selects a precompiled one that works on sm89+sm120. (Note:
   `VLLM_ATTENTION_BACKEND` is **not** a recognized env in this version — backend is
   `attention_config.backend` / `AttentionBackendEnum`; auto worked, so we left it.)
2. **Multiproc spawn** (PP uses spawn) re-imports the runner module → the run script
   MUST guard with `if __name__ == "__main__":`. Without it: "An attempt has been
   made to start a new process before the current process has finished its
   bootstrapping phase."
3. **Crashed runs leak GPU memory** (orphaned `VllmWorker`/`EngineCore` hold VRAM →
   next run fails in `init_device`). Before every run, kill stale PIDs:
   `for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p; done`.

### The harness used
`/root/vllm-dev/e3_run.py` (mode `baseline`|`spec`, writes token ids to JSON) +
`e3.sh`/`spec.sh`. Oracle = greedy `base.json` (no spec) vs `spec.json` (MTP),
token-identical = pass. SamplingParams(temperature=0, max_tokens=40).

---

## 2. Baseline = PASS (table stakes, NOT the result)

PP=2, no spec, greedy → generated 5 sequences fine. Confirms build/env/attention on
this hardware and gives the oracle. (Prod already runs this; baseline proves nothing
about Design C — it only un-blocks the comparison.)

Model facts learned here: **checkpoint 20.35 GiB**, **64 layers**, hidden 5120,
**vocab 248320** (huge), `mtp_num_hidden_layers=1`, quantization `awq_marlin`,
`tie_word_embeddings=False`. rank0 (32 layers default split) load ≈ 10.72 GiB.
**Qwen3.5 is a HYBRID model** — it has **GDN (Gated Delta Net / mamba-style linear
attention) layers** (logs: "Setting attention block size … mamba page size",
"Using Triton/FLA GDN prefill kernel"). This matters for KV/memory and spec.

---

## 3. The memory wall (Q13) — confirmed HARD, with numbers

Design C loads the **draft on the last rank**: the log confirms
`llm_base_proposer.py:1339` "draft model's vocab embedding will be loaded
separately" + `:1385` "Detected MTP model. **Sharing target model lm_head** weights
with the draft model." So the draft's extra footprint = its **own vocab embedding
(248320×5120×2 B = 2.37 GiB)** + 1 layer + fc ≈ **2.7 GiB**, all on the last rank.
lm_head is shared (no extra).

Empirical per-rank reality (the planner under-estimates this — real overhead higher):
- per transformer layer ≈ 0.244 GiB; embed = lm_head = 2.37 GiB (fp16, even AWQ).
- **~2 GiB/rank fixed CUDA context** + a **~144 MiB profiling workspace** (does NOT
  shrink with `max_num_batched_tokens` — it's kernel workspace, not activation).

Splits tried (`VLLM_PP_LAYER_PARTITION`, draft on rank1):
| split | result |
|---|---|
| default 32/32, util 0.96 | rank1 OOM loading draft (needs 2.37, no room) |
| 38/26, util 0.80 | rank1 OOM (14.80 GiB used, draft can't fit) |
| 44/20, util 0.90 | rank1 OOM, short **0.36 GiB** |
| 45/19, util 0.85 | rank1 OOM, short **0.16 GiB** |
| 46/18, util 0.88–0.92 | rank1 draft FITS, but **rank0 OOM** in profiling (needs 144 MiB, ~80 free) |

**Conclusion:** rank0 fits ≤45 layers; rank1 (with draft) fits ≤18 layers →
**45+18 = 63 < 64. One layer too many.** No PP split fits. `VLLM_PP_LAYER_PARTITION`
cannot solve it. The 27B+MTP-draft is ~0.5 GiB over what 2×16 GiB physically holds.

### cpu_offload — fits, but fragile with the hybrid model
`cpu_offload_gb=8` (UVAOffloader) offloads ~7 GiB/rank to CPU RAM → rank0 load drops
to 3.35 GiB, KV profiles at 7.96 GiB. **This is the practical Q13 mitigation**
(PP=2 + offload weights to RAM, keep KV/activations/draft in VRAM). BUT it then
exposed the execution-bug cascade below, and one run hit a GDN `causal_conv1d_update`
**device-side assert** (later got past it under `CUDA_LAUNCH_BLOCKING=1` — see §4).

### The real fix for Q13 (efficiency lens, stakeholder idea)
The draft embed is a **duplicate** of the target embed (same checkpoint weights;
`mtp.*` has no embed of its own). Under PP they sit on different no-NVLink GPUs, so
vLLM loads two copies (`load_eagle_model` only shares when `world_size==1`). The
draft needs the embed on the **last** rank (its input) while the target needs it on
the **first** rank (pipeline start) — the draft fundamentally needs inputs from
**both ends** of the pipeline (final hidden state on the last rank — free under C;
token embedding on the first rank — paid as the 2.37 GiB duplicate). Removing the
duplicate (share/stream the embed) is THE fix that makes 27B+MTP fit without offload.
Not done — real code work, the recommended next unblock.

---

## 4. The spec+PP(+async) execution-bug CASCADE (V1 runner)

Reaching the forward (via cpu_offload) revealed that **MTP+PP+async spec was never
actually run on V1** — every code path that touches the drafter or spec output on a
non-last rank is broken. Auto-async is on for us (`vllm.py:969-997`: mtp ∈
EagleModelTypes + MultiprocExecutor.supports_async_scheduling). V2 runner is
unavailable (quantized Qwen3_5).

### FIXED this session — non-last-rank `self.drafter` accesses
The drafter is created **only on the last PP rank** (`gpu_model_runner.py:542`, with
the comment "currently we put the entire draft model on the last PP rank" — i.e.
vLLM already intends Design-C placement). But these ran on **all** ranks and crashed
with `AttributeError: 'GPUModelRunner' object has no attribute 'drafter'`:
1. `_dummy_run` (~5892) — drafter dummy-run during profiling.
2. `initialize_attn_backend` (~6805) — `self.drafter.initialize_attn_backend`.
3. `_check_and_update_cudagraph_mode` (~6858) — `initialize_cudagraph_keys`.
4. `initialize_kv_cache` extract-hidden-states branch (~7320).
5. `_build_attention_metadata` (~2422) — `isinstance(self.drafter, …)` in the hot path.

**Fix applied (uncommitted, in `vllm/v1/worker/gpu_model_runner.py`):** guard the
init/assert sites with `get_pp_group().is_last_rank`, and set **`self.drafter = None`
on non-last ranks** in `__init__` (so the many `isinstance(self.drafter, …)` hot-path
checks are safe). These advanced the run from "crash at profiling" → "executes the
forward pass". 15 `is_last_rank` occurrences in the file after the fix.

### GDN/mamba `causal_conv1d_update` device-side assert — NOT actually the wall
One offload run died with a Triton device-side assert in
`mamba/ops/causal_conv1d.py:1193` (`_causal_conv1d_update_kernel`). Suspected root
(by code read): under spec (`IS_SPEC_DECODING`), `conv_state_token_offset =
num_accepted_tokens[seq] - 1` (`:850`); if `num_accepted_tokens == 0` → offset −1 →
the col0..col4 loads at `:867-879` use `mask_w = idx_feats < dim` which does **not**
guard the negative token offset → OOB read. **However**, under
`CUDA_LAUNCH_BLOCKING=1` the kernel JIT-compiled and **ran** (log: "Triton kernel JIT
compilation during inference: _causal_conv1d_update_kernel"), and the run proceeded
past it — so the earlier assert was likely an async-timing artifact on a different
step, not a hard mamba blocker. Flagged as a lead to re-check (the −1 offset is still
a real latent hazard if `num_accepted_tokens` can be 0).

### NOT FIXED — PP+async sampled_token_ids shape for spec (current front)
After mamba, the run hit:
`RuntimeError: PP+async expects sampled_token_ids to have shape [num_reqs, 1]`
at `gpu_model_runner.py:4653` in `_pp_broadcast_prev_sampled_token_ids`. Under
PP+async the **last** rank broadcasts its sampled tokens to the other ranks (the
GPU-broadcast path that propagates tokens to non-last ranks), and asserts shape
`[num_reqs, 1]`. **Spec decode produces `[num_reqs, num_spec+1]`** (accepted drafts +
bonus), so the assert fails. This is the genuine **brick-40 lead #3** ("non-last-rank
accepted-draft accounting") — NOT already handled, contrary to the earlier optimistic
read. Fixing it correctly means teaching the PP token-broadcast to carry the variable
spec width (non-last ranks need the accepted tokens to advance positions/KV). This is
non-trivial and was NOT attempted blind (can't validate without a fitting model).

---

## 5. Honest status & recommended order

- **Design C code is sound** at the unit level and **loads + runs into the forward**
  on the real model. The remaining problems are NOT Design-C correctness.
- **Two independent walls remain:** (1) the spec+PP+async execution cascade on V1
  (≥1 more real bug after the 5 fixed — the sampled_token_ids broadcast; likely more
  after that), and (2) the Q13 memory duplication.
- **They're coupled:** you can't validate execution fixes without the model running,
  which needs memory solved. So the order is:
  1. **Solve memory** — implement the embed-sharing/streaming so 27B+MTP fits without
     offload (or get a smaller MTP model / more VRAM to validate correctness in
     isolation).
  2. **Then** systematically fix the execution cascade with real-run feedback
     (start: the `[num_reqs, 1]` broadcast at 4653; re-check the conv1d −1 offset).
  3. Greedy-equivalence vs `base.json` is the bar.
- **Re-scoping:** this is NOT "small flag + done". It is real multi-session
  engineering on two axes. The big positive: Design C placement is what vLLM already
  intends, and we've turned "unknown" into a precise, code-referenced bug list.

## 6. Exact deliverables (uncommitted; user is the PR submitter)
- `vllm/model_executor/models/qwen3_5_mtp.py` — Design C standalone-draft forward flag.
- `vllm/v1/worker/gpu_model_runner.py` — 5 non-last-rank drafter guards + `drafter=None`.
- `tests/v1/spec_decode/test_qwen3_5_mtp_standalone.py` — Design C forward unit test.
- `tests/v1/core/test_pp_spec_batch_queue.py` + `tests/v1/core/utils.py` — brick-40
  scheduler tests (green) + harness tweaks (ngram_gpu under async, mp backend pp>1).
- `docs/superpowers/tools/` — `plan-pp-memory.py` (memory planner; recalibrate the
  ~4 GiB last-rank overhead), `sync-gpu-wb.sh`, `deploy-gpu-wb.md`.
