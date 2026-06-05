# Brick 70 — Memory fix (A1c) & cheap validation (A3): unblocking the two walls

Status: **RESEARCHED (session 4)** · Grounds the chosen first branches A3 + A1c ·
Verified against current code (paths relative to repo root).

> Session-3 left two coupled walls (e3-execution-log): Q13 memory (27B+draft >
> 2×16 GiB) and the spec+PP+async execution cascade. This brick grounds the two
> branches the stakeholder chose to attack them: **A3** (validate the cascade on a
> cheap model, decoupling it from memory) and **A1c** (quantize the draft embed to
> int8 — the real, correctness-safe memory fix). Both are code-referenced below.

---

## A3 — cheap validation of the execution cascade

### A3.1 Greedy-equivalence is weight-agnostic (the key enabler)
The correctness oracle is "spec output ≡ non-spec greedy output of the **same**
model." Under greedy (`temperature==0`) the rejection sampler accepts a draft
token **iff it equals the target's argmax**, else substitutes the target argmax:
- `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:224-253` — greedy
  path: `accepted &= target_argmax == draft_sampled`; output `draft_sampled if
  accepted else target_argmax`.
- `vllm/v1/sample/rejection_sampler.py:452` — `target_argmax =
  target_logits.argmax(dim=-1)`; greedy kernel keyed on argmax, not magnitudes.

**→ implication:** the output is a pure function of (target argmax sequence, draft
token sequence). Weight *quality* changes only the argmax *values* and the
acceptance *rate*, never the spec-vs-non-spec *equality*. So **`load_format=
"dummy"`** (random weights, `vllm/config/load.py:42-43`) gives a VALID
greedy-equivalence test. → We do NOT need the real 27B (or any good weights) to
validate execution correctness. **This is what decouples the cascade from Q13.**

### A3.2 Caveat — dummy weights give acceptance ≈ 0 (coverage gap)
With random weights the draft argmax matches the target argmax with prob ≈
1/vocab → **almost every draft token is rejected.** Consequences:
- **Still reproduces the broadcast bug:** `sampled_token_ids` has a **fixed**
  width `[num_reqs, num_spec+1]` regardless of how many are accepted (the accepted
  count is applied *after*, in accounting), so the `:4653` assert
  `shape[-1]==1` fails the same way. ✓ Dummy suffices to repro + verify the fix's
  shape handling.
- **Does NOT exercise accepted-token accounting:** the non-last-rank
  position/KV advance by the **variable accepted count** (the receiver
  `gpu_model_runner.py:4663-4691` currently appends exactly one `-1`/req). With
  ~0 acceptance this path is barely hit. → To cover it, use **real small weights**
  so acceptance > 0.

### A3.3 Smallest real MTP model + PP precedent
- **MiMo-7B-Base** (`tests/v1/spec_decode/test_mtp.py:30`,
  `tests/v1/spec_decode/test_max_len.py:88`) — the only small real MTP checkpoint
  used in tests (`MiMoMTP`, `vllm/model_executor/models/mimo_mtp.py:154`). **Does
  NOT declare `SupportsPP`** → would trip the guard (`vllm/config/model.py:1199`).
  Under **Design C (`draft_pp=1`) the guard does not fire** (pp=1 short-circuits),
  so MiMo-7B may run PP=2+MTP without adding SupportsPP — confirm.
- **Qwen3_5MTP / Qwen3_5MoeMTP** declare `SupportsPP` (`qwen3_5_mtp.py:365,477`)
  but are 27B/35B — our prod target, not "cheap".
- **PP-parametrized spec test exists but is config-only:**
  `tests/v1/spec_decode/test_eagle.py:738` `@parametrize("pp_size",[1,2])` →
  `test_load_model` mocks `get_pp_group`, asserts embed/lm_head sharing under PP
  (`:827-831`), **never runs forward/propose**. Useful template for a PP-config
  unit test; not an execution test.

### A3.4 Recommended A3 ladder (cheap → complete)
1. **Tiny synthetic Qwen3.5 config + `load_format=dummy` + PP=2** (gpu-wb) — same
   `Qwen3_5MTP` code path we are fixing, at tiny scale, near-zero memory → repro
   the full cascade (drafter guards already fixed; next: the `:4653` broadcast)
   with real-run feedback, no Q13. (Needs a tiny config.json with the Qwen3.5
   arch + `mtp_num_hidden_layers=1`; check whether it instantiates GDN layers —
   if so we also exercise the conv1d `-1` hazard, e3-log §4.)
2. **MiMo-7B-Base real weights + PP=2** (gpu-wb) — acceptance > 0 → covers the
   accepted-token accounting paths dummy can't.
3. **Real Qwen3.5-27B PP=2** — final greedy-equivalence vs `base.json` (needs A1c
   or offload to fit).

### A3.5 Local proxy for the broadcast fix (no GPU window) — "C1"
`_pp_broadcast_prev_sampled_token_ids` (`gpu_model_runner.py:4646`) +
`_pp_receive_prev_sampled_token_ids_to_input_batch` (`:4663`) are a
`torch.distributed.broadcast` between PP ranks. The **shape logic** (sender packs
variable spec width; receiver advances `output_token_ids`/`num_tokens_no_spec` by
the per-request accepted count instead of a single `-1`) is unit-testable with a
**2-rank gloo group on CPU** + synthetic tensors — no model, no CUDA. This TDDs
the fix before the gpu-wb run.

---

## A1c — quantize the draft embed to int8 (the memory fix)

### A1c.1 The duplicate, confirmed
`Qwen3_5MultiTokenPredictor.__init__` creates `embed_tokens` with **no
quant_config** (`qwen3_5_mtp.py:91-94`) → `UnquantizedEmbeddingMethod`, loaded at
the checkpoint dtype (**fp16**, `vocab_parallel_embedding.py:437,449`). vocab
248320 × hidden 5120 × 2 B = **2.37 GiB**. lm_head is shared (e3-log §3), embed is
the only duplicate.

### A1c.2 The hook is the SEPARATE-LOAD path, not the share path (correction)
`_maybe_share_embeddings` (`llm_base_proposer.py:1275-1342`) only shares when
`get_pp_group().world_size == 1` (`:1282`). **Under PP (`world_size>1`) it takes
the `else` (`:1338`): "draft model's vocab embedding will be loaded separately."**
So for our PP case the embed is the draft's **own** module — the A1c hook must be:
- (a) in the draft model construction/weight-load (`qwen3_5_mtp.py`), or
- (b) a post-load int8 cast of the draft's own `embed_tokens` after the draft
  loads (in the proposer, on the `world_size>1` branch),

NOT in the share branch (that path is dead under PP — a hook there is a no-op).

### A1c.3 No off-the-shelf quantized embedding
`VocabParallelEmbedding` accepts `quant_config` (`vocab_parallel_embedding.py:240`)
but **only GGUF implements the `embedding()` quant method**
(`quantization/gguf.py`); AWQ/GPTQ/fp8 do not → passing their quant_config raises
`NotImplementedError` (`vocab_parallel_embedding.py:283-287`). Embeddings are
normally in the quant skip-list (`is_layer_skipped`,
`quantization/utils/quant_utils.py:499-553`; test
`tests/quantization/test_lm_head.py:46` asserts `UnquantizedEmbeddingMethod`).
**→ we must add a minimal int8 embedding path ourselves.**

### A1c.IMPL — numerical core DONE (session 4, TDD)
`vllm/model_executor/layers/quantized_draft_embedding.py` —
`QuantizedVocabEmbedding(weight, bits)`: per-row symmetric quant + lookup/dequant.
- `bits=8`: int8, 1 byte/weight (~2x, saves ~1.18 GiB).
- `bits=4`: int4 packed 2 nibbles/uint8 byte, 0.5 byte/weight (~4x, saves ~1.78
  GiB → 27B+draft fits with margin). Requires even hidden (always true).
Tests (CPU, green): `tests/v1/spec_decode/test_quantized_draft_embedding.py` —
lookup ≈ fp16 within quant error (int8 <0.03, int4 <0.5) + storage byte-width.
The int4 lookup test passing also validates nibble pack/unpack ordering.
**A1c integration DONE locally (session 4, TDD — on the WORKING branch, not
foundation):**
1. **Config knob** — `SpeculativeConfig.draft_embed_quant_bits` (None/8/4) +
   `_verify_draft_embed_quant_bits` validator + `__post_init__` call
   (`vllm/config/speculative.py`).
2. **Proposer wiring** — `SpecDecodeBaseProposer._maybe_quantize_draft_embed`
   (`vllm/v1/spec_decode/llm_base_proposer.py`), called in `load_model` after the
   share steps; gated on `bits is not None and get_pp_group().world_size > 1`
   (PP separate-load path only — under world_size==1 the embed is shared with the
   target and must NOT be mutated). Swaps the draft's own `embed_tokens` for a
   `QuantizedVocabEmbedding` built from the loaded fp16 weight.
   Tests (CPU, green): `tests/v1/spec_decode/test_draft_embed_quant_integration.py`
   — validator (None/8/4/reject) + swap-under-PP + no-op when bits=None + no-op
   when world_size==1. Full A1c suite 17 green; ruff check+format clean. (Also
   reformatted the foundation embed module to satisfy `ruff format` and re-amended
   foundation `16503aada`.)
**RUN-A FINDING (session 4, real 27B PP=2, gpu-wb) — post-load swap is TOO LATE.**
Ran spec + `draft_embed_quant_bits=4`, `cpu_offload_gb=0`, util 0.90. **rank1 OOM
at MODEL LOAD** (`gpu_model_runner.py:5186`: "Tried to allocate 140 MiB, 80 MiB
free, 15.39 in use"), before forward; the "Quantized draft" log line never
appeared → `_maybe_quantize_draft_embed` never ran. Root cause: the draft builds a
full **fp16 `VocabParallelEmbedding` (2.37 GiB) on the GPU at construction/load**,
and the post-load swap can only free it *after* the load peak — but the OOM is *at*
that peak (exactly session-3's 63<64 deadlock; the peak still carries fp16). →
**A1c must quantize at LOAD time** (construct int storage directly + a quantizing
`weight_loader` so the fp16 [V,H] never materializes on GPU), NOT post-load. Numbers
confirm load-time int4 fits: rank1 ~15.4→~13.6 GiB, room for KV. Post-load proposer
hook (`_maybe_quantize_draft_embed`) is being replaced by construction-time quant in
`qwen3_5_mtp.py`.

**RUN-A v2 FINDING (load-time int4 works; SECOND duplicate surfaced).** With
construction-time int4 embed (no offload), rank1 OOM moved from "140 MiB short,
15.39 in use" → "**Tried to allocate 2.37 GiB**, 1.16 free, **14.31 in use**". So
the int4 embed worked (in-use dropped ~1.1 GiB; the embed is no longer the failing
alloc), but a **second 2.37 GiB fp16 table** now OOMs: the draft's **own lm_head**,
constructed in fp16 at __init__ *before* `_maybe_share_lm_head` swaps it for the
target's (shared) lm_head. e3-log §3 called lm_head "shared, no extra" — true for
steady state, but the **construction transient** is a real peak. So the memory wall
is TWO construction-time 2.37 GiB duplicates (embed — fixed by A1c; lm_head — open).
The lm_head transient is FREED after sharing, so it only needs load-peak headroom →
trying `VLLM_PP_LAYER_PARTITION=40,24` (shift target layers off rank1) + int4 embed,
no offload, to fit. Proper fix later: don't construct the draft's own fp16 lm_head
when it will be shared (it's on the same rank as the target lm_head → shareable).

**RUN-A v5–v7 — A1c VALIDATED + memory wall solved; execution cascade reached.**
The OOM marched downstream each fix (proof A1c works): v2 lm_head-transient → (add
lm_head skip) v5 model loads at default 32/32, only 432 MiB short at KV-profiling →
v6 144 MiB short → **v7 (int4 embed + lm_head skip + `cpu_offload_gb=3`) FITS**:
`[OK] engine constructed (model + KV fit)`, gpu0 9.30 / gpu1 12.67 GiB, KV 804
tokens / 6.29x, GDN `causal_conv1d` kernels JIT-compiled & **ran** (mamba NOT the
wall, confirmed). Reached `generate` → hit **exactly** the broadcast cascade:
`AssertionError: PP+async expects sampled_token_ids to have shape [num_reqs, 1]`
(`gpu_model_runner.py:4653`) — brick-40 lead #3, `[FAIL@GENERATE]` not `[FAIL@LOAD]`
(past memory). So: **A1c + a SMALL offload (3, vs session-3's fragile 8) gets the
real 27B running**; without offload we were 144 MiB short (A1c did the heavy lift
from a hard "one layer too many" deadlock). **Next axis = the broadcast fix (B1).**
MiMo-7B-Base (15G, has MTP, MiMoMTP already standalone — no Design-C fix needed)
downloaded as the fast/clean cascade-iteration vehicle (fits PP=2 with huge margin).

**Remaining A1c work (needs gpu-wb / real model):** (3) confirm last-rank VRAM
actually drops + the **acceptance vs bits sweep** {fp16, int8, int4, (nf4?)} on
MiMo-7B / 27B — correctness holds at all, pick the lowest bits that keeps
acceptance ~71%. nf4 is a future cycle if int4 acceptance is too low.

### A1c.4 Simplest correctness-safe design (TP=1)
Since our config is **TP=1**, `VocabParallelEmbedding.forward` is a plain lookup
(no mask, no all-reduce — `vocab_parallel_embedding.py:477-497`). So a draft-only
int8 embed needs only: store int8 weight + per-row (or per-column) fp16 scale;
on lookup, gather rows and dequant to fp16. No all-reduce concerns (those are
TP>1). Correctness is guaranteed by rejection sampling (A3.1) — a lossy draft
embed only lowers acceptance. int8 saves ~1.18 GiB (need ~0.5) → 27B+draft fits
without cpu_offload. int4 (~1.78 GiB saved) is the stretch option if int8 is tight.

---

## Open questions spawned / updated
- **Q13 → path to closure:** A1c (int8 draft embed) is the real fix; A3 +
  `load_format=dummy` lets us validate the cascade *without* closing Q13 first.
- **Q8 (the cascade):** next concrete front is the `:4653` broadcast shape; repro
  via A3.1 (dummy) and/or TDD via A3.5 (2-rank gloo).
- **New Q14:** does a tiny synthetic Qwen3.5 config instantiate GDN/mamba layers
  (so we also test the conv1d `-1` hazard), or pure attention? (A3.4 step 1.)
- **New Q15:** does MiMo-7B run PP=2+MTP under `draft_pp=1` without `SupportsPP`
  (guard short-circuits at pp=1)? (A3.3.)

## Session 9 — gateway-deploy memory map + prod-regime correctness (2026-06-05)

**Correctness under the FULL prod regime — VALIDATED.** The gateway (`/opt/llm`, router +
systemd vLLM backends, all `--pipeline-parallel-size 2`, currently NO spec) runs 27B with
`--max-num-seqs 16 --enable-prefix-caching --kv-cache-dtype fp8 --max-model-len 16384`.
Ran 27B PP=2 + MTP at batch=16 + prefix-caching + fp8 (cpu_offload to fit): **spec ==
baseline 5/5 token-identical.** Crucially, `--enable-prefix-caching` flips
`mamba_cache_mode` to **'align'** (config.py:355) — the mode the s9 GDN fix flagged as
untested. **Align is now tested under PP+MTP+batch and works** → s9 caveat resolved. Net:
the fix is correct in the exact prod engine config; the ONLY blocker is VRAM.

**Memory wall on 15.5 GiB GPUs (why MTP won't fit without offload at prod batch/context).**
Weight breakdown (`tools/gpu-harness/wbreakdown.py`, reads safetensors headers): total weights
**20.35 GiB** for "27B-AWQ":
| component | GiB | % |
|---|--:|--:|
| target GDN linear-attn (48 layers) | 9.47 | 46.5 |
| target full-attn (16 layers) | 4.50 | 22.1 |
| embed_tokens (UN-quantized) | 2.37 | 11.6 |
| lm_head (UN-quantized, NOT tied) | 2.37 | 11.6 |
| vision/MM tower (333 tensors) | 0.86 | 4.2 |
| MTP/draft | 0.79 | 3.9 |

Per-rank (default 32/32): rank0 = embed 2.37 + vision 0.86 + ~7.0 layers ≈ 10.2; **rank1
(the OOM rank) = lm_head 2.37 + draft 0.79 + ~7.0 layers ≈ 10.2 + runtime GDN state(×16) +
KV.** Empirical [MEM]: gpu0 10.61 / gpu1 12.85 (with offload=5). At batch=16 even BASELINE
fits only ~4704 ctx at util 0.95 (not the prod 16384 — that flag may be aspirational/untested).

**Compression levers to fit MTP WITHOUT offload (ranked by impact on rank1):**
1. **Quantize lm_head** (2.37 GiB UN-quantized, sits on rank1 = the OOM rank) → int8 ~1.2 / int4
   ~0.6 GiB, frees ~1.2-1.8 GiB exactly where needed. Natural A1c extension (A1c already int4s
   the draft embed + shares lm_head). Highest-impact, targeted.
2. **Drop the vision tower** (0.86 GiB, loaded but unused at `--limit-mm 0`) — free rank0 so Q13
   can shift more layers off rank1.
3. **Q13 `VLLM_PP_LAYER_PARTITION`** — move target layers rank1→rank0 (38/26 helped baseline but
   draft still OOM'd alone; combine with #1+#2).
4. embed_tokens quant (2.37, rank0 — lower priority, rank0 has headroom).
5. GDN runtime state fp32→bf16 (`mamba_ssm_dtype`) — risky for SSM stability; last resort.

**Bench under batch=16+offload:** baseline 5.48 vs spec 8.12 tok/s = **1.48× greedy-equiv**.
=> Deploy options for the gateway: (a) enable MTP with cpu_offload now (works, 1.48×, slower
abs), or (b) implement lm_head-quant (#1) + vision-drop (#2) to fit without offload at full
speed. The fix itself is done; this is deployment memory-tuning.
