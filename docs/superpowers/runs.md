# gpu-wb run journal

One row per real run on `gpu-wb` (RTX 4060 Ti 16G sm89 + RTX 5060 Ti 16G sm120, PCIe,
no NVLink). Append-only; the prose blow-by-blow is archived (`archive/2026-06-04-e3-execution-log.md`).
Marker convention: `[FAIL@X]` = how far it got. gpu-wb = standing auth; free GPUs after
(kill stale `--query-compute-apps` PIDs). rsync WITHOUT `--delete`. NEVER `pkill -f e3_run.py`
inline in an ssh string (kills the ssh shell, exit 255).

| Date | Model / config | Build under test | Marker / how far | Conclusion |
|---|---|---|---|---|
| 2026-06-04 (s3) | 27B-AWQ PP=2 TP=1, no spec | baseline | ✅ generates | oracle saved → `base.json` |
| 2026-06-04 (s3) | 27B-AWQ PP=2 + MTP, async | Design C flag + 5 drafter guards | loads + executes forward; **[FAIL@MEM]** then **[FAIL@:4653]** | Q13 wall (one layer too many on 2×16G); broadcast asserts `[num_reqs,1]` vs spec width |
| 2026-06-05 (s4) | 27B-AWQ PP=2 + MTP, async, v1→v7 | A1c (int4 embed + skip lm_head) | v7 (int4 + lm_head skip + `cpu_offload_gb=3`) **fits + reaches `generate`** (gpu0 9.30 / gpu1 12.67 GiB), GDN conv1d ran; **[FAIL@:4653]** | Q13 SOLVED; next gate = broadcast width (B1a). Async state machine wired for ngram_gpu only, not MTP |
| 2026-06-05 (s5) | MiMo-7B PP=2 + MTP, async, `CUDA_LAUNCH_BLOCKING=1` | A1c + B1a (width-agnostic broadcast) | **PAST `:4653`**; **[FAIL@break#2]** `indexSelectSmallIndex` on rank0 (non-last) in forward | B1a works; break #2 mapped: non-last `input_ids` carry a `-1` on the non-common path. B1b tested+FAILED (over-advances count). sync mode tested → DEADLOCKS |
| 2026-06-05 (s6) | MiMo-7B PP=2 + MTP, async, `CUDA_LAUNCH_BLOCKING=1`, no offload | A1c + B1a + C3 (scheduler discipline) | engine constructs (gpu0 12.50 / gpu1 14.42 GiB, KV 72,368 tok), reaches `generate`; **[FAIL@break#2]** rank0 `embed_tokens(input_ids)` (`mimo.py:73`→`vocab_parallel_embedding.py:491`) | **Q16 = NO.** C3 fixes scheduler-side `-1`; break #2's `-1` is worker-side → C3 necessary-but-not-sufficient. Next = C4 (worker-side value back-write) |
| 2026-06-05 (s7) | MiMo-7B PP=2 + MTP, async, `CUDA_LAUNCH_BLOCKING=1`, no offload | A1c + B1a + C3 + **C4 (A) single-pos value back-write** | engine constructs (gpu0 12.50 / gpu1 14.42 GiB, KV 72,368 tok), reaches `generate`; **[FAIL@break#2]** still — Worker_PP0 (non-last) asserts in forward; Worker_PP1 just `irecv_tensor_dict` broken-pipe (`gpu_worker.py:831`); KeyError `scheduler.py:1431` = fallout | **(A)-single-pos INSUFFICIENT (as brick 40 §F2 predicted).** Root cause CONFIRMED by static read: the embedded `-1` originates from the **optimistic-extend** `gpu_model_runner.py:1319` `output_token_ids.extend([-1]*optimistic_num_accepted)` → copied to `token_ids_cpu` → embedded on the non-common path. Fix must backfill ALL v positions with `recv[i,0:v]` + reconcile count (B). |

| 2026-06-05 (s7-dbg) | MiMo-7B PP=2 + MTP, async, `VLLM_PP_SPEC_DEBUG=1` (throwaway, reverted) | A1c + B1a + C3 + C4(A) + instrumentation | break #2 reproduced; PPDBG captured the position trajectory | **DATA:** non-common read = `token_ids_cpu[num_computed_tokens]`; C4(A) wrote `token_ids_cpu[num_tokens_no_spec]` (misses it); the `-1` is the optimistic-extend draft placeholder (`:1319`), never backfilled. Holistic C4 design grounded (brick 40 §Session 7). (1st attempt died on `NameError: os` not imported → local `import os`.) |
| 2026-06-05 (s8) | MiMo-7B PP=2 + MTP, async, **`VLLM_USE_V2_MODEL_RUNNER=1`** (Q18 V2-viability probe), `run_mimo_v2.sh` | **clean-for-MiMo tree** (B1a/C3/C4 stashed; committed A1c/standalone-flag are Qwen3.5-only → inert for MiMo) | `gpu_worker.py:292 Using V2 Model Runner`; both ranks load (PP0 7.11 / PP1 8.66 GiB), KV cache **72,448 tok**, engine-core init; **[FAIL@HANG]** — `[OK]` (construction-complete) NEVER printed; EngineCore `shm_broadcast.py:705 No available shared memory broadcast block` repeats 5×60s (12:05→12:11), Worker_PP1 in `futex_wait_queue`, PP0 wchan=0; killed → `exit=137`, "Worker proc died unexpectedly" | **Q18 RESOLVED = stay V1.** V2 is the "right" arch on paper (MTP+PP>1 supported, quant gate liftable) but **DEADLOCKS** on our config out of the box (PP collective step-sync mismatch at warmup/first-step — the V2 analog of V1 break#2, hangs instead of OOB-crashing). Pivot ≠ free fix: swaps "finish C4 (root cause pinned)" for "debug fresh V2 deadlock". No break#2 / no `indexSelectSmallIndex` (different failure mode). gdb py-bt/native-bt yielded nothing (no debug syms); py-spy absent. GPUs freed. |

| 2026-06-05 (s8-traj) | MiMo-7B PP=2 + MTP, async, V1, `VLLM_PP_SPEC_DEBUG=1` (probe, env-gated) | A1c+B1a+C3 + **C4(A) single-slot** + PPDBG probe | captured the full multi-step write(recv)/read trajectory; **[FAIL@break#2 s6]** | **GROUND TRUTH.** `num_computed_tokens` grows by EXACTLY the prev step's valid count v (Δnct=+2,+2,+1,+1,+2 == recv-v). Single-slot C4(A) writes the bonus at the wrong (early) position and leaves accepted-draft positions = -1. recv s6 = `gathered=None` (all-chunked) → writes -1 → read at nct hits it. |
| 2026-06-05 (s8-holA) | same, **C4 holistic value-backfill** (write recv[i,0:v] at [ntns:ntns+v], advance ntns by v; output_token_ids del+extend) | C4 holistic v1 | **s1,s2 reads now REAL** (was -1); **[FAIL@break#2 s3]** | (A) value/position fix WORKS for clean decode steps. But del+extend grew output_token_ids ahead of the 1-step-lagging nct → `num_tokens(10) > optimistic_seq_len(9)` → **spurious all-chunked at s3** (`discard_request_mask`, `:2045`) → -1 written. |
| 2026-06-05 (s8-holB) | same, **decoupled** (keep token_ids_cpu multi-write + ntns+=v; output_token_ids = original single bonus append) | C4 holistic v2 | **s1–s5 reads now REAL**; **[FAIL@break#2 s6]** — SAME step the original C4(A) failed at | discard misfire pushed from s3→s6. recv s6 = `gathered=None` again — and the s8-traj (C4(A)) run ALSO had recv s6 chunked → **s6-chunked is PRE-EXISTING, not a regression.** Root = (B): optimistic-extend `-1`s accumulate in output_token_ids on the non-last rank (no sampler → `correct_spec_decode_token_counts` doesn't run there) → num_tokens inflates → eventual spurious discard. **(B) fix: trim by `correction = prev_num_draft_len - (v-1)` at the receiver (non-last analogue of correct_spec_decode_token_counts).** GPUs freed. |

| 2026-06-05 (s8-holB2) | MiMo-7B PP=2 + MTP, async, V1, PPDBG | C4 holistic **(A)+(B)**: value backfill + receiver trims output_token_ids by `prev_num_draft_len` placeholders then extends `recv[i,0:v]` | **break#2 GONE — generate completes, `exit=0`, all 5 prompts × 40 tok** | **(B) fix WORKS for the crash.** discard probe: `discard=False` every step (num_tokens now tracks optimistic_seq; was mis-firing). reads clean through s19+ (neg only on the draft/query_pos=1 slot, overlaid). FIRST end-to-end MiMo PP=2+MTP run ever. |
| 2026-06-05 (s8-base) | MiMo-7B PP=2, **no spec** (greedy oracle) | baseline | `exit=0`, 5×40 tok -> `mimo_base.json` | greedy oracle for the equiv check. |
| 2026-06-05 (s8-equiv) | compare mimo_base.json vs mimo_spec_dbg.json | — | **GREEDY-EQUIV FAILS** — all 5 seqs diverge at ~token 2 (seq0 base `[12095,13,1084,...]` vs spec `[12095,13,315,...]`) | **break#2 closed but output WRONG.** Non-last reconstruction wrote `recv[s1]col1=0` at pos6 where the true committed 2nd token is 13 -> target saw bad context -> wrongly accepted draft 315 as out[2] (base=1084). => the broadcast grid's non-bonus columns are NOT all "committed", OR my probe's recv-step vs read-step labels are misaligned. NEW GATE = grid-content semantics + step alignment (needs sender-side instrumentation). |

| 2026-06-05 (s8-det) | MiMo-7B PP=2 baseline (no spec) ×3 | baseline | `a==b==c` all 5 seqs token-identical | **Oracle is DETERMINISTIC** (het pair / bf16 / PP=2 add no run-to-run variance). So greedy-equiv divergence is a REAL bug, not a flaky oracle. |
| 2026-06-05 (s8-widthpad) | MiMo-7B PP=2 + MTP async, PPDBG | + **sender width-pad** (pad sampled_token_ids to num_spec+1 with -1 before broadcast) | `exit=0`; recv s1 now `[12095,-1]` (was `[12095,0]` garbage); **output BYTE-IDENTICAL to pre-fix; greedy-equiv STILL fails @tok2** | Sender probe pinned it: send s0 = `[[12095]]` (WIDTH-1, first decode has no scheduled spec) but receiver always reads num_spec+1 -> col1 was uninit garbage `0` committed as a token. Padding fixes that REAL bug. BUT output unchanged => **decisive: the spec output is INVARIANT to every non-last reconstruction fix (C4A -> A+B -> width-pad).** In PP the non-last GPU inputs are overlaid correct in the common case (prev_sampled + draft scatter); my cpu reconstruction only matters for the CRASH (non-common path). So the `[12095,13,315,..]` vs greedy `[..,1084,..]` divergence is the spec MECHANISM producing non-greedy tokens = a separate (likely pre-existing) MTP+PP verification/acceptance correctness bug, NOT my input reconstruction. (cf. gibberish #36872.) |

| 2026-06-05 (s8-iso) | **single-GPU** MiMo MTP (pp=1, cpu_offload=7) spec vs baseline | isolation test (PP_SIZE env added to e3_run.py) | **single-GPU spec == baseline for 4/5 seqs; seq0 diverges only @ tok29 (late)** | **DECISIVE: the gross PP=2 divergence (@tok2, all seqs) is PP-SPECIFIC.** MTP itself is ~greedy-equiv without PP (199/200 tokens; the lone tok29 diff is a minor near-tie/offload edge, not the gross corruption). So the correctness bug lives in the **PP spec plumbing**, not MTP. (single-GPU base != PP=2 base = offload/PP numeric diff, expected; the valid compare is spec-vs-base on the SAME config.) Lead: draft slot (query_pos=1) reads -1 on the non-last rank; `update_req_spec_token_ids` (gpu_input_batch.py:506) writes the draft at `num_tokens_no_spec` as an async PLACEHOLDER meant to be overwritten by the GPU draft-scatter in `_prepare_input_ids` — but on the non-last rank `_draft_token_ids is None` so the scatter is skipped. Why the placeholder is -1 (not the scheduled draft) is the next thread. |

| 2026-06-05 (s8-spectok) | MiMo-7B PP=2 + MTP async, PPDBG + spectok probe | + spectok probe in update_req_spec_token_ids | `exit=0`; **`scheduled_spec_decode_tokens = [-1]` every step on BOTH ranks** | **GREEDY-EQUIV ROOT CAUSE FULLY PINNED (PP-specific, distinct from C4).** The scheduler carries a `-1` placeholder for the draft (async design); the REAL draft lives in `_draft_token_ids` only on the LAST rank (drafter gated to last rank, `gpu_model_runner.py:547`). On the non-last rank `_draft_token_ids is None` -> the GPU draft-scatter (`:1814`) is skipped -> the non-last rank embeds the `-1` placeholder at the spec position -> wrong hidden states -> wrong verification logits on the last rank -> non-greedy acceptance. Single-GPU has no split so the draft is local -> greedy-equiv. **FIX (new piece, analogous to B1a sampled-token broadcast): broadcast draft_token_ids from the last rank to the non-last ranks + scatter them into the spec positions there.** |

| 2026-06-05 (s8-upstream) | **PURE UPSTREAM** main (merge-base 68f5e565c, fresh worktree, our foundation REMOVED) | MiMo MTP+PP=2 async | **[FAIL@LOAD] `NotImplementedError: Pipeline parallelism is not supported for this model. Supported models implement the SupportsPP interface.`** | **Answers "does upstream silently corrupt prod?": NO.** Upstream HARD-BLOCKS MTP+PP at load (model.py:1200 is_pp_supported_model gate) — not a crash, not silent garbage. Our foundation (b93190138 config-decouple + SupportsPP on Qwen3.5MTP + draft_pp wiring) is what LIFTS the gate and makes MTP+PP runnable; break#2 + the draft-broadcast gap are bugs we EXPOSE by opening the path, latent-but-unreachable on upstream. So MTP+PP is an UNSUPPORTED combo we're enabling, not a silently-broken shipped feature. (Remote restored to our tree after; worktree removed.) |

## Next planned run
**Draft-token broadcast to non-last ranks (the greedy-equiv fix; a NEW piece, analogous to B1a).**
ROOT CAUSE fully pinned (s8-spectok): the non-last rank embeds the `-1` spec placeholder because the
real draft (`_draft_token_ids`) exists only on the last rank (drafter gated `:547`); the GPU
draft-scatter (`:1814`) is skipped off the last rank. Implement: (1) last rank broadcasts its
`_draft_token_ids` (the proposed drafts for the next step) to the PP group, like
`broadcast_sampled_token_ids`; (2) the non-last rank receives them and scatters into the spec
positions in `_prepare_input_ids` (replacing the `-1` placeholder), so its verification-forward
hidden states match single-GPU. Oracle: PP=2 spec == base_a.json (modulo the tok29-class edge that
single-GPU also shows). This is a distinct contribution from the C4 input-reconstruction (which closes
break#2). Remove all PPDBG probes (read/recv/discard/send/spectok) + run_mimo_dbg/v2/sg.sh before PR.

--- superseded leads ---
**C4 greedy-equiv — PP-specific verification bug (narrowed by s8-iso).** MTP is ~greedy-equiv single-GPU;
PP=2 grossly diverges → the bug is in how the non-last rank feeds the spec/draft positions under PP.
Read-only + instrument: on the non-last rank log `scheduled_spec_decode_tokens`, the
`update_req_spec_token_ids` write range, the draft-position values in token_ids_cpu, and whether the
GPU draft-scatter runs (it needs `_draft_token_ids`, which is None off the last rank). Hypothesis: the
non-last rank embeds a wrong/-1 token at the draft (query_pos=1) position → wrong hidden states for the
spec position → wrong verification logits on the last rank → wrong acceptance. Confirm, then ensure the
non-last rank embeds the scheduled draft token. Oracle: PP=2 spec == base_a.json (modulo the same
tok29-class edge single-GPU shows). Likely a distinct area from the C4 input-reconstruction work.

--- superseded: C4 greedy-equivalence (thought to be in reconstruction; now isolated to PP plumbing) ---
**C4 greedy-equivalence — now scoped to VERIFICATION correctness (a distinct layer).** The non-last
input reconstruction (A+B+width-pad) is done and makes MTP+PP spec decode RUN without crashing, but
the spec output is invariant to those fixes and still != greedy baseline. So the remaining bug is in
the spec MECHANISM (draft proposal / verification / rejection / acceptance under MTP+PP), which my
work doesn't touch. NEXT (read-only first): map the MTP draft→verify→accept path under PP — does the
non-last rank feed the draft tokens into its embedding (read trajectory shows the draft/query_pos=1
slot = -1 on the non-last rank; if its hidden states for the spec positions are wrong, the last
rank's verification logits are wrong)? Or is the rejection comparison itself wrong under PP? Compare
to a single-GPU MTP run (should be greedy-equiv) to isolate PP-specific breakage. Likely a separate
PR/area from C4. Keep A+B+width-pad (real fixes, locally green). Remove PPDBG probe before any PR.

--- superseded: C4 greedy-equivalence (when it was thought to be in the reconstruction) ---
**C4 greedy-equivalence** — break#2 is CLOSED (A+B); the remaining gate is correctness. The non-last
rank's `recv[i,0:v]` backfill produces output that diverges from greedy baseline at ~token 2. Pin the
exact `sampler_output.sampled_token_ids` grid layout (the sender at `:4470`/`:4684`) — is column 0 the
bonus and the rest the *next* (uncommitted) draft, or are they all accepted-then-bonus? — and the
step/timing alignment between the broadcast (sender step) and the receiver/read labels (my PPDBG
recv-step ≠ read-step). Instrument the SENDER (last rank): log the engine step, per-req
`sampled_token_ids` row + `num_computed_tokens`, alongside the receiver, so recv rows map 1:1 to
committed baseline tokens. Then write only the genuinely-committed tokens. Oracle: token-identical to
`mimo_base.json`. Keep (A)+(B) (they close the crash + are locally green); this refines WHICH values
to write. Remove the PPDBG probe before any PR.

--- superseded plan (B count reconciliation — DONE) ---
**C4 (B) count reconciliation** — the ONLY remaining piece of break#2 (the (A) value/position
backfill is DONE + hardware-confirmed for all clean decode steps s1–s5). On the non-last rank the
optimistic-extend (`gpu_model_runner.py:1319`) appends `prev_num_draft_len` `-1`s to
`output_token_ids` each step but they are never corrected (the deferred
`correct_spec_decode_token_counts` runs only on the last rank, which has the sampler). They
accumulate → `num_tokens` inflates → `discard_request_mask = optimistic_seq_len < num_tokens`
(`:2045`) eventually fires spuriously → request mis-marked all-chunked → broadcast suppressed →
`-1` written → break#2 at the first such step (empirically s6 for this prompt). **Fix:** in the
receiver, trim `output_token_ids` by `correction = prev_num_draft_len - (v - 1)` (rejected drafts),
the non-last-rank analogue of `correct_spec_decode_token_counts`, so `num_tokens` tracks the real
committed length and the discard mask stops misfiring. CAUTION: this is the exact B1b double-count
site — get `prev_num_draft_len` provenance right (it may be reset/restored in `_update_states`
1438-1442). Then re-run MiMo (probe on) → expect no `gathered=None` misfire, generate completes →
greedy-equiv vs a `mode=baseline` MiMo run → then 27B vs `base.json`. Same deploy
(rsync without `--delete`; `bash run_mimo_dbg.sh` keeps the env-gated PPDBG probe;
`run_mimo.sh` for the clean greedy-equiv compare). Remove the PPDBG probe before any PR.
