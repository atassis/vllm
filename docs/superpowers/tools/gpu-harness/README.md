# gpu-wb harness (PP+MTP spec-decode)

Reusable dev/test scripts for MiMo / Qwen3.5-27B PP=2 + MTP on gpu-wb. **Not part of
any upstream PR** — these live here (KB) for version control + re-deploy; the working
copies sit at `/root/vllm-dev/` on gpu-wb (untracked there). Regenerate/scp as needed.

- `e3_run.py` — core harness. Env: `MODE`(arg1 baseline|spec) `OUT`(arg2) `MODEL`
  `PP_SIZE` `CPU_OFFLOAD_GB` `MAX_TOKENS` `MAX_MODEL_LEN` `MAX_NUM_SEQS` `NUM_SPEC`
  `QUANT_BITS`(A1c int4/8 draft). Prints `[PERF ...]` tok/s; `disable_log_stats=False`
  so `SpecDecoding metrics` (acceptance, mean accept length) hit the INFO log.
- `run_bench.sh` — baseline+spec, tokens/s + acceptance. `MODEL/OFFLOAD/QUANT/MAXT/TAG`.
- `run_accept.sh` — spec-only acceptance (short), when baseline tok/s already known.
- `run_kspec.sh` — num_speculative_tokens sweep (k=1,2,3): greedy-equiv + speedup + accept.

Oracle = greedy-equiv: spec output token-identical to a same-config baseline (modulo the
fp near-tie floor). Cleanup gotcha: never `pkill -f e3_run.py` inline in an ssh string
(kills the ssh shell, exit 255) — kill via the `nvidia-smi --query-compute-apps=pid` loop
(as the scripts do).
