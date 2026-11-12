# Zero-stall lean Foundry LOAD at 83 capture sizes (vLLM 0.31, GLM-5.2-NVFP4 TP4+EP, 2026-11-12)

Pod rp-ldf-w1 (4x GB200, w0e-...-mb4d), branch rvinodkumar/v031-nowarmup. Same load, engine config, chain and
analysis as RESULTS_SBS.md (83 sizes 1..1024, 32K/1K at 0.2 req/s, seeds 20261107/20261208/20261309, stall =
inter-chunk gap > 0.5 s). Raw data: ~/load-diff-20261111/results-nowarm/ (run1-skipjit/,
run2-skipjit-skipprofile/, table.txt).

## Attribution (before)
In RESULTS_SBS's lean arm every stall > 1 s starts 0.3-2 s after GMS KV init and ends before `capture_model`, i.e.
in vLLM 0.31's V2 `warmup_kernels` (eager prefill over 2 x max_num_seqs tokens, 3 decode steps over max_num_seqs
requests, a mixed and a cleanup step). On GB200 time-slicing each eager forward with collectives costs the serving
engine roughly one slice per collective (~0.35 s per GLM forward), so the stall count follows the number of eager
forwards, not their size: capping the warmup batch would not help.

## Changes (matched on SAVE and LOAD, Foundry TOML keys; `v031-foundry/overlay/.../foundry_adapter/{config,hooks}.py`)
- `skip_jit_warmup`: skip V2 `warmup_kernels`; without it the pass is now timed in the log.
- `skip_profile_run`: skip the runner's memory-profiling forward (one eager 8192-token forward, attention
  skipped, plus a dummy sampler). Valid with a fixed KV geometry (`num_gpu_blocks_override`); LOAD reuses SAVE's
  archived KV budget. After `skip_jit_warmup` alone the two remaining stalls (1.1 s, 1.3 s) sat in this forward.
- Harness: `setup031.sh` appends `$FOUNDRY_TOML_EXTRA` to both TOMLs and also kills leftover `VLLM::` workers;
  `run_nowarm.sh` (setup, primary probe, optional baseline, 3-cycle chain, probe + burst on the promoted engine),
  `burst.py` (512 concurrent short + 48 concurrent ~8K-token requests), `deploy_f10.sh`.

## Results (replenish window = replacement start -> parked)

| arm (3 cycles each) | replenishment | kill -> first token | window TTFT p50 / p90 | ITL p50 / p99 | stalls (total, max) |
|---|---|---|---|---|---|
| lean before (RESULTS_SBS) | 86.6-88.7 s | 6.6-14.0 s | 1.48-2.17 / 2.93-7.34 s | 13-14 / 16-330 ms | 5 (9.8 s, 3.3 s) |
| + skip_jit_warmup | 83.4-84.4 s | 6.7-14.1 s | 1.42-2.02 / 2.18-8.31 s | 13-15 / 18-320 ms | 2 (2.4 s, 1.3 s) |
| **+ skip_profile_run** | **80.4-81.4 s** | 7.9-15.2 s | 1.33-2.02 / 2.39-8.11 s | 13-15 / 18-317 ms | **0** |
| no-failure baseline (same seeds) | - | - | 1.30-1.82 / 1.72-6.46 s | 12-13 / 55-316 ms | 0-1 |

Cycle-2/3 seeds carry heavier load (baseline TTFT p90 3.0/6.5 s), so compare per seed: windows match baseline.
Requests: 0 errors in every cycle; in-flight requests at the kill are still truncated (proxy), as before.

## Correctness and safety
- Promoted lean engine vs the primary (`probe.py`): solo 4/4, pair 4/4, long 2/2 identical, Δlogprob 0 (both
  runs). batch8: 8/8 (run 1), 2/8 (run 2); the promoted engine against itself gives 1/8 and 8/8 on repeats, so
  batch8 is concurrent-batching noise (the Foundry README saw native-vs-native 2/8).
- V2 locks the workspace after capture; growth after the lock would assert. Burst on the promoted engine (512
  concurrent short + 48 concurrent ~8K-token requests) and the 32K-prompt load: 0 errors, no assertion.

## Cost / open
- With the warmups skipped, a few Triton kernels compile on the first requests after promotion (vLLM's JIT
  monitor warns: sampler `_gumbel_sample`/`_bias`/`_topk_log_softmax`/`_ranks`, `sparse_mla_index_remap`,
  `_fused_norm_rope`); kill -> first token is ~1 s higher than before (kill -> promoted unchanged, 5.4-6.1 s).
  Options: ship the Triton cache with the archive, or run a small warmup right after wake (no co-resident engine).
- `skip_profile_run` relies on `num_gpu_blocks_override` (fixed KV). Without it the KV budget would need another
  source.
- n = 3 cycles per arm on one node; Snapshot still not measured on the matched config.
