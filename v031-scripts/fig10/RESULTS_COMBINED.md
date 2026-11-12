# Combined: gated pre-started spare + all four matched lean switches + bytecode cache (GLM-5.2, vLLM 0.31)

2026-11-12, node w0e-...-mb4d (pod rp-ldf-c2). Merged branch `rvinodkumar/gms-v0-vllm031` at 01aa50e482 plus
`run_combined.sh`. Same load and window definitions as RESULTS_SBS.md: GLM-5.2 NVFP4 TP4+EP, 83 capture sizes,
32K/1K at 0.2 req/s, seeds 20261107 / 20261208, kill at t0+120 s; analyzed with `sbs_analyze.py` together with the
earlier arms (`--ref=combined` aligns the baseline windows to this arm's replenish window).

Shadow configuration:
- GMS load-diff RO shadow, fixed VA, scratch KV;
- lean Foundry LOAD with `skip_cudagraph_memory_profile`, `skip_capture_warmup`, `skip_jit_warmup` and `skip_profile_run`,
  matched on SAVE and LOAD;
- a pre-started spare held after weight attach (`DYN_GMS_STARTUP_GATE`; `chain031g.sh`);
- bytecode cache (`F10_PYCACHE`).

| Arm (cycles) | Replenishment | Kill->promoted | Replenish-window TTFT p50 / p90 | ITL p50 / p99 | Stalls >0.5 s | Spare pre-start (during serving) |
|---|---|---|---|---|---|---|
| **Combined (2)** | **11.4, 12.4 s** | 6.1, 5.8 s | 1.35-2.17 / 2.60-3.77 s (2-3 req) | 10-15 / 11-319 ms | **0** | 59.7 s, 0 stalls |
| Gated spare only (3), RESULTS_STARTUP | 16.5-19.9 s | 5.4-6.3 s | 1.41-5.01 / 1.53-6.22 s | 12-15 / 60-331 ms | 5 (max 3.9 s) | 58-62 s, 1 stall |
| Zero-stall cold (3), RESULTS_NOWARM | 80.4-81.4 s | 5.4-6.1 s | 1.33-2.02 / 2.39-8.11 s | 13-15 / 18-317 ms | 0 | - |
| Lean cold (3), RESULTS_SBS | 86.6-88.7 s | 5.3-6.1 s | 1.48-2.17 / 2.93-7.34 s | 13-14 / 16-330 ms | 5 (max 3.3 s) | - |
| No-failure baseline (3), RESULTS_SBS | - | - | 1.31-1.58 / 1.31-1.79 s | 11-13 / 15-317 ms | 0 | - |
| Snapshot + GMS v1, pc2 run (6), **unmatched** | 9.2-10.4 s | 1.0-13.7 s | 1.31-1.38 / 1.35-1.60 s | 8-9 / 10-313 ms | 0 | - |

Replenishment = gate release (trigger) -> replacement parked (release -> ready 11.2 / 12.2 s). Errors: 0 in every
cycle; requests in flight at the kill are still truncated (5 / 10), as in every GMS arm. Kill->first token
(8.8 / 15.2 s) depends on when the next request arrives after the kill; same seeds give the same values in all arms.

Correctness (final promoted engine vs the primary, `probe.py`): solo 4/4, pair 4/4, long 2/2 identical (logprob delta
0); batch8 6/8 (concurrent batching; native-vs-native was 2/8 earlier). Burst on the promoted engine (512 concurrent
short + 48 concurrent ~8K-token requests): 0 errors, no workspace-lock assertion.

Caveats:
- 2 cycles, not 3. The first attempt (pod rp-ldf-c1 on 46k1) disappeared mid-run, results lost; most likely
  preempted, as a higher-priority nightly job preempted another pod in this namespace at the time; the event had
  expired. The rerun was cut to 2 cycles and no baseline cycle to fit the Teleport window. The baseline rows are the
  RESULTS_SBS cycles with the same seeds.
- The 11-12 s replenish window contains only 2-3 completed requests per cycle. The stall count covers all streaming
  requests in flight during the window, but the TTFT percentiles are thin.
- Snapshot: both snapshot-agent nodes were still SchedulingDisabled (checked 07:30 and 10:03 UTC), so there is no
  matched Snapshot run. The harness is ready in `snapshot-sbs/`.
- Not done: a post-wake warmup for the first-request Triton JIT (RESULTS_NOWARM open issue).

Reproduce: deploy with `deploy_f10.sh`, fill the bytecode cache (`PYTHONPYCACHEPREFIX=/results/pycache python3 -m
compileall -q -j 32 <site dirs>`), then `NO_BASE=1 NCYC=2 bash /results/f10/run_combined.sh`. Results:
`~/load-diff-20261111/results-combined/` (`arms/`, `all.json`, `table.txt`, `run_combined.out`).
