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

## Update 2026-11-12: n=5 (3 more cycles + 2 fresh baselines, same node)

Same node (w0e-...-mb4d, pod rp-ldf-n1), same config, same load. Cycles c3-c5 (seeds 20261309 / 20261410 / 20261511)
were run with `run_n5.sh` (`K0=3 NCYC=3`; `chain031g.sh` now takes `K0`). Fresh no-failure baselines for seeds c4/c5;
c1-c3 baselines are the RESULTS_SBS cycles with the same seeds. Analysis: `sbs_analyze.py --ref=combined` over all
five cycles (`results-combined-n5/table.txt`), plus `windows_vs_baseline.txt` (same-seed baseline at the same
offsets from t0).

| Cycle | Replenishment (release -> parked) | Kill -> promoted | Kill -> first token | Replenish window: n, TTFT p50/p90 (same-seed baseline) | Stalls >0.5 s (replenish / pre-start) | Pre-start (s): n, TTFT p50/p90 (baseline) | Err / truncated |
|---|---|---|---|---|---|---|---|
| c1 | 11.4 s | 6.1 s | 8.8 s | 3, 2.17/3.77 (1.58/1.72) | 0 / 0 | 59.7: 4, 1.32/1.81 (1.30/1.80) | 0 / 5 |
| c2 | 12.4 s | 5.8 s | 15.2 s | 2, 1.35/2.60 (1.30/1.31) | 0 / 0 | 59.7: 12, 1.71/3.28 (1.88/2.99) | 0 / 10 |
| c3 | 11.4 s | 6.3 s | 9.2 s | 3, 1.57/1.79 (1.58/1.79) | 0 / 0 | 59.1: 17, 2.17/8.08 (2.11/6.46) | 0 / 3 |
| c4 | 12.4 s | 5.8 s | 8.6 s | 3, 2.94/4.05 (1.28/1.28) | 0 / 0 | 61.8: 16, 2.20/7.64 (1.54/2.63) | 0 / 8 |
| c5 | 11.4 s | 5.3 s | 8.0 s | 2, 1.32/3.12 (1.35/1.36) | 0 / 0 | 61.2: 12, 1.35/1.58 (1.32/1.57) | 0 / 2 |
| **n=5** | **11.4-12.4 s (mean 11.8)** | 5.3-6.3 s | 8.0-15.2 s | | **0 / 0** | 59-62 s | **0** / 2-10 |

Replenishment is bimodal (11.4 or 12.4 s); the difference is in the API-server phase and polling, not in KV/graphs.

Post-release phases (released spares of c3-c5, seconds after `startup gate: released`, ts.py-stamped logs,
`startup_phases.txt`): KV ready +0.0-0.1, graphs restored +4.8-5.1, engine init done +6.5-6.9, API server serving
+10.8-11.3 s; event-measured release->ready 11.17-12.19 s (gate-file and readiness polling add 0.4-0.9 s), park +0.22 s.
So the ~11.5 s is ~5 s KV allocation + graph restore, ~1.7 s remaining engine init, ~4.4 s API-server startup after
the engine is ready, ~0.6 s polling + park. Everything before the gate (spawn, imports, NCCL, weight attach: 55-58 s
in these logs) is pre-paid while the spare starts during serving.

Correctness, final promoted engine (after c5) vs the primary: solo 4/4, long 2/2 identical (logprob delta 0); pair 3/4
(diverges at token 8, delta 0.0009 on the matched prefix, a near-tie); batch8 6/8 (concurrent batching). Errors: 0
in all cycles.

Notes:
- 0 decode stalls >0.5 s in every replenish window and every spare pre-start window (5/5 cycles).
- TTFT in the replenish window is above the same-seed baseline in c1/c4 (and c2 p90). These are the first 2-3
  requests on the just-promoted engine (first-request Triton JIT, ~1 s, RESULTS_NOWARM; plus a prefix cache that is
  cold relative to the baseline's primary). Not separated in this data.
- Pre-start window TTFT p90 is above baseline in c3/c4 (8.08 vs 6.46, 7.64 vs 2.63 s) with no decode gaps >0.5 s;
  c1/c2/c5 match baseline. It overlaps the promoted engine's first minute, so spare pre-start and cold-engine effects
  are not separated either.
- Snapshot: the snapshot-agent nodes (o7v xjjk / v6sj) are quarantined (`FAILURE_NVLINK` maintenance pending,
  SchedulingDisabled), so there is still no matched Snapshot run.

