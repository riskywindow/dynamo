# Replenishment under load on vLLM 0.31 (GMS load-diff shadow + lean Foundry LOAD), GLM-5.2-NVFP4 TP4+EP

2026-11-11/12, pod rp-ldf-d (4x GB200, o7v-...-cc2r), image vllm-runtime:1.6.0, branch rvinodkumar/v031-fig10.
Raw data: ~/load-diff-20261111/results-fig10/ (arms/<arm>/requests.jsonl with per-chunk times,
events.txt, shadow.log with epoch-ms line timestamps; analysis.txt, promote_stats.txt).

## Setup
- Primary: `vllm serve` GMS writer (load_diff) on :8000. Shadow: `vllm serve` GMS RO load_diff shadow on :8001,
  `DYN_GMS_FIXED_VA=1`, scratch KV, parked with `POST /sleep?level=1` (`VLLM_SERVER_DEV_MODE=1`). Same engine config
  for both (foundry_shadow.sh): FULL_AND_PIECEWISE, 10 capture sizes 1..512, NCCL all-reduce, custom AR off,
  5641 KV blocks, max_num_batched_tokens 8192, torch.compile off.
- Shadow arms: `foundry` = lean Foundry LOAD (`skip_cudagraph_memory_profile`, `skip_capture_warmup`, matched SAVE
  made by setup031.sh next to the idle primary); `native` = same shadow, native graph capture, no Foundry.
- Load: loadgen_perchunk.py, 32K-token prompts / 1024 forced output tokens, Poisson 0.2 req/s for 420 s (480 s in
  the promotion arm), seed 20261107 in every arm, prefix cache reset per arm. Shadow launched at t0+120 s.
  0.2 req/s does not saturate this engine: baseline TTFT p50 1.31 s over the whole run (paper's pre-failure: 1.33 s).
- Window = shadow launch -> parked; baseline uses the foundry arm's window offsets. Stall = inter-chunk gap > 0.5 s
  in any decoding stream, merged across streams.

## Results (one run per arm)

| arm | window | n req | TTFT p50/p90 (s) | ITL p50/p90/p99 (ms) | stalls > 0.5 s | stall time | max stall |
|---|---|---|---|---|---|---|---|
| no-failure baseline | 81 s | 11 | 1.31 / 1.79 | 13 / 13 / 283 | 0 | 0 | - |
| lean Foundry LOAD shadow | 81 s | 11 | 1.39 / 1.96 | 13 / 13 / 284 | **0** | 0 | - |
| native-capture shadow | 120 s | 14 | 1.72 / **6.21** | 13 / 13 / 313 | **9** | **30.4 s** | **14.0 s** |
| promotion arm, window (lean Foundry) | 82 s | 11 | 1.43 / 1.86 | 13 / 13 / 284 | 0 | 0 | - |

Pre-window (t0+30 s .. launch) is the same in every arm: TTFT 1.31-1.35 / 2.08-2.11 s, ITL 13/15/276-280 ms, 0 stalls.

Shadow phases (s after launch; TP0 log lines):

| phase start | lean Foundry | native |
|---|---|---|
| distributed init | 34.6 | 34.7 |
| weight attach (load diff) | 57.0 | 56.0 |
| profile run (+ profiling capture in native) | 65.3 | 64.1 |
| KV init | 69.1 | 82.3 |
| warmups + graphs | (no separate line) | 84.7 |
| graphs done | 73.2 | 111.7 |
| API ready / parked | 80.5 / 81.3 | 118.9 / 119.7 |

Stall attribution (native): 4 stalls (7.5 s total, max 4.1 s) between attach end and KV init, i.e. vLLM's
`profile_cudagraph_memory` throwaway captures with their warmups (the phase also holds the 8192-token
profile_run, which in the lean arm takes ~4 s with no stall); 5 stalls (22.9 s, max 14.0 s) in the warmup +
capture phase (per-size eager warmups and captures at 10 sizes, plus V2 warmup_kernels). Lean Foundry LOAD
removes both phases' graph work; the remaining 8192-token profile_run and V2 warmup_kernels (2048-token prefill
+ 3x1024-token eager decode) produced no stall > 0.5 s and no change in ITL p99 at this load.

Promotion (lean Foundry shadow parked; primary SIGKILLed at t0+300 s; proxy switched to :8001; `POST /wake_up`):
- wake_up returned 5.06 s after the kill (scratch KV -> real KV); first token of a request sent after the kill:
  6.81 s after the kill.
- The 7 requests in flight at the kill were cut off (81-1014 of 1024 tokens; the proxy closes cleanly, so loadgen
  records them as truncated, not as errors). 30 requests sent after the kill: 0 errors.
- Post-promotion (29 requests sent after the shadow served): TTFT p50/p90 1.34/1.71 s, ITL p50/p90/p99
  11/12/14 ms, i.e. at the primary's baseline (a Foundry-loaded engine serves at native speed on 0.31).

## Harness (v031-scripts/fig10/)
serve031.sh (one engine), setup031.sh (GMS + primary + matched SAVE), arm031.sh (one arm), promote031.sh
(+ proxy.py: kill/promote), loadgen_perchunk.py (from the 0.27.1 lean-load work), ts.py (ms line timestamps),
analyze031.py (windows, stalls, phase attribution), promote_stats.py.

## Open
- One run per arm; 10-size capture config (the 83-size production config is being measured separately).
- Stall attribution is by phase time windows (log-line resolution), not per forward.
- Truncated in-flight requests at kill are not retried; kill->first-token includes only wake_up (no detection or
  failover-lock latency).
