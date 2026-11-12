# Side-by-side replenishment under load, vLLM 0.31, GLM-5.2-NVFP4 TP4+EP (2026-11-12)

GMS arms: pod rp-ldf-s1 (4x GB200, o7v-...-cc2r), image vllm-runtime:1.6.0, branch rvinodkumar/v031-sidebyside.
Raw data and analysis: ~/load-diff-20261111/results-sbs/ (arms/, sbs.json, sbs_table.txt).

## Setup
- Engine config (all GMS arms): vLLM's default 83 capture sizes (1..1024), FULL_AND_PIECEWISE, breakable graphs,
  torch.compile off, NCCL all-reduce (custom AR off, `VLLM_ALLREDUCE_USE_FLASHINFER=0`), 5641 KV blocks.
- Load: per-chunk loadgen, 32K/1K, Poisson 0.2 req/s, 420 s of arrivals; seeds 20261107, 20261208, 20261309 for
  cycles 1-3 of every GMS arm.
- Cycle (Snapshot arm's structure, `chain031.sh`): SIGKILL the serving engine at t0+120 s, point the proxy at the
  parked shadow and wake it (promotion), then start a replacement shadow next to it and park it (replenishment).
  `foundry` = GMS load-diff shadows with lean Foundry LOAD (matched SAVE, `skip_cudagraph_memory_profile`,
  `skip_capture_warmup`); `native` = the same shadows capturing natively; `base` = no kill, no shadow, window at the
  foundry arm's mean offsets (t0+126 .. t0+214 s).
- Windows (`sbs_analyze.py`, same definition for every arm): replenish = [replacement start, replacement parked];
  pre = [t0+30, kill); post = (parked, t0+420]. Stall = inter-chunk gap > 0.5 s in any stream, merged.

## Results

| arm (n cycles) | replenishment | kill->promoted | replenish-window TTFT p50 / p90 | ITL p50 / p99 | stalls (total, max) | in-flight lost |
|---|---|---|---|---|---|---|
| no-failure baseline (3) | - | - | 1.30-1.82 / 1.72-6.46 s | 12-13 / 52-316 ms | 1 (0.5 s, 0.5 s) | 0 |
| **lean Foundry LOAD (3)** | **86.6-88.7 s** | 5.3-6.1 s | 1.48-2.17 / 2.93-7.34 s | 13-14 / 16-330 ms | **5 (9.8 s, 3.3 s)** | 16 truncated |
| native capture (3) | 317-353 s | 5.2-6.1 s | **94-143 / 134-179 s** | 15 / 1326-1385 ms | **263 (567 s, 23.7 s)** | 17 truncated |
| Snapshot + GMS v1, pc2 run (6)* | **9.2-10.4 s** | 1.0-13.7 s | 1.31-1.38 / 1.35-1.60 s (0-3 req) | 8-9 / 10-313 ms | **0** | 37 errors |

Per cycle: `sbs_table.txt`. Pre-kill windows match across GMS arms (TTFT p90 1.78-2.56 s, ITL p99 14-318 ms); the
seed-3 load is heavier (baseline TTFT p90 6.46 s in the window), so compare cycles with the same seed.

\* Snapshot was not run in this session: both snapshot-agent nodes (o7v-...-xjjk, -v6sj; `snapshot-lab=true`) were
cordoned and NotReady/unreachable for the whole session (checked over ~2 h; pods stayed Pending: "didn't have
enough resources", node unschedulable). The row is the 2026-11-11 pc2 run, analyzed with the same `sbs_analyze.py`:
same image (vLLM 0.31), same per-chunk loadgen, corpus, rate and 420 s cycle, same 83 capture sizes, compile off,
but different seeds, the default all-reduce backends (FlashInfer/custom AR/symm-mem/NCCL), the Dynamo frontend
instead of a TCP proxy, and node xjjk. The prepared NCCL-matched harness is in `snapshot-sbs/` (seeds as above).

## Findings
- Lean Foundry vs native at the production capture config: replenishment 87 s vs 317-353 s under load, and the
  window goes from collapsed serving (native: 81-94 stalls per cycle, TTFT p50 94-143 s, ITL p99 1.3 s) to near
  baseline (1-2 stalls per cycle). Native at 83 sizes is far worse than at 10 sizes (RESULTS.md: 9 stalls, 30 s).
- Lean Foundry is not stall-free at 83 sizes: all 5 stalls > 1 s fall 69.8-75 s after replacement start, between
  GMS KV init and graph restore, i.e. the V2 `warmup_kernels` pass (eager 2048-token prefill + 3 x 1024-token
  decode) and JIT kernel warmup; the 8192-token profile_run (63-69 s) caused none. The remaining one (0.78 s, c3)
  is during spawn and matches baseline noise (base c2: 0.51 s).
- Snapshot still leads: ~10 s replenishment with no stalls, vs 87 s with 1-2 stalls of up to 3.3 s. The GMS
  path's ~57 s before weight attach is process spawn, imports and NCCL init.
- Promotion: kill -> woken 5.3-6.1 s on the GMS path (scratch KV -> real KV) in every cycle; Snapshot's GMS v1 wake
  is 1.0 s in 4/6 cycles and 9.4/13.7 s in 2/6. In-flight requests at the kill are lost in all arms (GMS: truncated
  by the proxy; Snapshot: errors through the frontend).

## Open
- Snapshot on the NCCL-matched config with the same seeds (needs a schedulable snapshot-agent node).
- Remove or shrink V2 `warmup_kernels` at LOAD (source of the remaining stalls).
- Port the startup stack (bytecode cache, zygote/forkw) to 0.31 for the ~57 s before attach.
- One window definition, but per-arm n is 3 (Snapshot 6, older run).
