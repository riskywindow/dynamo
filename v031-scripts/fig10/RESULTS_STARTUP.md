# Shadow startup on vLLM 0.31: bytecode cache and a gated pre-started spare (2026-11-12)

GLM-5.2-NVFP4 TP4+EP, GMS load-diff shadow with lean Foundry LOAD, vLLM's default 83 capture sizes (same engine
config as RESULTS_SBS.md). Pod rp-ldf-u1 (4x GB200, w0e-...-9crm), branch rvinodkumar/v031-startup.
Raw data: ~/load-diff-20261111/results-startup/ (logs/, arms/, gated.json, startup_phases.txt).

## Where a cold shadow's time goes (no load, next to the live primary; `startup031.sh` + `startup_phases.py`)

Seconds after `vllm serve` launch, one run each:

| | API args logged | engine core init | workers dist-init | load start | attach done | KV ready | graphs restored | init done | serving | parked |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 cold (as in RESULTS_SBS) | - | 35.2 | 50.8 | 55.8 | 61.0 | 66.7 | 72.7 | 75.0 | 81.9 | **82.4** |
| E1 + bytecode cache | - | 30.5 | 43.9 | 48.7 | 54.6 | 58.6 | 64.1 | 65.4 | 70.5 | **71.2** |
| E3 + fork (vLLM forced spawn) | 14.9 | 29.4 | 42.6 | 48.1 | 53.2 | 58.0 | 64.0 | 65.5 | 69.8 | 70.2 |

- Before weight attach a cold shadow is three serial Python bootstraps: API server import (~15 s), engine-core
  spawn + import (~10-15 s), worker spawn + import (~12-15 s), then NCCL/dist init (~3 s).
- **Bytecode cache** (`F10_PYCACHE` -> `PYTHONPYCACHEPREFIX`, filled once with `compileall`, 771 MB, 25 s): the
  image's site-packages ship few `.pyc` (vllm 381/2771, flashinfer 0/1824) and are not writable, so every process
  recompiled them. Saves ~11 s per cold start (82.4 -> 71.2 s).
- **Fork** (`F10_FORK`, with Foundry's hook preloaded for the whole tree): vLLM overrides it to spawn because CUDA is
  already initialized in the API server process ("Reasons: CUDA is initialized"). Making that work needs the 0.27.1
  cudaclean import surgery; not done.
- After the engine is initialized, ~5 s go to engine-core post-init and API-server setup (chat template, routes).

## Gated pre-started spare (`DYN_GMS_STARTUP_GATE`, `F10_GATE`)

A spare is launched ahead of time and holds in `GMSWorker.load_model` after the weight attach (spawn, imports, NCCL
and attach done; profiling, KV, warmup, graph restore and the API server not yet). Releasing the gate (touching
the file) is the replenishment trigger. No load: release -> serving 17.3 s, release -> parked **17.5 s**
(KV 2.7 s, warmup + restore 6.7 s, init 1.6 s, API server 5.5 s).

**Memory:** a gated spare holds **~12.7 GiB per GPU** (device use 134,439 -> 147,455 MiB per GPU; a parked spare
holds the same, 147,471 MiB). Weights are shared through GMS. Serving + parked + gated: 166.8 GB of 184 GB per GPU.

## Under load (`chain031g.sh`, 3 cycles, same seeds/rate/load as RESULTS_SBS.md)

Cycle: SIGKILL the serving engine at t0+120 s, promote the parked spare (wake), release the gated spare and park it
(replenishment), then pre-start the next gated spare next to the promoted engine (prestart window).

| arm (n) | replenishment | kill->promoted | replenish window TTFT p50 / p90 | ITL p50 / p99 | stalls >0.5 s (total, max) | in-flight lost |
|---|---|---|---|---|---|---|
| lean Foundry, cold replacement (RESULTS_SBS, 3) | 86.6-88.7 s | 5.3-6.1 s | 1.48-2.17 / 2.93-7.34 s | 13-14 / 16-330 ms | 5 (9.8 s, 3.3 s) | 16 truncated |
| **lean Foundry, gated spare (3)** | **16.5-19.9 s** | 5.4-6.3 s | 1.41-5.01 / 1.53-6.22 s (2-5 req) | 12-15 / 60-331 ms | 5 (8.3 s, 3.9 s) | 18 truncated |
| gated spare prestart window (3) | 58.4-61.5 s, during serving | - | 1.39-2.94 / 1.91-11.51 s | 13-16 / 66-322 ms | 1 (0.5 s, 0.5 s) | - |
| Snapshot + GMS v1 (pc2, unmatched; RESULTS_SBS) | 9.2-10.4 s | 1.0-13.7 s | 1.31-1.38 / 1.35-1.60 s | 8-9 / 10-313 ms | 0 | 37 errors |

Per cycle (stalls in the replenish window): c1 2 (4.5 s, max 3.9 s), c2 0, c3 3 (3.8 s, max 1.9 s); 0 request
errors. Cycle 3's seed is the heavier load (also the baseline's worst cycle in RESULTS_SBS).

- Replenishment drops from ~87 s to ~17-20 s; the spare's ~60 s startup runs ahead of time next to the serving
  engine with 1 stall >0.5 s over 3 cycles.
- The replenish-window stalls are about the same as the cold lean arm's: they come from the post-gate phases
  (KV init, V2 `warmup_kernels`, graph restore), which the gate does not move; that is the parallel warmup worker's
  target. Window percentiles are from 2-5 requests (the window is 16-20 s).
- Snapshot is still ~2x faster (10 s vs 17-20 s). Of the remaining 17.5 s: ~5.5 s API-server/post-init overhead,
  ~6.7 s warmup + restore, ~2.7 s profile + KV.
