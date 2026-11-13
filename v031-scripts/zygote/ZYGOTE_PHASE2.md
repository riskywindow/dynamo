# Zygote / fork startup on vLLM 0.31: phase 2 (lean Foundry LOAD forked, zygote, under load)

2026-11-12/13, pod rp-ldf-z2 (w0e-46k1), vllm-runtime:1.6.0 (vLLM 0.31.0), GLM-5.2-NVFP4 TP4+EP, 83 capture sizes,
lean Foundry SAVE/LOAD with all four matched switches (skip_cudagraph_memory_profile, skip_capture_warmup,
skip_jit_warmup, skip_profile_run), bytecode cache, GMS load-diff shadow (fixed VA, scratch KV). No pre-started spare.
Results: `~/load-diff-20261111/results-zygote2/` (spawn/, fork/, zygote-noload/, load-zygote/).

## Fix needed for fork: Foundry adapter after-fork reinstall

With `VLLM_WORKER_MULTIPROC_METHOD=fork` the engine core (forked from the API server) installs Foundry's hooks and the
workers are forked from it. The adapter's `os.register_at_fork` handler reset `_INSTALLED` in every child, so each
worker installed the hooks again on top of the inherited, already-patched functions: `setup_graph_extension` ran twice
per worker and LOAD failed with `[HOOK] ERROR: Binaries already loaded` (SAVE silently ran its VMM setup twice). Fix
(`v031-foundry/overlay/vllm/compilation/foundry_adapter/hooks.py`): the after-fork handler keeps the parent's install
state; a forked child inherits every patch. The SAVE was redone with the fix (matched trajectory).

## No-load phase timeline, cold lean LOAD shadow next to the live primary (seconds after launch)

| Start | API up | engine core | worker dist | load start | attach done | KV | restore done | init done | serving | parked |
|---|---|---|---|---|---|---|---|---|---|---|
| spawn | 15.1 | 29.8 | 43.1 | 48.7 | 53.7 | 56.6 | 61.7 | 63.5 | 67.4 | **68.2** |
| fork + cudaclean | 15.9 | 19.4 | 21.0 | 25.5 | 30.6 | 32.7 | 37.9 | 39.6 | 42.4 | **42.9** |
| zygote (fork from pre-imported process) | 2.2 | 5.8 | 7.3 | 11.7 | 17.1 | 19.7 | 25.9 | 27.5 | 29.8 | **30.7** |

All three restored 332/332 breakable sets with no layout mismatch. Graph restore 4 s (spawn) / 5 s (fork) / 4 s
(zygote), init engine 6.9 / 6.9 / 7.9 s: **restore is not slowed by fork.** The ~15% capture slowdown seen in phase 1
persists in the forked SAVE (88-93 s vs 77-81 s spawn) but LOAD does not capture, so it was not profiled (step 2 skipped).

Zygote: `zygote.py` (started by `zystart.sh` with the LOAD shadow's environment: Foundry hook in LD_PRELOAD,
FOUNDRY_MODE=load, fork, cudaclean, OPENBLAS_NUM_THREADS=1) pre-imports the `vllm serve` closure plus engine-core,
worker, GMS and Foundry-adapter modules in 14 s, stays CUDA-clean (`torch.cuda.is_initialized()` False, no
/dev/nvidia fds), and forks `vllm serve` on a unix-socket request (`zyctl.py`; `serve031.sh F10_ZYGOTE=<sock>`).
Cost: host memory only, RSS 1.6 GiB (PSS 0.66 GiB), 0 GPU memory. Observation: it carries one non-Python thread
named `cuda0000280000b` (no GPU state, origin not identified); forks were unaffected.

## Under load: 3 cycles, replacement forked from the zygote after each kill (no spare)

`run_zy_load.sh zygote`: GMS writer primary (forked), 1 no-failure baseline cycle, `chain031.sh foundry 3` (32K/1K,
0.2 req/s, seeds 20261107/20261208/20261309; SIGKILL at t0+120 s; parked shadow woken; replacement launched through
the zygote and parked). Analysis: `sbs_analyze.py` (same window definition as RESULTS_SBS/NOWARM/COMBINED).

| Arm | n | Replenishment | Kill -> promoted | Kill -> first token | Replenish window TTFT p50/p90 | ITL p50/p99 ms | Stalls >0.5 s (all windows) | Err / trunc | GPU memory held |
|---|---|---|---|---|---|---|---|---|---|
| **Zygote, no spare (this)** | 3 | **33.8, 32.8, 33.8 s** | 5.28, 5.22, 5.25 s | 8.0, 15.6, 7.8 s | 2.52/5.20, 1.46/2.95, 1.78/2.32 | 12-14 / 16-327 | **0** | 0 / 5, 10, 3 | none |
| Baseline, same node (c1 seed) | 1 | - | - | - | 1.55/1.68 | 14/287 | 0 | 0/0 | - |
| Spawn, no spare (RESULTS_NOWARM) | 3 | 80.4-81.4 s | 5.4-6.1 s | 7.9-15.2 s | 1.33-2.02 / 2.39-8.11 | 13-15 / 18-317 | 0 | 0 | none |
| Gated pre-started spare (RESULTS_COMBINED) | 5 | 11.4-12.4 s | 5.3-6.3 s | 8.0-15.2 s | see RESULTS_COMBINED | 10-15 / 11-331 | 0 | 0 | ~12.7 GiB/GPU |

Replacement phases under load (seconds after the zygote request, c1/c2/c3): API up 2.2, engine core 5.7-6.0, worker
dist 8.0-8.1, load start 12.0-12.9, attach done 19.8-21.7, KV 22.2-24.2, restore done 27.6-29.6, init done 29.2-31.0,
serving 31.6-33.4, parked +0.2. Weight attach is ~8 s under load vs ~5.5 s idle.

Correctness (final promoted engine vs the primary): solo 4/4, pair 4/4, long 2/2 identical (Δ 0); batch8 6/8
(concurrent-batching noise, as in every earlier arm).

As in every earlier GMS arm: replenish-window TTFT in c1 is above the same-seed baseline (first requests on the
just-promoted engine; ~1 s first-request Triton JIT and a cold prefix cache), and in-flight requests at the kill are
truncated (5/10/3).

## Open

- One run per variant for the no-load timeline; 3 cycles under load on one node. Matched Snapshot still blocked
  (o7v nodes quarantined).
- The remaining ~33 s: zygote fork -> engine core -> workers (~8 s), NCCL (~3 s), GMS attach (~8 s under load),
  KV (~3 s), restore (~5 s), API server finish (~2-4 s).
- Forked native capture is ~15-25% slower (phase 1, forked SAVE): unexplained; irrelevant for LOAD.
- The zygote's `cuda0000280000b` thread: origin not identified.
- Matched SAVE was produced by a forked (non-zygote) API server; LOAD forked from the zygote restored 332/332 with no
  layout mismatch, so the allocation trajectory matched.

Files: `zygote.py`, `zystart.sh`, `zyctl.py`, `run_fork_lean.sh`, `run_zy_load.sh`; `v031-scripts/fig10/serve031.sh`
(F10_ZYGOTE); Foundry fix in `v031-foundry/overlay/vllm/compilation/foundry_adapter/hooks.py`.
