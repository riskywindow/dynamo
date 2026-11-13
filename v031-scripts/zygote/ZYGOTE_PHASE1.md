# Zygote / fork startup on vLLM 0.31: phase 1 (CUDA-init audit, cudaclean port, forked workers)

2026-11-12, pod rp-ldf-z1 (w0e-46k1), vllm-runtime:1.6.0 (vLLM 0.31.0), GLM-5.2-NVFP4 TP4+EP, `vllm serve` via
`v031-scripts/fig10/serve031.sh primary` (GMS writer, load_diff mode). Goal: replace the pre-started spare (holds
~12.7 GiB/GPU) with a fork/zygote startup that costs host memory only.

## Why 0.31 forced spawn

`vllm/utils/system_utils.py::_maybe_force_spawn` overrides `VLLM_WORKER_MULTIPROC_METHOD=fork` to spawn when
`cuda_is_initialized()` in the process about to create children (Ray / NUMA bind / WSL are irrelevant here). Run 1
(`F10_FORK=1`, no cudaclean) logged: `We must use the spawn multiprocessing start method ... Reasons: CUDA is
initialized` in the API server.

## Audit (`audit/sitecustomize.py`, ZY_AUDIT=1)

A `sitecustomize` on PYTHONPATH logs, per process, every change in open `/dev/nvidia*` fds and in
`torch.cuda.is_initialized()` with the Python stack (import hook after each module exec + a 20 ms poller on the main
thread). Same findings in the API server and the engine core (run 1, `results-zygote/audit1`):

| # | What | Call chain | Effect |
|---|---|---|---|
| 1 | NVML in platform detection | `vllm/platforms/__init__.py:64/84/257/286` -> `vllm.third_party.pynvml` init/shutdown; `vllm/platforms/cuda.py:1055/1062` | balanced: fds return to 0. Harmless |
| 2 | NVML leak + driver init at `import vllm.config` | `vllm/config/vllm.py:26` -> `vllm/triton_utils/__init__.py:5` -> `importing.py:34` -> triton backend `driver.is_active()` (`triton/backends/amd/driver.py:447` and the nvidia backend) -> `torch.cuda.is_available()` -> `device_count` -> `_raw_device_count_nvml` (`torch/cuda/__init__.py:951`, nvmlInit without nvmlShutdown) | 15 nvidia fds stay open after the import |
| 3 | torch CUDA `_lazy_init` | `vllm_omni/quantization/quack_fp8.py:35/44` -> `torch.cuda.get_device_capability/get_device_properties` (`torch/cuda/__init__.py:721/741 -> 505`), reached during API-server and engine-core imports (vllm-omni plugin) | `torch.cuda.is_initialized()` = True: this is what trips `_maybe_force_spawn` |

\#2 is 0.27.1's cause (A) (same chain via `vllm.triton_utils`); \#3 is new on 0.31 (0.27.1's equivalent was
flashinfer gdn's import-time `get_device_properties(0)`). `/dev/nvidia-uvm` also opens transiently inside NVML init, so
it is not by itself proof of `cuInit`.

## Port: 0.27.1 `rp_cudaclean.py`, unchanged

`cudaclean/rp_cudaclean.py` is the 0.27.1 module (`batch-a/startup2/cudaclean`), vendored byte-identical. It answers,
while "deferring", torch's NVML count (+ the missing nvmlShutdown), torch `get_device_properties/capability/name/
current_device`, cuda-python's `cuInit`/device queries and triton's driver check from a device table read by a
throwaway `python -I` helper. It releases at the engine core's executor `get_mp_context` and finalizes forked
workers in multiprocessing's after-fork hook. All three 0.31 causes are covered by it without changes.

0.31 wiring (in `audit/sitecustomize.py`, `RP_CUDA_CLEAN_IMPORTS=1`, only in the `vllm` entry process):
- `install()` in the API server at interpreter start;
- the engine core is now *forked* from the API server (0.27.1 spawned it), so `install()` is re-run in multiprocessing's
  after-fork hook, which arms release/finalize in the process named `EngineCore`;
- workers forked from the engine core finalize in the after-fork hook (real driver init happens there).
- Not in other Python processes: vLLM's usage-stats thread runs `cpuinfo --json` in a child Python and parses its
  stdout; cudaclean's status line broke that (JSONDecodeError in the usage thread; harmless but noisy).

## Result (run 2, audited; runs 3/4 clean: see below)

- No `must use spawn` warning: the API server forks the engine core, the engine core forks 4 workers
  (`VLLM::Worker_TP*` have the engine core as parent). cudaclean: `answered torch.get_device_capability x12,
  get_device_properties x1, fallbacks 0`, NVML balanced; release at `executor start (get_mp_context)`.
- The GMS writer loads, captures graphs (83 sizes) and serves; greedy outputs deterministic across repeats.
- Startup milestones (seconds after launch; run 1 spawn vs run 2 fork, both audited, run 1 had a cold page cache):
  engine core first log 50.4 vs 21.3, first worker log 64.7 vs 21.7, rank-0 distributed init 70.5 vs 23.1,
  load_model start 77.0 vs 27.7.

Clean comparison (runs 3 spawn / 4 fork+cudaclean, no audit): see RESULTS section appended below.

## RESULTS: clean runs (no audit hook), GLM-5.2 primary from disk, 83 capture sizes

Seconds after launch (`zyms.py`):

| Run | engine core first log | first worker log | rank-0 dist init | load_model start | model loaded | capture (profiling + final) | ready |
|---|---|---|---|---|---|---|---|
| serve3 spawn (reference) | 32.0 | 45.8 | 48.8 | 55.9 | 120.7 | 37 + 66 s | 248.1 |
| serve4 fork + cudaclean | 21.8 | 22.2 | 23.5 | **28.1** | 96.0 | 46 + 83 s | 247.3 |
| serve5 fork + cudaclean + gc.freeze before fork | 22.8 | 23.2 | 24.5 | 27.7 | 94.7 | 46 + 87 s | 243.2 |
| serve6 **GMS load-diff shadow**, fork + cudaclean (scratch KV, fixed VA, native capture) | 21.3 | 21.6 | 23.0 | **27.4** | 33.3 (attach 6.7 s) | 45 + 81 s | 171.7 |

- Fork removes the engine-core and worker Python bootstraps: weights start loading **~28 s earlier** (55.9 -> 28.1 s).
  Earlier cold spawn shadows started weight attach at ~55.8 s (RESULTS_STARTUP.md), so the shadow gains the same.
- Outputs: greedy probe (4 prompts, 48 tokens, logprobs) identical, Δ 0, spawn vs fork (`out-spawn.json` vs
  `out-fork.json`), fork primary vs forked shadow, and forked shadow after park -> kill primary -> wake
  (sleep 0.24 s, wake 4.22 s via the dev endpoints).
- **Open: graph capture is ~25% slower in forked workers** (46+83 vs 37+66 s), which cancels the gain for a native
  capture. Not GC copy-on-write (gc.freeze() before fork: no change), not torch threads or CPU affinity (same: torch
  threads 1, affinity 0-139 in both). Not yet explained. Candidates: something inherited from the API server / engine
  core heap or C-level state, or cudaclean leftovers in workers (the audit hook stays installed for the process
  lifetime; it returns early unless the event is `ctypes.dl*`). For the lean Foundry LOAD shadow this matters less:
  LOAD restores graphs (~3 s) instead of capturing, so the ~28 s pre-load gain should carry over. Needs measuring.

## Phase 2 plan

1. **Lean Foundry LOAD with fork.** `serve031.sh F10_FORK=1` already preloads `libcuda_hook.so` for the whole tree
   with the env Foundry's spawn-site injection would set (FOUNDRY_MODE, early reserve). Matched SAVE and LOAD both
   forked (the allocation trajectory must match), four lean switches, bytecode cache. Measure the no-load timeline
   (expect cold replacement ~80 -> ~50 s) and check that restore is not slowed the way capture is.
2. **Find the capture slowdown** only if it also hits LOAD (warmups are skipped at lean LOAD, so it may not):
   py-spy/perf of one worker during capture, fork vs spawn; try `RP_CUDA_CLEAN_IMPORTS=0` with spawn-forced engine
   core to split API-server-inherited state from cudaclean.
3. **Zygote**: a long-lived, pre-imported, CUDA-clean process (the API server's import closure plus
   `vllm.v1.engine.core`, the GMS worker and the Foundry adapter) that forks `vllm serve` on trigger. It saves the
   remaining ~16 s of API-server Python startup (first API log at 15-17 s in every run). Port from 0.27.1
   `batch-a/startup2/zygote` (thread hygiene: 0.27.1 refused the zygote for 16 OpenBLAS threads; set
   OPENBLAS_NUM_THREADS/OMP_NUM_THREADS=1 in the zygote and check `/proc/self/task` before fork). Host memory only.
4. **Under load**: 3 cycles at 83 sizes, same seeds as RESULTS_NOWARM / RESULTS_COMBINED, lean Foundry shadow
   started cold after the failure (no spare), fork + zygote + bytecode cache. Compare replenishment and stalls with
   RESULTS_NOWARM (80.4-81.4 s, 0 stalls) and RESULTS_COMBINED (spare, 11.4-12.4 s).

Files: `audit/sitecustomize.py` (audit, cudaclean wiring, gc.freeze option), `cudaclean/rp_cudaclean.py` (0.27.1,
unchanged). Use: `RP_CUDA_CLEAN_IMPORTS=1 PYTHONPATH=<this dir>/audit F10_FORK=1 serve031.sh ...`.
Results: `~/load-diff-20261111/results-zygote/`.
