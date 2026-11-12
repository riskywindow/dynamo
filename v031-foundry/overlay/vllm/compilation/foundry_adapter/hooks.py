# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project
"""Runtime monkey-patch installer for the foundry vLLM integration.

Pattern: wrap orig + insert foundry calls. For functions that must be
skipped on SAVE/LOAD (non-deterministic allocations), the patch replaces
the callee with a no-op dummy. The runner keeps its native profiling
and capture lifecycle.
"""

from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from vllm.compilation.foundry_adapter import runtime as rt
from vllm.compilation.foundry_adapter.config import (
    CUDAGraphExtensionMode,
    get_config,
    get_graph_extension_mode,
    get_workspace_root,
    load_graph_extension_config,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import CompilationConfig

log = init_logger("vllm.foundry.hooks")

_INSTALLED = False


def _reset_install_state_after_fork() -> None:
    """Reinstall hooks in a worker forked after the engine process.

    A child forked from a process that already installed the hooks inherits
    every patch (patched functions, loaded config, the capability quarantine).
    Reinstalling there wraps each patched function a second time: with vLLM
    0.31's fork start method (engine core forked from the API server, workers
    from the engine core) setup_graph_extension then ran twice per worker and
    LOAD failed with "[HOOK] ERROR: Binaries already loaded". So only reset
    when the parent had not installed the hooks, which leaves _INSTALLED
    False anyway: the child keeps the parent's state either way.
    """
    return


os.register_at_fork(after_in_child=_reset_install_state_after_fork)


def _quarantine_get_device_capability() -> None:
    """Serve `torch.cuda.get_device_capability` from NVML so flashinfer's
    module-level init can't trigger `torch.cuda._lazy_init` in the parent.

    If the parent ever initializes CUDA, vLLM's `_maybe_force_spawn`
    overrides our forkserver choice to spawn — killing the fork-based
    fast-start path entirely. NVML reads the same (major, minor) tuple
    without creating a CUDA runtime context, so the parent stays
    fork-safe.
    """
    import torch

    try:
        import pynvml
    except ImportError:
        log.info(
            "[foundry] pynvml not available; cannot quarantine "
            "get_device_capability — parent may still init CUDA"
        )
        return
    orig = torch.cuda.get_device_capability
    _nvml_started = {"v": False}

    def _resolve_index(device) -> int:
        if device is None:
            return 0
        if isinstance(device, int):
            return device
        if hasattr(device, "index"):
            return device.index or 0
        return int(device)

    def patched(device=None):
        try:
            if not _nvml_started["v"]:
                pynvml.nvmlInit()
                _nvml_started["v"] = True
            idx = _resolve_index(device)
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            return pynvml.nvmlDeviceGetCudaComputeCapability(h)
        except Exception:
            return orig(device)

    torch.cuda.get_device_capability = patched
    log.info(
        "[foundry] torch.cuda.get_device_capability → NVML (keeps parent fork-safe)"
    )


def install_hooks(compilation_config: CompilationConfig) -> None:
    """Idempotent. Loads the TOML and installs all patches."""
    global _INSTALLED
    if _INSTALLED:
        return
    cfg_path = compilation_config.graph_extension_config_path
    if not cfg_path:
        return

    # Must run BEFORE anything that might import flashinfer. Cheap + safe to
    # run in every process: when called in the engine child (after fork),
    # CUDA init is already OK to happen, the patch just keeps returning the
    # correct capability without paying for it.
    _quarantine_get_device_capability()

    # Measure subprocess spawn cost (parent stamped FOUNDRY_SPAWN_T0_NS right
    # before fork; we are at the first foundry code that runs in the child).
    # This window is Python + torch + vllm cold import with the LD_PRELOAD'd
    # hook, plus vLLM's pre-install_hooks scaffolding — the biggest single
    # chunk of LOAD time on remote servers.
    t0_ns_str = os.environ.get("FOUNDRY_SPAWN_T0_NS")
    if t0_ns_str:
        spawn_elapsed_ms = (time.perf_counter_ns() - int(t0_ns_str)) / 1e6
        log.info(
            "[foundry TIMING] subprocess spawn → install_hooks: %.1f ms "
            "(cold Python + torch + vllm import with LD_PRELOAD'd hook)",
            spawn_elapsed_ms,
        )

    load_graph_extension_config(cfg_path)
    log.info(
        "[foundry] install_hooks: mode=%s workspace=%s",
        get_graph_extension_mode().value,
        get_workspace_root(),
    )

    if get_graph_extension_mode() == CUDAGraphExtensionMode.NONE:
        _INSTALLED = True
        return

    from vllm.config import CompilationMode

    # LOAD reconstructs CUDA kernels without rebuilding the Torch compiled
    # model. Keep SAVE's uncaptured forwards (notably prefill) on that same
    # path: compiled SAVE and uncompiled LOAD can select different kernels
    # and produce different tokens before the restored decode graph runs.
    # FULL CUDA graphs remain enabled independently of Torch compilation.
    compilation_config.mode = CompilationMode.NONE
    log.info("[foundry] disabled Torch compilation for both SAVE and LOAD")

    _patch_init_worker_distributed_environment()
    _patch_kernel_warmup()
    _patch_jit_warmup()
    _patch_profile_run()
    _patch_load_model()
    _patch_capture_model()
    _patch_profile_cudagraph_memory()
    _patch_memory_profiling()
    _patch_subprocess_spawn_sites()
    _patch_deepep()
    # vLLM 0.31 port: breakable (PIECEWISE) CUDA graph SAVE/LOAD (segments + eager-closure archive)
    from vllm.compilation.foundry_adapter import breakable

    breakable.install()

    _INSTALLED = True
    log.info("[foundry] all patches installed")


# ---------------------------------------------------------------------------
# VMM region setup around NCCL.
# ---------------------------------------------------------------------------


def _patch_init_worker_distributed_environment() -> None:
    from vllm.v1.worker import gpu_worker as gw

    orig = gw.init_worker_distributed_environment

    @functools.wraps(orig)
    def patched(
        vllm_config, rank, distributed_init_method=None, local_rank=-1, backend="nccl"
    ):
        active = get_graph_extension_mode() != CUDAGraphExtensionMode.NONE
        t0 = time.perf_counter()
        if active:
            rt.setup_graph_extension(vllm_config.parallel_config)
        result = orig(vllm_config, rank, distributed_init_method, local_rank, backend)
        if active:
            rt.skip_to_scratch_boundary()
            log.info(
                "[foundry TIMING] init_worker_distributed_environment total: "
                "%.3f s (setup_graph_extension + NCCL + skip_to_scratch)",
                time.perf_counter() - t0,
            )
        return result

    gw.init_worker_distributed_environment = patched


# ---------------------------------------------------------------------------
# Skip-with-dummy patches for non-deterministic functions.
# ---------------------------------------------------------------------------


def _patch_kernel_warmup() -> None:
    # Captured kernels are archived by Foundry; keep the separate autotune
    # warmup from changing the SAVE/LOAD allocation trajectory.
    import vllm.v1.worker.gpu_worker as gw

    orig = gw.kernel_warmup

    @functools.wraps(orig)
    def patched(worker):
        # Foundry archives the CUDA modules and fatbins used by captured
        # graphs. Running vLLM's warmup/autotune pass again on SAVE or LOAD
        # allocates process-dependent workspaces and breaks the deterministic
        # VMM trajectory required by archived replay.
        if get_graph_extension_mode() != CUDAGraphExtensionMode.NONE:
            return None
        return orig(worker)

    gw.kernel_warmup = patched  # type: ignore[assignment]


def _patch_jit_warmup() -> None:
    """Time vLLM 0.31's V2 ``warmup_kernels`` and, with ``skip_jit_warmup``, skip it.

    The pass runs ~6 eager forwards (a 2 x max_num_seqs-token prefill, 3 decode
    steps over max_num_seqs requests, a mixed and a cleanup step) before graph
    capture. Every eager forward with collectives costs a serving engine on the
    same GPUs about one time slice per collective, so this pass is where a lean
    LOAD shadow stalls serving. Skipping must be matched on SAVE and LOAD to keep
    the deterministic allocation trajectory; capture locks the workspace after.
    """
    import vllm.v1.worker.gpu_worker as gw

    orig = getattr(gw, "warmup_kernels", None)
    if orig is None:
        return

    @functools.wraps(orig)
    def patched(*args, **kwargs):
        cfg = rt.get_config()
        active = cfg is not None and cfg.mode != CUDAGraphExtensionMode.NONE
        if active and cfg.skip_jit_warmup:
            log.info("[foundry] lean: skipped warmup_kernels (mode=%s)", cfg.mode.value)
            return None
        t = time.perf_counter()
        log.info("[foundry] warmup_kernels start")
        try:
            return orig(*args, **kwargs)
        finally:
            log.info("[foundry TIMING] warmup_kernels: %.3f s", time.perf_counter() - t)

    gw.warmup_kernels = patched  # type: ignore[assignment]


def _patch_profile_run() -> None:
    """With ``skip_profile_run``, skip the memory-profiling forward on SAVE and LOAD.

    The forward (max_num_batched_tokens tokens, attention skipped, plus a dummy
    sampler) only measures peak activation memory for the KV budget. With a fixed
    KV geometry that budget is not needed, and LOAD reuses SAVE's archived budget,
    but the forward's collectives stall a serving engine on the same GPUs.
    """
    from vllm.v1.worker.gpu import model_runner as gmr_v1

    cls = gmr_v1.GPUModelRunner
    orig = cls.profile_run

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        cfg = rt.get_config()
        if cfg is not None and cfg.mode != CUDAGraphExtensionMode.NONE and cfg.skip_profile_run:
            log.info("[foundry] lean: skipped profile_run (mode=%s)", cfg.mode.value)
            return None
        return orig(self, *args, **kwargs)

    cls.profile_run = patched  # type: ignore[method-assign]


def _load_model_with_overlap(
    runner, do_original_load: Callable, model_loader
) -> torch.nn.Module:
    """Load weights and start archived graph preparation for LOAD."""
    mode = get_graph_extension_mode()
    if mode == CUDAGraphExtensionMode.NONE:
        return do_original_load()

    cfg = get_config()
    if mode == CUDAGraphExtensionMode.LOAD and not (
        cfg is not None and cfg.recapture_on_load
    ):
        # Map the complete deterministic SAVE span before loading weights.
        # Weight tensors must receive the same VMM addresses as the capture
        # process; mapping only the tail after weight load leaves archived
        # graph parameters pointing at a different allocation layout.
        rt.preallocate_for_load_mode()
    result = do_original_load()
    from vllm.model_executor.model_loader.weight_cache.ipc_loader import IpcModelLoader

    if isinstance(model_loader, IpcModelLoader):
        rt.check_ipc_weight_layout(result)
    if mode == CUDAGraphExtensionMode.LOAD and not (
        cfg is not None and cfg.recapture_on_load
    ):
        # Build graph templates after weight loading has initialized the model.
        from vllm.compilation.foundry_adapter.graph_ops import start_graph_builds

        start_graph_builds()
    return result


def _patch_load_model() -> None:
    from vllm.v1.worker.gpu import model_runner as gmr_v1

    classes = [gmr_v1]

    for gmr in classes:
        cls = gmr.GPUModelRunner
        orig = cls.load_model

        @functools.wraps(orig)
        def patched(self, *args, _orig=orig, _gmr=gmr, **kwargs):
            if get_graph_extension_mode() == CUDAGraphExtensionMode.NONE:
                return _orig(self, *args, **kwargs)

            t0 = time.perf_counter()
            runner = self
            orig_get_loader = _gmr.get_model_loader

            def patched_get_loader(load_config):
                loader = orig_get_loader(load_config)
                orig_loader_load = loader.load_model

                def do_original_load():
                    return orig_loader_load(
                        vllm_config=runner.vllm_config,
                        model_config=runner.model_config,
                    )

                def patched_loader_load(*_args, **_kwargs):
                    return _load_model_with_overlap(runner, do_original_load, loader)

                loader.load_model = patched_loader_load
                return loader

            _gmr.get_model_loader = patched_get_loader
            try:
                result = _orig(self, *args, **kwargs)
            finally:
                _gmr.get_model_loader = orig_get_loader

            log.info(
                "[foundry TIMING] GPUModelRunner.load_model total: %.3f s "
                "(weight load + process_weights_after_loading + "
                "start_graph_builds)",
                time.perf_counter() - t0,
            )
            return result

        cls.load_model = patched


def _patch_memory_profiling() -> None:
    """Run native profiling and retain the archived KV allocation geometry."""
    import json
    from pathlib import Path

    from vllm.v1.worker.gpu_worker import Worker

    orig = Worker.determine_available_memory

    @functools.wraps(orig)
    def patched(self) -> int:
        mode = get_graph_extension_mode()
        started = time.perf_counter()
        available = orig(self)
        if mode == CUDAGraphExtensionMode.NONE:
            return available
        log.info(
            "[foundry] native memory profiling completed in %.3f s",
            time.perf_counter() - started,
        )
        cfg = get_config()
        assert cfg is not None and cfg.workspace_dir is not None
        path = Path(cfg.workspace_dir) / "kv_cache_budget.json"
        if mode == CUDAGraphExtensionMode.SAVE:
            path.write_text(json.dumps({"available_bytes": available}))
            return available
        # LOAD has already mapped the archived span, including KV storage.
        # Native profiling still runs to initialize scratch and kernel state,
        # but its budget counts that backing again. Use SAVE's budget so native
        # KV planning reconstructs the same cache geometry.
        archived = int(json.loads(path.read_text())["available_bytes"])
        log.info(
            "[foundry] reusing archived KV budget: %d bytes (profile returned %d)",
            archived,
            available,
        )
        return archived

    Worker.determine_available_memory = patched  # type: ignore[method-assign]


def _patch_profile_cudagraph_memory() -> None:
    """Keep the temporary graph-memory estimate out of the archive.

    The runner profiles graph memory before the real capture pass. SAVE
    would otherwise serialize temporary profile
    graphs and LOAD would replay them with the wrong allocation lifetime.
    """
    from vllm.v1.worker.gpu import model_runner as gmr_v1

    classes = [gmr_v1.GPUModelRunner]

    for cls in classes:
        orig = cls.profile_cudagraph_memory

        @functools.wraps(orig)
        def patched(self, *args, _orig=orig, **kwargs):
            cfg = rt.get_config()
            if cfg is None or cfg.mode == CUDAGraphExtensionMode.NONE:
                return _orig(self, *args, **kwargs)
            if cfg.skip_cudagraph_memory_profile:
                log.info(
                    "[foundry] lean: skipped profile_cudagraph_memory (mode=%s)",
                    cfg.mode.value,
                )
                return 0
            mode = cfg.mode
            cfg.mode = CUDAGraphExtensionMode.NONE
            try:
                return _orig(self, *args, **kwargs)
            finally:
                cfg.mode = mode

        cls.profile_cudagraph_memory = patched  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# GPUModelRunner.capture_model:
#   pre-orig:  prepare_graph_capture (cuBLAS + attention workspaces land at
#              deterministic offsets)
#   orig:      native manager captures or restores each FULL graph.
#   post-orig: SAVE → save_graph_manifest + pack_fatbins + capture_final_alloc_offset
#              LOAD → empty cached allocator segments
#              both → stop_allocation_region (post-capture allocs use the
#                     standard CUDA allocator; only captured-graph memory
#                     needs the deterministic VMM region)
# ---------------------------------------------------------------------------


def _patch_capture_model() -> None:
    from vllm.v1.worker.gpu import model_runner as gmr_v1

    classes = [gmr_v1.GPUModelRunner]

    for cls in classes:
        orig = cls.capture_model

        @functools.wraps(orig)
        def patched(
            self,
            *,
            profile_only: bool = False,
            _orig=orig,
        ):
            mode = get_graph_extension_mode()
            log.info(
                "[foundry] capture_model entered: mode=%s enforce_eager=%s "
                "cudagraph_mode=%s",
                mode.value,
                self.vllm_config.model_config.enforce_eager,
                self.compilation_config.cudagraph_mode,
            )
            if mode == CUDAGraphExtensionMode.NONE:
                return _orig(self, profile_only=profile_only)

            from vllm.compilation.foundry_adapter.graph_ops import (
                pack_fatbins,
                save_graph_manifest,
            )

            builders = [
                b
                for ag_list in self.attn_groups
                for ag in ag_list
                for b in ag.metadata_builders
            ]
            rt.prepare_graph_capture(
                metadata_builders=builders,
                device=self.device,
                vllm_config=self.vllm_config,
            )
            cfg = get_config()
            if mode == CUDAGraphExtensionMode.LOAD and not (
                cfg is not None and cfg.recapture_on_load
            ):
                # Ensure template construction has started before the capture loop.
                from vllm.compilation.foundry_adapter.graph_ops import (
                    start_graph_builds,
                )

                start_graph_builds()

            result = _orig(self, profile_only=profile_only)

            if mode == CUDAGraphExtensionMode.SAVE and not profile_only:
                save_graph_manifest()
                pack_fatbins()
            elif mode == CUDAGraphExtensionMode.LOAD:
                # Restored graphs are rebuilt by Foundry at the driver level;
                # unlike a native torch graph pool, the caching allocator does
                # not know which cached-but-free segments are still referenced
                # by those graphs. Return those segments before inference so a
                # later eager allocation cannot reuse an archived graph VA.
                torch.accelerator.empty_cache()
                log.info("[foundry] emptied CUDA cache after LOAD graph restore")

            # Only memory referenced by captured CUDA graphs needs the
            # deterministic VMM region. Past this point (sampler warmup +
            # inference allocations) the VMM overhead just slows things
            # down, so switch back to the regular CUDA caching allocator.
            from foundry import ops as fops

            fops.stop_allocation_region()
            log.info("[foundry] stopped allocation region after capture_model")

            # Record the deterministic watermark on SAVE. We do this AFTER
            # stop_allocation_region so the value reflects only the VMM cursor
            # at end of capture — sampler warmup and any later allocations no
            # longer advance it. Was previously a post-hook on
            # compile_or_warm_up_model; same result, one fewer monkey-patch.
            if mode == CUDAGraphExtensionMode.SAVE and not profile_only:
                rt.capture_final_alloc_offset()

            return result

        cls.capture_model = patched  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Subprocess spawn sites: LD_PRELOAD injection.
# ---------------------------------------------------------------------------


def _patch_subprocess_spawn_sites() -> None:
    # Multiproc executor worker spawn.
    from vllm.v1.executor import multiproc_executor as me

    if hasattr(me.WorkerProc, "make_worker_process"):
        orig = me.WorkerProc.make_worker_process

        @functools.wraps(orig)
        def patched(*args, **kwargs):
            rt.setup_ld_preload_env()
            return orig(*args, **kwargs)

        me.WorkerProc.make_worker_process = staticmethod(patched)

    # EngineCore proc manager.
    from vllm.v1.engine import utils as eu

    orig_init = eu.CoreEngineProcManager.__init__

    @functools.wraps(orig_init)
    def patched_init(self, *args, **kwargs):
        rt.setup_ld_preload_env()
        return orig_init(self, *args, **kwargs)

    eu.CoreEngineProcManager.__init__ = patched_init  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# DeepEP buffer kwargs: use_fabric=True, num_nvl_bytes=0 on any foundry mode.
# ---------------------------------------------------------------------------


def _patch_deepep() -> None:
    try:
        from vllm.distributed.device_communicators.all2all import (
            DeepEPLLAll2AllManager,
        )
    except ImportError:
        return
    cls = DeepEPLLAll2AllManager

    orig = cls._make_all2all_kwargs

    @functools.wraps(orig)
    def patched(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if (
            isinstance(out, dict)
            and get_graph_extension_mode() != CUDAGraphExtensionMode.NONE
        ):
            if os.environ.get("FOUNDRY_DEEPEP_NVL_IPC", "0") == "1":
                # Keep upstream's NVLink buffer (legacy cudaIpc sharing). The
                # hook's VMM-IPC layer transports the fds via SCM_RIGHTS and
                # maps peers at relocated VAs, which DeepEP absorbs through
                # its buffer_ptrs_gpu device table - peer VAs are never baked
                # into captured graphs. use_fabric stays off: fabric
                # cuMemCreate needs IMEX, and exercising the IPC path is the
                # point of this mode.
                out["use_fabric"] = False
            else:
                # Default: force VMM-fabric buffers
                # so the foundry cuMem hook tracks them, and disable the
                # NVLink buffer (LL-mode uses RDMA) to avoid fabric-cuMemCreate
                # collisions with preallocated region.
                out["use_fabric"] = True
                out["num_nvl_bytes"] = 0
        return out

    cls._make_all2all_kwargs = patched  # type: ignore[method-assign]
