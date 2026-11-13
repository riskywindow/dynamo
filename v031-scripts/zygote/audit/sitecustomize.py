"""CUDA/NVML-init audit for vLLM processes (ZY_AUDIT=1; put this dir first on PYTHONPATH).

Every Python process (API server, engine core, workers) writes $ZY_AUDIT_DIR/<pid>.log:
- import hook: after each module's exec, if /dev/nvidia* fds changed or torch.cuda became initialized,
  log the transition with the import stack (innermost module that finished = the one whose top level did it);
- poller (20 ms): same check, with the main thread's stack, for transitions outside imports.
/dev/nvidia-uvm appearing = cuInit (driver init); /dev/nvidiactl + /dev/nvidiaN without uvm = NVML.
"""
import os, sys
if os.environ.get("ZY_AUDIT") == "1":
    import threading, time, traceback
    import importlib.abc
    _OUT = os.environ.get("ZY_AUDIT_DIR", "/results/zy/audit")
    os.makedirs(_OUT, exist_ok=True)
    _f = open(f"{_OUT}/{os.getpid()}.log", "a", buffering=1)
    _f.write(f"pid {os.getpid()} ppid {os.getppid()} argv {sys.argv!r} t {time.time():.3f}\n")
    _main = threading.main_thread().ident

    def _nv():
        kinds = []
        try:
            for fd in os.listdir("/proc/self/fd"):
                try:
                    p = os.readlink(f"/proc/self/fd/{fd}")
                except OSError:
                    continue
                if p.startswith("/dev/nvidia"):
                    kinds.append(p)
        except OSError:
            pass
        return kinds

    def _tinit():
        m = sys.modules.get("torch.cuda")
        try:
            return bool(m is not None and m.is_initialized())
        except Exception:
            return False

    _st = {"n": len(_nv()), "t": False, "uvm": False}
    _lk = threading.Lock()

    def _check(where, frame=None):
        with _lk:
            kinds = _nv(); n = len(kinds); t = _tinit(); uvm = any("uvm" in k for k in kinds)
            if n != _st["n"] or t != _st["t"] or uvm != _st["uvm"]:
                _f.write(f"\n== {time.time():.3f} {where}: nvidia_fds {_st['n']}->{n} uvm(cuInit) {_st['uvm']}->{uvm} "
                         f"torch_cuda_init {_st['t']}->{t} devs {sorted(set(kinds))[:6]}\n")
                stack = traceback.format_stack(frame) if frame is not None else traceback.format_stack()[:-2]
                _f.write("".join(s for s in stack[-45:] if "importlib._bootstrap" not in s))
                _st.update(n=n, t=t, uvm=uvm)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                try:
                    spec = finder.find_spec(name, path, target)
                except Exception:
                    continue
                if spec is None:
                    continue
                ld = spec.loader
                if ld is not None and not isinstance(ld, type) and hasattr(ld, "exec_module") \
                        and not getattr(ld, "_zy_wrapped", False):
                    orig = ld.exec_module
                    def exec_module(module, _o=orig, _n=name):
                        _o(module)
                        _check(f"import {_n}")
                    try:
                        ld.exec_module = exec_module
                        ld._zy_wrapped = True
                    except Exception:
                        pass
                return spec
            return None

    sys.meta_path.insert(0, _Finder())

    def _poll():
        while True:
            time.sleep(0.02)
            fr = sys._current_frames().get(_main)
            _check("poll", fr)

    threading.Thread(target=_poll, daemon=True, name="zy-audit").start()

# RP_CUDA_CLEAN_IMPORTS=1: keep the API server and the engine core CUDA/NVML-clean until worker creation
# (0.27.1 startup2/cudaclean/rp_cudaclean.py, vendored unchanged in ../cudaclean). 0.31 difference: with
# VLLM_WORKER_MULTIPROC_METHOD=fork the engine core is *forked* from the API server, so re-run install() after
# fork; in a process named EngineCore* that arms release (executor get_mp_context) and finalize (_init_executor).
# Only in the `vllm` entry process; forked children inherit it. Other Python processes (cpuinfo's JSON helper,
# log filters) must not load it: its status line on stdout breaks callers that parse their child's stdout.
if (os.environ.get("RP_CUDA_CLEAN_IMPORTS") == "1" and "rp_cudaclean" not in sys.modules
        and os.path.basename(sys.argv[0] if sys.argv else "") == "vllm"):
    _cc_dir = os.environ.get("ZY_CUDACLEAN_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cudaclean"))
    sys.path.insert(0, _cc_dir)
    try:
        import multiprocessing.util, rp_cudaclean
        rp_cudaclean.install()

        class _ZyRearm:
            pass
        _zy_rearm = _ZyRearm()
        multiprocessing.util.register_after_fork(_zy_rearm, lambda _o: rp_cudaclean.install())
    except Exception as _e:
        print(f"ZY cudaclean install failed: {_e!r}", file=sys.stderr)
    finally:
        sys.path.remove(_cc_dir)

# ZY_GC_FREEZE=1: gc.freeze() right before every fork, so the child's garbage collector never traverses (and so never
# copy-on-write-faults) the heap it inherited from the API server / engine core.
if os.environ.get("ZY_GC_FREEZE") == "1" and os.path.basename(sys.argv[0] if sys.argv else "") == "vllm":
    import gc as _gc
    os.register_at_fork(before=_gc.freeze)
