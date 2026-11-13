#!/usr/bin/env python3
"""zygote.py SOCK: pre-imported, CUDA-clean parent that forks `vllm serve` processes on request (vLLM 0.31 shadows).

Start it with zystart.sh, which gives it a lean Foundry LOAD shadow's environment: Foundry's hook in LD_PRELOAD,
FOUNDRY_MODE=load, fork start method, cudaclean, GMS shadow env. It installs rp_cudaclean, so the process stays
CUDA/NVML-clean, and imports the `vllm serve` closure plus the engine-core, worker, GMS and Foundry-adapter modules
(ZY_PREIMPORT overrides the list). Then it waits on a unix socket:
  request, one JSON line: {"argv": ["serve", MODEL, ...], "env": {...}, "log": "/path/to/log"}
  reply, one JSON line:   {"pid": <child pid>, "zygote_threads": <threads in the zygote at fork>}
The child calls setsid (pid == pgid, so kill_engine works), applies the env, sends stdout/stderr through
`ts.py > log` (epoch timestamps, like serve031.sh) and runs vLLM's CLI main(). The zygote stays CUDA-clean and can
fork any number of engines. It must run with PYTHONUNBUFFERED=1 so that loggers created at import time write straight
to the child's redirected fds.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback

DEFAULT_MODULES = [
    "vllm",
    "vllm.entrypoints.cli.main",
    "vllm.entrypoints.cli.serve",
    "vllm.entrypoints.openai.api_server",
    "vllm.v1.engine.async_llm",
    "vllm.v1.engine.core",
    "vllm.v1.engine.core_client",
    "vllm.v1.executor.multiproc_executor",
    "vllm.v1.worker.gpu_worker",
    "vllm.v1.worker.gpu.model_runner",
    "vllm.compilation.foundry_adapter",
    "vllm.model_executor.models.deepseek_v2",
    "gpu_memory_service.integrations.vllm.worker",
    "gpu_memory_service.integrations.vllm.model_loader",
]


def log(msg):
    print(f"ZYGOTE {time.time():.3f} {msg}", flush=True)


def threads():
    out = []
    for t in sorted(os.listdir("/proc/self/task")):
        try:
            out.append(open(f"/proc/self/task/{t}/comm").read().strip())
        except OSError:
            pass
    return out


def install_cudaclean():
    d = os.environ.get("ZY_CUDACLEAN_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cudaclean"))
    sys.path.insert(0, d)
    try:
        import multiprocessing.util
        import rp_cudaclean
        rp_cudaclean.install()

        class _Rearm:  # register_after_fork keeps a weak reference: hold the object in a global
            pass
        globals()["_ZY_REARM"] = _Rearm()
        # The engine core is forked from the API server (the zygote's child) by multiprocessing: re-arm there,
        # exactly as audit/sitecustomize.py does for a plain `vllm serve` with F10_FORK=1.
        multiprocessing.util.register_after_fork(globals()["_ZY_REARM"], lambda _o: rp_cudaclean.install())
    finally:
        sys.path.remove(d)


def child(req):
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    os.setsid()
    os.environ.update({k: str(v) for k, v in req.get("env", {}).items()})
    logf = open(req["log"], "w")
    ts = subprocess.Popen([sys.executable, "-u", req.get("ts", "/results/f10/ts.py")], stdin=subprocess.PIPE,
                          stdout=logf, stderr=subprocess.STDOUT, close_fds=True)
    os.dup2(ts.stdin.fileno(), 1)
    os.dup2(ts.stdin.fileno(), 2)
    nul = os.open("/dev/null", os.O_RDONLY)
    os.dup2(nul, 0)
    print(f"ZYGOTE child pid {os.getpid()} forked {time.time():.3f} argv {req['argv'][:2]}", flush=True)
    sys.argv = ["vllm"] + list(req["argv"])
    code = 0
    try:
        from vllm.entrypoints.cli.main import main
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 0
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


def main():
    path = sys.argv[1]
    t0 = time.time()
    log(f"pid {os.getpid()} starting; LD_PRELOAD={os.environ.get('LD_PRELOAD', '')!r} "
        f"FOUNDRY_MODE={os.environ.get('FOUNDRY_MODE')} mp={os.environ.get('VLLM_WORKER_MULTIPROC_METHOD')}")
    install_cudaclean()
    mods = os.environ.get("ZY_PREIMPORT")
    mods = [m for m in mods.split(",") if m] if mods else DEFAULT_MODULES
    for m in mods:
        t = time.time()
        try:
            __import__(m)
            log(f"imported {m} {time.time() - t:.2f}s")
        except BaseException as e:  # keep going: a missing module only costs that import in the child
            log(f"import {m} FAILED {e!r}")
    try:
        import torch
        log(f"preimport done {time.time() - t0:.1f}s; torch.cuda.is_initialized={torch.cuda.is_initialized()} "
            f"threads={threads()}")
    except Exception as e:
        log(f"preimport done {time.time() - t0:.1f}s; torch check failed {e!r}")
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)  # auto-reap engines
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(8)
    log(f"ready on {path}")
    while True:
        conn, _ = srv.accept()
        try:
            req = json.loads(conn.makefile().readline())
        except Exception as e:
            log(f"bad request {e!r}")
            conn.close()
            continue
        th = threads()
        if len(th) > 1:
            log(f"WARNING forking with {len(th)} threads: {th}")
        pid = os.fork()
        if pid == 0:
            srv.close()
            conn.close()
            try:
                child(req)
            finally:
                os._exit(1)
        log(f"forked pid {pid} for {req.get('log')}")
        conn.sendall((json.dumps({"pid": pid, "zygote_threads": len(th)}) + "\n").encode())
        conn.close()


if __name__ == "__main__":
    main()
