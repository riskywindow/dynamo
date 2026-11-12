"""RP_CUDA_CLEAN_IMPORTS=1 (default off): keep a process CUDA/NVML-clean through its import closure, so it can fork
CUDA-using children (S-forkw: EngineCore -> workers; S-zygote) or be checkpointed host-only.

Design and evidence: batch-a/notes/cudaclean.md. Stdlib only. In the pod: /test/rp/cudaclean/rp_cudaclean.py.

While the process is 'deferring', these calls are answered with the value the real call would return, without
initializing anything in this process:
  nvml    torch.cuda._raw_device_count_nvml / _raw_device_uuid_nvml call nvmlInit and never nvmlShutdown, so each
          torch.cuda.is_available()/device_count() under PYTORCH_NVML_BASED_CUDA_CHECK=1 (set by vllm/env_override.py)
          leaves /dev/nvidia* fds open. Wrapped to add the one balancing nvmlShutdown (active until finalize).
          Also sets PYTORCH_NVML_BASED_CUDA_CHECK=1 if unset, as vLLM does at its first import anyway.
  torch   torch.cuda.get_device_properties / get_device_capability / get_device_name / current_device while torch
          CUDA is uninitialized: answered from the device table (no torch.cuda._lazy_init). An attribute the table
          lacks resolves through the real call (correct; logged as a fallback).
  driver  cuda-python (cuda.bindings.driver, legacy cuda.cuda): cuInit is only recorded ('virtual');
          cuDeviceGetCount / cuDeviceGet / cuDeviceGetAttribute are answered from the table; any other cu* call
          first runs the real cuInit (correct; logged as a fallback). Covers CUTLASS DSL's
          get_compute_capability_major_minor (cuda.cuInit + cuDeviceGet + 2 attributes).
  triton  triton.backends.nvidia.driver._cuda_driver_is_active (newer triton: ctypes cuInit) answered from the table.

The device table comes from a throwaway `python -I` helper that runs cuInit in its own process, never in this one.
It is cached per pod: RP_CUDA_CLEAN_PROPS, default /tmp/rp-cudaclean/props-<CUDA_VISIBLE_DEVICES>.json.

Lifecycle (log, never abort; whenever an answer is not possible the real call runs):
  install()      -> 'deferring'
  release(why)   -> 'released': every wrapper calls the real function (after a real cuInit if one was virtual).
  finalize(why)  -> real cuInit if one was only virtual, then restore every patched attribute that is still ours and
                    every module global that captured a wrapper: the process is as if this was never installed.
  EngineCore: rp_r_entry.py installs at __mp_main__; release at the executor's first get_mp_context() (right before the
  workers are created), finalize when MultiprocExecutor._init_executor returns. Workers forked from it finalize in
  multiprocessing's after-fork hook (Process._bootstrap), before WorkerProc runs. A zygote installs before its
  preload and never releases: its EngineCore child keeps deferring until its own executor; main finalizes at
  imports_done (rp_r_entry.py).

Fork-segfault work (batch-a/notes/forkseg.md; all only while installed, i.e. only with RP_CUDA_CLEAN_IMPORTS=1):
  recorder   every cuda-python driver call that reached the real driver before finalize ('real_calls'), and every
             ctypes dlopen of libcuda / libnvidia-ml / libcudart and dlsym of a cu* / nvml* name ('ctypes'), with the
             first caller. Reported by status() (and so by rp_forkw's probe line).
  RP_CUDA_CLEAN_FINALIZE=default|skip-cuinit|cuinit-after-restore|ctypes-cuinit|no-driver-restore|none
             (default 'default' = the order above). Selects the finalize order in a forked child, for the fork
             repro kit (startup2/forkrepro) and for one R in the GPU window. The cuInit return code is logged.
  RP_CUDA_CLEAN_STRICT=1 (default off): cuda-python cuDriverGetVersion is also answered from the table while deferring,
             so cuda-python never resolves its driver table (cuGetProcAddress x ~500) in the deferring process.
  RP_CUDA_CLEAN_NVML_DEFER=1 (default off; needs part nvml): NVML is not initialized at all while deferring. vLLM's
             pynvml (vllm.third_party.pynvml, and a top-level pynvml) answers nvmlInit/Shutdown (virtual refcount),
             DeviceGetCount, GetHandleByIndex (a virtual handle), GetName, GetUUID, GetCudaComputeCapability,
             GetMemoryInfo().total and GetP2PStatus from the table's 'nvml' section, which the helper reads from NVML in
             its own process; torch's _raw_device_count_nvml / _raw_device_uuid_nvml answer from it too. Anything else
             initializes NVML for real first (logged fallback; virtual handles become real ones). finalize() makes any
             still-open virtual init real BEFORE restoring the originals (log 'nvml_materialized'), and a virtual
             handle kept past finalize resolves to the real one through ctypes' _as_parameter_. Why: NVML's first
             init dlopens libcuda and calls cuGetExportTable in a per-process pthread_once (nvml_checkpoint.c),
             and libcuda starts its own thread; a forked child inherits "done" without the thread (forkseg H1,
             zythreads2 §2).
  driver_threads() / state()['driver_threads']: OS threads named cuda* (libcuda's own threads). problems() reports
             them: a libcuda thread before cuInit means the driver did partial internal init in this process.
"""
import ctypes
import importlib
import importlib.abc
import inspect
import json
import multiprocessing
import multiprocessing.util
import os
import subprocess
import sys
import time
import traceback

FLAG = 'RP_CUDA_CLEAN_IMPORTS'
PARTS = ('nvml', 'torch', 'driver', 'triton')
DRIVER_MODULES = ('cuda.bindings.driver', 'cuda.cuda')
TRITON_DRIVER = 'triton.backends.nvidia.driver'
EXECUTOR = 'vllm.v1.executor.multiproc_executor'
SAFE_DRIVER = ('cuDriverGetVersion', 'cuGetErrorString', 'cuGetErrorName')   # valid before cuInit
# CUdevice_attribute id -> torch _CudaDeviceProperties field
TORCH_ATTRS = {75: 'major', 76: 'minor', 16: 'multi_processor_count', 18: 'is_integrated', 84: 'is_multi_gpu_board',
               39: 'max_threads_per_multi_processor', 10: 'warp_size', 82: 'regs_per_multiprocessor',
               38: 'L2_cache_size', 1: 'max_threads_per_block', 8: 'shared_memory_per_block',
               97: 'shared_memory_per_block_optin', 81: 'shared_memory_per_multiprocessor', 33: 'pci_bus_id',
               34: 'pci_device_id', 50: 'pci_domain_id', 13: 'clock_rate', 36: 'memory_clock_rate',
               37: 'memory_bus_width'}
BOOL_FIELDS = ('is_integrated', 'is_multi_gpu_board')

FINALIZE_VARIANTS = ('default', 'skip-cuinit', 'cuinit-after-restore', 'ctypes-cuinit', 'no-driver-restore', 'none')
CTYPES_LIBS = ('libcuda', 'libnvidia-ml', 'libcudart')
CTYPES_SYMS = ('cu', 'nvml')            # cu* covers the driver (cuInit, cuGetProcAddress...) and cudart (cuda*) names

_S = dict(installed=False, mode='off', role=None, parts=(), t_install=None, armed=False, env_set=False,
          table=None, table_tried=False, table_src=None, counts={}, first_caller={}, answers={}, fallbacks=[],
          logged_fallbacks=0, virtual_cuinit=False, real_cuinit=None, driver_init=None,
          nvml=dict(balanced=0, unbalanced=0, shutdown_rc=[], skipped=[]), patches=[], marks=[],
          strict=False, real_calls={}, ctypes={}, audit=False, cuinit_rc=None, cuinit_how=None, finalize_variant=None,
          nvml_defer=False)
_NV = dict(virtual=0, real=False, mods=[], orig={})            # RP_CUDA_CLEAN_NVML_DEFER state
_WRAPPERS = {}          # id(wrapper) -> (wrapper, original)
_REAL = {}              # name -> original torch function (DeviceProps fallbacks)
_NVML = {}


def enabled():
    return os.environ.get(FLAG) == '1'


def _fmt(v):
    return v if isinstance(v, (int, float)) or (isinstance(v, str) and v and ' ' not in v) else json.dumps(v,
                                                                                                          default=str)


def log(msg, **kw):
    """One greppable line: 'RP_CUDACLEAN <msg> pid=<pid> k=v ...' (values with spaces are JSON)."""
    try:
        print('RP_CUDACLEAN %s pid=%d %s' % (msg, os.getpid(), ' '.join('%s=%s' % (k, _fmt(v)) for k, v in kw.items())),
              flush=True)
    except Exception:
        pass


def _proc_name():
    try:
        return multiprocessing.current_process().name
    except Exception:
        return '?'


def caller(limit=12):
    """Short Python stack of the current call, without importlib and this module's frames (file:line function)."""
    try:
        frames = traceback.extract_stack(limit=60)[:-1]
    except Exception:
        return []
    keep = [f for f in frames if not f.filename.startswith('<frozen')
            and not f.filename.endswith(('rp_cudaclean.py', 'cudaclean_probe.py'))]
    return ['%s:%d %s' % (f.filename, f.lineno, f.name) for f in keep[-limit:]]


def _count(kind):
    _S['counts'][kind] = _S['counts'].get(kind, 0) + 1
    if kind not in _S['first_caller']:
        _S['first_caller'][kind] = caller(8)


def _real_call(kind):
    """A driver call that reached the real driver before finalize (recorder; forkseg)."""
    _S['real_calls'][kind] = _S['real_calls'].get(kind, 0) + 1
    if 'real.' + kind not in _S['first_caller'] and len(_S['first_caller']) < 96:
        _S['first_caller']['real.' + kind] = caller(8)


def _audit(event, args):
    """sys.addaudithook: ctypes access to libcuda / NVML / cudart while installed and not final (recorder only)."""
    if not event.startswith('ctypes.dl') or _S['mode'] not in ('deferring', 'released'):
        return
    try:
        if event == 'ctypes.dlopen':
            name = args[0] if args else None
            name = name.decode() if isinstance(name, bytes) else name
            base = os.path.basename(name) if isinstance(name, str) else None
            if base and base.startswith(CTYPES_LIBS):
                key = 'dlopen:' + base
            else:
                return
        elif event.startswith('ctypes.dlsym') and len(args) > 1 and isinstance(args[1], str) \
                and args[1].startswith(CTYPES_SYMS):
            key = 'dlsym:' + args[1]
        else:
            return
        _S['ctypes'][key] = _S['ctypes'].get(key, 0) + 1
        if 'ctypes.' + key not in _S['first_caller'] and len(_S['first_caller']) < 96:
            _S['first_caller']['ctypes.' + key] = caller(8)
    except Exception:
        pass


def _answer(key, value):
    if key not in _S['answers'] and len(_S['answers']) < 128:
        _S['answers'][key] = value


def _fallback(fn, why):
    row = dict(fn=fn, why=why, mode=_S['mode'], stack=caller())
    if len(_S['fallbacks']) < 20:
        _S['fallbacks'].append(row)
    if _S['mode'] == 'deferring' and _S['logged_fallbacks'] < 5:
        _S['logged_fallbacks'] += 1
        log('fallback', fn=fn, why=why, at=row['stack'][-4:])


# ---------------------------------------------------------------- device table (helper process, cached per pod)
HELPER = r'''
import ctypes, json
lib = ctypes.CDLL("libcuda.so.1")
def chk(rc, what):
    if rc: raise SystemExit("%s rc=%d" % (what, rc))
chk(lib.cuInit(0), "cuInit")
n = ctypes.c_int(); chk(lib.cuDeviceGetCount(ctypes.byref(n)), "cuDeviceGetCount")
devices = []
for i in range(n.value):
    dev = ctypes.c_int(); chk(lib.cuDeviceGet(ctypes.byref(dev), i), "cuDeviceGet")
    name = ctypes.create_string_buffer(256); chk(lib.cuDeviceGetName(name, 256, dev), "cuDeviceGetName")
    mem = ctypes.c_size_t(); chk(lib.cuDeviceTotalMem_v2(ctypes.byref(mem), dev), "cuDeviceTotalMem_v2")
    attrs = {}
    for a in range(1, 200):
        v = ctypes.c_int()
        if lib.cuDeviceGetAttribute(ctypes.byref(v), a, dev) == 0:
            attrs[str(a)] = v.value
    devices.append(dict(index=i, name=name.value.decode(), total_memory=mem.value, attrs=attrs))
ver = ctypes.c_int(); lib.cuDriverGetVersion(ctypes.byref(ver))
nvml = None
try:                                   # NVML's own answers (RP_CUDA_CLEAN_NVML_DEFER), read here, never in an R process
    nv = ctypes.CDLL("libnvidia-ml.so.1")
    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]
    if nv.nvmlInit_v2() == 0:
        c = ctypes.c_uint(); nv.nvmlDeviceGetCount_v2(ctypes.byref(c))
        hs, nd = [], []
        for i in range(c.value):
            h = ctypes.c_void_p(); nv.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)); hs.append(h)
            nm = ctypes.create_string_buffer(96); nv.nvmlDeviceGetName(h, nm, 96)
            uu = ctypes.create_string_buffer(96); nv.nvmlDeviceGetUUID(h, uu, 96)
            mj, mn = ctypes.c_int(), ctypes.c_int(); nv.nvmlDeviceGetCudaComputeCapability(h, ctypes.byref(mj), ctypes.byref(mn))
            m = Mem(); nv.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m))
            nd.append(dict(name=nm.value.decode(), uuid=uu.value.decode(), cc=[mj.value, mn.value], memory_total=m.total))
        p2p = {}
        for idx in range(5):
            rows = []
            for a in hs:
                row = []
                for b in hs:
                    s = ctypes.c_int(-1)
                    row.append(s.value if nv.nvmlDeviceGetP2PStatus(a, b, idx, ctypes.byref(s)) == 0 else None)
                rows.append(row)
            p2p[str(idx)] = rows
        nvml = dict(count=c.value, devices=nd, p2p=p2p)
        nv.nvmlShutdown()
except Exception as e:
    nvml = dict(error=repr(e))
print(json.dumps(dict(devices=devices, driver_version=ver.value, nvml=nvml)))
'''


def props_path():
    p = os.environ.get('RP_CUDA_CLEAN_PROPS')
    if p:
        return p
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    key = 'all' if cvd is None else (''.join(c if c.isalnum() else '_' for c in cvd) or 'none')
    return '/tmp/rp-cudaclean/props-%s.json' % key


def run_helper(timeout=120):
    """-> (table dict, None) or (None, reason). The helper runs cuInit in its own process (no LD_PRELOAD hook)."""
    env = {k: v for k, v in os.environ.items() if k != 'LD_PRELOAD'}
    try:
        r = subprocess.run([sys.executable, '-I', '-c', HELPER], capture_output=True, text=True, timeout=timeout,
                           env=env)
        if r.returncode:
            return None, 'helper rc=%d %s' % (r.returncode, (r.stderr or r.stdout).strip()[-300:])
        return json.loads(r.stdout), None
    except Exception as e:
        return None, '%s: %s' % (type(e).__name__, e)


def table():
    """The device table ({devices: [{index, name, total_memory, attrs: {id: value}}], ...}) or None. Loaded once:
    from the cache file, else from the helper (then cached). RP_CUDA_CLEAN_HELPER=0 disables the helper."""
    if _S['table_tried']:
        return _S['table']
    _S['table_tried'] = True
    t0, path, data, src = time.time(), props_path(), None, None
    try:
        with open(path) as f:
            data, src = json.load(f), 'file'
    except FileNotFoundError:
        pass
    except Exception as e:
        src = 'bad-file:%s' % type(e).__name__
    if (isinstance(data, dict) and _S['nvml_defer'] and 'nvml' not in data
            and os.environ.get('RP_CUDA_CLEAN_HELPER', '1') != '0'):
        data, src = None, 'file without nvml section'      # E0-prime's cache predates it: rebuild once
    if data is None and os.environ.get('RP_CUDA_CLEAN_HELPER', '1') != '0':
        data, err = run_helper()
        src = 'helper' if data else 'helper-failed:%s' % err
        if data:
            data.update(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), created=time.time())
            try:
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                tmp = '%s.%d.tmp' % (path, os.getpid())
                with open(tmp, 'w') as f:
                    json.dump(data, f)
                os.replace(tmp, path)
            except OSError as e:
                src += ' (not cached: %s)' % type(e).__name__
    if not (isinstance(data, dict) and isinstance(data.get('devices'), list) and data['devices']):
        data = None
    _S['table'], _S['table_src'] = data, '%s %s' % (src or 'missing', path)
    devs = (data or {}).get('devices') or []
    log('table', source=_S['table_src'], devices=len(devs), seconds=round(time.time() - t0, 3),
        name=devs[0].get('name') if devs else None,
        cc='%s.%s' % (devs[0]['attrs'].get('75'), devs[0]['attrs'].get('76')) if devs else None,
        sms=devs[0]['attrs'].get('16') if devs else None)
    return data


def _row(device):
    i = _index(device)
    t = table()
    devs = (t or {}).get('devices') or []
    return i, (devs[i] if i is not None and 0 <= i < len(devs) else None)


def _index(device):
    try:
        if device is None:
            return 0                               # current device; 0 until set_device (CUDA's default)
        if isinstance(device, int):
            return device
        if isinstance(device, str):
            return int(device.split(':')[1]) if ':' in device else 0
        idx = getattr(device, 'index', None)
        return 0 if idx is None else int(idx)
    except Exception:
        return None


def torch_props(row):
    a = row.get('attrs') or {}
    v = dict(name=row.get('name'), total_memory=row.get('total_memory'))
    for aid, field in TORCH_ATTRS.items():
        if str(aid) in a:
            v[field] = bool(a[str(aid)]) if field in BOOL_FIELDS else a[str(aid)]
    return v


class DeviceProps:
    """Stand-in for torch's _CudaDeviceProperties, filled from the device table. Any other attribute resolves through
    the real torch.cuda.get_device_properties (which initializes torch CUDA: correct, logged as a fallback)."""

    def __init__(self, index, values):
        self.__dict__.update(values)
        self.__dict__['_rp_cudaclean_index'] = index

    def __getattr__(self, name):
        if name.startswith('__') or name == '_rp_cudaclean_index':
            raise AttributeError(name)
        _fallback('torch.get_device_properties().' + name, 'attribute not in the device table')
        return getattr(_REAL['get_device_properties'](self.__dict__['_rp_cudaclean_index']), name)

    def __repr__(self):
        return ("_CudaDeviceProperties(name='%s', major=%s, minor=%s, total_memory=%sMB, multi_processor_count=%s, "
                "rp_cudaclean=True)" % (self.__dict__.get('name'), self.__dict__.get('major'),
                                        self.__dict__.get('minor'), (self.__dict__.get('total_memory') or 0) >> 20,
                                        self.__dict__.get('multi_processor_count')))


# ---------------------------------------------------------------- post-import hooks (no import-order change)
class _HookLoader(importlib.abc.Loader):
    def __init__(self, inner, name, finder):
        self.inner, self.name, self.finder = inner, name, finder

    def __getattr__(self, name):            # get_source, get_filename, is_package, ... of the wrapped loader
        return getattr(self.inner, name)

    def create_module(self, spec):
        create = getattr(self.inner, 'create_module', None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.inner.exec_module(module)
        self.finder.run(self.name, module)


class _AfterImport(importlib.abc.MetaPathFinder):
    """Runs hooks right after a named module executes. The spec still comes from the finders behind this one
    (qualified-load's parent_launch, rp_forkw's hook, the Foundry bridge), so nothing they do is skipped."""

    def __init__(self):
        self.hooks, self.busy = {}, set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.hooks or fullname in self.busy:
            return None
        self.busy.add(fullname)
        try:
            spec = None
            for finder in list(sys.meta_path):
                if finder is self or not hasattr(finder, 'find_spec'):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    break
        finally:
            self.busy.discard(fullname)
        if spec is not None and spec.loader is not None and hasattr(spec.loader, 'exec_module'):
            spec.loader = _HookLoader(spec.loader, fullname, self)
        return spec

    def run(self, fullname, module):
        for fn in self.hooks.pop(fullname, ()):
            try:
                fn(module)
            except Exception as e:      # log, don't abort: the real functions stay in place
                log('hook_failed', module=fullname, error=repr(e))


_FINDER = []


def after_import(name, fn):
    """fn(module) right after `name` is imported (at once if it already is)."""
    mod = sys.modules.get(name)
    if mod is not None:
        try:
            fn(mod)
        except Exception as e:
            log('hook_failed', module=name, error=repr(e))
        return
    if not _FINDER or _FINDER[0] not in sys.meta_path:
        _FINDER[:] = [_AfterImport()]
        sys.meta_path.insert(0, _FINDER[0])
    _FINDER[0].hooks.setdefault(name, []).append(fn)


def _patch(owner, name, wrapper, orig):
    wrapper._rp_cudaclean = True
    wrapper.__wrapped__ = orig
    for attr in ('__name__', '__qualname__', '__doc__'):
        try:
            setattr(wrapper, attr, getattr(orig, attr))
        except Exception:
            pass
    setattr(owner, name, wrapper)
    _S['patches'].append((owner, name, orig, wrapper))
    _WRAPPERS[id(wrapper)] = (wrapper, orig)


def _mark(mod):
    try:
        mod._rp_cudaclean = True
        _S['marks'].append(mod)
    except Exception:
        pass


def _ours(f):
    return id(f) in _WRAPPERS and _WRAPPERS[id(f)][0] is f


# ---------------------------------------------------------------- nvml + torch
def _nvml_shutdown():
    fn = _NVML.get('fn')
    if fn is None:
        alt = os.environ.get('RP_CUDA_CLEAN_NVML_SHUTDOWN')        # CPU tests only: 'module:function'
        if alt:
            mod, _, attr = alt.partition(':')
            fn = getattr(importlib.import_module(mod), attr)
        else:
            fn = ctypes.CDLL('libnvidia-ml.so.1').nvmlShutdown      # the library torch's ctypes code initialized
        _NVML['fn'] = fn
    return fn()


def _called_names(src):
    """Names called in a function's source (`nvml_h.nvmlInit()` -> 'nvmlInit'); comments and strings ignored."""
    import ast
    import textwrap
    out = set()
    for node in ast.walk(ast.parse(textwrap.dedent(src))):
        if isinstance(node, ast.Call):
            f = node.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None))
    return out


def _nvml_tab():
    """The table's NVML section (RP_CUDA_CLEAN_NVML_DEFER), or None."""
    nv = (table() or {}).get('nvml')
    if isinstance(nv, dict) and isinstance(nv.get('devices'), list) and nv.get('count') == len(nv['devices']):
        return nv
    return None


def _nvml_torch_answer(name):
    nv = _nvml_tab()
    if nv is None:
        return None
    if name == '_raw_device_count_nvml':
        _count('nvml.torch_count')
        _answer(('nvml_count',), nv['count'])
        return nv['count']
    uu = [d.get('uuid') for d in nv['devices']]
    if uu and all(isinstance(u, str) and u for u in uu):
        _count('nvml.torch_uuids')
        return list(uu)
    return None


def _patch_nvml(cuda):
    for name, ok in (('_raw_device_count_nvml', lambda r: isinstance(r, int) and r >= 0),
                     ('_raw_device_uuid_nvml', lambda r: r is not None)):
        orig = getattr(cuda, name, None)
        if orig is None or _ours(orig):
            continue
        balance = True
        try:
            called = _called_names(inspect.getsource(orig))
        except Exception as e:
            _S['nvml']['skipped'].append('%s: no source (%s)' % (name, type(e).__name__))
            balance = False
        else:
            if not called & {'nvmlInit', 'nvmlInit_v2'} or called & {'nvmlShutdown'}:
                _S['nvml']['skipped'].append('%s: shape differs' % name)   # we balance exactly one init, no shutdown
                balance = False
        if not balance and not _S['nvml_defer']:
            continue

        def wrapper(*a, _orig=orig, _ok=ok, _name=name, _bal=balance, **k):
            if _S['nvml_defer'] and _S['mode'] == 'deferring' and not _NV['real'] and not a and not k:
                ans = _nvml_torch_answer(_name)
                if ans is not None:
                    return ans                              # no NVML init at all (RP_CUDA_CLEAN_NVML_DEFER)
            r = _orig(*a, **k)
            if not _bal:
                return r
            if _ok(r):
                _S['nvml']['balanced'] += 1
                try:
                    rc = _nvml_shutdown()
                    if rc and len(_S['nvml']['shutdown_rc']) < 8:
                        _S['nvml']['shutdown_rc'].append(rc)
                except Exception as e:
                    _S['nvml']['shutdown_rc'].append(repr(e)[:80])
            else:
                _S['nvml']['unbalanced'] += 1
            return r
        _patch(cuda, name, wrapper, orig)


def _patch_torch(cuda):
    if 'nvml' in _S['parts']:
        _patch_nvml(cuda)
    if 'torch' not in _S['parts'] or _ours(getattr(cuda, 'get_device_properties', None)):
        return
    is_init = getattr(cuda, 'is_initialized', None)

    def deferring():
        try:
            return _S['mode'] == 'deferring' and is_init is not None and not is_init()
        except Exception:
            return False

    orig_props = getattr(cuda, 'get_device_properties', None)
    _REAL['get_device_properties'] = orig_props
    if orig_props is not None:
        def get_device_properties(device=None):
            if deferring():
                i, row = _row(device)
                if row is not None:
                    _count('torch.get_device_properties')
                    values = torch_props(row)
                    _answer(('props', i), values)
                    return DeviceProps(i, values)
                _fallback('torch.get_device_properties', 'no table row for %r' % (device,))
            return orig_props(device)
        _patch(cuda, 'get_device_properties', get_device_properties, orig_props)

    orig_cap = getattr(cuda, 'get_device_capability', None)
    if orig_cap is not None:
        def get_device_capability(device=None):
            if deferring():
                i, row = _row(device)
                p = torch_props(row) if row is not None else {}
                if 'major' in p and 'minor' in p:
                    _count('torch.get_device_capability')
                    _answer(('cap', i), (p['major'], p['minor']))
                    return (p['major'], p['minor'])
                _fallback('torch.get_device_capability', 'no table row for %r' % (device,))
            return orig_cap(device)
        _patch(cuda, 'get_device_capability', get_device_capability, orig_cap)

    orig_name = getattr(cuda, 'get_device_name', None)
    if orig_name is not None:
        def get_device_name(device=None):
            if deferring():
                i, row = _row(device)
                if row is not None and row.get('name'):
                    _count('torch.get_device_name')
                    _answer(('name', i), row['name'])
                    return row['name']
                _fallback('torch.get_device_name', 'no table row for %r' % (device,))
            return orig_name(device)
        _patch(cuda, 'get_device_name', get_device_name, orig_name)

    orig_cur = getattr(cuda, 'current_device', None)
    if orig_cur is not None:
        def current_device():
            if deferring():                     # no set_device can have run before torch CUDA init
                _count('torch.current_device')
                return 0
            return orig_cur()
        _patch(cuda, 'current_device', current_device, orig_cur)
    log('patched', module=cuda.__name__, parts=[p for p in ('nvml', 'torch') if p in _S['parts']],
        nvml_skipped=_S['nvml']['skipped'] or None)


# ---------------------------------------------------------------- cuda-python driver
def _rc(r):
    """CUresult of a cuda-python tuple return (or a plain int from ctypes) as an int; None if not a result."""
    try:
        return int(r[0]) if isinstance(r, tuple) else int(r)
    except Exception:
        return None


def _real_cuinit(why):
    fn = _S['driver_init']
    _S['real_cuinit'] = why
    if fn is None:
        return None
    _real_call('cuInit')
    r = fn(0)
    _S['cuinit_rc'], _S['cuinit_how'] = _rc(r), 'cuda-python'
    return r


def _ctypes_cuinit(why):
    """Real cuInit through ctypes on the mapped libcuda (finalize variant 'ctypes-cuinit'): cuda-python's driver table
    is then never resolved in this process. RP_CUDA_CLEAN_CTYPES_CUINIT='module:function' replaces it (CPU tests)."""
    _S['real_cuinit'] = why
    alt = os.environ.get('RP_CUDA_CLEAN_CTYPES_CUINIT')
    if alt:
        mod, _, attr = alt.partition(':')
        fn = getattr(importlib.import_module(mod), attr)
    else:
        libs = mapped_libs('libcuda.so')
        fn = ctypes.CDLL(libs[0] if libs else 'libcuda.so.1').cuInit
    r = fn(0)
    _S['cuinit_rc'], _S['cuinit_how'] = _rc(r), 'ctypes'
    return r


def _driver_answer(mod, name, a, k):
    t = table()
    devs = (t or {}).get('devices') or []
    if not devs or k:
        return None
    try:
        ok = mod.CUresult.CUDA_SUCCESS
        if name == 'cuDeviceGetCount' and not a:
            _answer(('count',), len(devs))
            return (ok, len(devs))
        if name == 'cuDeviceGet' and len(a) == 1:
            i = int(a[0])
            if 0 <= i < len(devs):
                return (ok, mod.CUdevice(i))
        if name == 'cuDeviceGetAttribute' and len(a) == 2:
            aid, i = int(a[0]), int(a[1])
            if 0 <= i < len(devs):
                v = (devs[i].get('attrs') or {}).get(str(aid))
                if v is not None:
                    _answer(('attr', aid, i), v)
                    return (ok, v)
    except Exception:
        return None
    return None


def _strict_version(mod, a, k):
    """RP_CUDA_CLEAN_STRICT=1: cuDriverGetVersion() from the table (no real driver call), else None."""
    ver = (table() or {}).get('driver_version')
    if a or k or not isinstance(ver, int) or ver <= 0:
        return None
    try:
        _count('driver.cuDriverGetVersion')
        _answer(('driver_version',), ver)
        return (mod.CUresult.CUDA_SUCCESS, ver)
    except Exception:
        return None


def _driver_wrapper(mod, name, orig):
    def real(*a, **k):
        if _S['mode'] != 'final':
            _real_call(name)               # recorder: this call reaches the driver in a not-yet-finalized process
        return orig(*a, **k)

    def wrapper(*a, **k):
        if name in SAFE_DRIVER:
            if name == 'cuDriverGetVersion' and _S['strict'] and _S['mode'] == 'deferring':
                ans = _strict_version(mod, a, k)
                if ans is not None:
                    return ans
            return real(*a, **k)
        if _S['mode'] != 'deferring':
            if name == 'cuInit':
                _S['real_cuinit'] = _S['real_cuinit'] or 'cuInit after release'
            elif _S['virtual_cuinit'] and not _S['real_cuinit']:
                _real_cuinit('%s after release' % name)
            return real(*a, **k)
        if name == 'cuInit':
            if _S['real_cuinit']:
                return real(*a, **k)
            if table() is None:
                _fallback('driver.cuInit', 'no device table')
                _S['real_cuinit'] = 'cuInit (no table)'
                return real(*a, **k)
            _S['virtual_cuinit'] = True
            _count('driver.cuInit')
            return (mod.CUresult.CUDA_SUCCESS,)
        if not _S['virtual_cuinit'] or _S['real_cuinit']:
            return real(*a, **k)           # nothing claimed: the real driver answers (NOT_INITIALIZED before cuInit)
        ans = _driver_answer(mod, name, a, k)
        if ans is not None:
            _count('driver.' + name)
            return ans
        _fallback('driver.' + name, 'not answerable while cuInit is deferred')
        _real_cuinit(name)
        return real(*a, **k)
    return wrapper


def _patch_driver(mod):
    if getattr(mod, '_rp_cudaclean', False):
        return
    init = getattr(mod, 'cuInit', None)
    if init is None or _ours(init):       # e.g. legacy cuda.cuda star-imported our wrappers: nothing to add
        _mark(mod)
        return
    if _S['driver_init'] is None:
        _S['driver_init'] = init
    n = 0
    for name in dir(mod):
        if not name.startswith('cu'):
            continue
        f = getattr(mod, name, None)
        if not callable(f) or isinstance(f, type) or _ours(f):
            continue
        _patch(mod, name, _driver_wrapper(mod, name, f), f)
        n += 1
    _mark(mod)
    log('patched', module=mod.__name__, functions=n)


# ---------------------------------------------------------------- triton (newer CudaDriver.is_active: ctypes cuInit)
def _patch_triton(mod):
    f = getattr(mod, '_cuda_driver_is_active', None)
    if f is None or _ours(f):
        log('patched', module=mod.__name__, note='no _cuda_driver_is_active: is_active uses torch.cuda.is_available')
        return

    def _cuda_driver_is_active(*a, **k):
        if _S['mode'] == 'deferring' and not a and not k:
            t = table()
            if t is not None:
                _count('triton._cuda_driver_is_active')
                _answer(('triton_active',), True)
                return True
            _fallback('triton._cuda_driver_is_active', 'no device table')
        return f(*a, **k)
    _patch(mod, '_cuda_driver_is_active', _cuda_driver_is_active, f)
    log('patched', module=mod.__name__, functions=1)


# ---------------------------------------------------------------- pynvml (RP_CUDA_CLEAN_NVML_DEFER=1)
NVML_INIT = ('nvmlInit', 'nvmlInit_v2', 'nvmlInitWithFlags')
_NO = object()


class NvmlHandle:
    """Virtual nvmlDevice_t for device `index` while NVML is deferred. Turned into the real handle (real nvmlInit
    first) as soon as it reaches a function the table cannot answer, or any call after release."""
    __slots__ = ('index',)

    def __init__(self, index):
        self.index = index

    def __repr__(self):
        return '<rp_cudaclean virtual nvmlDevice %d>' % self.index

    @property
    def _as_parameter_(self):
        """ctypes protocol: a virtual handle that reaches the real pynvml (e.g. one kept past finalize) is passed as
        the real handle of the same index. NVML must be really initialized by then (finalize materializes any
        outstanding virtual init)."""
        for name in _NV['mods']:
            mod = sys.modules.get(name)
            fn = _NV['orig'].get(id(mod), {}).get('nvmlDeviceGetHandleByIndex') if mod is not None else None
            if fn is not None:
                return fn(self.index)
        raise TypeError('rp_cudaclean virtual NVML handle %d: no pynvml to resolve it' % self.index)


class NvmlMemory:
    """nvmlDeviceGetMemoryInfo() result: .total from the table; any other field (free, used, ...) is a real query."""

    def __init__(self, mod, index, total, args, kwargs):
        self.__dict__.update(total=total, _rp=(mod, index, args, kwargs))

    def __getattr__(self, name):
        if name.startswith('__') or name == '_rp':
            raise AttributeError(name)
        mod, index, args, kwargs = self.__dict__['_rp']
        _nvml_go_real(mod, 'nvmlDeviceGetMemoryInfo().' + name)
        real = _NV['orig'][id(mod)]['nvmlDeviceGetMemoryInfo']
        return getattr(real(_nvml_real_handle(mod, NvmlHandle(index)), *args[1:], **kwargs), name)


def _nvml_real_handle(mod, x):
    if isinstance(x, NvmlHandle):
        return _NV['orig'][id(mod)]['nvmlDeviceGetHandleByIndex'](x.index)
    return x


def _nvml_go_real(mod, why, fallback=True):
    """From here on NVML is real in this process: one real init per outstanding virtual init. -> number made real.
    fallback=True: an unanswerable call (logged fallback). False: finalize materializing what is still open."""
    if _NV['real']:
        return 0
    _NV['real'] = True
    if fallback:
        _fallback('nvml.' + why, 'not answerable while NVML is deferred')
    init = _NV['orig'][id(mod)].get('nvmlInit')
    n, _NV['virtual'] = _NV['virtual'], 0
    for _ in range(n if init is not None else 0):
        _real_call('nvmlInit')
        init()
    return n if init is not None else 0


def _nvml_materialize(why):
    """finalize: virtual nvmlInit calls still open (a caller inside with_nvml_context, or an init without shutdown)
    become real BEFORE the originals are restored, so that caller's later real nvmlShutdown / handle use balances
    (zythreads2 R1). Normally nothing is open."""
    if _NV['virtual'] <= 0 or _NV['real']:
        return
    for name in _NV['mods']:
        mod = sys.modules.get(name)
        if mod is not None and id(mod) in _NV['orig']:
            try:
                n = _nvml_go_real(mod, 'finalize: %s' % why, fallback=False)
                log('nvml_materialized', why=why, process=_proc_name(), real_inits=n)
            except Exception as e:
                log('nvml_materialize_failed', error=repr(e)[:200])
            return


def _nvml_answer(mod, name, a, k):
    nv = _nvml_tab()
    if nv is None or k:
        return _NO
    devs = nv['devices']

    def dev(h):
        return devs[h.index] if isinstance(h, NvmlHandle) and 0 <= h.index < len(devs) else None
    try:
        if name in ('nvmlDeviceGetCount', 'nvmlDeviceGetCount_v2') and not a:
            return nv['count']
        if name in ('nvmlDeviceGetHandleByIndex', 'nvmlDeviceGetHandleByIndex_v2') and len(a) == 1:
            i = int(a[0])
            return NvmlHandle(i) if 0 <= i < len(devs) else _NO
        d = dev(a[0]) if a else None
        if d is None:
            return _NO
        if name == 'nvmlDeviceGetName' and len(a) == 1 and isinstance(d.get('name'), str):
            return d['name']
        if name == 'nvmlDeviceGetUUID' and len(a) == 1 and isinstance(d.get('uuid'), str):
            return d['uuid']
        if name == 'nvmlDeviceGetCudaComputeCapability' and len(a) == 1 and isinstance(d.get('cc'), list):
            return tuple(d['cc'])
        if name == 'nvmlDeviceGetMemoryInfo' and isinstance(d.get('memory_total'), int):
            return NvmlMemory(mod, a[0].index, d['memory_total'], a, k)
        if name == 'nvmlDeviceGetP2PStatus' and len(a) == 3 and dev(a[1]) is not None:
            v = ((nv.get('p2p') or {}).get(str(int(a[2]))) or [])
            v = v[a[0].index][a[1].index] if a[0].index < len(v) and a[1].index < len(v[a[0].index]) else None
            return v if isinstance(v, int) else _NO
    except Exception:
        return _NO
    return _NO


def _pynvml_wrapper(mod, name, orig):
    def wrapper(*a, **k):
        defer = _S['mode'] == 'deferring' and not _NV['real']
        if name in NVML_INIT:
            if defer and _nvml_tab() is not None:
                _NV['virtual'] += 1
                _count('nvml.init')
                return None
            if _S['mode'] != 'final':
                _real_call(name)
            return orig(*a, **k)
        if name == 'nvmlShutdown':
            if _NV['virtual'] > 0:
                _NV['virtual'] -= 1
                _count('nvml.shutdown')
                return None
            return orig(*a, **k)
        if defer and _NV['virtual'] > 0:
            ans = _nvml_answer(mod, name, a, k)
            if ans is not _NO:
                _count('nvml.' + name)
                return ans
            _nvml_go_real(mod, name)
        elif _NV['virtual'] > 0 or any(isinstance(x, NvmlHandle) for x in a):
            _nvml_go_real(mod, name)                # after release with virtual state: make it real first
        if _S['mode'] != 'final':
            _real_call(name)
        return orig(*[_nvml_real_handle(mod, x) for x in a], **k)
    return wrapper


def _patch_pynvml(mod):
    """vllm.third_party.pynvml / pynvml: every nvml* function (module globals, so pynvml's own internal calls too)."""
    if getattr(mod, '_rp_cudaclean', False):
        return
    orig = _NV.setdefault('orig', {}).setdefault(id(mod), {})
    n = 0
    for name in dir(mod):
        f = getattr(mod, name, None)
        if not name.startswith('nvml') or not callable(f) or isinstance(f, type) or _ours(f):
            continue
        orig[name] = f
        _patch(mod, name, _pynvml_wrapper(mod, name, f), f)
        n += 1
    _NV['mods'].append(mod.__name__)
    _mark(mod)
    log('patched', module=mod.__name__, functions=n, nvml_table=_nvml_tab() is not None)


# ---------------------------------------------------------------- EngineCore executor: release / finalize points
def _patch_executor(mod):
    if getattr(mod, '_rp_cudaclean', False):
        return
    orig_ctx = mod.get_mp_context
    executor = mod.MultiprocExecutor
    orig_init = executor._init_executor

    def get_mp_context(*a, **k):
        if _S['mode'] == 'deferring':
            release('executor start (get_mp_context)')
        return orig_ctx(*a, **k)

    def _init_executor(self):
        try:
            return orig_init(self)
        finally:
            finalize('executor ready')
    _patch(mod, 'get_mp_context', get_mp_context, orig_ctx)
    _patch(executor, '_init_executor', _init_executor, orig_init)
    _mark(mod)
    log('armed', module=mod.__name__, process=_proc_name())


# ---------------------------------------------------------------- lifecycle
class _AfterFork:
    pass


_AFTER_FORK = _AfterFork()


def _after_mp_fork(_obj):
    """multiprocessing's after-fork hook (Process._bootstrap, normal Python context, child process)."""
    if _S['installed'] and _S['mode'] != 'final':
        name = _proc_name()
        if _S['mode'] == 'released' or not name.startswith('EngineCore'):
            finalize('forked child %s' % name, variant=os.environ.get('RP_CUDA_CLEAN_FINALIZE'))


def install(role=None, parts=None):
    """Idempotent. Starts deferring; in an EngineCore also arms release/finalize at its executor."""
    if not _S['installed']:
        want = parts or os.environ.get('RP_CUDA_CLEAN_PARTS') or ','.join(PARTS)
        _S.update(installed=True, mode='deferring', role=role or _proc_name(), t_install=time.time(),
                  parts=tuple(p for p in want.split(',') if p in PARTS),
                  strict=os.environ.get('RP_CUDA_CLEAN_STRICT') == '1')
        _S['nvml_defer'] = os.environ.get('RP_CUDA_CLEAN_NVML_DEFER') == '1' and 'nvml' in _S['parts']
        if not _S['audit']:
            try:
                sys.addaudithook(_audit)       # recorder only; inert once final (audit hooks cannot be removed)
                _S['audit'] = True
            except Exception:
                pass
        if 'nvml' in _S['parts'] and 'PYTORCH_NVML_BASED_CUDA_CHECK' not in os.environ:
            os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
            _S['env_set'] = True
        if 'nvml' in _S['parts'] or 'torch' in _S['parts']:
            after_import('torch.cuda', _patch_torch)
        if _S['nvml_defer']:
            for name in ('vllm.third_party.pynvml', 'pynvml'):
                after_import(name, _patch_pynvml)
        if 'driver' in _S['parts']:
            for name in DRIVER_MODULES:
                after_import(name, _patch_driver)
        if 'triton' in _S['parts']:
            after_import(TRITON_DRIVER, _patch_triton)
        multiprocessing.util.register_after_fork(_AFTER_FORK, _after_mp_fork)
        log('installed', role=_S['role'], parts=','.join(_S['parts']), torch_imported='torch' in sys.modules,
            props=props_path(), env_set=_S['env_set'], **({'strict': True} if _S['strict'] else {}),
            **({'nvml_defer': True} if _S['nvml_defer'] else {}))
    if _proc_name().startswith('EngineCore') and not _S['armed'] and _S['mode'] == 'deferring':
        _S['armed'] = True
        after_import(EXECUTOR, _patch_executor)
    return status(brief=True)


def install_for_entry():
    """rp_r_entry.py at __mp_main__: install in an EngineCore; elsewhere only finish an inherited deferral."""
    name = _proc_name()
    if name.startswith('EngineCore'):
        return install()
    if _S['installed'] and _S['mode'] != 'final':
        finalize('entry in %s' % name)
    return None


def release(why):
    if _S['mode'] != 'deferring':
        return False
    _S['mode'] = 'released'
    log('release', why=why, process=_proc_name(), answered=_S['counts'], fallbacks=len(_S['fallbacks']),
        virtual_cuinit=_S['virtual_cuinit'], real_cuinit=_S['real_cuinit'])
    return True


def _rebind(skip=()):
    """Module globals that captured a wrapper (`from torch.cuda import get_device_properties`) -> the original.
    Wrappers whose id is in `skip` stay (finalize variant 'no-driver-restore')."""
    out = []
    for mname, mod in list(sys.modules.items()):
        d = getattr(mod, '__dict__', None) if mod is not None else None
        if not isinstance(d, dict):
            continue
        for k, v in list(d.items()):
            hit = _WRAPPERS.get(id(v))
            if hit is not None and hit[0] is v and id(v) not in skip:
                d[k] = hit[1]
                out.append('%s.%s' % (mname, k))
    return out


def _is_driver_owner(owner):
    return getattr(owner, '__name__', None) in DRIVER_MODULES


def _cuinit(why, how):
    try:
        if how == 'ctypes':
            _ctypes_cuinit('finalize(ctypes): %s' % why)
        else:
            _real_cuinit('finalize: %s' % why)
    except Exception as e:
        log('finalize_cuinit_failed', how=how, error=repr(e))


def finalize(why, variant=None):
    """End of deferral in this process: real cuInit if one was only virtual, then remove every wrapper.
    variant (default 'default'; forked children get RP_CUDA_CLEAN_FINALIZE via _after_mp_fork, so the EngineCore's
    and main's own finalize always use the default order) picks the order (notes/forkseg.md):
      default               real cuInit (cuda-python), then restore            (the original behaviour)
      skip-cuinit           restore only; torch/cudart run cuInit at their own first use
      cuinit-after-restore  restore, then real cuInit (cuda-python's original function)
      ctypes-cuinit         real cuInit through ctypes on the mapped libcuda, then restore
      no-driver-restore     real cuInit, restore everything except the cuda-python wrappers (they pass through)
      none                  stop answering (released) and leave every wrapper in place; nothing is called"""
    if not _S['installed'] or _S['mode'] == 'final':
        return False
    variant = variant or 'default'
    if variant not in FINALIZE_VARIANTS:
        log('finalize_variant_unknown', variant=variant, used='default')
        variant = 'default'
    _S['finalize_variant'] = variant
    if variant == 'none':
        release('finalize variant none: %s' % why)
        log('finalize', why=why, variant='none', process=_proc_name(), mode=_S['mode'], note='wrappers left (released)')
        return False
    t0 = time.time()
    _S['mode'] = 'released'
    need = _S['virtual_cuinit'] and not _S['real_cuinit']
    if need and variant in ('default', 'no-driver-restore', 'ctypes-cuinit'):
        _cuinit(why, 'ctypes' if variant == 'ctypes-cuinit' else 'cuda-python')
    _nvml_materialize(why)                   # outstanding virtual nvmlInit -> real, before the originals come back
    kept, skip = [], set()
    for owner, name, orig, wrapper in reversed(_S['patches']):
        if variant == 'no-driver-restore' and _is_driver_owner(owner):
            skip.add(id(wrapper))
            continue
        if getattr(owner, name, None) is wrapper:
            setattr(owner, name, orig)
        else:
            kept.append(name)            # someone wrapped on top (e.g. rp_forkw): ours stays, passing through
    if need and variant == 'cuinit-after-restore':
        _cuinit(why, 'cuda-python')
    rebound = _rebind(skip)
    if _FINDER and _FINDER[0] in sys.meta_path:      # hooks not yet run (e.g. cuda.cuda never imported) are dropped
        sys.meta_path.remove(_FINDER[0])
    for mod in _S['marks']:
        try:
            del mod._rp_cudaclean
        except Exception:
            pass
    _S['marks'] = []
    _S['mode'] = 'final'
    extra = {} if variant == 'default' else dict(variant=variant, left=len(skip) or None)
    log('finalize', why=why, process=_proc_name(), answered=_S['counts'], fallbacks=len(_S['fallbacks']),
        virtual_cuinit=_S['virtual_cuinit'], real_cuinit=_S['real_cuinit'], cuinit_rc=_S['cuinit_rc'],
        nvml=_S['nvml'], restored=len(_S['patches']) - len(kept) - len(skip), kept=kept or None,
        rebound=rebound[:12] or None, real_calls=_S['real_calls'] or None, ctypes=_S['ctypes'] or None,
        seconds=round(time.time() - t0, 3), **extra)
    return True


def status(brief=False):
    st = dict(installed=_S['installed'], mode=_S['mode'], role=_S['role'], pid=os.getpid(), process=_proc_name(),
              parts=list(_S['parts']), answered=dict(_S['counts']), virtual_cuinit=_S['virtual_cuinit'],
              real_cuinit=_S['real_cuinit'], fallbacks=len(_S['fallbacks']), table=_S['table_src'],
              nvml=dict(_S['nvml']))
    for key in ('real_calls', 'ctypes'):              # recorder (forkseg): only when something was recorded
        if _S[key]:
            st[key] = dict(_S[key])
    for key in ('strict', 'cuinit_rc', 'cuinit_how', 'finalize_variant', 'nvml_defer'):
        if _S[key] not in (None, False):
            st[key] = _S[key]
    if _S['nvml_defer']:
        st['nvml_deferred'] = dict(virtual=_NV['virtual'], real=_NV['real'], modules=list(_NV['mods']))
    if brief:
        if _S['fallbacks']:
            f = _S['fallbacks'][0]
            st['first_fallback'] = dict(fn=f['fn'], why=f['why'], at=f['stack'][-3:])
        return st
    st.update(fallback_rows=list(_S['fallbacks']), first_caller=dict(_S['first_caller']),
              answers={repr(k): v for k, v in _S['answers'].items()},
              patched=['%s.%s' % (getattr(o, '__name__', type(o).__name__), n) for o, n, _f, _w in _S['patches']])
    return st


# ---------------------------------------------------------------- process state (probe, tests, rp_forkw)
DEV_PREFIX = os.environ.get('RP_CUDACLEAN_DEV_PREFIX', '/dev/nvidia')
_DRV = {}


def open_fds():
    out = {}
    for name in os.listdir('/proc/self/fd'):
        try:
            out[int(name)] = os.readlink('/proc/self/fd/' + name)
        except OSError:
            pass
    return out


def nvidia_fds():
    return sorted([fd, path] for fd, path in open_fds().items() if path.startswith(DEV_PREFIX))


def threads():
    out = []
    for tid in sorted(os.listdir('/proc/self/task'), key=int):
        try:
            with open('/proc/self/task/%s/comm' % tid) as f:
                out.append([int(tid), f.read().strip()])
        except OSError:
            out.append([int(tid), '?'])
    return out


DRIVER_THREAD_PREFIX = 'cuda'           # libcuda names its threads cuda-EvtHandlr, cuda0000280000b, ...


def driver_threads():
    """[tid, comm] of every OS thread named like a CUDA driver thread. Such a thread before cuInit means libcuda did
    partial internal init in this process (E2-prime's EngineCore had cuda00002c0000c at the fork; notes/forkseg.md)."""
    return [t for t in threads() if t[1].startswith(DRIVER_THREAD_PREFIX)]


def child_pids():
    pids = []
    for tid in os.listdir('/proc/self/task'):
        try:
            with open('/proc/self/task/%s/children' % tid) as f:
                pids += [int(x) for x in f.read().split()]
        except OSError:
            pass
    return sorted(pids)


def mapped_libs(fragment):
    """Every distinct mapped shared object whose basename starts with `fragment` (e.g. 'libcuda.so')."""
    out = []
    try:
        with open('/proc/self/maps') as f:
            for line in f:
                parts = line.split(None, 5)
                if len(parts) == 6:
                    p = parts[5].strip()
                    if os.path.basename(p).startswith(fragment) and p not in out:
                        out.append(p)
    except OSError:
        pass
    return out


def driver_initialized(paths=None):
    """True if cuInit has run in this process (cuDeviceGetCount on an already-mapped libcuda returns CUDA_SUCCESS;
    CUDA_ERROR_NOT_INITIALIZED=3 before). None if no libcuda is mapped. Never loads libcuda, never calls cuInit.
    Checks every mapped libcuda (a compat and a host copy may both be mapped)."""
    paths = mapped_libs('libcuda.so') if paths is None else paths
    if not paths:
        return None
    if os.environ.get('RP_CUDACLEAN_PASSIVE_DRIVER_CHECK') == '1':
        # forkseg: no libcuda call at all (a pre-cuInit cuDeviceGetCount is itself a driver touch). True means an
        # open /dev/nvidia-uvm fd, which cuInit opens and keeps. It is NOT proof of cuInit: E0's
        # foundry-enginecore-off had /dev/nvidia-uvm open with cuInit=False (torch's leaked NVML attach; zythreads2
        # R2). problems() words it as "uvm open", not "cuInit ran".
        return any(p.startswith(DEV_PREFIX + '-uvm') for _fd, p in nvidia_fds())
    res = []
    for p in paths:
        try:
            lib = _DRV.get(p)
            if lib is None:
                lib = _DRV[p] = ctypes.CDLL(p).cuDeviceGetCount
            n = ctypes.c_int(0)
            rc = lib(ctypes.byref(n))
            res.append({0: True, 3: False}.get(rc, 'rc=%d' % rc))
        except Exception as e:
            res.append('error:%s' % type(e).__name__)
    if True in res:
        return True
    return False if all(r is False for r in res) else (res[0] if len(res) == 1 else res)


def torch_cuda_initialized():
    t = sys.modules.get('torch')
    if t is None or not hasattr(t, 'cuda'):
        return None
    try:
        return bool(t.cuda.is_initialized())
    except Exception as e:
        return 'error:%s' % type(e).__name__


def state():
    th = threads()
    return dict(pid=os.getpid(), t=time.time(), torch_cuda_initialized=torch_cuda_initialized(),
                driver_initialized=driver_initialized(), libcuda=mapped_libs('libcuda.so'),
                nvml=mapped_libs('libnvidia-ml.so'), nvidia_fds=nvidia_fds(), threads=th,
                driver_threads=[t for t in th if t[1].startswith(DRIVER_THREAD_PREFIX)],
                children=child_pids(), modules=len(sys.modules))


def problems(st, threads_ok=True):
    """Reasons this process must not fork CUDA-using children / be checkpointed host-only (empty = clean)."""
    p = []
    if st.get('torch_cuda_initialized') is True:
        p.append('torch.cuda initialized')
    if st.get('driver_initialized') is True:
        p.append('driver initialized? (/dev/nvidia-uvm open; passive check)'
                 if os.environ.get('RP_CUDACLEAN_PASSIVE_DRIVER_CHECK') == '1' else 'cuInit ran')
    if st.get('nvidia_fds'):
        p.append('nvidia fds %d' % len(st['nvidia_fds']))
    if st.get('driver_threads'):              # a driver thread, even with threads_ok (forkseg: partial libcuda init)
        p.append('cuda driver threads %d (%s)' % (len(st['driver_threads']),
                                                  ','.join(t[1] for t in st['driver_threads'][:3])))
    if not threads_ok and len(st.get('threads') or ()) > 1:
        p.append('threads=%d' % len(st['threads']))
    return p


if __name__ == '__main__':      # python3 rp_cudaclean.py: build (or show) this pod's device table
    print(json.dumps(dict(path=props_path(), table=table()), indent=1, default=str))
