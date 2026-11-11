# SPDX-License-Identifier: Apache-2.0
"""Foundry SAVE/LOAD for vLLM 0.31 breakable (PIECEWISE) CUDA graphs.

A breakable capture (``compilation/breakable_cudagraph.py``) is an ordered chain of plain CUDA graph
segments with eager closures between them: every ``@eager_break_during_capture`` op (attention, the
DSA indexer+attention method, ...) ends the current segment, runs once on the capture stream, is
recorded as ``lambda: fn(*weak_args, **weak_kwargs)`` for replay, and a new segment begins.

SAVE  each segment is captured into a Foundry graph (same pool, current stream) and archived with the
      adapter's filename scheme, mode PIECEWISE and the segment index in the piecewise slot, so LOAD's
      ``start_graph_builds`` / ``finish_one_graph_load`` pick it up in capture order. Every eager
      closure is described in ``bcg_<descriptor>.json``: the callee (module + qualname), the
      argument structure with tensors as (ptr, shape, stride, dtype), modules by their qualified name
      in the model, LayerName wrappers and scalars by value. The entry's output tensors are recorded the same way.
LOAD  no capture: the segment graphs are restored in archived order and each closure is rebuilt from
      the archive (tensor views at the archived addresses, modules looked up by name in the LOAD
      model, the callee re-imported). The capture loop's eager warmup forward still runs on LOAD, so
      the deterministic allocation trajectory matches SAVE's.

Under Foundry's VMM region the activation addresses are identical on SAVE and LOAD; the LOAD path
checks each rebuilt tensor's address against the archive. Unsupported closure arguments (objects that
are not tensors, modules, scalars, strings, None or containers of those) fail the SAVE.
"""

from __future__ import annotations

import functools
import importlib
import json
import os
import time
from typing import Any

import torch

from vllm.logger import init_logger

log = init_logger(__name__)

_INSTALLED = False
_STATS = {"saved": 0, "restored": 0, "segments": 0, "breaks": 0, "empty": 0, "save_s": 0.0, "restore_s": 0.0}


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------

def _desc_key(bd) -> str:
    reqs = bd.num_reqs if bd.num_reqs is not None else "N"
    return "t%d_r%s_%s_l%d" % (bd.num_tokens, reqs, "U" if bd.uniform else "X", bd.num_active_loras)


def _meta_path(workspace_dir: str, bd) -> str:
    return os.path.join(workspace_dir, "bcg_%s.json" % _desc_key(bd))


def _tensor_meta(t: torch.Tensor) -> dict[str, Any]:
    return dict(ptr=t.data_ptr(), shape=list(t.shape), stride=list(t.stride()),
                dtype=str(t.dtype).removeprefix("torch."), device=t.device.index or 0,
                numel=t.numel())


def tensor_from_ptr(ptr: int, shape, stride, dtype: torch.dtype, device: int) -> torch.Tensor:
    """Non-owning tensor view of device memory at ``ptr`` (memory is owned by the graph pool / VMM
    arena). Built as a uint8 view through __cuda_array_interface__ (which cannot describe bf16/fp8)
    and then reinterpreted with the archived dtype, shape and stride."""
    itemsize = torch.empty((), dtype=dtype).element_size()
    if any(s == 0 for s in shape):
        return torch.empty(shape, dtype=dtype, device="cuda:%d" % device)
    span = 1 + sum((s - 1) * st for s, st in zip(shape, stride))
    nbytes = span * itemsize

    class _View:
        __cuda_array_interface__ = dict(shape=(nbytes,), typestr="|u1", data=(int(ptr), False), version=3,
                                        strides=None)

    with torch.cuda.device(device):
        base = torch.as_tensor(_View(), device="cuda:%d" % device)
    return base.view(dtype).as_strided(list(shape), list(stride))


class _Spec:
    """Archive the argument structure of an eager closure."""

    def __init__(self, module_names: dict[int, str]):
        self.module_names = module_names
        self.tensors: list[dict[str, Any]] = []

    def walk(self, o: Any) -> Any:
        if isinstance(o, torch.Tensor):
            self.tensors.append(_tensor_meta(o))
            return {"t": len(self.tensors) - 1}
        if o is None or isinstance(o, (bool, int, float, str)):
            return {"v": o}
        if isinstance(o, torch.dtype):
            return {"dtype": str(o).removeprefix("torch.")}
        if isinstance(o, tuple):
            return {"tuple": [self.walk(e) for e in o]}
        if isinstance(o, list):
            return {"list": [self.walk(e) for e in o]}
        if isinstance(o, dict):
            return {"dict": {k: self.walk(v) for k, v in o.items()}}
        from vllm.utils.torch_utils import LayerName

        if isinstance(o, LayerName):                 # attention ops pass the layer name as an opaque wrapper
            return {"layer_name": o.value}
        if isinstance(o, torch.nn.Module):
            name = self.module_names.get(id(o))
            if name is None:
                raise TypeError("breakable SAVE: closure holds a module that is not part of the model: %s"
                                % type(o).__qualname__)
            return {"module": name}
        raise TypeError("breakable SAVE: closure holds an unsupported object %s (%r); closure "
                        "synthesis cannot archive it" % (type(o).__qualname__, repr(o)[:80]))


def _build(spec: Any, tensors: list[torch.Tensor], root: torch.nn.Module) -> Any:
    if "t" in spec:
        return tensors[spec["t"]]
    if "v" in spec:
        return spec["v"]
    if "dtype" in spec:
        return getattr(torch, spec["dtype"])
    if "tuple" in spec:
        return tuple(_build(e, tensors, root) for e in spec["tuple"])
    if "list" in spec:
        return [_build(e, tensors, root) for e in spec["list"]]
    if "dict" in spec:
        return {k: _build(v, tensors, root) for k, v in spec["dict"].items()}
    if "module" in spec:
        return root if spec["module"] == "" else root.get_submodule(spec["module"])
    if "layer_name" in spec:
        from vllm.utils.torch_utils import LayerName

        return LayerName(spec["layer_name"])
    raise ValueError("bad spec node %r" % (spec,))


def _rebuild_tensors(metas: list[dict[str, Any]], where: str) -> list[torch.Tensor]:
    out = []
    for m in metas:
        t = tensor_from_ptr(m["ptr"], m["shape"], m["stride"], getattr(torch, m["dtype"]), m["device"])
        if m["numel"] and t.data_ptr() != m["ptr"]:
            raise RuntimeError("breakable LOAD %s: rebuilt tensor at 0x%x, archive 0x%x" % (where, t.data_ptr(), m["ptr"]))
        out.append(t)
    return out


def _callee_ref(fn) -> dict[str, str]:
    mod = getattr(fn, "__module__", None)
    qual = getattr(fn, "__qualname__", None)
    if not mod or not qual or "<locals>" in qual or "<lambda>" in qual:
        raise TypeError("breakable SAVE: eager callee %r is not importable by module + qualname" % (fn,))
    return {"module": mod, "qualname": qual}


def _resolve_callee(ref: dict[str, str]):
    """The importable attribute is the decorated wrapper; outside a capture it calls the op directly."""
    obj: Any = importlib.import_module(ref["module"])
    for part in ref["qualname"].split("."):
        obj = getattr(obj, part)
    return obj


def _closure_parts(fn) -> tuple[Any, tuple, dict]:
    """``lambda: fn(*weak_args, **weak_kwargs)`` from eager_break_during_capture."""
    names = fn.__code__.co_freevars
    cells = dict(zip(names, (c.cell_contents for c in (fn.__closure__ or ()))))
    if not {"fn", "weak_args", "weak_kwargs"} <= set(cells):
        raise TypeError("breakable SAVE: unexpected eager closure shape, free vars %r" % (names,))
    return cells["fn"], cells["weak_args"], cells["weak_kwargs"]


def _model_root(wrapper) -> torch.nn.Module:
    root = wrapper.unwrap() if hasattr(wrapper, "unwrap") else wrapper.runnable
    while not isinstance(root, torch.nn.Module) and hasattr(root, "__self__"):
        root = root.__self__  # bound forward
    if not isinstance(root, torch.nn.Module):
        raise TypeError("breakable: cannot find the model module behind %r" % (wrapper,))
    return root


# --------------------------------------------------------------------------------------------------
# SAVE: Foundry-backed segments + closure archive
# --------------------------------------------------------------------------------------------------

def _make_save_capture(base_cls, bd, workspace_dir: str, module_names: dict[int, str]):
    from foundry.graph import CUDAGraph as FoundryCUDAGraph

    from vllm.compilation.foundry_adapter.graph_ops import _build_graph_filename
    from vllm.compilation.foundry_adapter.runtime import get_state
    from vllm.config import CUDAGraphMode

    state = get_state()

    class _SaveCapture(base_cls):
        def __init__(self, pool):
            super().__init__(pool=pool)
            self.items: list[list[Any]] = []       # ["g", seg_index] | ["empty"] | ["e", break_index]
            self.breaks: list[dict[str, Any]] = []
            self._seg = 0

        def _begin_segment(self) -> None:
            assert not self._capturing
            g = FoundryCUDAGraph()
            if self.pool is not None:
                g.capture_begin(pool=self.pool)
            else:
                g.capture_begin()
            self._current_graph = g
            self._capturing = True

        def _end_segment(self) -> None:
            if not self._capturing:
                return
            g = self._current_graph
            g.capture_end()
            t0 = time.perf_counter()
            name = _build_graph_filename(state.capture_index, CUDAGraphMode.PIECEWISE, bd, self._seg)
            try:
                g.save(os.path.join(workspace_dir, name), None)
            except RuntimeError as e:
                if "no nodes" not in str(e):
                    raise
                self.items.append(["empty"])          # nothing captured between two breaks
                _STATS["empty"] += 1
            else:
                self.items.append(["g", self._seg])
                state.capture_index += 1
                self._seg += 1
            _STATS["save_s"] += time.perf_counter() - t0
            self.segments.append(g.replay)
            self._num_graphs += 1
            self._current_graph = None
            self._capturing = False

        def add_eager(self, fn):
            callee, weak_args, weak_kwargs = _closure_parts(fn)
            spec = _Spec(module_names)
            rec = dict(callee=_callee_ref(callee), args=spec.walk(tuple(weak_args)),
                       kwargs=spec.walk(dict(weak_kwargs)))
            rec["tensors"] = spec.tensors
            self._end_segment()
            self.items.append(["e", len(self.breaks)])
            self.breaks.append(rec)
            result = fn()
            self.segments.append(fn)
            self._num_eager_breaks += 1
            self._begin_segment()
            return result

    return _SaveCapture


# --------------------------------------------------------------------------------------------------
# wrapper patch
# --------------------------------------------------------------------------------------------------

def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    import gc

    from vllm.compilation import breakable_cudagraph as bcg
    from vllm.compilation.foundry_adapter.config import (
        CUDAGraphExtensionMode,
        get_config,
        get_graph_extension_mode,
    )
    from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
    from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
    from vllm.model_executor.offloader.base import get_offloader
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import weak_ref_tensors

    cls = bcg.BreakableCUDAGraphWrapper
    orig_capture = cls._capture

    @functools.wraps(orig_capture)
    def patched_capture(self, entry, args, kwargs):
        mode = get_graph_extension_mode()
        if mode == CUDAGraphExtensionMode.NONE:
            return orig_capture(self, entry, args, kwargs)
        cfg = get_config()
        ws = cfg.workspace_dir
        bd = entry.batch_descriptor
        from vllm.compilation.foundry_adapter import runtime as rt

        if mode == CUDAGraphExtensionMode.SAVE:
            validate_cudagraph_capturing_enabled()
            entry.input_addresses = self._collect_tensor_addresses(args, kwargs)
            set_graph_pool_id(self.graph_pool if self.graph_pool is not None
                              else current_platform.graph_pool_handle())
            if gc.isenabled():
                gc.collect()
                torch.accelerator.empty_cache()
            get_offloader().sync_prev_onload()
            root = _model_root(self)
            module_names = {id(m): n for n, m in root.named_modules()}
            capture = _make_save_capture(bcg.BreakableCUDAGraphCapture, bd, ws, module_names)(pool=self.graph_pool)
            rt.set_capture_active(True)
            try:
                with capture:
                    output = self.runnable(*args, **kwargs)
                    get_offloader().join_after_forward()
                    output = weak_ref_tensors(output)
            finally:
                rt.set_capture_active(False)
            spec = _Spec({})
            out_spec = spec.walk(output)
            meta = dict(schema=1, descriptor=_desc_key(bd), items=capture.items, breaks=capture.breaks,
                        output=dict(spec=out_spec, tensors=spec.tensors),
                        input_addresses=entry.input_addresses)
            with open(_meta_path(ws, bd), "w") as f:
                json.dump(meta, f)
            entry.capture = capture
            entry.output = weak_ref_tensors(output)
            _STATS["saved"] += 1
            _STATS["segments"] += sum(1 for i in capture.items if i[0] == "g")
            _STATS["breaks"] += len(capture.breaks)
            log.info("[foundry] breakable SAVE %s: %d segments, %d breaks, %d empty",
                     _desc_key(bd), sum(1 for i in capture.items if i[0] == "g"), len(capture.breaks),
                     sum(1 for i in capture.items if i[0] == "empty"))
            return output

        if mode == CUDAGraphExtensionMode.LOAD:
            from foundry.graph import CUDAGraph as FoundryCUDAGraph

            from vllm.compilation.foundry_adapter import graph_ops
            from vllm.config import CUDAGraphMode

            t0 = time.perf_counter()
            entry.input_addresses = self._collect_tensor_addresses(args, kwargs)
            set_graph_pool_id(self.graph_pool if self.graph_pool is not None
                              else current_platform.graph_pool_handle())
            with open(_meta_path(ws, bd)) as f:
                meta = json.load(f)
            if meta.get("input_addresses") and meta["input_addresses"] != entry.input_addresses:
                raise RuntimeError("breakable LOAD %s: input addresses differ from SAVE" % _desc_key(bd))
            pending = graph_ops._pending_graph_builds
            if pending is None:
                raise RuntimeError("breakable LOAD: no pending graph builds (start_graph_builds not called)")
            state = rt.get_state()
            root = _model_root(self)
            capture = bcg.BreakableCUDAGraphCapture(pool=self.graph_pool)
            for item in meta["items"]:
                if item[0] == "empty":
                    continue
                if item[0] == "g":
                    ident = (CUDAGraphMode.PIECEWISE, bd, item[1])
                    index = pending.identifier_to_index.get(ident)
                    if index is None:
                        raise RuntimeError("breakable LOAD: segment not in archive: %r" % (ident,))
                    graph, out = FoundryCUDAGraph.finish_one_graph_load(pending.pending, index)
                    state.loaded_graphs[ident] = (graph, out)
                    capture.segments.append(graph.replay)
                    capture._num_graphs += 1
                else:
                    b = meta["breaks"][item[1]]
                    tensors = _rebuild_tensors(b["tensors"], "%s break %d" % (_desc_key(bd), item[1]))
                    callee = _resolve_callee(b["callee"])
                    a = _build(b["args"], tensors, root)
                    kw = _build(b["kwargs"], tensors, root)
                    capture.segments.append(functools.partial(callee, *a, **kw))
                    capture._num_eager_breaks += 1
            outs = _rebuild_tensors(meta["output"]["tensors"], "%s output" % _desc_key(bd))
            output = _build(meta["output"]["spec"], outs, root)
            state.loaded_graphs[("breakable-output", bd)] = (None, output)
            entry.capture = capture
            entry.output = output
            _STATS["restored"] += 1
            _STATS["restore_s"] += time.perf_counter() - t0
            log.info("[foundry] breakable LOAD %s: %r in %.3f s", _desc_key(bd), capture,
                     time.perf_counter() - t0)
            return output

        return orig_capture(self, entry, args, kwargs)

    cls._capture = patched_capture
    _INSTALLED = True
    log.info("[foundry] breakable CUDA graph SAVE/LOAD hooks installed")


def stats() -> dict[str, Any]:
    return dict(_STATS)
