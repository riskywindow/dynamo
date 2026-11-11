# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic read-only attach: the writer publishes what loading changed.

Selected with ``model_loader_extra_config={"gms_reconstruction_mode": "load_diff"}``
on both the writer and the readers. No model, GPU or parallelism is pinned.

Writer: snapshots every module (and its ``quant_method`` and attention ``impl``)
right after construction and again after ``load_weights`` +
``process_weights_after_loading``, and publishes the final state plus which
attributes loading changed (the load diff), next to the GMS tensors.

Reader:
1. builds the model on meta and attaches the GMS tensors (params, buffers and
   tensor attributes; parameters that loading deleted are dropped),
2. rebuilds Python-side objects (MoE kernels, KV-scale floats) by running each
   quantization method's ``process_weights_after_loading`` under vLLM's
   ``weights_already_processed()``; other post-processing is skipped because
   its outputs are attached tensors and the weight mapping is read-only,
3. applies the load diff: every simple value loading set is taken from the
   writer; attributes loading deleted are deleted,
4. checks the result against the writer. Objects or tensors that loading
   created and the reader could not rebuild are errors (the engine would not
   serve correctly); value disagreements are logged.
"""

from __future__ import annotations

import enum
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import torch
from gpu_memory_service.client.torch.module import (
    _iter_module_tensors,
    _resolve_module_attr,
)
from gpu_memory_service.client.torch.tensor import GMSTensorSpec

logger = logging.getLogger(__name__)

KEY = "__gms__.load_diff.v1"
MODE = "load_diff"
OWNERS = ("module", "quant_method", "impl")
ABSENT = {"absent": True}
_SIMPLE = (bool, int, float, str, type(None))
_MAX_SEQ = 64
# CPU tensors are not in GMS; small ones travel by value in the descriptor.
_MAX_CPU_VALUES = 4096


# =============================================================================
# Snapshots and the load diff (no vLLM dependency; unit-tested on CPU)
# =============================================================================


def _cls(value) -> str:
    return type(value).__module__ + "." + type(value).__qualname__


def describe(value) -> dict:
    """JSON description of one attribute value."""
    if torch.is_tensor(value):
        desc = {"t": [list(value.shape), str(value.dtype), value.device.type]}
        if value.device.type == "cpu" and value.numel() <= _MAX_CPU_VALUES:
            try:
                desc["data"] = value.detach().reshape(-1).tolist()
            except (RuntimeError, TypeError):
                pass  # dtypes without a Python scalar type (e.g. fp8)
        return desc
    if type(value) is float and value != value:
        return {"token": "float:nan"}
    if type(value) in _SIMPLE:
        return {"v": value}
    if (
        isinstance(value, (tuple, list))
        and type(value) in (tuple, list)
        and len(value) <= _MAX_SEQ
        and all(type(x) in _SIMPLE for x in value)
    ):
        return {"v": list(value), "seq": type(value).__name__}
    if isinstance(value, (enum.Enum, torch.dtype, torch.device)):
        return {"token": _cls(value) + ":" + str(value)}
    return {"o": _cls(value)}


def _owners(module):
    yield "module", module
    for name in ("quant_method", "impl"):
        owner = vars(module).get(name)
        if owner is not None:
            yield name, owner


def snapshot(model: torch.nn.Module) -> dict:
    """{"<module>|<owner>": {attr: description}} for every module.

    An object reachable from two attributes of the same module's owners also
    records ``ref``: the other attribute, so a reader can re-link an object
    that loading wired into a new place (e.g. a kernel stamping itself onto
    its layer).
    """
    state = {}
    for name, module in model.named_modules():
        owners = list(_owners(module))
        places = {}
        for owner_name, owner in sorted(owners, key=lambda o: OWNERS[::-1].index(o[0])):
            for key, value in vars(owner).items():
                if "o" in describe(value):
                    places.setdefault(id(value), []).append(f"{owner_name}.{key}")
        for owner_name, owner in owners:
            skip = set()
            if owner_name == "module":
                skip = set(module._parameters) | set(module._buffers) | set(module._modules)
            row = {}
            for key, value in vars(owner).items():
                if key.startswith("__") or key in skip:
                    continue
                desc = describe(value)
                if "o" in desc:
                    other = [p for p in places.get(id(value), []) if p != f"{owner_name}.{key}"]
                    if other:
                        desc["ref"] = other[0]
                row[key] = desc
            state[f"{name}|{owner_name}"] = row
    return state


def diff(construct: dict, final: dict) -> dict:
    """{"<path>": {attr: construct description or ABSENT}} for attributes that
    loading added, changed or deleted."""
    changed = {}
    for path in set(construct) | set(final):
        before, after = construct.get(path, {}), final.get(path, {})
        row = {
            key: before.get(key, ABSENT)
            for key in set(before) | set(after)
            if before.get(key, ABSENT) != after.get(key, ABSENT)
        }
        if row:
            changed[path] = row
    return changed


def _resolve_owner(model, path: str):
    name, _, owner_name = path.rpartition("|")
    module = model.get_submodule(name) if name else model
    return module if owner_name == "module" else vars(module).get(owner_name)


def _tensor(desc: dict) -> torch.Tensor:
    shape, dtype, _ = desc["t"]
    dtype = getattr(torch, dtype.split(".", 1)[1])
    return torch.tensor(desc["data"], dtype=dtype).reshape(shape)


def _value(desc: dict, current):
    value = desc["v"]
    if desc.get("seq") == "tuple":
        return tuple(value)
    return value


@dataclass
class Report:
    applied: list = field(default_factory=list)
    restored: list = field(default_factory=list)
    linked: list = field(default_factory=list)
    deleted: list = field(default_factory=list)
    disagreements: list = field(default_factory=list)
    unapplied_tokens: list = field(default_factory=list)
    nondeterministic: list = field(default_factory=list)
    missing_objects: list = field(default_factory=list)
    missing_tensors: list = field(default_factory=list)
    missing_owners: list = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"applied {len(self.applied)} load-time values, restored "
            f"{len(self.restored)} CPU tensors, linked {len(self.linked)} objects, "
            f"deleted {len(self.deleted)}; "
            f"disagreements {len(self.disagreements)}, unapplied tokens "
            f"{len(self.unapplied_tokens)}, nondeterministic {len(self.nondeterministic)}; "
            f"missing objects {len(self.missing_objects)}, tensors "
            f"{len(self.missing_tensors)}, owners {len(self.missing_owners)}"
        )

    @property
    def fatal(self) -> list:
        return self.missing_owners + self.missing_objects + self.missing_tensors


def apply(model: torch.nn.Module, final: dict, changed: dict) -> Report:
    """Make the reader's state match the writer's ``final`` snapshot.

    Simple values that loading set are written from ``final``. Everything else
    is compared: objects and tensors must have the writer's class / layout.
    """
    report = Report()
    for path, writer_row in final.items():
        owner = _resolve_owner(model, path)
        loaded = changed.get(path, {})
        if owner is None:
            report.missing_owners.append(path)
            continue
        for key, want in writer_row.items():
            have = describe(vars(owner)[key]) if key in vars(owner) else ABSENT
            where = f"{path}.{key}"
            if have == want:
                continue
            if "v" in want:
                if key in loaded:
                    if have not in (ABSENT, loaded[key]):
                        report.disagreements.append((where, have.get("v", have), want["v"]))
                    setattr(owner, key, _value(want, vars(owner).get(key)))
                    report.applied.append(where)
                else:
                    report.nondeterministic.append((where, have.get("v", have), want["v"]))
            elif "token" in want:
                (report.unapplied_tokens if key in loaded else report.nondeterministic).append(
                    (where, have, want["token"]))
            elif "o" in want:
                if have.get("o") == want["o"]:
                    continue  # same class; ``ref`` differences are wiring detail
                if key in loaded and "ref" in want:
                    ref_owner, _, ref_key = want["ref"].partition(".")
                    source = _resolve_owner(model, path.rpartition("|")[0] + "|" + ref_owner)
                    target = vars(source).get(ref_key) if source is not None else None
                    if target is not None and _cls(target) == want["o"]:
                        setattr(owner, key, target)
                        report.linked.append(f"{where} -> {want['ref']}")
                        continue
                if key in loaded and not want["o"].startswith("builtins."):
                    report.missing_objects.append((where, have.get("o", have), want["o"]))
                elif key in loaded:
                    # A container loading created (e.g. a lazily filled cache).
                    report.disagreements.append((where, have.get("o", have), want["o"]))
                else:
                    report.nondeterministic.append((where, have, want["o"]))
            elif "t" in want:
                value = vars(owner).get(key)
                present = torch.is_tensor(value) and not value.is_meta
                if not present and "data" in want:
                    setattr(owner, key, _tensor(want))
                    report.restored.append(where)
                    continue
                if torch.is_tensor(value) and value.is_meta:
                    continue  # reported by the remaining-meta scan
                if present and "t" in have and have["t"] == want["t"]:
                    # Same layout, different CPU values.
                    if key in loaded:
                        value.copy_(_tensor(want))
                        report.disagreements.append((where, "cpu tensor values", "writer's"))
                    else:
                        report.nondeterministic.append((where, "cpu tensor values", "writer's"))
                    continue
                if key in loaded:
                    report.missing_tensors.append((where, have.get("t", have), want["t"]))
                else:
                    report.nondeterministic.append((where, have, want["t"]))
        for key, before in loaded.items():
            if key not in writer_row and before != ABSENT and key in vars(owner):
                delattr(owner, key)
                report.deleted.append(f"{path}.{key}")
    return report


# =============================================================================
# Identity: generic fingerprint instead of pinned model/GPU/parallelism
# =============================================================================


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def identity(vllm_config, model_config) -> dict:
    from vllm.distributed import (
        get_ep_group,
        get_pp_group,
        get_tensor_model_parallel_rank,
    )

    parallel = vllm_config.parallel_config
    config = json.dumps(model_config.hf_config.to_dict(), sort_keys=True, default=str)
    try:
        ep_rank = get_ep_group().rank_in_group
    except Exception:
        ep_rank = None
    kernel = getattr(vllm_config, "kernel_config", None)
    return dict(
        schema=1,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model=str(model_config.model),
        revision=model_config.revision,
        architectures=list(model_config.hf_config.architectures or []),
        config_sha256=hashlib.sha256(config.encode()).hexdigest(),
        versions={k: _version(k) for k in ("torch", "vllm", "transformers", "flashinfer-python")},
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        dtype=str(model_config.dtype),
        quantization=model_config.quantization,
        kv_cache_dtype=vllm_config.cache_config.cache_dtype,
        max_model_len=model_config.max_model_len,
        max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        tp=parallel.tensor_parallel_size,
        pp=parallel.pipeline_parallel_size,
        dp=parallel.data_parallel_size,
        enable_expert_parallel=parallel.enable_expert_parallel,
        expert_placement=getattr(parallel, "expert_placement_strategy", None),
        enable_eplb=getattr(parallel, "enable_eplb", False),
        tp_rank=get_tensor_model_parallel_rank(),
        pp_rank=get_pp_group().rank_in_group,
        ep_rank=ep_rank,
        kernel_config=(
            json.loads(json.dumps(vars(kernel), default=str, sort_keys=True))
            if kernel is not None else None
        ),
    )


def _source_modules(model: torch.nn.Module, final: dict | None = None) -> set:
    """Python modules defining the classes that build and post-process this
    model: modules, quant methods, attention impls and objects loading made."""
    names = set()
    for _, module in model.named_modules():
        for _, owner in _owners(module):
            names.add(type(owner).__module__)
    for row in (final or {}).values():
        for desc in row.values():
            if "o" in desc:
                names.add(desc["o"].rsplit(".", 1)[0])
    return {n for n in names if n.split(".")[0] in ("vllm", "flashinfer", "gpu_memory_service")}


def _source_hashes(names) -> dict:
    hashes = {}
    for name in sorted(names):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        origin = getattr(spec, "origin", None)
        if origin and origin.endswith(".py"):
            hashes[name] = hashlib.sha256(Path(origin).read_bytes()).hexdigest()
    return hashes


def _check_identity(published: dict, expected: dict, what: str) -> None:
    if published != expected:
        keys = sorted(k for k in set(published) | set(expected)
                      if published.get(k) != expected.get(k))
        raise RuntimeError(f"Load-diff artifact/runtime {what} mismatch: {keys[:20]}")


# =============================================================================
# Writer
# =============================================================================


def selected(vllm_config) -> bool:
    load = getattr(vllm_config, "load_config", None)
    extra = getattr(load, "model_loader_extra_config", None) or {}
    return extra.get("gms_reconstruction_mode") == MODE


def _quant(module) -> str:
    return type(getattr(module, "quant_method", None)).__name__


def _routing_digest(module) -> dict | None:
    buffer = module._buffers.get("_expert_map")
    if buffer is None or buffer.is_meta:
        return None
    raw = buffer.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    return {"shape": list(buffer.shape), "dtype": str(buffer.dtype),
            "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest()}


def publish_descriptor(manager, model, vllm_config, model_config, construct: dict) -> None:
    final = snapshot(model)
    changed = diff(construct, final)
    descriptor = dict(
        identity=identity(vllm_config, model_config),
        sources=_source_hashes(_source_modules(model, final)),
        modules={},
        aliases={},
        storage_aliases={},
        final=final,
        changed=changed,
    )
    registered = set(manager.metadata_list())
    objects, storages = {}, {}
    for name, tensor, kind in _iter_module_tensors(model):
        if name not in registered:
            continue
        descriptor["aliases"][name] = objects.setdefault(int(tensor._cdata), name)
        if kind != "parameter":
            canonical, pointer = storages.setdefault(
                int(tensor.untyped_storage()._cdata), (name, tensor.data_ptr()))
            descriptor["storage_aliases"][name] = [canonical, tensor.data_ptr() - pointer]
    for name, module in model.named_modules():
        descriptor["modules"][name] = dict(
            cls=_cls(module),
            quant=_quant(module),
            impl=_cls(module.impl) if hasattr(module, "impl") else None,
            params=[k for k, v in module._parameters.items() if v is not None],
            routing=_routing_digest(module),
        )
    payload = json.dumps(descriptor, sort_keys=True).encode()
    mapping = next(iter(manager.mappings.values()))
    manager.metadata_put(KEY, mapping.allocation_id, 0, payload)
    n_changed = sum(len(r) for r in changed.values())
    logger.info(
        "[GMS] load diff published: %d modules, %d attributes changed by loading "
        "in %d owners, %d source modules, %.1f KiB",
        len(descriptor["modules"]), n_changed, len(changed),
        len(descriptor["sources"]), len(payload) / 1024,
    )
    counts = {}
    for path, row in changed.items():
        for key in row:
            name = path.rpartition("|")[2] + "." + key
            counts[name] = counts.get(name, 0) + 1
    logger.info("[GMS] load diff attributes changed by loading (owner.attr: count): %s",
                sorted(counts.items(), key=lambda kv: -kv[1])[:40])


# =============================================================================
# Reader
# =============================================================================


@contextmanager
def _construction_context(device: torch.device):
    """vLLM 0.31 formats the EP expert map with ``.item()`` while constructing;
    build that small map on the real device (newer vLLM does this itself)."""
    try:
        from vllm.model_executor.layers.fused_moe.expert_map_manager import (
            ExpertMapManager,
        )
    except ImportError:
        with nullcontext():
            yield
        return
    original = ExpertMapManager._calculate_expert_maps

    def calculate(manager):
        with device:
            return original(manager)

    with patch.object(ExpertMapManager, "_calculate_expert_maps", calculate):
        yield


def _attach(manager, model, descriptor, device_index) -> None:
    specs = GMSTensorSpec.load_all(manager)
    if set(specs) != set(descriptor["aliases"]):
        missing = sorted(set(descriptor["aliases"]) ^ set(specs))
        raise RuntimeError(f"Load-diff tensor schema mismatch: {missing[:10]}")
    attached = {}
    with torch.inference_mode():
        for name, spec in specs.items():
            owner, key = _resolve_module_attr(model, name)
            canonical = descriptor["aliases"][name]
            tensor = attached.get(canonical)
            if tensor is None:
                source = spec.materialize(manager, device_index)
                if spec.meta.tensor_type == "parameter":
                    tensor = torch.nn.Parameter(source, requires_grad=False)
                else:
                    current = owner[int(key)] if isinstance(owner, (list, tuple)) \
                        else getattr(owner, key, None)
                    if current is None or current.is_meta:
                        # Private (mutable) state: the reader owns a copy.
                        tensor = source.clone()
                    else:
                        if current.shape != source.shape or current.dtype != source.dtype:
                            raise RuntimeError(f"Load-diff private tensor layout mismatch: {name}")
                        current.copy_(source)
                        tensor = current
                attached[canonical] = tensor
            if isinstance(owner, list):
                owner[int(key)] = tensor
            elif spec.meta.tensor_type == "parameter":
                owner._parameters[key] = tensor
            elif key in owner._buffers:
                owner._buffers[key] = tensor
            else:
                setattr(owner, key, tensor)
    for name, (canonical, offset) in descriptor["storage_aliases"].items():
        a, ka = _resolve_module_attr(model, name)
        b, kb = _resolve_module_attr(model, canonical)
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            continue
        if getattr(a, ka).data_ptr() - getattr(b, kb).data_ptr() != offset:
            raise RuntimeError(f"Load-diff private storage alias mismatch: {name}")


def _rebuild(modules: dict) -> None:
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.utils import weights_already_processed

    unsupported = sorted({
        f"{type(m.quant_method).__name__} ({name})"
        for name, m in modules.items()
        if isinstance(getattr(m, "quant_method", None), QuantizeMethodBase)
        and not m.quant_method.supports_pre_processed_weights
    })
    if unsupported:
        raise RuntimeError(
            "Load-diff attach needs quant methods that support pre-processed "
            f"weights; unsupported: {unsupported[:10]}"
        )
    with torch.inference_mode(), weights_already_processed():
        for module in modules.values():
            method = getattr(module, "quant_method", None)
            if isinstance(method, QuantizeMethodBase):
                method.process_weights_after_loading(module)


def _remaining_meta(model, device) -> list:
    """Names of tensors still on meta. Zero-element tensors carry no data (GMS
    publishes nothing for them, e.g. a rotary cache with ``rotary_dim=0``), so
    they are materialized empty on ``device`` instead of being reported."""
    for module in model.modules():
        for table in (module._parameters, module._buffers):
            for key, value in list(table.items()):
                if value is not None and value.is_meta and value.numel() == 0:
                    empty = torch.empty_like(value, device=device)
                    table[key] = (torch.nn.Parameter(empty, requires_grad=False)
                                  if isinstance(value, torch.nn.Parameter) else empty)
    missing = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
               if t.is_meta]
    for name, module in model.named_modules():
        for owner_name, owner in _owners(module):
            for key, value in list(vars(owner).items()):
                if not (torch.is_tensor(value) and value.is_meta):
                    continue
                if value.numel() == 0 and not isinstance(value, torch.nn.Parameter):
                    setattr(owner, key, torch.empty_like(value, device=device))
                else:
                    missing.append(f"{name}|{owner_name}.{key}")
    return missing


def restore(manager, vllm_config, model_config, device_index: int):
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    got = manager.metadata_get(KEY)
    if got is None:
        raise RuntimeError("Artifact has no load-diff descriptor (writer not in load_diff mode)")
    descriptor = json.loads(got[2])
    _check_identity(descriptor["identity"], identity(vllm_config, model_config), "identity")

    device = torch.device("cuda", device_index)
    with _construction_context(device), set_default_torch_dtype(model_config.dtype), \
            torch.device("meta"):
        model = initialize_model(vllm_config=vllm_config, model_config=model_config)
    modules = dict(model.named_modules())
    if set(modules) != set(descriptor["modules"]):
        raise RuntimeError("Load-diff module schema mismatch")
    for name, module in modules.items():
        row = descriptor["modules"][name]
        impl = _cls(module.impl) if hasattr(module, "impl") else None
        if (row["cls"], row["quant"], row["impl"]) != (_cls(module), _quant(module), impl):
            raise RuntimeError(f"Load-diff module class mismatch: {name}")
        for key in list(module._parameters):
            if module._parameters[key] is not None and key not in row["params"]:
                del module._parameters[key]

    _attach(manager, model, descriptor, device_index)
    for name, module in modules.items():
        if descriptor["modules"][name]["routing"] != _routing_digest(module):
            raise RuntimeError(f"Load-diff expert routing mismatch: {name}")

    _rebuild(modules)
    report = apply(model, descriptor["final"], descriptor["changed"])
    _check_identity(descriptor["sources"],
                    _source_hashes(set(descriptor["sources"]) | _source_modules(model)),
                    "source")

    for name, module in modules.items():
        actual = {k for k, v in module._parameters.items() if v is not None}
        if actual != set(descriptor["modules"][name]["params"]):
            raise RuntimeError(f"Load-diff parameter schema mismatch: {name}")
        if hasattr(module, "update_param_tp_status"):
            module.update_param_tp_status()
    missing = _remaining_meta(model, device)

    logger.info("[GMS] load diff: %s", report.summary())
    for label in ("applied", "linked", "deleted"):
        items = getattr(report, label)
        if items:
            logger.info("[GMS] load diff %s (%d): %s", label, len(items), items[:30])
    for label in ("disagreements", "unapplied_tokens", "nondeterministic"):
        items = getattr(report, label)
        if items:
            logger.warning("[GMS] load diff %s (%d): %s", label, len(items), items[:30])
    if report.fatal or missing:
        raise RuntimeError(
            "Load-diff attach could not rebuild state that loading created: "
            f"owners {report.missing_owners[:5]}, objects {report.missing_objects[:10]}, "
            f"tensors {report.missing_tensors[:10]}, still meta {missing[:10]}"
        )
    return model.eval()
