# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in finalized attachment through the experimental vLLM component API.

Admission remains the existing pinned GLM policy. This module transfers the
artifact schema; component code owns construction and runtime binding. The
tested writer ownership/copy path is intentionally retained for this milestone.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from gpu_memory_service.client.torch.module import _iter_module_tensors, _resolve_module_attr
from gpu_memory_service.client.torch.tensor import GMSTensorSpec
from gpu_memory_service.integrations.vllm import glm_reconstruction as policy

KEY = "__gms__.component_reconstruction.v1"


def selected(vllm_config) -> bool:
    load = getattr(vllm_config, "load_config", None)
    extra = getattr(load, "model_loader_extra_config", None) or {}
    mode = extra.get("gms_reconstruction_mode", "reference")
    if mode not in ("reference", "components", "load_diff"):
        raise ValueError("Unsupported GMS reconstruction mode")
    if mode in ("reference", "load_diff"):
        return False
    if not policy.selected_profile(vllm_config):
        raise RuntimeError("Component reconstruction requires explicit pinned admission")
    _api()
    return True


def _api():
    try:
        from vllm.model_executor.layers import finalized_weights as api
    except ImportError as exc:
        raise RuntimeError("The required vLLM finalized-weight API is unavailable") from exc
    if api.API_VERSION != 1 or not isinstance(api.CONTRACT_ID, str):
        raise RuntimeError("Unsupported vLLM finalized-weight API")
    return api


def identity(vllm_config, model_config):
    return {
        "policy": policy.identity(vllm_config, model_config),
        "contract": _api().CONTRACT_ID,
        "reader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _cls(value):
    return type(value).__module__ + "." + type(value).__name__


def _routing(layer):
    # Facades can expose expert_map_manager through a property. The routed
    # component owns both the actual attribute and its registered map buffer.
    manager = vars(layer).get("expert_map_manager")
    if manager is None or "_expert_map" not in layer._buffers:
        return None
    if layer._expert_map is not manager.expert_map:
        raise RuntimeError("Expert map owner/reference mismatch")
    config = layer.moe_config.moe_parallel_config
    tensors = [layer._expert_map]
    tables = layer._expert_routing_tables()
    if tables is not None:
        tensors.extend(tables)
    rows = []
    for tensor in tensors:
        if tensor is None:
            rows.append(None)
        else:
            if tensor.is_meta:
                raise RuntimeError("Expert routing metadata is still meta")
            raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
            rows.append({
                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
            })
    return {"ep_rank": config.ep_rank, "ep_size": config.ep_size,
            "local_experts": layer.local_num_experts, "tensors": rows}


def publish_descriptor(manager, model, vllm_config, model_config):
    api = _api()
    descriptor = {"identity": identity(vllm_config, model_config),
                  "modules": {}, "aliases": {}, "storage_aliases": {}}
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
        quant = type(getattr(module, "quant_method", None)).__name__
        if quant not in policy.SUPPORTED_METHODS:
            raise RuntimeError(f"Unsupported component quantization at {name}: {quant}")
        descriptor["modules"][name] = {
            "cls": _cls(module), "quant": quant,
            "backend": _cls(module.impl) if hasattr(module, "impl") else None,
            "params": [n for n, p in module._parameters.items() if p is not None],
            "kv": api.export_kv_state(module),
            "routing": _routing(module),
            "padding": vars(module).get("weights_padding_cols"),
        }
    from gpu_memory_service.integrations.vllm.load_diff import descriptor_allocation_id

    manager.metadata_put(KEY, descriptor_allocation_id(manager, model), 0,
                         json.dumps(descriptor, sort_keys=True).encode())


def _preflight(manager, descriptor):
    specs = GMSTensorSpec.load_all(manager)
    if set(specs) != set(descriptor["aliases"]):
        raise RuntimeError("Component artifact tensor schema mismatch")
    handles = {str(h.allocation_id): int(h.aligned_size) for h in manager.list_handles()}
    for name, spec in specs.items():
        target = descriptor["aliases"][name]
        if target not in specs:
            raise RuntimeError("Missing canonical component alias")
        other = specs[target]
        if (spec.allocation_id, spec.offset_bytes, spec.meta) != (
                other.allocation_id, other.offset_bytes, other.meta):
            raise RuntimeError("Component alias metadata mismatch")
        shape, stride = spec.meta.shape, spec.meta.stride
        if (len(shape) != len(stride) or any(type(d) is not int or d <= 0 for d in shape)
                or any(type(s) is not int or s < 0 for s in stride)):
            raise RuntimeError("Invalid component tensor layout")
        element = torch.empty((), dtype=spec.meta.dtype, device="meta").element_size()
        extent = (1 + sum((d - 1) * s for d, s in zip(shape, stride))) * element
        if (spec.meta.tensor_type not in ("parameter", "buffer", "tensor_attr")
                or spec.offset_bytes < 0 or spec.offset_bytes % element
                or spec.allocation_id not in handles
                or spec.offset_bytes + extent > handles[spec.allocation_id]):
            raise RuntimeError("Component tensor exceeds its allocation")
    return specs


def restore(manager, vllm_config, model_config, device_index):
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    api = _api()
    got = manager.metadata_get(KEY)
    if got is None:
        raise RuntimeError("Artifact has no component reconstruction descriptor")
    descriptor = json.loads(got[2])
    if descriptor["identity"] != identity(vllm_config, model_config):
        raise RuntimeError("Component artifact/runtime identity mismatch")
    specs = _preflight(manager, descriptor)
    # Only component weight creation enters meta. Private buffers, expert-map
    # owners and backend objects follow normal CUDA construction.
    device = torch.device("cuda", device_index)
    with set_default_torch_dtype(model_config.dtype), device, api.weight_construction("meta"):
        model = initialize_model(vllm_config=vllm_config, model_config=model_config)
    modules = dict(model.named_modules())
    if set(modules) != set(descriptor["modules"]):
        raise RuntimeError("Component module schema mismatch")
    for name, module in modules.items():
        row = descriptor["modules"][name]
        if (row["cls"] != _cls(module)
                or row["quant"] != type(getattr(module, "quant_method", None)).__name__
                or row["backend"] != (_cls(module.impl) if hasattr(module, "impl") else None)):
            raise RuntimeError("Component module ABI mismatch: " + name)
        if _routing(module) != row["routing"]:
            raise RuntimeError("Constructor expert routing mismatch: " + name)
        for key in list(module._parameters):
            if module._parameters[key] is not None and key not in row["params"]:
                del module._parameters[key]
        padding = row["padding"]
        if padding is not None:
            if type(padding) is not int or padding < 0:
                raise RuntimeError("Invalid finalized linear padding")
            module.weights_padding_cols = padding

    # Enforce constructor aliases before mutation; never repair a cached owner
    # by silently replacing a private object.
    private = {}
    for name, spec in specs.items():
        if spec.meta.tensor_type == "parameter":
            continue
        owner, attr = _resolve_module_attr(model, name)
        current = getattr(owner, attr)
        if (current.is_meta or tuple(current.shape) != tuple(spec.meta.shape)
                or current.dtype != spec.meta.dtype):
            raise RuntimeError("Constructor private layout mismatch: " + name)
        private[name] = current
    for name, current in private.items():
        canonical = descriptor["aliases"][name]
        if canonical not in private or current._cdata != private[canonical]._cdata:
            raise RuntimeError("Constructor private alias mismatch: " + name)
    for name, (canonical, offset) in descriptor["storage_aliases"].items():
        if private[name].data_ptr() - private[canonical].data_ptr() != offset:
            raise RuntimeError("Constructor private view mismatch: " + name)

    attached = {}
    with torch.inference_mode():
        for name, spec in specs.items():
            owner, attr = _resolve_module_attr(model, name)
            canonical = descriptor["aliases"][name]
            if canonical not in attached:
                value = spec.materialize(manager, device_index)
                if spec.meta.tensor_type == "parameter":
                    attached[canonical] = torch.nn.Parameter(value, requires_grad=False)
                else:
                    private[name].copy_(value)
                    attached[canonical] = private[name]
            if spec.meta.tensor_type == "parameter":
                owner._parameters[attr] = attached[canonical]
        for name, module in modules.items():
            row = descriptor["modules"][name]
            if row["kv"] is not None:
                api.restore_kv_state(module, row["kv"])
            if {n for n, p in module._parameters.items() if p is not None} != set(row["params"]):
                raise RuntimeError("Final component parameter schema mismatch: " + name)
            method = getattr(module, "quant_method", None)
            if row["quant"] == "ModelOptNvFp4FusedMoE":
                hook = getattr(method, "attach_finalized_weights", None)
                if hook is None:
                    raise RuntimeError("Missing finalized MoE attach hook")
                hook(module)
            if hasattr(module, "update_param_tp_status"):
                module.update_param_tp_status()
    missing = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    if missing:
        raise RuntimeError("Component reconstruction left meta tensors: " + str(missing[:10]))
    return model.eval()
