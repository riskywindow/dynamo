# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental GLM attach adapter for one pinned vLLM/GB200 execution profile.

Weights in the artifact are already packed. Replaying generic quantization
post-load hooks would pack them twice and mutate read-only weights. The adapter
instead attaches the final parameter schema and reconstructs kernel references.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from gpu_memory_service.client.torch.module import (
    _iter_module_tensors,
    _resolve_module_attr,
)
from gpu_memory_service.client.torch.tensor import GMSTensorSpec

PROFILE = "glm52-nvfp4-vllm0271-gb200-tp4-ep4"
METADATA_KEY = "__gms__.glm_reconstruction"
REVISION = "53e0691e21895a3863a606dfd12910c69eba94ab"
QUALIFIED_MAX_MODEL_LEN = 2048
SUPPORTED_METHODS = {
    "NoneType",
    "UnquantizedLinearMethod",
    "UnquantizedEmbeddingMethod",
    "ModelOptNvFp4FusedMoE",
    "ModelOptKVCacheMethod",
}
# Values computed during normal loading and read directly by the runtime.
SCALAR_FIELDS = {
    "_k_scale_float",
    "_v_scale_float",
    "_q_scale_float",
    "_prob_scale_float",
    "calculate_kv_scales",
    "weights_padding_cols",
}


def selected_profile(vllm_config) -> bool:
    extra = vllm_config.load_config.model_loader_extra_config or {}
    selected = extra.get("gms_reconstruction_profile")
    if selected is None:
        return False
    if selected != PROFILE:
        raise ValueError(f"Unsupported GMS reconstruction profile: {selected!r}")
    _require_qualified_context(vllm_config.model_config)
    return True


def _require_qualified_context(model_config) -> None:
    """Reject the sparse-attention context range pending native qualification."""
    if model_config.max_model_len > QUALIFIED_MAX_MODEL_LEN:
        raise RuntimeError(
            "GLM GMS reconstruction is qualified only for max_model_len <= 2048; "
            "longer sparse-attention contexts require separate qualification"
        )


def identity(vllm_config, model_config) -> dict:
    parallel = vllm_config.parallel_config
    if (
        importlib.metadata.version("vllm") != "0.27.1"
        or importlib.metadata.version("torch") != "2.13.0+cu130"
        or importlib.metadata.version("transformers") != "5.15.0"
        or not model_config.enforce_eager
        or model_config.hf_config.architectures != ["GlmMoeDsaForCausalLM"]
        or parallel.tensor_parallel_size != 4
        or not parallel.enable_expert_parallel
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.enable_eplb
        or parallel.expert_placement_strategy != "linear"
        or torch.cuda.get_device_name() != "NVIDIA GB200"
    ):
        raise RuntimeError(
            "GLM reconstruction requires pinned vLLM 0.27.1, GLM DSA, "
            "GB200, TP4/EP4, PP1/DP1 with linear expert placement, "
            "without EPLB in eager mode"
        )
    _require_qualified_context(model_config)
    if REVISION not in model_config.model and model_config.revision != REVISION:
        raise RuntimeError(
            "GLM reconstruction requires the qualified immutable model revision"
        )
    from vllm.distributed import get_ep_group

    config = json.dumps(model_config.hf_config.to_dict(), sort_keys=True, default=str)
    return dict(
        schema=2,
        profile=PROFILE,
        model_revision=REVISION,
        config_sha256=hashlib.sha256(config.encode()).hexdigest(),
        versions={
            key: importlib.metadata.version(key)
            for key in ("torch", "vllm", "transformers", "flashinfer-python")
        },
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        source_hashes={
            name: hashlib.sha256(
                Path(importlib.util.find_spec(name).origin).read_bytes()
            ).hexdigest()
            for name in (
                "vllm.model_executor.layers.quantization.modelopt",
                "vllm.model_executor.layers.fused_moe.expert_map_manager",
                "vllm.model_executor.layers.fused_moe.oracle.nvfp4",
                "vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe",
                "vllm.model_executor.layers.attention.mla_attention",
                "vllm.model_executor.models.deepseek_v2",
            )
        },
        ep_rank=get_ep_group().rank_in_group,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        expert_placement=parallel.expert_placement_strategy,
        tp=4,
        ep=4,
        max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        max_model_len=model_config.max_model_len,
        kv_cache_dtype=vllm_config.cache_config.cache_dtype,
        dtype=str(model_config.dtype),
        kernel_config=json.loads(
            json.dumps(vars(vllm_config.kernel_config), default=str, sort_keys=True)
        ),
    )


def publish_descriptor(manager, model, vllm_config, model_config) -> None:
    descriptor = dict(
        identity=identity(vllm_config, model_config),
        modules={},
        aliases={},
        storage_aliases={},
    )
    objects = {}
    storages = {}
    registered = set(manager.metadata_list())
    for name, tensor, kind in _iter_module_tensors(model):
        if name not in registered:
            continue
        descriptor["aliases"][name] = objects.setdefault(int(tensor._cdata), name)
        if kind != "parameter":
            canonical, pointer = storages.setdefault(
                int(tensor.untyped_storage()._cdata), (name, tensor.data_ptr())
            )
            descriptor["storage_aliases"][name] = [
                canonical,
                tensor.data_ptr() - pointer,
            ]
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        kind = type(method).__name__
        if kind not in SUPPORTED_METHODS:
            raise RuntimeError(f"Unsupported GLM quantization method at {name}: {kind}")
        descriptor["modules"][name] = dict(
            cls=type(module).__module__ + "." + type(module).__name__,
            quant=kind,
            attention_backend=(
                type(module.impl).__module__ + "." + type(module.impl).__name__
                if hasattr(module, "impl")
                else None
            ),
            params=[
                key for key, value in module._parameters.items() if value is not None
            ],
            scalars={
                key: vars(module)[key] for key in SCALAR_FIELDS if key in vars(module)
            },
        )
    mapping = next(iter(manager.mappings.values()))
    manager.metadata_put(
        METADATA_KEY,
        mapping.allocation_id,
        0,
        json.dumps(descriptor, sort_keys=True).encode(),
    )


@contextmanager
def construction_context(device: torch.device):
    # Expert placement is data-dependent even during skeleton construction.
    # Construct this small engine-owned object on the actual device, so its
    # properties and cached users all receive the same valid tensors.
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager

    original = ExpertMapManager._calculate_expert_maps

    def calculate(manager):
        with device:
            return original(manager)

    with patch.object(ExpertMapManager, "_calculate_expert_maps", calculate):
        yield


def restore(manager, vllm_config, model_config, device_index: int):
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE,
        make_nvfp4_moe_kernel,
    )
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    got = manager.metadata_get(METADATA_KEY)
    if got is None:
        raise RuntimeError("GLM artifact has no reconstruction descriptor")
    descriptor = json.loads(got[2])
    expected = identity(vllm_config, model_config)
    if descriptor["identity"] != expected:
        raise RuntimeError("GLM artifact/runtime identity mismatch")
    device = torch.device("cuda", device_index)
    with construction_context(device), set_default_torch_dtype(
        model_config.dtype
    ), torch.device("meta"):
        model = initialize_model(vllm_config=vllm_config, model_config=model_config)
    modules = dict(model.named_modules())
    if set(modules) != set(descriptor["modules"]):
        raise RuntimeError("GLM module schema mismatch")
    for name, module in modules.items():
        row = descriptor["modules"][name]
        if (
            row["cls"] != type(module).__module__ + "." + type(module).__name__
            or row["quant"] != type(getattr(module, "quant_method", None)).__name__
        ):
            raise RuntimeError(f"GLM module ABI mismatch: {name}")
        backend = (
            type(module.impl).__module__ + "." + type(module.impl).__name__
            if hasattr(module, "impl")
            else None
        )
        if row["attention_backend"] != backend:
            raise RuntimeError(f"GLM attention backend mismatch: {name}")
        for key in list(module._parameters):
            if module._parameters[key] is not None and key not in row["params"]:
                del module._parameters[key]
        for key, value in row["scalars"].items():
            if key not in SCALAR_FIELDS or type(value) not in (int, float, bool):
                raise RuntimeError(f"Unsupported GLM scalar state: {name}.{key}")
            setattr(module, key, value)

    specs = GMSTensorSpec.load_all(manager)
    if set(specs) != set(descriptor["aliases"]):
        raise RuntimeError("GLM artifact tensor schema mismatch")
    for name, canonical in descriptor["aliases"].items():
        if canonical not in specs:
            raise RuntimeError(f"GLM alias target missing: {name}")
        a, b = specs[name], specs[canonical]
        if (a.allocation_id, a.offset_bytes, a.meta) != (
            b.allocation_id,
            b.offset_bytes,
            b.meta,
        ):
            raise RuntimeError(f"GLM alias metadata mismatch: {name}")
    attached = {}
    with torch.inference_mode():
        for name, spec in specs.items():
            module, key = _resolve_module_attr(model, name)
            canonical = descriptor["aliases"][name]
            tensor = attached.get(canonical)
            if tensor is None:
                source = spec.materialize(manager, device_index)
                if spec.meta.tensor_type == "parameter":
                    tensor = torch.nn.Parameter(source, requires_grad=False)
                else:
                    current = getattr(module, key)
                    if current.is_meta:
                        tensor = source.clone()
                    else:
                        if (
                            current.shape != source.shape
                            or current.dtype != source.dtype
                        ):
                            raise RuntimeError(
                                f"GLM private tensor layout mismatch: {name}"
                            )
                        current.copy_(source)
                        tensor = current
                attached[canonical] = tensor
            if spec.meta.tensor_type == "parameter":
                module._parameters[key] = tensor
            elif key in module._buffers:
                module._buffers[key] = tensor
            else:
                setattr(module, key, tensor)
    for name, (canonical, offset) in descriptor["storage_aliases"].items():
        owner, attr = _resolve_module_attr(model, name)
        canonical_owner, canonical_attr = _resolve_module_attr(model, canonical)
        if (
            getattr(owner, attr).data_ptr()
            - getattr(canonical_owner, canonical_attr).data_ptr()
            != offset
        ):
            raise RuntimeError(f"GLM private storage alias mismatch: {name}")
    for name, module in modules.items():
        row = descriptor["modules"][name]
        actual = [key for key, value in module._parameters.items() if value is not None]
        if set(actual) != set(row["params"]):
            raise RuntimeError(f"GLM parameter schema mismatch: {name}")
        method = getattr(module, "quant_method", None)
        if isinstance(method, ModelOptNvFp4FusedMoE):
            if method.nvfp4_backend.name != "FLASHINFER_TRTLLM":
                raise RuntimeError("Unsupported GLM MoE backend")
            method.moe_quant_config = method.get_fused_moe_quant_config(module)
            method.moe_kernel = make_nvfp4_moe_kernel(
                moe_quant_config=method.moe_quant_config,
                moe_config=method.moe,
                experts_cls=method.experts_cls,
                backend=method.nvfp4_backend,
                routing_tables=module._expert_routing_tables(),
                layer=module,
            )
            experts = method.moe_kernel.fused_experts
            for key in ("g1_scale_c", "gemm1_clamp_limit", "gemm1_beta", "gemm1_alpha"):
                if key in module._parameters:
                    setattr(experts, key, module._parameters[key])
        if hasattr(module, "update_param_tp_status"):
            module.update_param_tp_status()
    missing = [
        name
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor.is_meta
    ]
    if missing:
        raise RuntimeError(f"GLM restore left meta state: {missing[:10]}")
    return model.eval()
