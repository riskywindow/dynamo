# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental GLM attach adapter for pinned vLLM/GB200 execution profiles.

Weights in the artifact are already packed. Replaying generic quantization
post-load hooks would pack them twice and mutate read-only weights. The adapter
instead attaches the final parameter schema and reconstructs kernel references.

Two profiles are supported:

- ``glm52-nvfp4-vllm0271-gb200-tp4-ep4`` (vLLM 0.27.1): the adapter rebuilds
  the NVFP4 MoE kernel itself and copies the published runtime scalars.
- ``glm52-nvfp4-vllm031-gb200-tp4-ep4`` (vLLM 0.31.0): after attaching the
  published tensors, the adapter re-runs each quantization method's
  ``process_weights_after_loading`` under vLLM's ``weights_already_processed``
  context. That is vLLM's own contract for loading post-processed weights:
  methods skip tensor transforms and only rebuild host-side state (MoE kernel,
  KV-scale floats and CPU copies) from the attached tensors. The writer's
  published scalars are then compared against the rebuilt ones.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch
from gpu_memory_service.client.torch.module import (
    _iter_module_tensors,
    _resolve_module_attr,
)
from gpu_memory_service.client.torch.tensor import GMSTensorSpec

logger = logging.getLogger(__name__)

PROFILE = "glm52-nvfp4-vllm0271-gb200-tp4-ep4"
PROFILE_V031 = "glm52-nvfp4-vllm031-gb200-tp4-ep4"
METADATA_KEY = "__gms__.glm_reconstruction"
AUDIT_KEY = "__gms__.glm_reconstruction.audit"
REVISION = "53e0691e21895a3863a606dfd12910c69eba94ab"
# The hub snapshot that revision 53e0691e is a projection of (same files).
REVISIONS = (REVISION, "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa")
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
# Rebuilt by vLLM's pre-processed KV-cache path from the attached tensors.
REBUILT_SCALAR_FIELDS = {"_k_scale_float", "_v_scale_float", "_q_scale_float"}
_SIMPLE = (bool, int, float, str, type(None))


@dataclass(frozen=True)
class Profile:
    name: str
    versions: dict
    requires_eager: bool
    max_model_len: int
    source_modules: tuple
    pre_processed_finalize: bool
    supported_methods: frozenset


PROFILES = {
    PROFILE: Profile(
        name=PROFILE,
        versions={
            "vllm": "0.27.1",
            "torch": "2.13.0+cu130",
            "transformers": "5.15.0",
        },
        requires_eager=True,
        max_model_len=QUALIFIED_MAX_MODEL_LEN,
        source_modules=(
            "vllm.model_executor.layers.quantization.modelopt",
            "vllm.model_executor.layers.fused_moe.expert_map_manager",
            "vllm.model_executor.layers.fused_moe.oracle.nvfp4",
            "vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe",
            "vllm.model_executor.layers.attention.mla_attention",
            "vllm.model_executor.models.deepseek_v2",
        ),
        pre_processed_finalize=False,
        supported_methods=frozenset(SUPPORTED_METHODS),
    ),
    PROFILE_V031: Profile(
        name=PROFILE_V031,
        versions={
            "vllm": "0.31.0",
            "torch": "2.13.0+cu130",
            "transformers": "5.14.1",
        },
        # CUDA graphs are allowed: graph capture only reads the attached
        # weights, and the mode is part of the identity both sides must share.
        requires_eager=False,
        # Qualified on GB200 TP4/EP4 with prompts up to ~32K tokens.
        max_model_len=40960,
        source_modules=(
            "vllm.model_executor.layers.quantization.modelopt",
            "vllm.model_executor.layers.quantization.kv_cache",
            "vllm.model_executor.layers.fused_moe.expert_map_manager",
            "vllm.model_executor.layers.fused_moe.oracle.nvfp4",
            "vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe",
            "vllm.model_executor.layers.attention.mla_attention",
            "vllm.model_executor.models.deepseek_v2",
            "vllm.models.deepseek_v32.attention",
            "vllm.models.deepseek_v32.nvidia.model",
        ),
        pre_processed_finalize=True,
        supported_methods=frozenset(SUPPORTED_METHODS),
    ),
}


def _extra(vllm_config) -> dict:
    return vllm_config.load_config.model_loader_extra_config or {}


def selected_profile(vllm_config) -> str | None:
    """Return the selected profile name, or None when not configured."""
    selected = _extra(vllm_config).get("gms_reconstruction_profile")
    if selected is None:
        return None
    if selected not in PROFILES:
        raise ValueError(f"Unsupported GMS reconstruction profile: {selected!r}")
    _require_qualified_context(vllm_config.model_config, PROFILES[selected])
    return selected


def _profile(vllm_config) -> Profile:
    name = selected_profile(vllm_config)
    if name is None:
        raise RuntimeError("No GMS reconstruction profile selected")
    return PROFILES[name]


def _runtime_profile() -> Profile | None:
    """The profile whose pinned versions match the installed packages."""
    for profile in PROFILES.values():
        if all(
            importlib.metadata.version(key) == value
            for key, value in profile.versions.items()
        ):
            return profile
    return None


def _require_qualified_context(model_config, profile: Profile | None = None) -> None:
    """Reject sparse-attention context ranges pending native qualification."""
    limit = (profile or PROFILES[PROFILE]).max_model_len
    if model_config.max_model_len > limit:
        raise RuntimeError(
            f"GLM GMS reconstruction is qualified only for max_model_len <= {limit}; "
            "longer sparse-attention contexts require separate qualification"
        )


def _model_revision(model_config) -> str:
    for revision in REVISIONS:
        if revision in model_config.model or model_config.revision == revision:
            return revision
    raise RuntimeError(
        "GLM reconstruction requires the qualified immutable model revision"
    )


def identity(vllm_config, model_config, profile: Profile | None = None) -> dict:
    parallel = vllm_config.parallel_config
    profile = profile or _runtime_profile()
    if (
        profile is None
        or any(
            importlib.metadata.version(key) != value
            for key, value in profile.versions.items()
        )
        or (profile.requires_eager and not model_config.enforce_eager)
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
            "GLM reconstruction requires pinned vLLM 0.27.1 (eager) or 0.31.0, "
            "GLM DSA, GB200, TP4/EP4, PP1/DP1 with linear expert placement, "
            "without EPLB"
        )
    _require_qualified_context(model_config, profile)
    revision = _model_revision(model_config)
    from vllm.distributed import get_ep_group

    config = json.dumps(model_config.hf_config.to_dict(), sort_keys=True, default=str)
    result = dict(
        schema=2,
        profile=profile.name,
        model_revision=revision,
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
            for name in profile.source_modules
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
    if not profile.requires_eager:
        result["enforce_eager"] = bool(model_config.enforce_eager)
        result["cudagraph_mode"] = str(vllm_config.compilation_config.cudagraph_mode)
    return result


def _quant_name(module) -> str:
    return type(getattr(module, "quant_method", None)).__name__


def _cls(value) -> str:
    return type(value).__module__ + "." + type(value).__name__


def _simple_state(obj) -> dict:
    """Simple-typed instance attributes, used to audit reconstructed state."""
    return {
        key: value
        for key, value in vars(obj).items()
        if type(value) in _SIMPLE and not key.startswith("__")
    }


def _audit(model) -> dict:
    audit = {}
    for name, module in model.named_modules():
        row = {"module": _simple_state(module)}
        method = getattr(module, "quant_method", None)
        if method is not None:
            row["quant_method"] = _simple_state(method)
        audit[name] = row
    return audit


def _routing_digest(module) -> dict | None:
    """Hash of a MoE layer's expert map, as published by the writer."""
    buffer = module._buffers.get("_expert_map")
    if buffer is None or buffer.is_meta:
        return None
    raw = buffer.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    return {
        "shape": list(buffer.shape),
        "dtype": str(buffer.dtype),
        "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
    }


def publish_descriptor(manager, model, vllm_config, model_config) -> None:
    profile = _profile(vllm_config)
    descriptor = dict(
        identity=identity(vllm_config, model_config, profile),
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
    unsupported = sorted(
        {
            f"{name}: {_quant_name(module)}"
            for name, module in model.named_modules()
            if _quant_name(module) not in profile.supported_methods
        }
    )
    if unsupported:
        raise RuntimeError(
            f"Unsupported GLM quantization methods ({len(unsupported)}): "
            + "; ".join(unsupported[:20])
        )
    for name, module in model.named_modules():
        descriptor["modules"][name] = dict(
            cls=_cls(module),
            quant=_quant_name(module),
            attention_backend=_cls(module.impl) if hasattr(module, "impl") else None,
            params=[
                key for key, value in module._parameters.items() if value is not None
            ],
            scalars={
                key: vars(module)[key] for key in SCALAR_FIELDS if key in vars(module)
            },
            routing=_routing_digest(module),
        )
    from gpu_memory_service.integrations.vllm.load_diff import descriptor_allocation_id

    allocation_id = descriptor_allocation_id(manager, model)
    manager.metadata_put(
        METADATA_KEY,
        allocation_id,
        0,
        json.dumps(descriptor, sort_keys=True).encode(),
    )
    if profile.pre_processed_finalize:
        manager.metadata_put(
            AUDIT_KEY,
            allocation_id,
            0,
            json.dumps(_audit(model), sort_keys=True, default=str).encode(),
        )


@contextmanager
def construction_context(device: torch.device):
    # Expert placement is data-dependent even during skeleton construction
    # (vLLM formats the map with .item() for its EP log line). The map is a
    # pure function of (ep_size, ep_rank, placement), so construct this small
    # engine-owned object on the actual device; its properties and cached
    # users all receive the same valid tensors. The writer's published
    # _expert_map is copied over it and its digest checked during restore.
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager

    original = ExpertMapManager._calculate_expert_maps

    def calculate(manager):
        with device:
            return original(manager)

    with patch.object(ExpertMapManager, "_calculate_expert_maps", calculate):
        yield


def _finalize_legacy(module, method) -> None:
    """vLLM 0.27.1: rebuild the NVFP4 MoE kernel from attached weights."""
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE,
        make_nvfp4_moe_kernel,
    )

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


def _finalize_pre_processed(modules: dict) -> None:
    """vLLM >= 0.31: rebuild runtime state through vLLM's pre-processed path.

    Only quantization methods run; attention ``process_weights_after_loading``
    (W_UK_T/W_UV derivation) is skipped because those tensors are attached
    from the artifact and the read-only mapping must not be rewritten.
    """
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.utils import weights_already_processed

    with weights_already_processed():
        for name, module in modules.items():
            method = getattr(module, "quant_method", None)
            if not isinstance(method, QuantizeMethodBase):
                continue
            if not method.supports_pre_processed_weights:
                raise RuntimeError(
                    f"{type(method).__name__} at {name} does not support "
                    "pre-processed weights"
                )
            method.process_weights_after_loading(module)


def _check_scalars(modules: dict, descriptor: dict) -> None:
    """Apply the writer's scalars; values vLLM rebuilt must already agree.

    The KV-scale floats are rebuilt from the attached ``_k_scale``/``_v_scale``
    /``_q_scale`` tensors by the pre-processed KV-cache path, so a difference
    means the attached state diverged. Other fields are carried from the
    writer, as in the 0.27.1 profile.
    """
    mismatched = []
    for name, module in modules.items():
        for key, value in descriptor["modules"][name]["scalars"].items():
            if key not in SCALAR_FIELDS or type(value) not in (int, float, bool):
                raise RuntimeError(f"Unsupported GLM scalar state: {name}.{key}")
            if (
                key in REBUILT_SCALAR_FIELDS
                and key in vars(module)
                and vars(module)[key] != value
            ):
                mismatched.append((name, key, vars(module)[key], value))
            setattr(module, key, value)
    if mismatched:
        raise RuntimeError(f"GLM rebuilt scalar mismatch: {mismatched[:10]}")


def _compare_audit(manager, model) -> list:
    """Log simple attributes that differ from the writer (diagnostic only)."""
    got = manager.metadata_get(AUDIT_KEY)
    if got is None:
        return []
    published = json.loads(got[2])
    local = json.loads(json.dumps(_audit(model), sort_keys=True, default=str))
    diffs = []
    for name, row in published.items():
        mine = local.get(name, {})
        for part, values in row.items():
            ours = mine.get(part, {})
            for key in sorted(set(values) | set(ours)):
                if values.get(key, "<absent>") != ours.get(key, "<absent>"):
                    diffs.append(
                        (name, part, key, values.get(key, "<absent>"),
                         ours.get(key, "<absent>"))
                    )
    if diffs:
        logger.warning(
            "[GMS] GLM reconstruction audit: %d simple attributes differ from "
            "the writer, e.g. %s",
            len(diffs),
            diffs[:20],
        )
    else:
        logger.info("[GMS] GLM reconstruction audit: all simple attributes match")
    return diffs


def _remaining_meta(model, attributes: bool, device: torch.device) -> list:
    """Names of tensors still on meta after restore.

    With ``attributes``, plain tensor attributes of modules and their quant
    methods are checked too. Empty placeholders (e.g. the ``kv_cache``
    tensors vLLM binds after KV allocation) carry no data and are moved to
    ``device`` instead of being reported.
    """
    missing = [
        name
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor.is_meta
    ]
    if not attributes:
        return missing
    for name, module in model.named_modules():
        for owner in (module, getattr(module, "quant_method", None)):
            if owner is None:
                continue
            for key, value in list(vars(owner).items()):
                if not (torch.is_tensor(value) and value.is_meta):
                    continue
                if value.numel() == 0 and not isinstance(value, torch.nn.Parameter):
                    setattr(owner, key, torch.empty_like(value, device=device))
                else:
                    missing.append(f"{name}.{key}")
    return missing


def restore(manager, vllm_config, model_config, device_index: int):
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    got = manager.metadata_get(METADATA_KEY)
    if got is None:
        raise RuntimeError("GLM artifact has no reconstruction descriptor")
    descriptor = json.loads(got[2])
    expected = identity(vllm_config, model_config)
    if descriptor["identity"] != expected:
        keys = sorted(
            key
            for key in set(descriptor["identity"]) | set(expected)
            if descriptor["identity"].get(key) != expected.get(key)
        )
        raise RuntimeError(f"GLM artifact/runtime identity mismatch: {keys}")
    profile = PROFILES[expected["profile"]]
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
        if row["cls"] != _cls(module) or row["quant"] != _quant_name(module):
            raise RuntimeError(f"GLM module ABI mismatch: {name}")
        backend = _cls(module.impl) if hasattr(module, "impl") else None
        if row["attention_backend"] != backend:
            raise RuntimeError(f"GLM attention backend mismatch: {name}")
        for key in list(module._parameters):
            if module._parameters[key] is not None and key not in row["params"]:
                del module._parameters[key]
        if not profile.pre_processed_finalize:
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
        published = descriptor["modules"][name].get("routing")
        if published is not None and _routing_digest(module) != published:
            raise RuntimeError(f"GLM expert routing mismatch: {name}")

    if profile.pre_processed_finalize:
        with torch.inference_mode():
            _finalize_pre_processed(modules)
        _check_scalars(modules, descriptor)
    for name, module in modules.items():
        row = descriptor["modules"][name]
        actual = [key for key, value in module._parameters.items() if value is not None]
        if set(actual) != set(row["params"]):
            raise RuntimeError(f"GLM parameter schema mismatch: {name}")
        if not profile.pre_processed_finalize:
            _finalize_legacy(module, getattr(module, "quant_method", None))
        if hasattr(module, "update_param_tp_status"):
            module.update_param_tp_status()
    missing = _remaining_meta(model, profile.pre_processed_finalize, device)
    if missing:
        kinds = sorted({name.rsplit(".", 1)[-1] for name in missing})
        raise RuntimeError(
            f"GLM restore left meta state ({len(missing)}; kinds {kinds}): "
            f"{missing[:10]}"
        )
    if profile.pre_processed_finalize:
        _compare_audit(manager, model)
    return model.eval()
