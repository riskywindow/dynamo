# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regressions for GMS pool ownership and reconstruction admission."""

from types import SimpleNamespace

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS or not HAS_TORCH:
    pytest.skip("GMS and Torch are required", allow_module_level=True)

import torch
from gpu_memory_service.client.torch import allocator, module
from gpu_memory_service.integrations.vllm import model_loader
from gpu_memory_service.integrations.vllm.glm_reconstruction import selected_profile

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.none,
]


def test_module_traversal_does_not_evaluate_tensor_properties(monkeypatch):
    class WithProperty(torch.nn.Module):
        @property
        def expert_map(self):
            raise AssertionError("The owning expert map manager must be used")

    model = WithProperty()
    model.register_buffer("scratch", torch.ones(4))
    monkeypatch.setattr(module, "_is_accelerator_tensor", lambda t: True)
    assert [(name, kind) for name, _, kind in module._iter_module_tensors(model)] == [
        ("scratch", "buffer")
    ]


def test_materialization_rejects_missing_weights(monkeypatch):
    model = torch.nn.Linear(2, 3, device="meta")
    monkeypatch.setattr(module.GMSTensorSpec, "load_all", lambda manager: {})
    with pytest.raises(RuntimeError, match="left meta tensors"):
        module.materialize_module_from_gms(object(), model, device_index=0)


def test_publication_keeps_initial_value_storage_alive(monkeypatch):
    owner = torch.ones(8).untyped_storage()
    manager = SimpleNamespace(tag="weights")
    monkeypatch.setattr(
        allocator,
        "_tag_states",
        {"weights": SimpleNamespace(manager=manager, callback_failure=None)},
    )
    monkeypatch.setattr(model_loader, "_pending_gms_client", manager)
    monkeypatch.setattr(model_loader, "_pending_retained_gms_tensors", [owner])
    monkeypatch.setattr(model_loader, "_published_gms_tensor_owners", [])

    def publish(actual):
        assert actual is manager
        assert owner.nbytes() == 32

    monkeypatch.setattr(model_loader, "publish_gms_write", publish)
    assert model_loader.publish_pending_gms_write()
    assert model_loader._published_gms_tensor_owners == [owner]


def test_pool_finalization_preserves_live_mapping_and_surfaces_free_failure(
    monkeypatch,
):
    events = []
    manager = SimpleNamespace(tag="weights", mappings={1: "live"})

    class Pool:
        def __del__(self):
            events.append("torch_released_cached_blocks")

    state = SimpleNamespace(
        manager=manager, mem_pool=Pool(), callback_failure=RuntimeError("free failed")
    )
    monkeypatch.setattr(allocator, "_tag_states", {"weights": state})
    with pytest.raises(RuntimeError, match="free callback failed") as error:
        allocator.release_gms_mem_pool(manager)
    assert str(error.value.__cause__) == "free failed"
    assert events == ["torch_released_cached_blocks"]
    assert manager.mappings == {1: "live"}
    assert state.mem_pool is None


def test_reconstruction_rejects_unknown_profile():
    config = SimpleNamespace(
        load_config=SimpleNamespace(
            model_loader_extra_config={"gms_reconstruction_profile": "unknown"}
        )
    )
    with pytest.raises(ValueError, match="Unsupported GMS reconstruction profile"):
        selected_profile(config)


@pytest.mark.vllm
@pytest.mark.core
def test_required_meta_postload_error_propagates(monkeypatch):
    pytest.importorskip("vllm")
    from vllm.model_executor.model_loader import utils

    monkeypatch.setattr(model_loader, "setup_meta_tensor_workaround", lambda: None)
    monkeypatch.setattr(utils, "initialize_model", lambda **kw: torch.nn.Linear(2, 2))

    def fail(*args):
        raise RuntimeError("required quantization initialization failed")

    monkeypatch.setattr(utils, "process_weights_after_loading", fail)
    with pytest.raises(
        RuntimeError, match="required quantization initialization failed"
    ):
        model_loader._create_meta_model(object(), SimpleNamespace(dtype=torch.float32))


def test_unqualified_runtime_version_rejected(monkeypatch):
    from gpu_memory_service.integrations.vllm import glm_reconstruction

    monkeypatch.setattr(
        glm_reconstruction.importlib.metadata, "version", lambda name: "0.0.0"
    )
    with pytest.raises(RuntimeError, match="requires pinned vLLM"):
        glm_reconstruction.identity(SimpleNamespace(parallel_config=object()), object())


@pytest.mark.vllm
@pytest.mark.core
def test_artifact_identity_mismatch_rejected_before_model_construction(monkeypatch):
    pytest.importorskip("vllm")
    import json

    from gpu_memory_service.integrations.vllm import glm_reconstruction

    manager = SimpleNamespace(
        metadata_get=lambda key: (
            "weights",
            0,
            json.dumps({"identity": {"generation": "old"}}).encode(),
        )
    )
    monkeypatch.setattr(
        glm_reconstruction, "identity", lambda *args: {"generation": "new"}
    )
    with pytest.raises(RuntimeError, match="artifact/runtime identity mismatch"):
        glm_reconstruction.restore(manager, object(), object(), 0)


def test_native_free_failure_is_preserved_for_the_publication_barrier(monkeypatch):
    def fail(va):
        raise RuntimeError("sidecar release failed")

    manager = SimpleNamespace(
        tag="weights",
        mappings={4096: object()},
        destroy_scratch_mapping=lambda va: False,
        destroy_mapping=fail,
    )
    state = SimpleNamespace(manager=manager, callback_failure=None)
    monkeypatch.setattr(allocator, "_tag_states", {"weights": state})
    allocator._gms_free(4096, 64, 0, 0)
    with pytest.raises(RuntimeError, match="free callback failed") as error:
        allocator.raise_if_gms_allocator_failed(manager)
    assert str(error.value.__cause__) == "sidecar release failed"


def test_profile_rejects_the_unqualified_sparse_context_range():
    from gpu_memory_service.integrations.vllm import glm_reconstruction

    def config(length):
        return SimpleNamespace(
            load_config=SimpleNamespace(
                model_loader_extra_config={
                    "gms_reconstruction_profile": glm_reconstruction.PROFILE,
                }
            ),
            model_config=SimpleNamespace(max_model_len=length),
        )

    assert glm_reconstruction.selected_profile(config(1024))
    with pytest.raises(RuntimeError, match="longer sparse-attention contexts"):
        glm_reconstruction.selected_profile(config(4096))


def test_v031_profile_admits_the_qualified_context_range():
    from gpu_memory_service.integrations.vllm import glm_reconstruction

    def config(length):
        return SimpleNamespace(
            load_config=SimpleNamespace(
                model_loader_extra_config={
                    "gms_reconstruction_profile": glm_reconstruction.PROFILE_V031,
                }
            ),
            model_config=SimpleNamespace(max_model_len=length),
        )

    assert glm_reconstruction.selected_profile(config(40960)) == (
        glm_reconstruction.PROFILE_V031
    )
    with pytest.raises(RuntimeError, match="longer sparse-attention contexts"):
        glm_reconstruction.selected_profile(config(65536))


def test_unconfigured_profile_is_not_selected():
    from gpu_memory_service.integrations.vllm import glm_reconstruction

    config = SimpleNamespace(load_config=SimpleNamespace(model_loader_extra_config={}))
    assert glm_reconstruction.selected_profile(config) is None


@pytest.mark.vllm
@pytest.mark.core
def test_meta_kv_scale_processing_is_deferred_until_attach():
    pytest.importorskip("vllm")
    try:
        from vllm.model_executor.utils import weights_already_processed  # noqa: F401
    except ImportError:
        pytest.skip("vLLM has no pre-processed weights API")
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

    layer = torch.nn.Module()
    for name in ("k_scale", "v_scale", "q_scale", "prob_scale"):
        setattr(layer, name, torch.nn.Parameter(torch.empty((), device="meta")))
    method = BaseKVCacheMethod.__new__(BaseKVCacheMethod)
    with model_loader._deferred_meta_kv_scales() as deferred:
        BaseKVCacheMethod.process_weights_after_loading(method, layer)
    assert deferred == [(method, layer)]
    assert not hasattr(layer, "k_scale")

    # After attach, the folded scales come from the published tensors.
    layer._k_scale = torch.tensor(0.5)
    layer._v_scale = torch.tensor(0.25)
    layer._q_scale = torch.tensor(1.0)
    model_loader._process_deferred_kv_scales(deferred)
    assert (layer._k_scale_float, layer._v_scale_float) == (0.5, 0.25)
