# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load diff: writer snapshots, the diff, and how the reader applies it."""

import json

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS or not HAS_TORCH:
    pytest.skip("GMS and Torch are required", allow_module_level=True)

import torch
from gpu_memory_service.integrations.vllm import load_diff

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.none]


class Kernel:
    pass


class Method:
    def __init__(self):
        self.backend_ready = False


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
        self.quant_method = Method()
        self.calculate_scales = True
        self.placeholder = 1.0
        self.dims = (2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Layer()


def _load(model):
    """What a writer's load_weights + process_weights_after_loading does."""
    layer = model.layer
    layer._scale_float = 0.5
    layer.calculate_scales = False
    del layer.placeholder
    layer.dims = (4, 2)
    layer.quant_method.kernel = Kernel()
    layer.quant_method.backend_ready = True


def _rebuild(model):
    """What the reader's pre-processed quant rebuild recreates."""
    model.layer.quant_method.kernel = Kernel()
    model.layer.quant_method.backend_ready = True


def _published():
    writer = Model()
    construct = load_diff.snapshot(writer)
    _load(writer)
    final = load_diff.snapshot(writer)
    # The descriptor travels as JSON through GMS metadata.
    return json.loads(json.dumps({"final": final, "changed": load_diff.diff(construct, final)}))


def test_diff_lists_only_what_loading_changed():
    changed = _published()["changed"]
    assert set(changed["layer|module"]) == {
        "_scale_float", "calculate_scales", "placeholder", "dims"}
    assert set(changed["layer|quant_method"]) == {"kernel", "backend_ready"}
    assert "|module" not in changed  # the root module was not touched


def test_reader_applies_load_time_values_and_deletions():
    published = _published()
    reader = Model()
    _rebuild(reader)
    report = load_diff.apply(reader, published["final"], published["changed"])
    assert not report.fatal and not report.disagreements and not report.nondeterministic
    assert reader.layer._scale_float == 0.5
    assert reader.layer.calculate_scales is False
    assert reader.layer.dims == (4, 2) and isinstance(reader.layer.dims, tuple)
    assert not hasattr(reader.layer, "placeholder")
    assert report.deleted == ["layer|module.placeholder"]


def test_object_loading_created_but_reader_did_not_rebuild_is_fatal():
    published = _published()
    reader = Model()
    report = load_diff.apply(reader, published["final"], published["changed"])
    assert report.missing_objects == [
        ("layer|quant_method.kernel", {"absent": True}, f"{__name__}.Kernel")]


def test_rebuild_value_that_differs_from_the_writer_is_recorded_and_overwritten():
    published = _published()
    reader = Model()
    _rebuild(reader)
    reader.layer._scale_float = 0.25
    report = load_diff.apply(reader, published["final"], published["changed"])
    assert report.disagreements == [("layer|module._scale_float", 0.25, 0.5)]
    assert reader.layer._scale_float == 0.5


def test_constructor_difference_is_recorded_not_applied():
    published = _published()
    reader = Model()
    _rebuild(reader)
    reader.flag = 3
    published["final"]["|module"]["flag"] = {"v": 7}
    report = load_diff.apply(reader, published["final"], published["changed"])
    assert report.nondeterministic == [("|module.flag", 3, 7)]
    assert reader.flag == 3


def test_tensor_attribute_loading_created_must_exist_on_the_reader():
    writer = Model()
    construct = load_diff.snapshot(writer)
    big = load_diff._MAX_CPU_VALUES + 1  # too large to travel by value
    writer.layer.cpu_copy = torch.zeros(big)
    final = load_diff.snapshot(writer)
    published = json.loads(json.dumps({"final": final, "changed": load_diff.diff(construct, final)}))
    report = load_diff.apply(Model(), published["final"], published["changed"])
    assert [r[0] for r in report.missing_tensors] == ["layer|module.cpu_copy"]
    reader = Model()
    reader.layer.cpu_copy = torch.zeros(big)
    assert not load_diff.apply(reader, published["final"], published["changed"]).fatal


def test_mode_selection():
    from types import SimpleNamespace

    def config(extra):
        return SimpleNamespace(load_config=SimpleNamespace(model_loader_extra_config=extra))

    assert load_diff.selected(config({"gms_reconstruction_mode": "load_diff"}))
    assert not load_diff.selected(config({}))
    assert not load_diff.selected(config(None))


def test_constructor_cpu_tensor_left_meta_is_restored_from_the_writer():
    """E.g. attention ``_k_scale_cpu``: built on CPU by the writer's constructor,
    left on meta by the reader's meta construction when no quant method rebuilds it."""
    writer = Model()
    writer.layer._k_scale_cpu = torch.tensor([1.0])
    final = load_diff.snapshot(writer)
    published = json.loads(json.dumps({"final": final, "changed": {}}))
    reader = Model()
    reader.layer._k_scale_cpu = torch.empty(1, device="meta")
    report = load_diff.apply(reader, published["final"], published["changed"])
    assert report.restored == ["layer|module._k_scale_cpu"] and not report.fatal
    assert reader.layer._k_scale_cpu.device.type == "cpu"
    assert reader.layer._k_scale_cpu.tolist() == [1.0]


def test_writer_copy_out_skips_tensors_without_storage():
    """The writer's process-wide tensor scan can see torch.compile FakeTensors
    (Qwen3-30B-A3B-FP8 on vLLM 0.31); they have no data pointer."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from gpu_memory_service.client.torch import storage

    with FakeTensorMode():
        fake = torch.empty(4)
    assert storage._containing_mapping(fake, (), ()) is None
    assert storage._containing_mapping(torch.empty(4, device="meta"), (), ()) is None


def test_object_loading_wired_into_a_new_attribute_is_relinked():
    """E.g. vLLM's DeepGEMM FP8 kernel stamps itself onto its layer as
    ``deep_gemm_warmup_provider`` during loading; the pre-processed rebuild
    does not, but the kernel object already exists on the quant method."""
    def build():
        model = Model()
        model.layer.quant_method.fp8_linear = Kernel()
        return model

    writer = build()
    construct = load_diff.snapshot(writer)
    writer.layer.provider = writer.layer.quant_method.fp8_linear
    final = load_diff.snapshot(writer)
    assert final["layer|module"]["provider"]["ref"] == "quant_method.fp8_linear"
    published = json.loads(json.dumps({"final": final, "changed": load_diff.diff(construct, final)}))
    reader = build()
    report = load_diff.apply(reader, published["final"], published["changed"])
    assert not report.fatal
    assert report.linked == ["layer|module.provider -> quant_method.fp8_linear"]
    assert reader.layer.provider is reader.layer.quant_method.fp8_linear


def test_zero_element_meta_buffers_are_materialized_not_reported():
    """GLM-5.3-Flash's indexer rope has rotary_dim=0: its cos_sin_cache has no
    elements, so the writer publishes nothing for it."""
    model = Model()
    model.layer.register_buffer("cos_sin_cache", torch.empty(8, 0, device="meta"), persistent=False)
    model.layer.register_buffer("real", torch.empty(8, 2, device="meta"), persistent=False)
    missing = load_diff._remaining_meta(model, torch.device("cpu"))
    assert missing == ["layer.real"]
    assert model.layer.cos_sin_cache.device.type == "cpu"
    assert tuple(model.layer.cos_sin_cache.shape) == (8, 0)


def test_writer_publishes_the_identity_taken_before_construction(monkeypatch):
    """vLLM's DeepSeek-V2 mutates the HF config while constructing/loading, so the
    writer must publish the fingerprint it took before construction (readers check
    theirs before constructing); recomputing at publish time broke the match."""
    from types import SimpleNamespace

    published = {}
    manager = SimpleNamespace(
        metadata_list=lambda: [],
        mappings={0x1000: SimpleNamespace(allocation_id="a0")},
        metadata_put=lambda key, alloc, off, payload: published.update({key: payload}),
    )
    monkeypatch.setattr(load_diff, "identity", lambda *a: {"config_sha256": "after-load"})
    writer = Model()
    construct = load_diff.snapshot(writer)
    _load(writer)
    load_diff.publish_descriptor(manager, writer, None, None, construct,
                                 identity_before_construction={"config_sha256": "before"})
    descriptor = json.loads(published[load_diff.KEY])
    assert descriptor["identity"] == {"config_sha256": "before"}
