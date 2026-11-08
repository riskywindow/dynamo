"""Integration admission and allocation-bound preflight regressions."""
from types import SimpleNamespace
import json
import pytest
import torch
from gpu_memory_service.integrations.vllm import component_reconstruction as c
from gpu_memory_service.client.torch.tensor import GMSTensorSpec,TensorMetadata

pytestmark=[pytest.mark.pre_merge,pytest.mark.unit,pytest.mark.gpu_0,pytest.mark.none]

def config(mode,profile=None):
    return SimpleNamespace(load_config=SimpleNamespace(model_loader_extra_config={
        "gms_reconstruction_mode":mode,"gms_reconstruction_profile":profile}))

def test_unknown_component_mode_rejected():
    with pytest.raises(ValueError,match="Unsupported"):
        c.selected(config("unknown"))

def test_component_mode_requires_explicit_admission():
    with pytest.raises(RuntimeError,match="explicit pinned"):
        c.selected(config("components"))

def test_reference_mode_has_no_new_runtime_dependency(monkeypatch):
    monkeypatch.setattr(c,"_api",lambda:pytest.fail("Reference must not require new API"))
    assert not c.selected(config("reference"))

def spec(name,offset):
    return GMSTensorSpec(name,name,"allocation",offset,TensorMetadata(shape=[2],stride=[1],dtype=torch.float32,tensor_type="parameter"))

def test_out_of_bounds_rejected_before_materialization(monkeypatch):
    monkeypatch.setattr(GMSTensorSpec,"load_all",lambda _:{"a":spec("a",60)})
    manager=SimpleNamespace(list_handles=lambda:[SimpleNamespace(allocation_id="allocation",aligned_size=64)])
    with pytest.raises(RuntimeError,match="exceeds"):
        c._preflight(manager,{"aliases":{"a":"a"}})

def test_inconsistent_alias_rejected(monkeypatch):
    monkeypatch.setattr(GMSTensorSpec,"load_all",lambda _:{"a":spec("a",0),"b":spec("b",8)})
    manager=SimpleNamespace(list_handles=lambda:[SimpleNamespace(allocation_id="allocation",aligned_size=64)])
    with pytest.raises(RuntimeError,match="alias metadata"):
        c._preflight(manager,{"aliases":{"a":"a","b":"a"}})

def test_component_identity_rejected_before_construction(monkeypatch):
    pytest.importorskip("vllm")
    monkeypatch.setattr(c,"_api",lambda:object())
    monkeypatch.setattr(c,"identity",lambda *args:{"generation":"new"})
    manager=SimpleNamespace(metadata_get=lambda _:(0,0,json.dumps({"identity":{"generation":"old"}}).encode()))
    with pytest.raises(RuntimeError,match="identity mismatch"):
        c.restore(manager,object(),object(),0)


def test_routing_inspects_owner_not_facade_properties():
    class Facade(torch.nn.Module):
        @property
        def expert_map_manager(self):
            raise AssertionError("Derived facade property must not be evaluated")
    assert c._routing(Facade()) is None
