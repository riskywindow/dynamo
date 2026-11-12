# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DYN_GMS_STARTUP_GATE: a pre-started spare holds after weight attach until the gate file exists."""

import threading
import time

import pytest

worker = pytest.importorskip("gpu_memory_service.integrations.vllm.worker")

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.none]


def test_no_gate_returns_immediately(monkeypatch):
    monkeypatch.delenv("DYN_GMS_STARTUP_GATE", raising=False)
    t = time.monotonic()
    worker._wait_at_startup_gate()
    assert time.monotonic() - t < 0.1


def test_gate_holds_until_the_file_exists(monkeypatch, tmp_path):
    gate = tmp_path / "gate"
    monkeypatch.setenv("DYN_GMS_STARTUP_GATE", str(gate))
    threading.Timer(0.3, gate.touch).start()
    t = time.monotonic()
    worker._wait_at_startup_gate()
    assert 0.25 < time.monotonic() - t < 2.0
