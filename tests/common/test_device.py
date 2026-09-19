"""Выбор устройства torch и подготовка окружения (torch подменён пустышкой)."""

import os
import sys
import types

import pytest

from common import device


def fake_torch(monkeypatch, cuda=False, mps=False, has_mps_backend=True):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    torch.backends = types.SimpleNamespace(**({"mps": types.SimpleNamespace(is_available=lambda: mps)}
                                              if has_mps_backend else {}))
    monkeypatch.setitem(sys.modules, "torch", torch)


class TestPickDevice:
    def test_cuda_wins(self, monkeypatch):
        fake_torch(monkeypatch, cuda=True, mps=True)
        assert device.pick_device() == "cuda"

    def test_mps_when_no_cuda(self, monkeypatch):
        fake_torch(monkeypatch, cuda=False, mps=True)
        assert device.pick_device() == "mps"

    def test_cpu_when_nothing_available(self, monkeypatch):
        fake_torch(monkeypatch)
        assert device.pick_device() == "cpu"

    def test_old_torch_without_mps_backend(self, monkeypatch):
        fake_torch(monkeypatch, has_mps_backend=False)
        assert device.pick_device() == "cpu"


class TestPrepareTorchEnvironment:
    def test_enables_mps_fallback_on_macos(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
        device.prepare_torch_environment()
        assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"

    def test_user_value_is_not_overridden(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
        device.prepare_torch_environment()
        assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"

    @pytest.mark.parametrize("platform", ["win32", "linux"])
    def test_nothing_is_set_elsewhere(self, monkeypatch, platform):
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
        device.prepare_torch_environment()
        assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
