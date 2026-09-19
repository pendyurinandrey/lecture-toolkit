"""Проверки окружения: совместимость ffmpeg и токен Hugging Face."""

import os
import sys
import types

import pytest

from common.ffmpeg import FFmpegError
from speech_to_text_gigaam import environment
from speech_to_text_gigaam.environment import EnvironmentCheckError, check_ffmpeg_compatibility, ensure_hf_token


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def fake_version(monkeypatch, value):
    def get():
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(environment, "get_ffmpeg_major_version", get)


class TestCheckFfmpegCompatibility:
    @pytest.mark.parametrize("major", [4, 6, 8])
    def test_supported_versions_pass(self, monkeypatch, major):
        fake_version(monkeypatch, major)
        check_ffmpeg_compatibility()

    @pytest.mark.parametrize("major", [3, 9, 12])
    def test_unsupported_versions_give_instructions(self, monkeypatch, major):
        fake_version(monkeypatch, major)
        with pytest.raises(EnvironmentCheckError) as info:
            check_ffmpeg_compatibility()
        message = str(info.value)
        assert f"версии {major}" in message and "4-8" in message
        assert "DYLD_FALLBACK_LIBRARY_PATH" in message and "LD_LIBRARY_PATH" in message

    def test_unknown_version_does_not_block(self, monkeypatch):
        fake_version(monkeypatch, FFmpegError("git-сборка"))
        check_ffmpeg_compatibility()

    @pytest.mark.parametrize("variable", ["DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH"])
    def test_user_configured_library_path_skips_the_check(self, monkeypatch, variable):
        monkeypatch.setenv(variable, "/opt/ffmpeg8/lib")
        fake_version(monkeypatch, 9)  # без переменной это была бы ошибка
        check_ffmpeg_compatibility()


class TestEnsureHfToken:
    @staticmethod
    def fake_hub(monkeypatch, token):
        hub = types.ModuleType("huggingface_hub")
        hub.get_token = lambda: token
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    def test_explicit_env_token_is_kept(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "explicit")
        self.fake_hub(monkeypatch, "cached")
        ensure_hf_token()
        assert os.environ["HF_TOKEN"] == "explicit"

    def test_cached_token_is_exported(self, monkeypatch):
        self.fake_hub(monkeypatch, "cached-token")
        ensure_hf_token()
        assert os.environ["HF_TOKEN"] == "cached-token"

    def test_no_token_anywhere_gives_instructions(self, monkeypatch):
        self.fake_hub(monkeypatch, None)
        with pytest.raises(EnvironmentCheckError) as info:
            ensure_hf_token()
        assert "hf auth login" in str(info.value) and "segmentation-3.0" in str(info.value)

    def test_missing_huggingface_hub_counts_as_no_token(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # import -> ImportError
        with pytest.raises(EnvironmentCheckError):
            ensure_hf_token()
