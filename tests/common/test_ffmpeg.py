"""Обёртки common/ffmpeg.py с подменённым subprocess.run — ffmpeg не нужен."""

import subprocess
from pathlib import Path

import pytest

from common import ffmpeg


def fake_run(monkeypatch, *, returncode=0, stdout="", stderr="", raises=None):
    """Подменяет subprocess.run; возвращает список вызовов (команды)."""
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if raises:
            raise raises
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr(ffmpeg.subprocess, "run", run)
    return calls


class TestErrors:
    def test_missing_binary_gives_hint(self, monkeypatch):
        fake_run(monkeypatch, raises=FileNotFoundError())
        with pytest.raises(ffmpeg.FFmpegError, match="Не найден ffprobe в PATH"):
            ffmpeg.get_media_duration("a.mp4")

    def test_nonzero_exit_includes_action_and_error_tail(self, monkeypatch):
        stderr = "\n".join(f"line {i}" for i in range(40))
        fake_run(monkeypatch, returncode=1, stderr=stderr)
        with pytest.raises(ffmpeg.FFmpegError) as info:
            ffmpeg.get_media_duration("a.mp4")
        message = str(info.value)
        assert "Не удалось определить длительность a.mp4" in message
        assert "line 39" in message and "line 25" in message  # последние 15 строк
        assert "line 24" not in message and "line 0" not in message

    def test_is_a_runtime_error(self):
        assert issubclass(ffmpeg.FFmpegError, RuntimeError)


class TestGetMediaDuration:
    def test_parses_seconds(self, monkeypatch):
        calls = fake_run(monkeypatch, stdout="11389.662667\n")
        assert ffmpeg.get_media_duration(Path("lecture.m4a")) == pytest.approx(11389.662667)
        assert calls[0][0] == "ffprobe" and calls[0][-1] == "lecture.m4a"

    def test_unparsable_output_is_an_ffmpeg_error(self, monkeypatch):
        fake_run(monkeypatch, stdout="N/A\n")
        with pytest.raises(ffmpeg.FFmpegError, match="N/A"):
            ffmpeg.get_media_duration("a.mp4")


class TestDetectSilences:
    LOG = (
        "size=N/A time=00:00:10 bitrate=N/A\n"
        "[silencedetect @ 0x1] silence_start: 12.5\n"
        "[silencedetect @ 0x1] silence_end: 20.25 | silence_duration: 7.75\n"
        "[silencedetect @ 0x1] silence_start: 100\n"
        "[silencedetect @ 0x1] silence_end: 130.5 | silence_duration: 30.5\n"
    )

    def test_pairs_starts_and_ends(self, monkeypatch):
        fake_run(monkeypatch, stderr=self.LOG)
        assert ffmpeg.detect_silences("a.mp4") == [(12.5, 20.25), (100.0, 130.5)]

    def test_no_silence_gives_empty_list(self, monkeypatch):
        fake_run(monkeypatch, stderr="size=N/A\n")
        assert ffmpeg.detect_silences("a.mp4") == []

    def test_command_uses_parameters_and_skips_video(self, monkeypatch):
        calls = fake_run(monkeypatch, stderr="")
        ffmpeg.detect_silences("a.mp4", min_duration=60, noise_db="-40dB")
        command = calls[0]
        assert "-vn" in command
        assert "silencedetect=noise=-40dB:d=60" in command

    def test_failure_is_not_silently_empty(self, monkeypatch):
        fake_run(monkeypatch, returncode=1, stderr="Output file does not contain any stream")
        with pytest.raises(ffmpeg.FFmpegError):
            ffmpeg.detect_silences("silent_video.mp4")


class TestGetKeyframeTimestamps:
    def test_keeps_only_keyframes_sorted(self, monkeypatch):
        fake_run(monkeypatch, stdout="2.000000,K__\n0.000000,K_\n0.033000,___\n\n1.000000,K__\n1.033000,_\n")
        assert ffmpeg.get_keyframe_timestamps("a.mp4") == [0.0, 1.0, 2.0]

    def test_no_keyframes(self, monkeypatch):
        fake_run(monkeypatch, stdout="")
        assert ffmpeg.get_keyframe_timestamps("a.mp4") == []


class TestConvertToWav:
    def test_builds_16k_mono_pcm16_command(self, monkeypatch, tmp_path):
        calls = fake_run(monkeypatch)
        result = ffmpeg.convert_to_wav_16k_mono("in.m4a", tmp_path / "out.wav")
        command = calls[0]
        assert result == tmp_path / "out.wav"
        assert command[command.index("-ar") + 1] == "16000"
        assert command[command.index("-ac") + 1] == "1"
        assert command[command.index("-c:a") + 1] == "pcm_s16le"
        assert "-vn" in command and command[-1] == str(tmp_path / "out.wav")

    def test_failure_raises(self, monkeypatch, tmp_path):
        fake_run(monkeypatch, returncode=1, stderr="Invalid data found")
        with pytest.raises(ffmpeg.FFmpegError, match="Invalid data found"):
            ffmpeg.convert_to_wav_16k_mono("in.m4a", tmp_path / "out.wav")
