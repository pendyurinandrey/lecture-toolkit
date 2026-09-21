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


class TestCheckFfmpeg:
    def test_found(self, monkeypatch):
        monkeypatch.setattr(ffmpeg.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        ffmpeg.check_ffmpeg()  # не бросает

    def test_missing_gives_install_hint(self, monkeypatch):
        monkeypatch.setattr(ffmpeg.shutil, "which", lambda name: None)
        with pytest.raises(ffmpeg.FFmpegError) as info:
            ffmpeg.check_ffmpeg()
        assert "Не найден ffmpeg в PATH" in str(info.value) and "brew install ffmpeg" in str(info.value)


class TestRunFfmpeg:
    def test_adds_overwrite_and_quiet_flags_before_arguments(self, monkeypatch):
        calls = fake_run(monkeypatch)
        ffmpeg.run_ffmpeg(["-i", "in.mp4", "out.mp4"])
        assert calls[0] == ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", "in.mp4", "out.mp4"]

    def test_error_names_the_action(self, monkeypatch):
        fake_run(monkeypatch, returncode=1, stderr="Invalid data found when processing input")
        with pytest.raises(ffmpeg.FFmpegError) as info:
            ffmpeg.run_ffmpeg(["-i", "x"], "Не удалось вырезать фрагмент")
        assert "Не удалось вырезать фрагмент" in str(info.value)
        assert "Invalid data found" in str(info.value)

    def test_default_action_text(self, monkeypatch):
        fake_run(monkeypatch, returncode=1, stderr="boom")
        with pytest.raises(ffmpeg.FFmpegError, match="Не удалось выполнить команду ffmpeg"):
            ffmpeg.run_ffmpeg(["-i", "x"])

    def test_missing_binary(self, monkeypatch):
        fake_run(monkeypatch, raises=FileNotFoundError())
        with pytest.raises(ffmpeg.FFmpegError, match="Не найден ffmpeg в PATH"):
            ffmpeg.run_ffmpeg(["-i", "x"])


class TestProbeMedia:
    M4A = '{"streams": [{"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 1, "disposition": {"attached_pic": 0}}]}'

    def test_returns_media_info(self, monkeypatch):
        calls = fake_run(monkeypatch, stdout=self.M4A)
        info = ffmpeg.probe_media(Path("lecture.m4a"))
        assert (info.kind, info.audio_codec, info.sample_rate, info.channels) == ("audio", "aac", 48000, 1)
        assert calls[0][0] == "ffprobe" and calls[0][-1] == "lecture.m4a" and "json" in calls[0]

    def test_requests_cover_art_flag_so_it_is_not_mistaken_for_video(self, monkeypatch):
        calls = fake_run(monkeypatch, stdout=self.M4A)
        ffmpeg.probe_media("a.m4a")
        assert "stream_disposition=attached_pic" in " ".join(calls[0])

    def test_file_without_streams_is_an_ffmpeg_error(self, monkeypatch):
        fake_run(monkeypatch, stdout='{"streams": []}')
        with pytest.raises(ffmpeg.FFmpegError, match="нет ни видео-, ни аудиопотока"):
            ffmpeg.probe_media("empty.bin")

    def test_garbage_output_is_an_ffmpeg_error(self, monkeypatch):
        fake_run(monkeypatch, stdout="not json")
        with pytest.raises(ffmpeg.FFmpegError, match="Не удалось определить тип файла"):
            ffmpeg.probe_media("a.m4a")

    def test_unreadable_file(self, monkeypatch):
        fake_run(monkeypatch, returncode=1, stderr="a.m4a: Invalid data found when processing input")
        with pytest.raises(ffmpeg.FFmpegError, match="Invalid data found"):
            ffmpeg.probe_media("a.m4a")


class TestGetFfmpegMajorVersion:
    @pytest.mark.parametrize("first_line, expected", [
        ("ffmpeg version 9.0.1 Copyright (c) 2000-2026 the FFmpeg developers", 9),
        ("ffmpeg version 8.0 Copyright (c) 2000-2025 the FFmpeg developers", 8),
        ("ffmpeg version 10.1.2 Copyright (c) 2000-2027 the FFmpeg developers", 10),
        ("ffmpeg version n12.0 Copyright (c) 2000-2029 the FFmpeg developers", 12),
        ("ffmpeg version n7.1 Copyright (c) 2000-2024 the FFmpeg developers", 7),
        ("ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers", 4),
        ("ffmpeg version 7.1.1-tessus  https://evermeet.cx/ffmpeg/  Copyright (c) 2000-2025", 7),
    ])
    def test_parses_the_major_version(self, monkeypatch, first_line, expected):
        fake_run(monkeypatch, stdout=first_line + "\nbuilt with Apple clang version 16.0.0\n")
        assert ffmpeg.get_ffmpeg_major_version() == expected

    @pytest.mark.parametrize("first_line", [
        "ffmpeg version N-118000-gabcdef1234 Copyright (c) 2000-2025 the FFmpeg developers",
        "ffmpeg version 2024-05-06-git-abc123-essentials_build-www.gyan.dev Copyright (c) 2000-2024",
        "something else entirely",
        "",
    ])
    def test_unparsable_version_is_an_ffmpeg_error(self, monkeypatch, first_line):
        fake_run(monkeypatch, stdout=first_line + "\n")
        with pytest.raises(ffmpeg.FFmpegError, match="Не удалось определить версию ffmpeg"):
            ffmpeg.get_ffmpeg_major_version()

    def test_missing_ffmpeg(self, monkeypatch):
        fake_run(monkeypatch, raises=FileNotFoundError())
        with pytest.raises(ffmpeg.FFmpegError, match="Не найден ffmpeg в PATH"):
            ffmpeg.get_ffmpeg_major_version()

    def test_command_and_timeout(self, monkeypatch):
        seen = {}

        def run(command, **kwargs):
            seen["command"], seen["timeout"] = command, kwargs.get("timeout")
            return subprocess.CompletedProcess(command, 0, "ffmpeg version 8.0\n", "")

        monkeypatch.setattr(ffmpeg.subprocess, "run", run)
        ffmpeg.get_ffmpeg_major_version()
        assert seen == {"command": ["ffmpeg", "-version"], "timeout": 10}

    def test_hung_ffmpeg_is_reported_not_awaited_forever(self, monkeypatch):
        fake_run(monkeypatch, raises=subprocess.TimeoutExpired(["ffmpeg", "-version"], 10))
        with pytest.raises(ffmpeg.FFmpegError, match="не ответил за 10 с"):
            ffmpeg.get_ffmpeg_major_version()


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
