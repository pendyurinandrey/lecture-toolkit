"""fork_join: проверка конфигурации и то, какие команды ffmpeg он формирует
(сам ffmpeg подменён — файлы и запуск процессов не нужны)."""

import pytest

from common.ffmpeg import FFmpegError
from fork_join import fork_join


class TestValidateConfig:
    @pytest.fixture
    def video(self, tmp_path):
        path = tmp_path / "in.mp4"
        path.write_bytes(b"")
        return str(path)

    def config(self, video, **overrides):
        config = {
            "segments": [{"path": video, "fragments": [{"start": "00:00:10", "end": "00:00:20"}]}],
            "output": {"videoPath": "out.mp4", "audioPath": "out.m4a"},
        }
        config.update(overrides)
        return config

    def test_valid_config_passes(self, video):
        fork_join.validate_config(self.config(video))

    def test_segments_must_be_a_list_of_1_to_20(self, video):
        valid = {"path": video, "fragments": [{"start": "00:00:00", "end": "00:00:01"}]}
        for segments in (None, [], [valid] * 21, "x"):
            with pytest.raises(fork_join.ConfigError, match="1..20 объектов"):
                fork_join.validate_config(self.config(video, segments=segments))
        fork_join.validate_config(self.config(video, segments=[valid] * 20))  # ровно 20 — допустимо

    def test_missing_source_file(self, tmp_path):
        config = self.config(str(tmp_path / "нет_такого.mp4"))
        with pytest.raises(fork_join.ConfigError, match="файл не найден"):
            fork_join.validate_config(config)

    def test_fragments_must_be_1_to_20(self, video):
        for fragments in (None, [], [{"start": "00:00:00", "end": "00:00:01"}] * 21):
            config = self.config(video, segments=[{"path": video, "fragments": fragments}])
            with pytest.raises(fork_join.ConfigError, match="1..20 объектов"):
                fork_join.validate_config(config)
        twenty = [{"start": "00:00:00", "end": "00:00:01"}] * 20
        fork_join.validate_config(self.config(video, segments=[{"path": video, "fragments": twenty}]))

    def test_fragment_needs_start_and_end(self, video):
        config = self.config(video, segments=[{"path": video, "fragments": [{"start": "00:00:10"}]}])
        with pytest.raises(fork_join.ConfigError, match="start"):
            fork_join.validate_config(config)

    def test_bad_time_format_is_a_config_error(self, video):
        config = self.config(video, segments=[{"path": video, "fragments": [{"start": "10", "end": "00:00:20"}]}])
        with pytest.raises(fork_join.ConfigError, match="HH:mm:ss"):
            fork_join.validate_config(config)

    def test_end_must_be_after_start(self, video):
        config = self.config(video, segments=[{"path": video, "fragments": [{"start": "00:00:20", "end": "00:00:20"}]}])
        with pytest.raises(fork_join.ConfigError, match="должен быть больше"):
            fork_join.validate_config(config)

    def test_output_paths_are_required(self, video):
        for output in ({}, {"videoPath": "a.mp4"}, {"audioPath": "a.m4a"}):
            with pytest.raises(fork_join.ConfigError, match="output"):
                fork_join.validate_config(self.config(video, output=output))


def test_escape_concat_path_escapes_single_quotes():
    assert fork_join.escape_concat_path("/a/b's/c.mp4") == "/a/b'\\''s/c.mp4"


class FakeFfmpeg:
    """Записывает вызовы run_ffmpeg(args, action)."""

    def __init__(self, monkeypatch):
        self.calls = []
        monkeypatch.setattr(fork_join, "run_ffmpeg", lambda args, action="": self.calls.append((args, action)))
        monkeypatch.setattr(fork_join, "check_ffmpeg", lambda: None)


class TestCommands:
    def test_cut_fragment_copies_streams_from_start_for_duration(self, monkeypatch):
        fake = FakeFfmpeg(monkeypatch)
        fork_join.cut_fragment("in.mp4", "00:01:00", "00:02:30", "out.mp4")
        args, action = fake.calls[0]
        assert args == ["-ss", "00:01:00", "-i", "in.mp4", "-t", "90.0", "-c", "copy",
                        "-avoid_negative_ts", "make_zero", "out.mp4"]
        assert "00:01:00" in action and "00:02:30" in action and "in.mp4" in action

    def test_extract_audio_copies_the_audio_stream_without_recoding(self, monkeypatch):
        fake = FakeFfmpeg(monkeypatch)
        fork_join.extract_audio("v.mp4", "a.m4a")
        assert fake.calls[0][0] == ["-i", "v.mp4", "-vn", "-acodec", "copy", "a.m4a"]

    def test_process_config_cuts_then_joins_then_extracts_audio(self, monkeypatch, tmp_path):
        fake = FakeFfmpeg(monkeypatch)
        video = tmp_path / "in.mp4"
        video.write_bytes(b"")
        config = {
            "segments": [{"path": str(video), "fragments": [
                {"start": "00:00:00", "end": "00:00:10"}, {"start": "00:01:00", "end": "00:01:30"}]}],
            "output": {"videoPath": str(tmp_path / "res" / "out.mp4"), "audioPath": str(tmp_path / "res" / "out.m4a")},
        }
        logs = []
        fork_join.process_config(config, log=logs.append)

        commands = [args for args, _ in fake.calls]
        assert len(commands) == 4                                   # 2 нарезки, склейка, звук
        assert commands[0][0] == "-ss" and commands[1][0] == "-ss"
        assert commands[2][:5] == ["-f", "concat", "-safe", "0", "-i"] and commands[2][-2:] == ["copy", str(tmp_path / "res" / "out.mp4")]
        assert commands[3] == ["-i", str(tmp_path / "res" / "out.mp4"), "-vn", "-acodec", "copy", str(tmp_path / "res" / "out.m4a")]
        assert (tmp_path / "res").is_dir()                          # выходная папка создаётся
        assert any("Аудио сохранено" in m for m in logs)

    def test_process_config_stops_when_ffmpeg_is_missing(self, monkeypatch, tmp_path):
        fake = FakeFfmpeg(monkeypatch)

        def missing():
            raise FFmpegError("Не найден ffmpeg в PATH")

        monkeypatch.setattr(fork_join, "check_ffmpeg", missing)
        video = tmp_path / "in.mp4"
        video.write_bytes(b"")
        config = {"segments": [{"path": str(video), "fragments": [{"start": "00:00:00", "end": "00:00:10"}]}],
                  "output": {"videoPath": "o.mp4", "audioPath": "o.m4a"}}
        with pytest.raises(FFmpegError):
            fork_join.process_config(config)
        assert fake.calls == []                                     # до нарезки дело не дошло
