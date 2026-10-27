"""fork_join: проверка конфигурации и то, какие команды ffmpeg он формирует
(сам ffmpeg подменён — файлы и запуск процессов не нужны)."""

from pathlib import Path

import pytest

from common.ffmpeg import FFmpegError
from common.media_info import MediaInfo
from fork_join import fork_join

VIDEO_INFO = MediaInfo(kind="video", audio_codec="aac", sample_rate=48000, channels=2,
                       video_codec="h264", width=1920, height=1080, pix_fmt="yuv420p", frame_rate="30/1")
AAC_INFO = MediaInfo(kind="audio", audio_codec="aac", sample_rate=48000, channels=1)
MP3_INFO = MediaInfo(kind="audio", audio_codec="mp3", sample_rate=48000, channels=1)


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
    """Записывает вызовы run_ffmpeg(args, action); probe_media отдаёт infos[путь] (по умолчанию — видео)."""

    def __init__(self, monkeypatch):
        self.calls = []
        self.infos = {}
        self.default_info = VIDEO_INFO
        monkeypatch.setattr(fork_join, "run_ffmpeg", lambda args, action="": self.calls.append((args, action)))
        monkeypatch.setattr(fork_join, "check_ffmpeg", lambda: None)
        monkeypatch.setattr(fork_join, "probe_media", lambda path: self.infos.get(str(path), self.default_info))
        self.durations = {}
        monkeypatch.setattr(fork_join, "get_media_duration", lambda path: self.durations.get(str(path), 100000.0))
        self.progress_calls = []       # (args, action) вызовов run_ffmpeg_with_progress
        self.progress_script = []      # секунды, которые «ffmpeg» сообщит в on_progress
        self.encoders = {"libx264"}
        monkeypatch.setattr(fork_join, "run_ffmpeg_with_progress", self._run_with_progress)
        monkeypatch.setattr(fork_join, "has_encoder", lambda name: name in self.encoders)

    def _run_with_progress(self, args, action, on_progress):
        self.progress_calls.append((args, action))
        for seconds in self.progress_script:
            on_progress(seconds)


class TestCommands:
    def test_cut_fragment_copies_streams_from_start_for_duration(self, monkeypatch):
        fake = FakeFfmpeg(monkeypatch)
        fork_join.cut_fragment("in.mp4", "00:01:00", "00:02:30", "out.mp4")
        args, action = fake.calls[0]
        assert args == ["-ss", "00:01:00", "-i", "in.mp4", "-t", "90.0", "-c", "copy",
                        "-avoid_negative_ts", "make_zero", "out.mp4"]
        assert "00:01:00" in action and "00:02:30" in action and "in.mp4" in action

    def test_cut_fragment_of_audio_drops_cover_art_streams(self, monkeypatch):
        fake = FakeFfmpeg(monkeypatch)
        fork_join.cut_fragment("in.m4a", "00:01:00", "00:02:30", "out.m4a", media_type="audio")
        assert fake.calls[0][0] == ["-ss", "00:01:00", "-i", "in.m4a", "-t", "90.0", "-vn", "-c", "copy",
                                    "-avoid_negative_ts", "make_zero", "out.m4a"]

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


class TestAudioMode:
    """Записи без видео: нарезка и склейка звука (mediaType = "audio")."""

    @pytest.fixture
    def files(self, tmp_path):
        paths = {}
        for name in ("a.m4a", "b.m4a", "c.mp3", "v.mp4"):
            (tmp_path / name).write_bytes(b"")
            paths[name] = str(tmp_path / name)
        return paths

    def config(self, files, sources=("a.m4a",), out="out.m4a", tmp_path=None, **extra):
        return {
            "mediaType": "audio",
            "segments": [{"path": files[name], "fragments": [{"start": "00:00:10", "end": "00:00:40"}]}
                         for name in sources],
            "output": {"audioPath": str(Path(files["a.m4a"]).parent / out)},
            **extra,
        }

    def audio_fake(self, monkeypatch, files):
        fake = FakeFfmpeg(monkeypatch)
        fake.infos = {files["a.m4a"]: AAC_INFO, files["b.m4a"]: AAC_INFO, files["c.mp3"]: MP3_INFO,
                      files["v.mp4"]: VIDEO_INFO}
        return fake

    # --- конфигурация
    def test_video_path_is_not_required_for_audio(self, files):
        fork_join.validate_config(self.config(files))

    def test_audio_path_is_required(self, files):
        config = self.config(files)
        config["output"] = {}
        with pytest.raises(fork_join.ConfigError, match="audioPath"):
            fork_join.validate_config(config)

    @pytest.mark.parametrize("out", ["out.wav", "out.mp4", "out"])
    def test_audio_output_must_be_m4a_or_mp3(self, files, out):
        with pytest.raises(fork_join.ConfigError, match=r"\.m4a"):
            fork_join.validate_config(self.config(files, out=out))

    def test_unknown_media_type_is_rejected(self, files):
        with pytest.raises(fork_join.ConfigError, match="mediaType"):
            fork_join.validate_config(self.config(files, mediaType="hologram"))

    def test_media_type_defaults_to_video(self):
        assert fork_join.media_type_of({}) == "video"
        assert fork_join.media_type_of({"mediaType": "audio"}) == "audio"

    # --- склейка
    def test_audio_is_cut_and_joined_without_extracting_a_track(self, monkeypatch, files, tmp_path):
        fake = self.audio_fake(monkeypatch, files)
        logs = []
        fork_join.process_config(self.config(files, sources=("a.m4a", "b.m4a")), log=logs.append)

        commands = [args for args, _ in fake.calls]
        assert len(commands) == 3                                       # 2 нарезки и склейка, без «извлечь звук»
        assert commands[0][-1].endswith("fragment_00_00.m4a") and commands[1][-1].endswith("fragment_01_00.m4a")
        assert "-vn" in commands[0] and "-c" in commands[0]
        assert commands[2][:5] == ["-f", "concat", "-safe", "0", "-i"]
        assert commands[2][-2:] == ["copy", str(tmp_path / "out.m4a")]
        assert any("Аудио сохранено" in m for m in logs) and not any("Видео сохранено" in m for m in logs)

    def test_mp3_fragments_keep_the_mp3_extension(self, monkeypatch, files, tmp_path):
        fake = self.audio_fake(monkeypatch, files)
        config = self.config(files, sources=("c.mp3",), out="out.mp3")
        fork_join.process_config(config, log=lambda m: None)
        assert fake.calls[0][0][-1].endswith("fragment_00_00.mp3") and fake.calls[-1][0][-1] == str(tmp_path / "out.mp3")

    # --- проверка совместимости
    def test_files_with_different_parameters_are_refused_before_cutting(self, monkeypatch, files):
        fake = self.audio_fake(monkeypatch, files)
        fake.infos[files["b.m4a"]] = MediaInfo(kind="audio", audio_codec="aac", sample_rate=44100, channels=1)
        with pytest.raises(fork_join.ConfigError) as info:
            fork_join.process_config(self.config(files, sources=("a.m4a", "b.m4a")), log=lambda m: None)
        assert "b.m4a" in str(info.value) and "44100 Гц вместо 48000 Гц" in str(info.value)
        assert fake.calls == []                                         # ffmpeg не запускался

    def test_m4a_and_mp3_cannot_be_joined(self, monkeypatch, files):
        fake = self.audio_fake(monkeypatch, files)
        with pytest.raises(fork_join.ConfigError, match="кодек звука MP3 вместо AAC"):
            fork_join.process_config(self.config(files, sources=("a.m4a", "c.mp3")), log=lambda m: None)
        assert fake.calls == []

    def test_video_in_an_audio_config_is_refused(self, monkeypatch, files):
        fake = self.audio_fake(monkeypatch, files)
        with pytest.raises(fork_join.ConfigError, match="v.mp4: это видео, а конфигурация для аудио"):
            fork_join.process_config(self.config(files, sources=("v.mp4",)), log=lambda m: None)

    def test_audio_in_a_video_config_is_refused(self, monkeypatch, files, tmp_path):
        fake = self.audio_fake(monkeypatch, files)
        config = {"segments": [{"path": files["a.m4a"], "fragments": [{"start": "00:00:00", "end": "00:00:10"}]}],
                  "output": {"videoPath": str(tmp_path / "o.mp4"), "audioPath": str(tmp_path / "o.m4a")}}
        with pytest.raises(fork_join.ConfigError, match="a.m4a: это аудио, а конфигурация для видео"):
            fork_join.process_config(config, log=lambda m: None)

    def test_output_extension_must_match_the_codec(self, monkeypatch, files):
        self.audio_fake(monkeypatch, files)
        with pytest.raises(fork_join.ConfigError, match=r"MP3.*\*\.mp3, а не \*\.m4a"):
            fork_join.process_config(self.config(files, sources=("c.mp3",), out="out.m4a"), log=lambda m: None)
        with pytest.raises(fork_join.ConfigError, match=r"AAC.*\*\.m4a, а не \*\.mp3"):
            fork_join.process_config(self.config(files, sources=("a.m4a",), out="out.mp3"), log=lambda m: None)

    def test_unsupported_audio_codec_is_explained(self, monkeypatch, files):
        fake = self.audio_fake(monkeypatch, files)
        fake.infos[files["a.m4a"]] = MediaInfo(kind="audio", audio_codec="opus", sample_rate=48000, channels=1)
        with pytest.raises(fork_join.ConfigError, match="AAC .* или MP3"):
            fork_join.process_config(self.config(files), log=lambda m: None)

    def test_video_files_with_different_parameters_are_refused_too(self, monkeypatch, files, tmp_path):
        fake = FakeFfmpeg(monkeypatch)
        (tmp_path / "v2.mp4").write_bytes(b"")
        fake.infos = {str(tmp_path / "v2.mp4"): MediaInfo(**{**VIDEO_INFO.__dict__, "width": 1280, "height": 720})}
        config = {"segments": [
            {"path": files["v.mp4"], "fragments": [{"start": "00:00:00", "end": "00:00:10"}]},
            {"path": str(tmp_path / "v2.mp4"), "fragments": [{"start": "00:00:00", "end": "00:00:10"}]}],
            "output": {"videoPath": str(tmp_path / "o.mp4"), "audioPath": str(tmp_path / "o.m4a")}}
        with pytest.raises(fork_join.ConfigError, match="разрешение 1280×720 вместо 1920×1080"):
            fork_join.process_config(config, log=lambda m: None)
        assert fake.calls == []

    # --- проверка при добавлении файла в интерфейсе
    def test_explain_incompatibility(self, monkeypatch, files):
        fake = self.audio_fake(monkeypatch, files)
        assert fork_join.explain_incompatibility(files["a.m4a"], files["b.m4a"]) is None
        assert fork_join.explain_incompatibility(files["a.m4a"], files["c.mp3"]) == "c.mp3: кодек звука MP3 вместо AAC"
        assert "смешивать видео и аудио нельзя" in fork_join.explain_incompatibility(files["a.m4a"], files["v.mp4"])


class TestCutToEndOfFile:
    """Конец фрагмента у самого конца файла — резать до конца, а не терять хвост из-за округления секунд."""

    def cut(self, monkeypatch, end, duration, media_type="video"):
        fake = FakeFfmpeg(monkeypatch)
        fork_join.cut_fragment("in.m4a", "00:00:00", end, "out.m4a", media_type, source_duration=duration)
        return fake.calls[0][0]

    def test_end_shown_as_the_file_end_cuts_to_the_end(self, monkeypatch):
        # запись 5296.363 с показывается как 01:28:16 — раньше хвост 0.363 с терялся
        args = self.cut(monkeypatch, "01:28:16", 5296.363)
        assert "-t" not in args and args[:4] == ["-ss", "00:00:00", "-i", "in.m4a"]

    def test_end_a_bit_after_the_file_end_also_cuts_to_the_end(self, monkeypatch):
        assert "-t" not in self.cut(monkeypatch, "01:28:17", 5296.363)     # округление вверх

    def test_end_exactly_at_the_tolerance_edge(self, monkeypatch):
        assert "-t" not in self.cut(monkeypatch, "00:01:40", 100.5)        # разница ровно 0.5 с
        assert "-t" in self.cut(monkeypatch, "00:01:40", 100.51)           # чуть больше — уже обычный конец

    def test_end_clearly_before_the_file_end_keeps_the_duration(self, monkeypatch):
        args = self.cut(monkeypatch, "01:28:15", 5296.363)                 # 1.363 с до конца
        assert args[args.index("-t") + 1] == "5295.0"

    def test_unknown_duration_keeps_the_old_behaviour(self, monkeypatch):
        fake = FakeFfmpeg(monkeypatch)
        fork_join.cut_fragment("in.mp4", "00:01:00", "00:02:30", "out.mp4")
        assert fake.calls[0][0][4:6] == ["-t", "90.0"]

    def test_works_for_audio_too(self, monkeypatch):
        args = self.cut(monkeypatch, "01:28:16", 5296.363, media_type="audio")
        assert "-t" not in args and "-vn" in args

    def test_join_uses_each_sources_own_duration(self, monkeypatch, tmp_path):
        fake = FakeFfmpeg(monkeypatch)
        a, b = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")
        fake.durations = {a: 5296.363, b: 5196.864}
        segments = [{"path": a, "fragments": [{"start": "00:00:00", "end": "01:28:16"}]},
                    {"path": b, "fragments": [{"start": "00:00:00", "end": "01:26:16"}]}]   # у b конец далеко
        fork_join.join_fragments(segments, str(tmp_path), str(tmp_path / "out.mp4"))
        cuts = [args for args, _ in fake.calls[:2]]
        assert "-t" not in cuts[0]                      # a: до конца файла
        assert cuts[1][cuts[1].index("-t") + 1] == "5176.0"   # b: обычный конец (60 с до конца)


class TestRecompressVideo:
    """Необязательное пережатие видео: x264 veryfast CRF 23 вместо копирования при склейке."""

    def config(self, tmp_path, **extra):
        video = tmp_path / "in.mp4"
        video.write_bytes(b"")
        return {
            "segments": [{"path": str(video), "fragments": [
                {"start": "00:00:00", "end": "00:00:10"}, {"start": "00:01:00", "end": "00:01:30"}]}],
            "output": {"videoPath": str(tmp_path / "out.mp4"), "audioPath": str(tmp_path / "out.m4a")},
            **extra,
        }

    # --- конфигурация
    def test_flag_is_optional_and_false_by_default(self, tmp_path):
        fork_join.validate_config(self.config(tmp_path))
        fork_join.validate_config(self.config(tmp_path, recompressVideo=False))
        fork_join.validate_config(self.config(tmp_path, recompressVideo=True))

    @pytest.mark.parametrize("value", ["true", 1, None, "yes"])
    def test_flag_must_be_a_boolean(self, tmp_path, value):
        with pytest.raises(fork_join.ConfigError, match="recompressVideo"):
            fork_join.validate_config(self.config(tmp_path, recompressVideo=value))

    def test_flag_makes_no_sense_for_audio(self, tmp_path):
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"")
        config = {"mediaType": "audio", "recompressVideo": True,
                  "segments": [{"path": str(audio), "fragments": [{"start": "00:00:00", "end": "00:00:10"}]}],
                  "output": {"audioPath": str(tmp_path / "out.m4a")}}
        with pytest.raises(fork_join.ConfigError, match="только для видео"):
            fork_join.validate_config(config)

    def test_false_flag_is_fine_for_audio(self, tmp_path):
        audio = tmp_path / "a.m4a"
        audio.write_bytes(b"")
        config = {"mediaType": "audio", "recompressVideo": False,
                  "segments": [{"path": str(audio), "fragments": [{"start": "00:00:00", "end": "00:00:10"}]}],
                  "output": {"audioPath": str(tmp_path / "out.m4a")}}
        fork_join.validate_config(config)

    # --- команды
    def test_cuts_stay_stream_copies_and_the_join_reencodes_video_only(self, monkeypatch, tmp_path):
        fake = FakeFfmpeg(monkeypatch)
        fork_join.process_config(self.config(tmp_path, recompressVideo=True), log=lambda m: None)

        cuts = [args for args, _ in fake.calls if args[0] == "-ss"]
        assert len(cuts) == 2 and all("copy" in c and "libx264" not in c for c in cuts)

        assert len(fake.progress_calls) == 1
        args, action = fake.progress_calls[0]
        assert args[:5] == ["-f", "concat", "-safe", "0", "-i"] and args[5].endswith("concat_list.txt")
        assert args[6:] == ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                            "-c:a", "copy", str(tmp_path / "out.mp4")]
        assert action == "Не удалось пережать видео"

    def test_audio_is_still_extracted_from_the_result_without_recoding(self, monkeypatch, tmp_path):
        fake = FakeFfmpeg(monkeypatch)
        fork_join.process_config(self.config(tmp_path, recompressVideo=True), log=lambda m: None)
        assert fake.calls[-1][0] == ["-i", str(tmp_path / "out.mp4"), "-vn", "-acodec", "copy", str(tmp_path / "out.m4a")]

    def test_no_plain_concat_copy_command_when_recompressing(self, monkeypatch, tmp_path):
        fake = FakeFfmpeg(monkeypatch)
        fork_join.process_config(self.config(tmp_path, recompressVideo=True), log=lambda m: None)
        assert not any(args[:2] == ["-f", "concat"] for args, _ in fake.calls)

    def test_without_the_flag_nothing_changes(self, monkeypatch, tmp_path):
        fake = FakeFfmpeg(monkeypatch)
        fork_join.process_config(self.config(tmp_path), log=lambda m: None)
        assert fake.progress_calls == []
        concat = [args for args, _ in fake.calls if args[:2] == ["-f", "concat"]]
        assert len(concat) == 1 and concat[0][-2:] == ["copy", str(tmp_path / "out.mp4")]

    # --- fail fast
    def test_missing_encoder_stops_before_any_work(self, monkeypatch, tmp_path):
        fake = FakeFfmpeg(monkeypatch)
        fake.encoders = set()
        with pytest.raises(FFmpegError) as info:
            fork_join.process_config(self.config(tmp_path, recompressVideo=True), log=lambda m: None)
        assert "libx264" in str(info.value) and "Отключите «Пережать видео»" in str(info.value)
        assert fake.calls == [] and fake.progress_calls == []

    def test_encoder_is_not_checked_when_not_recompressing(self, monkeypatch, tmp_path):
        fake = FakeFfmpeg(monkeypatch)
        fake.encoders = set()
        fork_join.process_config(self.config(tmp_path), log=lambda m: None)   # не бросает

    # --- журнал
    def test_progress_is_logged_every_ten_percent(self, monkeypatch, tmp_path):
        fake = FakeFfmpeg(monkeypatch)
        fake.durations = {}  # каждый фрагмент по умолчанию 100000 с, всего 200000 с
        fake.progress_script = [0, 5000, 20000, 20001, 100000, 200000]
        logs = []
        fork_join.process_config(self.config(tmp_path, recompressVideo=True), log=logs.append)
        assert [m for m in logs if m.startswith("Пережатие видео")] == [
            "Пережатие видео: 10%", "Пережатие видео: 50%", "Пережатие видео: 100%"]

    def test_log_announces_the_recompression(self, monkeypatch, tmp_path):
        FakeFfmpeg(monkeypatch)
        logs = []
        fork_join.process_config(self.config(tmp_path, recompressVideo=True), log=logs.append)
        assert any("пережимаю видео (x264 veryfast, CRF 23)" in m for m in logs)


class TestProgressLogger:
    def test_logs_only_when_a_new_step_is_crossed(self):
        logs = []
        callback = fork_join.progress_logger(100.0, logs.append)
        for seconds in (0, 3, 9.9, 10, 12, 19.9, 20, 55, 99, 100):
            callback(seconds)
        assert logs == ["Пережатие видео: 10%", "Пережатие видео: 20%", "Пережатие видео: 50%",
                        "Пережатие видео: 90%", "Пережатие видео: 100%"]

    def test_never_exceeds_one_hundred_and_never_repeats(self):
        logs = []
        callback = fork_join.progress_logger(10.0, logs.append)
        for seconds in (10, 11, 50, 10):          # ffmpeg может слегка превысить расчётную длительность
            callback(seconds)
        assert logs == ["Пережатие видео: 100%"]

    def test_zero_total_duration_is_harmless(self):
        logs = []
        fork_join.progress_logger(0.0, logs.append)(5.0)
        assert logs == []

    def test_custom_step(self):
        logs = []
        callback = fork_join.progress_logger(100.0, logs.append, step_percent=25)
        for seconds in (10, 30, 60, 80, 100):
            callback(seconds)
        assert logs == ["Пережатие видео: 25%", "Пережатие видео: 50%", "Пережатие видео: 75%", "Пережатие видео: 100%"]


class TestSizeReport:
    def test_logged_when_the_files_exist(self, monkeypatch, tmp_path):
        FakeFfmpeg(monkeypatch)
        # размеры подменены: два фрагмента 8 МБ в сумме -> результат 2 МБ
        monkeypatch.setattr(fork_join, "_size_mb", lambda paths: 8.0 if len(list(paths)) > 1 else 2.0)
        video = tmp_path / "in.mp4"
        video.write_bytes(b"")
        config = {"segments": [{"path": str(video), "fragments": [
            {"start": "00:00:00", "end": "00:00:10"}, {"start": "00:01:00", "end": "00:01:30"}]}],
            "output": {"videoPath": str(tmp_path / "out.mp4"), "audioPath": str(tmp_path / "out.m4a")},
            "recompressVideo": True}
        logs = []
        fork_join.process_config(config, log=logs.append)
        assert "Размер видео: 8 МБ → 2 МБ (25% от исходного)" in logs

    def test_skipped_silently_when_sizes_are_unknown(self, monkeypatch, tmp_path):
        FakeFfmpeg(monkeypatch)       # файлы не созданы -> os.path.getsize бросает OSError
        video = tmp_path / "in.mp4"
        video.write_bytes(b"")
        config = {"segments": [{"path": str(video), "fragments": [{"start": "00:00:00", "end": "00:00:10"}]}],
                  "output": {"videoPath": str(tmp_path / "out.mp4"), "audioPath": str(tmp_path / "out.m4a")},
                  "recompressVideo": True}
        logs = []
        fork_join.process_config(config, log=logs.append)
        assert not any(m.startswith("Размер видео") for m in logs)

    def test_size_helper(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.write_bytes(b"x" * 1048576)
        b.write_bytes(b"x" * 2097152)
        assert fork_join._size_mb([a, b]) == pytest.approx(3.0)
        assert fork_join._size_mb([tmp_path / "нет"]) is None

