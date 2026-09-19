"""Оркестрация transcribe.run: порядок этапов, аргументы воркера, файлы результата.
Всё тяжёлое (ffmpeg, диаризация, GigaAM) подменено."""

import json
import os
from pathlib import Path

import pytest

from diarization.diarize_pyannote import DiarizationError
from speech_to_text_gigaam import transcribe
from speech_to_text_gigaam.environment import EnvironmentCheckError

PLAIN_SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "Мы вот работаем."},
    {"start": 4.0, "end": 8.0, "text": ""},  # ничего не распознано (тишина) — результат не меняется
]
WORD_SEGMENTS = [
    {"start": 0.0, "end": 6.0, "text": "Привет, всем. Да.", "words": [
        {"text": "Привет,", "start": 0.0, "end": 1.0},
        {"text": "всем.", "start": 1.0, "end": 2.0},
        {"text": "Да.", "start": 5.0, "end": 6.0},
    ]},
]
TURNS = {"model": "m", "turns": [
    {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_A"},
    {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_B"},
]}


class FakePipeline:
    def __init__(self, monkeypatch, segments):
        self.calls = []          # порядок обращений
        self.modules = []        # (module, args) запусков подпроцессов
        self.wav_paths = []
        self.segments = segments
        self.fail_worker = False
        monkeypatch.setattr(transcribe, "check_ffmpeg_compatibility", lambda: self.calls.append("ffmpeg_check"))
        monkeypatch.setattr(transcribe, "ensure_hf_token", lambda: self.calls.append("hf_token"))
        monkeypatch.setattr(transcribe.diarize_pyannote, "check_model_available",
                            lambda: self.calls.append("model_check"))
        monkeypatch.setattr(transcribe, "get_media_duration", lambda path: 3725.0)
        monkeypatch.setattr(transcribe, "convert_to_wav_16k_mono", self.convert)
        monkeypatch.setattr(transcribe, "_run_module", self.run_module)

    def convert(self, src, dst):
        self.calls.append("convert")
        self.wav_paths.append(Path(dst))
        Path(dst).write_bytes(b"wav")

    def run_module(self, module, args, log):
        args = [str(a) for a in args]
        self.calls.append(module)
        self.modules.append((module, args))
        output = Path(args[args.index("--output") + 1])
        if module == "diarization.diarize_pyannote":
            output.write_text(json.dumps(TURNS), encoding="utf-8")
        else:
            if self.fail_worker:
                raise RuntimeError("Сбой worker (код 1)")
            output.write_text(json.dumps(self.segments), encoding="utf-8")
        log(f"вывод {module}")


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "lecture.m4a"
    path.write_bytes(b"audio")
    return path


class TestSidecarPaths:
    def test_paths_share_the_transcript_stem(self):
        assert transcribe.diarization_path("/out/lecture.txt") == Path("/out/lecture.diarization.json")
        assert transcribe.words_path("/out/lecture.txt") == Path("/out/lecture.words.json")

    def test_dots_in_the_name_are_kept(self):
        assert transcribe.diarization_path("a/lecture.v2.txt") == Path("a/lecture.v2.diarization.json")


class TestRunWithoutDiarization:
    def test_writes_transcript_and_no_sidecar_files(self, monkeypatch, tmp_path, audio):
        fake = FakePipeline(monkeypatch, PLAIN_SEGMENTS)
        out = tmp_path / "res.txt"
        assert transcribe.run(audio, out, keep_fillers=False, diarize=False, log=lambda m: None) == out
        text = out.read_text(encoding="utf-8")
        assert "# lecture.m4a" in text and "слова-паразиты удалены" in text
        assert "[00:00:00] Мы работаем." in text and "Спикер" not in text
        assert sorted(p.name for p in tmp_path.iterdir()) == ["lecture.m4a", "res.txt"]

    def test_empty_segments_change_nothing(self, monkeypatch, tmp_path, audio):
        with_empty = tmp_path / "with.txt"
        without_empty = tmp_path / "without.txt"
        FakePipeline(monkeypatch, PLAIN_SEGMENTS)
        transcribe.run(audio, with_empty, log=lambda m: None)
        FakePipeline(monkeypatch, PLAIN_SEGMENTS[:1] + [{"start": 5.0, "end": 6.0, "text": "   "}])
        transcribe.run(audio, without_empty, log=lambda m: None)
        assert with_empty.read_text(encoding="utf-8") == without_empty.read_text(encoding="utf-8")

    def test_only_the_worker_runs_without_word_timestamps(self, monkeypatch, tmp_path, audio):
        fake = FakePipeline(monkeypatch, PLAIN_SEGMENTS)
        transcribe.run(audio, tmp_path / "res.txt", log=lambda m: None)
        assert fake.calls == ["ffmpeg_check", "hf_token", "convert", "speech_to_text_gigaam.worker"]
        args = fake.modules[0][1]
        assert "--word-timestamps" not in args
        assert args[args.index("--model") + 1] == transcribe.MODEL_NAME

    def test_default_output_path_is_next_to_the_audio(self, monkeypatch, audio):
        FakePipeline(monkeypatch, PLAIN_SEGMENTS)
        assert transcribe.run(audio, log=lambda m: None) == audio.resolve().with_suffix(".gigaam.txt")

    def test_fillers_kept_on_request(self, monkeypatch, tmp_path, audio):
        FakePipeline(monkeypatch, PLAIN_SEGMENTS)
        out = tmp_path / "res.txt"
        transcribe.run(audio, out, keep_fillers=True, log=lambda m: None)
        assert "Мы вот работаем." in out.read_text(encoding="utf-8")

    def test_progress_is_logged(self, monkeypatch, tmp_path, audio):
        FakePipeline(monkeypatch, PLAIN_SEGMENTS)
        logs = []
        transcribe.run(audio, tmp_path / "res.txt", log=logs.append)
        assert "Длительность видеофайла после обрезки и склейки: 01:02:05" in logs
        assert any("вывод speech_to_text_gigaam.worker" in m for m in logs)
        assert logs[-1].startswith("Готово: ")


class TestRunWithDiarization:
    def test_diarization_runs_first_and_worker_gets_word_timestamps(self, monkeypatch, tmp_path, audio):
        fake = FakePipeline(monkeypatch, WORD_SEGMENTS)
        transcribe.run(audio, tmp_path / "res.txt", diarize=True, log=lambda m: None)
        assert fake.calls == ["ffmpeg_check", "hf_token", "model_check", "convert",
                              "diarization.diarize_pyannote", "speech_to_text_gigaam.worker"]
        assert "--word-timestamps" in fake.modules[1][1]

    def test_transcript_has_speaker_labels_and_sidecar_files(self, monkeypatch, tmp_path, audio):
        FakePipeline(monkeypatch, WORD_SEGMENTS)
        out = tmp_path / "res.txt"
        transcribe.run(audio, out, keep_fillers=True, diarize=True, log=lambda m: None)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert "[00:00:00] Спикер 1: Привет, всем." in lines and "[00:00:05] Спикер 2: Да." in lines
        assert json.loads((tmp_path / "res.diarization.json").read_text(encoding="utf-8")) == TURNS
        words = json.loads((tmp_path / "res.words.json").read_text(encoding="utf-8"))
        assert words["audio"] == "lecture.m4a" and words["model"] == transcribe.MODEL_NAME
        assert [w["text"] for w in words["words"]] == ["Привет,", "всем.", "Да."]

    def test_empty_segment_without_words_is_harmless(self, monkeypatch, tmp_path, audio):
        FakePipeline(monkeypatch, WORD_SEGMENTS + [{"start": 7.0, "end": 8.0, "text": "", "words": []}])
        out = tmp_path / "res.txt"
        transcribe.run(audio, out, keep_fillers=True, diarize=True, log=lambda m: None)
        assert "[00:00:05] Спикер 2: Да." in out.read_text(encoding="utf-8")
        words = json.loads((tmp_path / "res.words.json").read_text(encoding="utf-8"))["words"]
        assert len(words) == 3

    def test_diarization_output_path_is_passed_to_the_subprocess(self, monkeypatch, tmp_path, audio):
        fake = FakePipeline(monkeypatch, WORD_SEGMENTS)
        transcribe.run(audio, tmp_path / "res.txt", diarize=True, log=lambda m: None)
        args = fake.modules[0][1]
        assert args[args.index("--output") + 1] == str(tmp_path / "res.diarization.json")


class TestFailFastAndCleanup:
    def test_missing_model_stops_before_any_work(self, monkeypatch, tmp_path, audio):
        fake = FakePipeline(monkeypatch, WORD_SEGMENTS)

        def missing():
            fake.calls.append("model_check")
            raise DiarizationError("модель не найдена")

        monkeypatch.setattr(transcribe.diarize_pyannote, "check_model_available", missing)
        with pytest.raises(DiarizationError):
            transcribe.run(audio, tmp_path / "res.txt", diarize=True, log=lambda m: None)
        assert "convert" not in fake.calls and fake.modules == []

    def test_environment_error_stops_immediately(self, monkeypatch, tmp_path, audio):
        fake = FakePipeline(monkeypatch, PLAIN_SEGMENTS)

        def broken():
            raise EnvironmentCheckError("ffmpeg 9")

        monkeypatch.setattr(transcribe, "check_ffmpeg_compatibility", broken)
        with pytest.raises(EnvironmentCheckError):
            transcribe.run(audio, tmp_path / "res.txt", log=lambda m: None)
        assert fake.calls == []

    def test_missing_audio(self, monkeypatch, tmp_path):
        fake = FakePipeline(monkeypatch, PLAIN_SEGMENTS)
        with pytest.raises(FileNotFoundError):
            transcribe.run(tmp_path / "нет.m4a", tmp_path / "res.txt", log=lambda m: None)
        assert "convert" not in fake.calls

    def test_temporary_wav_is_removed_after_success(self, monkeypatch, tmp_path, audio):
        fake = FakePipeline(monkeypatch, PLAIN_SEGMENTS)
        transcribe.run(audio, tmp_path / "res.txt", log=lambda m: None)
        assert fake.wav_paths and not fake.wav_paths[0].exists() and not fake.wav_paths[0].parent.exists()

    def test_temporary_wav_is_removed_after_worker_failure(self, monkeypatch, tmp_path, audio):
        fake = FakePipeline(monkeypatch, PLAIN_SEGMENTS)
        fake.fail_worker = True
        with pytest.raises(RuntimeError, match="Сбой worker"):
            transcribe.run(audio, tmp_path / "res.txt", log=lambda m: None)
        assert not fake.wav_paths[0].parent.exists()
        assert not (tmp_path / "res.txt").exists()


class TestRunModule:
    """Настоящий подпроцесс `python -m`: вывод, окружение, ошибка."""

    @pytest.fixture
    def modules(self, tmp_path, monkeypatch):
        (tmp_path / "hello_mod.py").write_text(
            "import os, sys\n"
            "print('строка 1'); print(); print('окружение:', os.environ['TQDM_DISABLE'], os.environ['PYTHONUNBUFFERED'])\n"
            "print('аргументы:', *sys.argv[1:])\n", encoding="utf-8")
        (tmp_path / "boom_mod.py").write_text(
            "import sys\nfor i in range(30): print(f'line {i}')\nsys.exit(3)\n", encoding="utf-8")
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        return tmp_path

    def test_streams_non_empty_lines_to_log(self, modules):
        logs = []
        transcribe._run_module("hello_mod", ["a", Path("b")], logs.append)
        assert logs == ["строка 1", "окружение: 1 1", "аргументы: a b"]

    def test_failure_reports_exit_code_and_last_lines(self, modules):
        logs = []
        with pytest.raises(RuntimeError) as info:
            transcribe._run_module("boom_mod", [], logs.append)
        message = str(info.value)
        assert "Сбой boom_mod (код 3)" in message
        assert "line 29" in message and "line 15" in message and "line 14" not in message  # последние 15
        assert len(logs) == 30

    def test_repo_root_is_on_the_python_path_of_the_child(self, modules):
        (modules / "root_mod.py").write_text(
            "import os; print(os.environ['PYTHONPATH'])\n", encoding="utf-8")
        logs = []
        transcribe._run_module("root_mod", [], logs.append)
        assert logs[0].split(os.pathsep)[0] == str(transcribe.REPO_ROOT)
