"""Воркер GigaAM: JSON сегментов, прогресс, аргументы. Сам GigaAM не нужен —
модули gigaam подменяются пустышками."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from speech_to_text_gigaam import worker


def fake_gigaam(monkeypatch, n_segments):
    """Подставляет gigaam и gigaam.vad_utils; VAD «нарезает» n_segments сегментов."""
    package = types.ModuleType("gigaam")
    vad = types.ModuleType("gigaam.vad_utils")
    vad.segment_audio_file = lambda *args, **kwargs: (list(range(n_segments)), [(0.0, 1.0)] * n_segments)
    package.vad_utils = vad
    monkeypatch.setitem(sys.modules, "gigaam", package)
    monkeypatch.setitem(sys.modules, "gigaam.vad_utils", vad)
    return vad


class FakeModel:
    def __init__(self):
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return "encoded"


class TestInstallProgressReporting:
    def test_reports_vad_and_recognition_progress(self, monkeypatch, capsys):
        vad = fake_gigaam(monkeypatch, n_segments=40)   # 40 сегментов / 16 = 3 пачки
        model = FakeModel()
        worker.install_progress_reporting(model)

        segments, boundaries = vad.segment_audio_file("x.wav")
        assert len(segments) == 40 and len(boundaries) == 40   # результат VAD не искажается
        for _ in range(3):
            assert model.forward() == "encoded"                  # результат forward тоже

        out = capsys.readouterr().out
        assert "Нарезка на сегменты речи (VAD)..." in out
        assert "сегментов речи: 40" in out
        assert "Распознавание: 30% (пачка 1/3)" in out
        assert "Распознавание: 65% (пачка 2/3)" in out
        assert "Распознавание: 100% (пачка 3/3)" in out
        assert model.calls == 3

    def test_progress_lines_are_throttled(self, monkeypatch, capsys):
        vad = fake_gigaam(monkeypatch, n_segments=16 * 200)   # 200 пачек
        model = FakeModel()
        worker.install_progress_reporting(model)
        vad.segment_audio_file("x.wav")
        for _ in range(200):
            model.forward()
        lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("Распознавание:")]
        assert len(lines) == 100 // worker.PROGRESS_STEP_PERCENT + 1   # 0%..100% шагами по 5
        assert lines[-1].startswith("Распознавание: 100%")

    def test_no_speech_means_no_progress_lines_but_forward_still_works(self, monkeypatch, capsys):
        vad = fake_gigaam(monkeypatch, n_segments=0)
        model = FakeModel()
        worker.install_progress_reporting(model)
        vad.segment_audio_file("x.wav")
        assert model.forward() == "encoded"
        assert "Распознавание:" not in capsys.readouterr().out

    def test_without_gigaam_the_model_is_left_untouched(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "gigaam", None)        # import gigaam -> ImportError
        monkeypatch.setitem(sys.modules, "gigaam.vad_utils", None)
        model = FakeModel()
        original = model.forward
        worker.install_progress_reporting(model)
        assert model.forward == original


class TestSegmentsToJson:
    RESULT = [
        SimpleNamespace(start=0.0, end=2.0, text="Привет, всем.", words=[
            SimpleNamespace(text="Привет,", start=0.0, end=0.8), SimpleNamespace(text="всем.", start=0.9, end=1.5)]),
        SimpleNamespace(start=3.0, end=4.0, text="Да.", words=None),
    ]

    def test_without_word_timestamps(self):
        assert worker.segments_to_json(self.RESULT, word_timestamps=False) == [
            {"start": 0.0, "end": 2.0, "text": "Привет, всем."},
            {"start": 3.0, "end": 4.0, "text": "Да."},
        ]

    def test_with_word_timestamps(self):
        result = worker.segments_to_json(self.RESULT, word_timestamps=True)
        assert result[0]["words"] == [{"text": "Привет,", "start": 0.0, "end": 0.8},
                                      {"text": "всем.", "start": 0.9, "end": 1.5}]
        assert result[1]["words"] == []   # words=None -> пустой список

    def test_result_is_json_serialisable(self):
        json.dumps(worker.segments_to_json(self.RESULT, word_timestamps=True), ensure_ascii=False)


class TestMain:
    def test_passes_arguments_to_transcribe(self, monkeypatch, tmp_path):
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"")
        seen = {}
        monkeypatch.setattr(worker, "transcribe_to_json", lambda *args: seen.setdefault("args", args))
        monkeypatch.setattr(sys, "argv", ["worker", str(wav), "--output", str(tmp_path / "o.json"),
                                          "--model", "m1", "--word-timestamps"])
        worker.main()
        assert seen["args"] == (wav, tmp_path / "o.json", "m1", True)

    def test_word_timestamps_off_by_default(self, monkeypatch, tmp_path):
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"")
        seen = {}
        monkeypatch.setattr(worker, "transcribe_to_json", lambda *args: seen.setdefault("args", args))
        monkeypatch.setattr(sys, "argv", ["worker", str(wav), "--output", "o.json", "--model", "m1"])
        worker.main()
        assert seen["args"][3] is False

    def test_missing_wav(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "argv", ["worker", str(tmp_path / "нет.wav"), "--output", "o.json", "--model", "m"])
        with pytest.raises(SystemExit, match="Файл не найден"):
            worker.main()

    def test_model_is_required(self, monkeypatch, tmp_path, capsys):
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"")   # файл есть, значит падать может только отсутствие --model
        monkeypatch.setattr(worker, "transcribe_to_json", lambda *args: pytest.fail("не должен запускаться"))
        monkeypatch.setattr(sys, "argv", ["worker", str(wav), "--output", "o.json"])
        with pytest.raises(SystemExit) as info:
            worker.main()
        assert info.value.code == 2 and "--model" in capsys.readouterr().err
