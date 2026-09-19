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


def fake_torch_with_devices(monkeypatch):
    """torch.device(name) -> ("device", name): достаточно, чтобы видеть, какое устройство запросили."""
    torch = types.ModuleType("torch")
    torch.device = lambda name: ("device", name)
    monkeypatch.setitem(sys.modules, "torch", torch)


class TestRunVadOn:
    def make_vad(self, monkeypatch, fail_on=()):
        """Подставляет gigaam.vad_utils; запоминает, с каким устройством вызывали VAD."""
        fake_torch_with_devices(monkeypatch)
        vad = fake_gigaam(monkeypatch, n_segments=3)
        calls = []

        def segment(*args, **kwargs):
            calls.append(kwargs.get("device"))
            if kwargs.get("device") in fail_on:
                raise RuntimeError("нет операции на MPS")
            return ["s1", "s2", "s3"], [(0.0, 1.0)] * 3

        vad.segment_audio_file = segment
        return vad, calls

    def test_vad_runs_on_the_requested_accelerator(self, monkeypatch):
        vad, calls = self.make_vad(monkeypatch)
        worker.run_vad_on("mps")
        # gigaam передаёт свой device (CPU модели) — он заменяется на ускоритель
        segments, boundaries = vad.segment_audio_file("x.wav", 16000, device=("device", "cpu"))
        assert calls == [("device", "mps")]
        assert segments == ["s1", "s2", "s3"] and len(boundaries) == 3

    def test_cpu_leaves_vad_untouched(self, monkeypatch):
        vad, calls = self.make_vad(monkeypatch)
        original = vad.segment_audio_file
        worker.run_vad_on("cpu")
        assert vad.segment_audio_file is original

    def test_accelerator_failure_falls_back_to_cpu_with_a_warning(self, monkeypatch, capsys):
        vad, calls = self.make_vad(monkeypatch, fail_on={("device", "mps")})
        worker.run_vad_on("mps")
        segments, _ = vad.segment_audio_file("x.wav", 16000, device=("device", "cpu"))
        assert calls == [("device", "mps"), ("device", "cpu")]
        assert segments == ["s1", "s2", "s3"]
        out = capsys.readouterr().out
        assert "VAD на mps не удался" in out and "RuntimeError" in out and "повторяю на CPU" in out

    def test_failure_on_cpu_too_is_not_swallowed(self, monkeypatch):
        vad, _ = self.make_vad(monkeypatch, fail_on={("device", "mps"), ("device", "cpu")})
        worker.run_vad_on("mps")
        with pytest.raises(RuntimeError, match="нет операции"):
            vad.segment_audio_file("x.wav", 16000, device=("device", "cpu"))

    def test_other_arguments_are_passed_through(self, monkeypatch):
        vad, _ = self.make_vad(monkeypatch)
        seen = {}
        vad.segment_audio_file = lambda *args, **kwargs: (seen.update(args=args, kwargs=kwargs), ([], []))[1]
        worker.run_vad_on("cuda")
        vad.segment_audio_file("x.wav", 16000, device=("device", "cpu"), max_duration=10.0)
        assert seen["args"] == ("x.wav", 16000)
        assert seen["kwargs"] == {"device": ("device", "cuda"), "max_duration": 10.0}

    def test_without_gigaam_nothing_happens(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "gigaam", None)
        monkeypatch.setitem(sys.modules, "gigaam.vad_utils", None)
        fake_torch_with_devices(monkeypatch)
        worker.run_vad_on("mps")  # не бросает

    def test_fallback_keeps_the_progress_messages_single(self, monkeypatch, capsys):
        # порядок обёрток: сначала подмена устройства, поверх — прогресс; иначе при откате
        # на CPU «Нарезка на сегменты речи» печаталась бы дважды
        vad, calls = self.make_vad(monkeypatch, fail_on={("device", "mps")})
        worker.run_vad_on("mps")
        worker.install_progress_reporting(FakeModel())
        vad.segment_audio_file("x.wav", 16000, device=("device", "cpu"))
        out = capsys.readouterr().out
        assert out.count("Нарезка на сегменты речи (VAD)...") == 1 and out.count("VAD завершён") == 1
        assert "повторяю на CPU" in out

    def test_progress_reporting_still_works_on_top_of_the_device_override(self, monkeypatch, capsys):
        vad, calls = self.make_vad(monkeypatch)
        model = FakeModel()
        worker.run_vad_on("mps")
        worker.install_progress_reporting(model)   # порядок, как в transcribe_to_json
        vad.segment_audio_file("x.wav", 16000, device=("device", "cpu"))
        model.forward()
        out = capsys.readouterr().out
        assert calls == [("device", "mps")]
        assert "сегментов речи: 3" in out and "Распознавание: 100% (пачка 1/1)" in out


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


class TestTranscribeToJson:
    """Связка целиком с подставным GigaAM: устройства, порядок вызовов, файл результата."""

    @pytest.fixture
    def setup(self, monkeypatch):
        fake_torch_with_devices(monkeypatch)
        vad = fake_gigaam(monkeypatch, n_segments=3)
        events, vad_devices, load_args, fail_on = [], [], {}, set()

        def segment(*args, **kwargs):
            vad_devices.append(kwargs.get("device"))
            if kwargs.get("device") in fail_on:
                raise RuntimeError("нет операции на MPS")
            return ["s"] * 3, [(0.0, 1.0)] * 3

        vad.segment_audio_file = segment

        class Model(FakeModel):
            def transcribe_longform(self, path, word_timestamps):
                # как настоящий GigaAM: VAD получает устройство модели (CPU)
                sys.modules["gigaam"].vad_utils.segment_audio_file(path, 16000, device=("device", "cpu"))
                self.forward()
                return [SimpleNamespace(start=0.0, end=1.0, text="Привет.",
                                        words=[SimpleNamespace(text="Привет.", start=0.0, end=0.5)])]

        def load_model(name, **kwargs):
            load_args.update(name=name, **kwargs)
            return Model()

        sys.modules["gigaam"].load_model = load_model
        monkeypatch.setattr(worker, "prepare_torch_environment", lambda: events.append("prepare"))
        monkeypatch.setattr(worker, "pick_device", lambda: events.append("pick") or "mps")
        return SimpleNamespace(events=events, vad_devices=vad_devices, load_args=load_args, fail_on=fail_on)

    def test_vad_runs_on_the_accelerator_while_the_model_stays_on_cpu(self, setup, tmp_path):
        worker.transcribe_to_json(tmp_path / "a.wav", tmp_path / "o.json", "v3_test", word_timestamps=False)
        assert setup.vad_devices == [("device", "mps")]
        assert setup.load_args == {"name": "v3_test", "device": "cpu", "fp16_encoder": False}

    def test_torch_environment_is_prepared_before_the_device_is_picked(self, setup, tmp_path):
        worker.transcribe_to_json(tmp_path / "a.wav", tmp_path / "o.json", "v3_test", word_timestamps=False)
        assert setup.events == ["prepare", "pick"]

    def test_writes_segments_json(self, setup, tmp_path):
        out = tmp_path / "o.json"
        worker.transcribe_to_json(tmp_path / "a.wav", out, "v3_test", word_timestamps=True)
        assert json.loads(out.read_text(encoding="utf-8")) == [{
            "start": 0.0, "end": 1.0, "text": "Привет.",
            "words": [{"text": "Привет.", "start": 0.0, "end": 0.5}]}]

    def test_accelerator_failure_still_produces_the_result_with_single_progress_messages(self, setup, tmp_path, capsys):
        setup.fail_on.add(("device", "mps"))
        out = tmp_path / "o.json"
        worker.transcribe_to_json(tmp_path / "a.wav", out, "v3_test", word_timestamps=False)
        assert setup.vad_devices == [("device", "mps"), ("device", "cpu")]
        assert json.loads(out.read_text(encoding="utf-8"))[0]["text"] == "Привет."
        printed = capsys.readouterr().out
        assert "повторяю на CPU" in printed
        assert printed.count("Нарезка на сегменты речи (VAD)...") == 1   # прогресс не дублируется

    def test_reports_devices_and_progress(self, setup, tmp_path, capsys):
        worker.transcribe_to_json(tmp_path / "a.wav", tmp_path / "o.json", "v3_test", word_timestamps=False)
        out = capsys.readouterr().out
        assert "Модель: v3_test (CPU), VAD: MPS" in out
        assert "Распознавание: 100% (пачка 1/1)" in out and "Сегментов: 1" in out


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
