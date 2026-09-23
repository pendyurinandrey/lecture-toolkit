"""Пакетная обработка: план, сортировка/дедупликация списка, оркестрация
(`transcribe.run` подменяется — GigaAM не нужен)."""

from pathlib import Path

import pytest

from diarization.diarize_pyannote import DiarizationError
from speech_to_text_gigaam import batch
from speech_to_text_gigaam.environment import EnvironmentCheckError


class TestTranscriptPathFor:
    @pytest.mark.parametrize("name, expected", [
        ("lecture.m4a", "lecture.txt"),
        ("lecture.mp3", "lecture.txt"),
        ("lecture.mp4", "lecture.txt"),
        ("01 - тема： часть».m4a", "01 - тема： часть».txt"),   # необычные символы, реальный случай
        ("lecture..m4a", "lecture..txt"),                        # двойная точка перед расширением
        ("lecture.v2.m4a", "lecture.v2.txt"),
    ])
    def test_replaces_only_the_last_extension(self, name, expected):
        assert batch.transcript_path_for(Path("/a/b") / name) == Path("/a/b") / expected


class TestSortNewFiles:
    def test_sorted_by_name_ascending(self):
        paths = [Path("/a/03.m4a"), Path("/a/01.m4a"), Path("/a/02.m4a")]
        assert batch.sort_new_files(paths) == [Path("/a/01.m4a"), Path("/a/02.m4a"), Path("/a/03.m4a")]

    def test_sorts_by_name_not_by_full_path(self):
        # разные папки — сортировка всё равно по имени файла, не по полному пути
        paths = [Path("/z/01.m4a"), Path("/a/02.m4a")]
        assert batch.sort_new_files(paths) == paths

    def test_stable_for_equal_names_in_different_folders(self):
        a, b = Path("/z/same.m4a"), Path("/a/same.m4a")
        assert batch.sort_new_files([a, b]) == [a, b]

    def test_accepts_strings_too(self):
        assert batch.sort_new_files(["b.m4a", "a.m4a"]) == [Path("a.m4a"), Path("b.m4a")]

    def test_empty(self):
        assert batch.sort_new_files([]) == []


class TestAddFiles:
    def test_new_files_are_sorted_and_appended(self):
        existing = [Path("/a/z.m4a")]
        result = batch.add_files(existing, [Path("/a/c.m4a"), Path("/a/b.m4a")])
        assert result == [Path("/a/z.m4a"), Path("/a/b.m4a"), Path("/a/c.m4a")]

    def test_existing_order_is_never_changed(self):
        # даже если новый файл лексикографически должен был бы встать раньше — он всё равно в конце
        existing = [Path("/a/z.m4a"), Path("/a/y.m4a")]
        assert batch.add_files(existing, [Path("/a/a.m4a")])[:2] == existing

    def test_duplicate_of_an_existing_path_is_not_added_again(self):
        existing = [Path("/a/1.m4a")]
        assert batch.add_files(existing, [Path("/a/1.m4a")]) == existing

    def test_duplicate_within_the_same_new_selection_is_added_once(self):
        result = batch.add_files([], [Path("/a/1.m4a"), Path("/a/1.m4a")])
        assert result == [Path("/a/1.m4a")]

    def test_existing_list_is_not_mutated(self):
        existing = [Path("/a/1.m4a")]
        batch.add_files(existing, [Path("/a/2.m4a")])
        assert existing == [Path("/a/1.m4a")]

    def test_empty_addition_changes_nothing(self):
        existing = [Path("/a/1.m4a")]
        assert batch.add_files(existing, []) == existing


class TestDisplayName:
    def test_unique_name_is_shown_as_is(self):
        paths = [Path("/a/1.m4a"), Path("/b/2.m4a")]
        assert batch.display_name(Path("/a/1.m4a"), paths) == "1.m4a"

    def test_colliding_names_get_the_parent_folder(self):
        paths = [Path("/a/lecture.m4a"), Path("/b/lecture.m4a")]
        assert batch.display_name(Path("/a/lecture.m4a"), paths) == "lecture.m4a (a)"
        assert batch.display_name(Path("/b/lecture.m4a"), paths) == "lecture.m4a (b)"

    def test_three_way_collision(self):
        paths = [Path("/a/x.m4a"), Path("/b/x.m4a"), Path("/c/x.m4a")]
        assert batch.display_name(Path("/c/x.m4a"), paths) == "x.m4a (c)"


class TestPlanBatch:
    def test_already_done_reflects_the_filesystem(self, tmp_path):
        done = tmp_path / "done.m4a"
        done.write_bytes(b"")
        (tmp_path / "done.txt").write_text("готово", encoding="utf-8")
        pending = tmp_path / "pending.m4a"
        pending.write_bytes(b"")

        items = batch.plan_batch([done, pending])
        assert [i.already_done for i in items] == [True, False]
        assert items[0].transcript_path == tmp_path / "done.txt"

    def test_keeps_the_given_order(self, tmp_path):
        files = [tmp_path / f"{n}.m4a" for n in ("b", "a")]
        assert [i.path for i in batch.plan_batch(files)] == files

    def test_empty(self):
        assert batch.plan_batch([]) == []


class TestCheckEnvironment:
    def test_diarize_off_skips_the_model_check(self, monkeypatch):
        calls = []
        monkeypatch.setattr(batch, "check_ffmpeg_compatibility", lambda: calls.append("ffmpeg"))
        monkeypatch.setattr(batch, "ensure_hf_token", lambda: calls.append("token"))
        monkeypatch.setattr(batch.diarize_pyannote, "check_model_available", lambda: calls.append("model"))
        batch.check_environment(diarize=False)
        assert calls == ["ffmpeg", "token"]

    def test_diarize_on_checks_the_model_too_in_order(self, monkeypatch):
        calls = []
        monkeypatch.setattr(batch, "check_ffmpeg_compatibility", lambda: calls.append("ffmpeg"))
        monkeypatch.setattr(batch, "ensure_hf_token", lambda: calls.append("token"))
        monkeypatch.setattr(batch.diarize_pyannote, "check_model_available", lambda: calls.append("model"))
        batch.check_environment(diarize=True)
        assert calls == ["ffmpeg", "token", "model"]

    def test_ffmpeg_failure_stops_before_the_token_check(self, monkeypatch):
        calls = []
        monkeypatch.setattr(batch, "check_ffmpeg_compatibility",
                            lambda: (_ for _ in ()).throw(EnvironmentCheckError("ffmpeg 9")))
        monkeypatch.setattr(batch, "ensure_hf_token", lambda: calls.append("token"))
        with pytest.raises(EnvironmentCheckError, match="ffmpeg 9"):
            batch.check_environment(diarize=False)
        assert calls == []


class FakeTranscribe:
    """Подменяет speech_to_text_gigaam.transcribe.run: пишет фиктивный файл, запоминает вызовы."""

    def __init__(self, monkeypatch, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)  # имена файлов (path.name), на которых бросить исключение
        monkeypatch.setattr(batch, "transcribe", self)

    def run(self, audio_path, output_path=None, keep_fillers=False, diarize=False, log=print):
        self.calls.append({"audio_path": Path(audio_path), "output_path": Path(output_path),
                           "keep_fillers": keep_fillers, "diarize": diarize})
        if Path(audio_path).name in self.fail_on:
            raise RuntimeError(f"Сбой на {Path(audio_path).name}")
        Path(output_path).write_text(f"транскрипт {Path(audio_path).name}", encoding="utf-8")
        return Path(output_path)


@pytest.fixture
def files(tmp_path):
    paths = [tmp_path / name for name in ("01.m4a", "02.m4a", "03.m4a")]
    for p in paths:
        p.write_bytes(b"")
    return paths


class TestRunBatch:
    def test_processes_files_in_order_and_writes_transcripts(self, monkeypatch, files):
        fake = FakeTranscribe(monkeypatch)
        items = batch.plan_batch(files)
        result = batch.run_batch(items, keep_fillers=False, diarize=False, log=lambda m: None)

        assert result.processed == files and result.skipped == [] and result.failed == [] and not result.stopped
        assert [c["audio_path"] for c in fake.calls] == files
        assert all(p.with_suffix(".txt").exists() for p in files)

    def test_keep_fillers_and_diarize_are_passed_through(self, monkeypatch, files):
        fake = FakeTranscribe(monkeypatch)
        items = batch.plan_batch(files[:1])
        batch.run_batch(items, keep_fillers=True, diarize=True, log=lambda m: None)
        assert fake.calls[0]["keep_fillers"] is True and fake.calls[0]["diarize"] is True

    def test_already_done_files_are_skipped_without_calling_transcribe(self, monkeypatch, files):
        files[1].with_suffix(".txt").write_text("уже готово", encoding="utf-8")
        fake = FakeTranscribe(monkeypatch)
        items = batch.plan_batch(files)
        result = batch.run_batch(items, keep_fillers=False, diarize=False, log=lambda m: None)

        assert result.skipped == [files[1]] and result.processed == [files[0], files[2]]
        assert [c["audio_path"] for c in fake.calls] == [files[0], files[2]]

    def test_skip_check_is_re_evaluated_at_run_time_not_from_the_plan(self, monkeypatch, files):
        # already_done в плане было False, но файл появился до фактической обработки
        items = batch.plan_batch(files[:1])
        assert items[0].already_done is False
        files[0].with_suffix(".txt").write_text("успели создать", encoding="utf-8")
        fake = FakeTranscribe(monkeypatch)
        result = batch.run_batch(items, keep_fillers=False, diarize=False, log=lambda m: None)
        assert result.skipped == [files[0]] and fake.calls == []

    def test_error_on_one_file_does_not_stop_the_batch(self, monkeypatch, files):
        fake = FakeTranscribe(monkeypatch, fail_on={"02.m4a"})
        items = batch.plan_batch(files)
        result = batch.run_batch(items, keep_fillers=False, diarize=False, log=lambda m: None)

        assert result.processed == [files[0], files[2]]
        assert result.failed == [(files[1], "Сбой на 02.m4a")]
        assert not result.stopped
        assert [c["audio_path"] for c in fake.calls] == files  # все три файла были попытаны

    def test_stop_is_checked_before_each_file_not_mid_file(self, monkeypatch, files):
        fake = FakeTranscribe(monkeypatch)
        items = batch.plan_batch(files)
        seen = []

        def should_stop():
            seen.append(len(fake.calls))
            return len(fake.calls) >= 1  # остановиться после первого обработанного файла

        result = batch.run_batch(items, keep_fillers=False, diarize=False, log=lambda m: None, should_stop=should_stop)
        assert result.stopped is True
        assert result.processed == [files[0]]          # первый файл дообработан целиком
        assert [c["audio_path"] for c in fake.calls] == [files[0]]  # второй/третий не начинались

    def test_on_progress_reports_index_total_and_path(self, monkeypatch, files):
        FakeTranscribe(monkeypatch)
        items = batch.plan_batch(files)
        seen = []
        batch.run_batch(items, keep_fillers=False, diarize=False, log=lambda m: None,
                        on_progress=lambda i, t, p: seen.append((i, t, p)))
        assert seen == [(1, 3, files[0]), (2, 3, files[1]), (3, 3, files[2])]

    def test_log_messages_for_progress_skip_and_error(self, monkeypatch, files):
        files[1].with_suffix(".txt").write_text("готово", encoding="utf-8")
        FakeTranscribe(monkeypatch, fail_on={"03.m4a"})
        items = batch.plan_batch(files)
        logs = []
        batch.run_batch(items, keep_fillers=False, diarize=False, log=logs.append)
        assert "=== Файл 1/3: 01.m4a ===" in logs
        assert "Пропущено (уже обработано): 02.m4a" in logs
        assert "=== Файл 3/3: 03.m4a ===" in logs
        assert "Ошибка: 03.m4a: Сбой на 03.m4a" in logs

    def test_empty_items(self, monkeypatch):
        FakeTranscribe(monkeypatch)
        result = batch.run_batch([], keep_fillers=False, diarize=False, log=lambda m: None)
        assert result == batch.BatchResult()


class TestLogFilePath:
    def test_next_to_the_first_file_with_a_timestamp(self, tmp_path):
        from datetime import datetime
        first = tmp_path / "01.m4a"
        path = batch.log_file_path(first, when=datetime(2026, 9, 23, 14, 5, 9))
        assert path == tmp_path / "lecture-toolkit-batch-20260923-140509.log"

    def test_uses_the_parent_of_whatever_file_is_passed(self, tmp_path):
        other_dir = tmp_path / "elsewhere"
        other_dir.mkdir()
        path = batch.log_file_path(other_dir / "x.m4a")
        assert path.parent == other_dir


class TestBatchLogFile:
    def test_writes_lines_with_flush_and_still_calls_the_original_log(self, tmp_path):
        log_path = tmp_path / "sub" / "batch.log"
        received = []
        with batch.batch_log_file(log_path, received.append) as log:
            log("первая строка")
            log("вторая строка")
            assert log_path.read_text(encoding="utf-8") == "первая строка\nвторая строка\n"  # flush внутри контекста

        assert received == ["первая строка", "вторая строка"]
        assert log_path.read_text(encoding="utf-8") == "первая строка\nвторая строка\n"

    def test_creates_parent_directories(self, tmp_path):
        log_path = tmp_path / "a" / "b" / "c.log"
        with batch.batch_log_file(log_path, lambda m: None) as log:
            log("x")
        assert log_path.exists()

    def test_file_exists_even_with_zero_lines(self, tmp_path):
        log_path = tmp_path / "empty.log"
        with batch.batch_log_file(log_path, lambda m: None):
            pass
        assert log_path.exists() and log_path.read_text(encoding="utf-8") == ""

    def test_appends_across_separate_uses(self, tmp_path):
        log_path = tmp_path / "batch.log"
        with batch.batch_log_file(log_path, lambda m: None) as log:
            log("первый запуск")
        with batch.batch_log_file(log_path, lambda m: None) as log:
            log("второй запуск")
        assert log_path.read_text(encoding="utf-8") == "первый запуск\nвторой запуск\n"

    def test_integrates_with_run_batch(self, monkeypatch, files, tmp_path):
        FakeTranscribe(monkeypatch)
        items = batch.plan_batch(files)
        log_path = tmp_path / "run.log"
        with batch.batch_log_file(log_path, lambda m: None) as log:
            batch.run_batch(items, keep_fillers=False, diarize=False, log=log)
        content = log_path.read_text(encoding="utf-8")
        assert "=== Файл 1/3: 01.m4a ===" in content and "=== Файл 3/3: 03.m4a ===" in content
