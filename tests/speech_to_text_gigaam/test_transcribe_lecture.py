"""Сборка итогового транскрипта: абзацы без спикеров и со спикерами."""

from pathlib import Path

import pytest

pytest.importorskip("gigaam")  # transcribe_lecture тянет GigaAM (и torch) при импорте

from speech_to_text_gigaam.transcribe_lecture import (  # noqa: E402
    diarization_path,
    plain_paragraphs,
    speaker_paragraphs,
    words_path,
)


def word(start, end, text):
    return {"text": text, "start": start, "end": end}


def turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


class TestSidecarPaths:
    def test_paths_share_the_transcript_stem(self):
        assert diarization_path("/out/lecture.txt") == Path("/out/lecture.diarization.json")
        assert words_path("/out/lecture.txt") == Path("/out/lecture.words.json")

    def test_dots_in_the_name_are_kept(self):
        assert diarization_path("a/lecture.v2.txt") == Path("a/lecture.v2.diarization.json")


class TestPlainParagraphs:
    def test_words_get_interpolated_times_inside_a_segment(self):
        segments = [{"text": "Привет всем", "start": 10.0, "end": 14.0}]
        assert plain_paragraphs(segments, keep_fillers=True, log=lambda m: None) == [(10.0, "Привет всем")]

    def test_fillers_removed_and_counted(self):
        logs = []
        segments = [{"text": "Мы вот работаем", "start": 0.0, "end": 3.0}]
        assert plain_paragraphs(segments, keep_fillers=False, log=logs.append) == [(0.0, "Мы работаем")]
        assert any("удалено: 1" in m for m in logs)

    def test_fillers_kept_on_request(self):
        segments = [{"text": "Мы вот работаем", "start": 0.0, "end": 3.0}]
        assert plain_paragraphs(segments, keep_fillers=True, log=lambda m: None)[0][1] == "Мы вот работаем"


class TestSpeakerParagraphs:
    def run(self, words, turns, keep_fillers=True):
        logs = []
        return speaker_paragraphs(words, turns, keep_fillers, logs.append), logs

    def test_paragraph_breaks_on_speaker_change(self):
        words = [word(0, 1, "Привет,"), word(1, 2, "всем."), word(5, 6, "Да.")]
        paragraphs, _ = self.run(words, [turn(0, 4, "A"), turn(4, 8, "B")])
        assert paragraphs == [(0, "Привет, всем.", 1), (5, "Да.", 2)]

    def test_same_speaker_speaking_again_keeps_the_number(self):
        words = [word(0, 1, "Раз."), word(5, 6, "Два."), word(10, 11, "Три.")]
        turns = [turn(0, 3, "A"), turn(4, 8, "B"), turn(9, 12, "A")]
        paragraphs, _ = self.run(words, turns)
        assert [p[2] for p in paragraphs] == [1, 2, 1]

    def test_long_run_splits_at_sentence_end_within_one_speaker(self):
        words = [word(i, i + 0.5, "слово.") for i in range(100)]
        paragraphs, _ = self.run(words, [turn(0, 200, "A")])
        assert [p[2] for p in paragraphs] == [1, 1]
        assert paragraphs[1][0] == 72  # цель абзаца ~500 знаков: 72 слова по 7 знаков

    def test_fillers_removed_with_speaker_kept(self):
        words = [word(0, 1, "Ну,"), word(1, 2, "да."), word(5, 6, "Вот"), word(6, 7, "нет.")]
        paragraphs, logs = self.run(words, [turn(0, 4, "A"), turn(4, 8, "B")], keep_fillers=False)
        assert paragraphs == [(1, "Да.", 1), (6, "Нет.", 2)]
        assert any("удалено: 2" in m for m in logs)

    def test_micro_turn_jitter_does_not_split_the_paragraph(self):
        words = [word(9, 9.5, "перед"), word(10.15, 10.25, "шум"), word(11, 12, "после.")]
        turns = [turn(0, 10, "A"), turn(10.1, 10.3, "B"), turn(10.4, 20, "A")]
        paragraphs, logs = self.run(words, turns)
        assert len(paragraphs) == 1 and paragraphs[0][2] == 1
        assert any("после сглаживания микрореплик: 1" in m for m in logs)
