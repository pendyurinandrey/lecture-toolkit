"""Сборка итогового транскрипта: абзацы без спикеров и со спикерами, запись файла."""

import pytest

from speech_to_text_gigaam.transcript import (
    format_timestamp,
    group_into_sentence_paragraphs,
    plain_paragraphs,
    speaker_paragraphs,
    write_transcript,
)


def word(start, end, text):
    return {"text": text, "start": start, "end": end}


def turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


@pytest.mark.parametrize("seconds, expected", [
    (0, "00:00:00"), (59.9, "00:00:59"), (61, "00:01:01"), (3661.5, "01:01:01"), (11389.66, "03:09:49"),
])
def test_format_timestamp_truncates_fractions(seconds, expected):
    assert format_timestamp(seconds) == expected


class TestGroupIntoSentenceParagraphs:
    def test_short_text_is_one_paragraph(self):
        pairs = [("Привет.", 0.0), ("Как", 1.0), ("дела?", 2.0)]
        assert group_into_sentence_paragraphs(pairs) == [(0.0, "Привет. Как дела?")]

    def test_splits_only_after_sentence_end_once_target_is_reached(self):
        pairs = [("слово", float(i)) for i in range(10)]
        pairs[4] = ("слово,", 4.0)       # запятая — не конец предложения
        pairs[7] = ("слово.", 7.0)       # точка — можно резать
        result = group_into_sentence_paragraphs(pairs, target_chars=20)
        assert [start for start, _ in result] == [0.0, 8.0]
        assert result[0][1].endswith("слово.")

    @pytest.mark.parametrize("end", [".", "?", "!"])
    def test_every_sentence_ending_mark_allows_a_split(self, end):
        pairs = [("слово", 0.0), (f"слово{end}", 1.0), ("ещё", 2.0)]
        assert group_into_sentence_paragraphs(pairs, target_chars=5) == [(0.0, f"слово слово{end}"), (2.0, "ещё")]

    def test_other_punctuation_does_not_allow_a_split(self):
        pairs = [("слово,", 0.0), ("слово;", 1.0), ("слово:", 2.0), ("слово…", 3.0), ("конец", 4.0)]
        assert len(group_into_sentence_paragraphs(pairs, target_chars=5)) == 1

    def test_never_splits_mid_sentence_even_when_over_target(self):
        pairs = [("слово", float(i)) for i in range(50)]  # ни одной точки
        assert len(group_into_sentence_paragraphs(pairs, target_chars=10)) == 1

    def test_empty(self):
        assert group_into_sentence_paragraphs([]) == []


class TestPlainParagraphs:
    def test_words_get_interpolated_times_inside_a_segment(self):
        segments = [{"text": "Привет всем", "start": 10.0, "end": 14.0}]
        assert plain_paragraphs(segments, keep_fillers=True, log=lambda m: None) == [(10.0, "Привет всем")]

    def test_empty_and_blank_segments_are_skipped_without_errors(self):
        segments = [{"text": "", "start": 0.0, "end": 1.0}, {"text": "  ", "start": 1.0, "end": 2.0},
                    {"text": "Привет", "start": 2.0, "end": 3.0}, {"text": "", "start": 3.0, "end": 4.0}]
        assert plain_paragraphs(segments, keep_fillers=True, log=lambda m: None) == [(2.0, "Привет")]
        assert plain_paragraphs([{"text": "", "start": 0.0, "end": 1.0}], keep_fillers=True, log=lambda m: None) == []

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


class TestWriteTranscript:
    def test_plain_transcript(self, tmp_path):
        out = tmp_path / "t.txt"
        write_transcript(out, "lecture.m4a", [(0.0, "Привет."), (65.0, "Пока.")], "v3_test",
                         keep_fillers=False, diarize=False)
        assert out.read_text(encoding="utf-8") == (
            "# lecture.m4a\n"
            "# Транскрибация: GigaAM v3_test (longform), перегруппировано по предложениям, слова-паразиты удалены\n"
            "\n"
            "[00:00:00] Привет.\n\n"
            "[00:01:05] Пока.\n\n"
        )

    def test_fillers_kept_header(self, tmp_path):
        out = tmp_path / "t.txt"
        write_transcript(out, "a.m4a", [(0.0, "Да.")], "m", keep_fillers=True, diarize=False)
        header = out.read_text(encoding="utf-8").splitlines()[1]
        assert header.endswith("перегруппировано по предложениям") and "паразит" not in header

    def test_diarized_transcript_labels_speakers(self, tmp_path):
        out = tmp_path / "t.txt"
        write_transcript(out, "a.m4a", [(0.0, "Привет.", 1), (12.0, "Здравствуйте.", 2)], "m",
                         keep_fillers=True, diarize=True)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[2].startswith("# Говорящие: pyannote/") and "«Спикер N»" in lines[2]
        assert "[00:00:00] Спикер 1: Привет." in lines and "[00:00:12] Спикер 2: Здравствуйте." in lines
