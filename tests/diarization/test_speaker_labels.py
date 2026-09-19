"""Склейка слов с репликами диаризации и сглаживание микрореплик."""

import json

from diarization.speaker_labels import (
    assign_speaker,
    format_speaker,
    label_words,
    load_turns,
    smooth_turns,
)


def turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def word(start, end, text="x"):
    return {"text": text, "start": start, "end": end}


TURNS = [turn(0.0, 10.0, "T"), turn(12.0, 20.0, "S1"), turn(20.0, 30.0, "T"), turn(40.0, 50.0, "S2")]
STARTS = [t["start"] for t in TURNS]


def who(w):
    return assign_speaker(w, TURNS, STARTS)


class TestAssignSpeaker:
    def test_word_inside_turn(self):
        assert who(word(1, 2)) == "T"

    def test_word_on_boundary_goes_to_larger_overlap(self):
        assert who(word(19.0, 21.5)) == "T"    # 1.0 с в S1 против 1.5 с в T
        assert who(word(19.0, 20.4)) == "S1"   # 1.0 с против 0.4 с

    def test_word_in_gap_goes_to_nearest_turn(self):
        assert who(word(10.2, 10.6)) == "T"
        assert who(word(11.5, 11.8)) == "S1"

    def test_word_before_first_and_after_last_turn(self):
        assert who(word(-3, -2)) == "T"
        assert who(word(90, 91)) == "S2"

    def test_zero_length_word(self):
        assert who(word(15, 15)) == "S1"
        assert who(word(25, 25)) == "T"

    def test_word_spanning_several_turns_takes_the_longest_share(self):
        assert who(word(9.0, 31.0)) == "T"     # T: 1 + 10 с, S1: 8 с


class TestLabelWords:
    def test_numbers_follow_first_appearance_in_text(self):
        words = [word(1, 2), word(13, 14), word(21, 22), word(41, 42)]
        assert label_words(words, TURNS) == [1, 2, 1, 3]

    def test_speaker_without_words_does_not_consume_a_number(self):
        turns = [turn(0, 5, "A"), turn(5, 10, "C"), turn(10, 15, "B")]
        assert label_words([word(1, 2), word(11, 12)], turns) == [1, 2]

    def test_no_turns_means_single_speaker(self):
        assert label_words([word(0, 1), word(2, 3)], []) == [1, 1]

    def test_no_words(self):
        assert label_words([], TURNS) == []


def test_format_speaker():
    assert format_speaker(3) == "Спикер 3"


def test_load_turns_sorts_by_start(tmp_path):
    path = tmp_path / "d.diarization.json"
    path.write_text(json.dumps({"model": "m", "turns": [turn(5, 6, "B"), turn(1, 2, "A")]}), encoding="utf-8")
    assert [t["speaker"] for t in load_turns(path)] == ["A", "B"]


class TestSmoothTurns:
    def test_jitter_at_the_junction_is_absorbed_into_the_longer_neighbour(self):
        jitter = [turn(0, 10, "A"), turn(10.48, 10.51, "A"), turn(10.51, 10.61, "B"),
                  turn(10.61, 10.83, "A"), turn(10.83, 20, "B")]
        assert smooth_turns(jitter) == [turn(0, 10.83, "A"), turn(10.83, 20, "B")]

    def test_isolated_short_turn_among_silence_is_kept(self):
        turns = [turn(0, 10, "A"), turn(12, 12.4, "B"), turn(15, 25, "A")]
        assert smooth_turns(turns) == turns

    def test_turn_longer_than_threshold_is_kept(self):
        turns = [turn(0, 10, "A"), turn(10, 10.6, "B"), turn(10.6, 20, "A")]
        assert smooth_turns(turns) == turns

    def test_micro_turn_between_same_speaker_collapses_into_one(self):
        assert smooth_turns([turn(0, 5, "A"), turn(5, 5.2, "B"), turn(5.2, 9, "A")]) == [turn(0, 9, "A")]

    def test_micro_turn_goes_to_the_longer_neighbour(self):
        result = smooth_turns([turn(0, 3, "A"), turn(3, 3.2, "B"), turn(3.2, 30, "C")])
        assert [t["speaker"] for t in result] == ["A", "C"] and result[1]["start"] == 3.0

    def test_thresholds_are_configurable(self):
        turns = [turn(0, 10, "A"), turn(10, 10.6, "B"), turn(10.6, 20, "A")]
        assert smooth_turns(turns, min_duration=1.0) == [turn(0, 20, "A")]

    def test_input_is_not_mutated_and_empty_is_fine(self):
        turns = [turn(0, 10, "A"), turn(10.48, 10.51, "A"), turn(10.51, 20, "B")]
        smooth_turns(turns)
        assert turns[1] == turn(10.48, 10.51, "A")
        assert smooth_turns([]) == []
