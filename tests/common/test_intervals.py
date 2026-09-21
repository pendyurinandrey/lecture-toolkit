"""Чистая логика интервалов: ни ffmpeg, ни файлов не нужно."""

import pytest

from common.intervals import (
    IntervalRangeError,
    add_left_pad,
    edit_interval,
    keep_intervals_between_silences,
    snap_intervals_to_keyframes,
)


class TestKeepIntervalsBetweenSilences:
    def test_no_silences_keeps_whole_recording(self):
        assert keep_intervals_between_silences(100.0, []) == [(0.0, 100.0)]

    def test_silence_in_the_middle_splits_into_two(self):
        assert keep_intervals_between_silences(100.0, [(40.0, 50.0)]) == [(0.0, 40.0), (50.0, 100.0)]

    def test_silence_at_the_start_drops_leading_interval(self):
        # [0, 0.5] короче min_length=1 — не фрагмент, а огрызок
        assert keep_intervals_between_silences(100.0, [(0.5, 10.0)]) == [(10.0, 100.0)]

    def test_silence_at_the_end_drops_trailing_interval(self):
        assert keep_intervals_between_silences(100.0, [(60.0, 99.5)]) == [(0.0, 60.0)]

    def test_short_gap_between_two_silences_is_dropped(self):
        silences = [(10.0, 20.0), (20.5, 30.0)]  # между ними 0.5 с звука
        assert keep_intervals_between_silences(100.0, silences) == [(0.0, 10.0), (30.0, 100.0)]

    def test_min_length_is_configurable(self):
        silences = [(10.0, 20.0), (20.5, 30.0)]
        assert keep_intervals_between_silences(100.0, silences, min_length=0.4) == [
            (0.0, 10.0), (20.0, 20.5), (30.0, 100.0),
        ]

    def test_unsorted_and_overlapping_silences(self):
        silences = [(50.0, 60.0), (10.0, 20.0), (15.0, 55.0)]  # вместе: 10..60
        assert keep_intervals_between_silences(100.0, silences) == [(0.0, 10.0), (60.0, 100.0)]

    def test_silence_nested_inside_another_silence(self):
        # короткая пауза целиком внутри длинной не должна «вернуть» звук раньше конца длинной
        assert keep_intervals_between_silences(100.0, [(10.0, 50.0), (20.0, 30.0)]) == [(0.0, 10.0), (50.0, 100.0)]

    def test_everything_is_silence(self):
        assert keep_intervals_between_silences(100.0, [(0.0, 100.0)]) == []

    def test_recording_shorter_than_min_length(self):
        assert keep_intervals_between_silences(0.5, []) == []


class TestSnapIntervalsToKeyframes:
    KEYFRAMES = [0.0, 10.0, 20.0, 30.0, 40.0]

    def snap(self, intervals, left_pad=0.0, duration=45.0, keyframes=None):
        return snap_intervals_to_keyframes(
            intervals, self.KEYFRAMES if keyframes is None else keyframes, left_pad, duration,
        )

    def test_no_keyframes_returns_copy_unchanged(self):
        intervals = [[1.0, 2.0]]
        result = snap_intervals_to_keyframes(intervals, [], 1.0, 50.0)
        assert result == [[1.0, 2.0]] and result is not intervals and result[0] is not intervals[0]

    def test_left_goes_to_previous_keyframe_right_to_next(self):
        assert self.snap([[12.0, 27.0]]) == [[10.0, 30.0]]

    def test_boundary_exactly_on_keyframe_stays(self):
        assert self.snap([[10.0, 30.0]]) == [[10.0, 30.0]]

    def test_left_pad_takes_one_more_keyframe_when_margin_is_too_small(self):
        # старт 10.5, ближайший слева 10.0: запас 0.5 < 1 -> ещё один кадр левее
        assert self.snap([[10.5, 25.0]], left_pad=1.0) == [[0.0, 30.0]]

    def test_left_pad_satisfied_no_extra_step(self):
        assert self.snap([[12.0, 25.0]], left_pad=1.0) == [[10.0, 30.0]]

    def test_left_pad_only_one_step_not_a_loop(self):
        # огромный отступ всё равно даёт ровно один шаг назад
        assert self.snap([[32.0, 35.0]], left_pad=100.0) == [[20.0, 40.0]]

    def test_left_pad_does_not_apply_at_first_keyframe(self):
        assert self.snap([[0.5, 5.0]], left_pad=1.0) == [[0.0, 10.0]]

    def test_start_before_first_keyframe_becomes_zero(self):
        assert self.snap([[2.0, 12.0]], keyframes=[5.0, 15.0]) == [[0.0, 15.0]]

    def test_right_boundary_after_last_keyframe_is_duration(self):
        assert self.snap([[32.0, 44.0]], duration=45.0) == [[30.0, 45.0]]

    def test_overlapping_after_snapping_are_merged(self):
        # [12, 18] -> [10, 20]; [22, 28] -> [20, 30]: касаются в 20 -> один фрагмент
        assert self.snap([[12.0, 18.0], [22.0, 28.0]]) == [[10.0, 30.0]]

    def test_separate_intervals_stay_separate(self):
        assert self.snap([[2.0, 8.0], [32.0, 38.0]]) == [[0.0, 10.0], [30.0, 40.0]]

    def test_result_is_sorted_by_start(self):
        assert self.snap([[32.0, 38.0], [2.0, 8.0]]) == [[0.0, 10.0], [30.0, 40.0]]

    def test_input_is_not_mutated(self):
        intervals = [[12.0, 27.0]]
        self.snap(intervals)
        assert intervals == [[12.0, 27.0]]

    def test_empty_intervals(self):
        assert self.snap([]) == []


class TestEditInterval:
    """Ручная правка полей «Начало»/«Конец» в диалоге фрагментов."""

    DURATION = 12200.7   # 03:23:20.7 — в поле «Конец» показывается как 03:23:21

    def test_editing_only_the_start_keeps_the_end_exact(self):
        # ошибка из жизни: правка начала давала «Фрагмент должен быть в пределах…», потому что
        # нетронутый конец разбирался как 03:23:21 = 12201 с > длительности 12200.7 с
        result = edit_interval("00:10:00", "03:23:21", [0.0, self.DURATION], 0.0, self.DURATION)
        assert result == [600.0, self.DURATION]

    def test_typing_the_shown_end_explicitly_snaps_to_the_video_end(self):
        assert edit_interval("00:00:00", "03:23:21", [600.0, 900.0], 0.0, self.DURATION) == [0.0, self.DURATION]

    def test_start_equal_to_the_shown_boundary_of_the_previous_fragment_snaps_to_it(self):
        # предыдущий фрагмент кончается на 100.4 (показан как 01:40), предел — 100.4, а не 100
        assert edit_interval("00:01:40", "00:05:00", [150.0, 300.0], 100.4, 500.0) == [100.4, 300.0]

    def test_end_equal_to_the_shown_start_of_the_next_fragment_snaps_to_it(self):
        assert edit_interval("00:00:10", "00:08:21", [10.0, 300.0], 0.0, 500.6) == [10.0, 500.6]

    def test_unchanged_fields_keep_fractions(self):
        assert edit_interval("00:10:00", "00:20:00", [599.7, 1200.2], 0.0, 5000.0) == [599.7, 1200.2]

    def test_ordinary_edit(self):
        assert edit_interval("00:01:00", "00:02:30", [10.0, 20.0], 0.0, 500.0) == [60.0, 150.0]

    def test_end_beyond_the_limit_is_rejected_with_the_limits_in_the_message(self):
        with pytest.raises(IntervalRangeError, match="00:00:00–03:23:21"):
            edit_interval("00:00:10", "03:23:22", [0.0, self.DURATION], 0.0, self.DURATION)

    def test_start_before_the_limit_is_rejected(self):
        with pytest.raises(IntervalRangeError):
            edit_interval("00:01:00", "00:05:00", [150.0, 300.0], 100.4, 500.0)

    @pytest.mark.parametrize("start, end", [("00:05:00", "00:05:00"), ("00:06:00", "00:05:00")])
    def test_start_must_be_less_than_end(self, start, end):
        with pytest.raises(IntervalRangeError, match="начало должно быть меньше конца"):
            edit_interval(start, end, [10.0, 20.0], 0.0, 1000.0)

    def test_wrong_time_format_is_a_plain_value_error_not_a_range_error(self):
        with pytest.raises(ValueError) as info:
            edit_interval("abc", "00:05:00", [10.0, 20.0], 0.0, 1000.0)
        assert not isinstance(info.value, IntervalRangeError)
        with pytest.raises(ValueError, match="ожидается HH:mm:ss") as info:
            edit_interval("12:30", "00:05:00", [10.0, 20.0], 0.0, 1000.0)
        assert not isinstance(info.value, IntervalRangeError)

    def test_range_error_is_a_value_error(self):
        assert issubclass(IntervalRangeError, ValueError)   # вызывающий код может ловить оба одним `except ValueError`

    def test_input_list_is_not_mutated(self):
        current = [10.0, 20.0]
        edit_interval("00:00:15", "00:00:19", current, 0.0, 100.0)
        assert current == [10.0, 20.0]


class TestAddLeftPad:
    """Запас до начала речи для аудио (у аудио нет ключевых кадров)."""

    def test_start_moves_back_end_stays(self):
        assert add_left_pad([[100.0, 200.0]], 1.0) == [[99.0, 200.0]]

    def test_zero_pad_changes_nothing(self):
        assert add_left_pad([[100.0, 200.0], [300.0, 400.0]], 0.0) == [[100.0, 200.0], [300.0, 400.0]]

    def test_never_goes_below_zero(self):
        assert add_left_pad([[0.4, 10.0]], 1.0) == [[0.0, 10.0]]

    def test_pad_that_eats_the_gap_merges_neighbours(self):
        assert add_left_pad([[10.0, 20.0], [20.5, 30.0]], 1.0) == [[9.0, 30.0]]

    def test_touching_or_overlapping_intervals_merge(self):
        assert add_left_pad([[10.0, 20.0], [20.0, 30.0]], 0.0) == [[10.0, 30.0]]

    def test_gap_larger_than_the_pad_is_kept(self):
        assert add_left_pad([[10.0, 20.0], [25.0, 30.0]], 1.0) == [[9.0, 20.0], [24.0, 30.0]]

    def test_result_is_sorted_and_input_is_not_mutated(self):
        intervals = [[50.0, 60.0], [10.0, 20.0]]
        assert add_left_pad(intervals, 2.0) == [[8.0, 20.0], [48.0, 60.0]]
        assert intervals == [[50.0, 60.0], [10.0, 20.0]]

    def test_empty(self):
        assert add_left_pad([], 1.0) == []

