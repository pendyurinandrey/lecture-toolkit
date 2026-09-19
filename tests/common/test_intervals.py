"""Чистая логика интервалов: ни ffmpeg, ни файлов не нужно."""

from common.intervals import keep_intervals_between_silences, snap_intervals_to_keyframes


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
