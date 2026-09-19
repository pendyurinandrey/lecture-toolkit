"""Время: «ЧЧ:ММ:СС» (два способа округления) и длительности для журнала."""

import random

import pytest

from common.timecode import (
    format_duration,
    format_timestamp,
    hhmmss_to_seconds,
    parse_displayed_time,
    seconds_to_hhmmss,
)


class TestHhmmssToSeconds:
    def test_parses(self):
        assert hhmmss_to_seconds("01:02:03") == 3723
        assert hhmmss_to_seconds(" 00:00:30.5 ") == 30.5
        assert hhmmss_to_seconds("100:00:00") == 360000

    @pytest.mark.parametrize("value", ["12:30", "1:2:3:4", ""])
    def test_wrong_number_of_parts_is_rejected_with_an_explanation(self, value):
        with pytest.raises(ValueError, match="ожидается HH:mm:ss"):
            hhmmss_to_seconds(value)

    def test_non_numeric_parts_are_rejected(self):
        with pytest.raises(ValueError):
            hhmmss_to_seconds("aa:bb:cc")


class TestSecondsToHhmmss:
    """До ближайшей секунды: границы фрагментов и длительности."""

    @pytest.mark.parametrize("seconds, expected", [
        (0, "00:00:00"), (0.4, "00:00:00"), (0.6, "00:00:01"), (3723, "01:02:03"),
        (59.4, "00:00:59"), (59.6, "00:01:00"), (11389.66, "03:09:50"), (360000, "100:00:00"),
    ])
    def test_rounds_to_the_nearest_second(self, seconds, expected):
        assert seconds_to_hhmmss(seconds) == expected

    @pytest.mark.parametrize("seconds, expected", [(0.5, "00:00:01"), (1.5, "00:00:02"), (2.5, "00:00:03"), (59.5, "00:01:00")])
    def test_exact_half_rounds_up(self, seconds, expected):
        # встроенный round() округлил бы 0.5 и 2.5 «до чётного» вниз — здесь предсказуемо вверх
        assert seconds_to_hhmmss(seconds) == expected

    def test_error_is_at_most_half_a_second_and_symmetric(self):
        random.seed(3)
        errors = [hhmmss_to_seconds(seconds_to_hhmmss(x)) - x for x in (random.uniform(0, 20000) for _ in range(2000))]
        assert max(abs(e) for e in errors) <= 0.5 and min(errors) < 0 < max(errors)


class TestFormatTimestamp:
    """Вниз: метка в транскрипте никогда не позже начала речи."""

    @pytest.mark.parametrize("seconds, expected", [
        (0, "00:00:00"), (59.9, "00:00:59"), (61, "00:01:01"), (3661.5, "01:01:01"), (11389.66, "03:09:49"),
    ])
    def test_truncates_fractions(self, seconds, expected):
        assert format_timestamp(seconds) == expected

    def test_never_later_than_the_moment(self):
        random.seed(4)
        for x in (random.uniform(0, 20000) for _ in range(2000)):
            assert hhmmss_to_seconds(format_timestamp(x)) <= x

    def test_differs_from_nearest_rounding_only_by_policy(self):
        # одна и та же длительность: в диалоге и журнале — до ближайшей, метка в транскрипте — вниз
        assert seconds_to_hhmmss(11389.66) == "03:09:50"
        assert format_timestamp(11389.66) == "03:09:49"


class TestFormatDuration:
    @pytest.mark.parametrize("seconds, expected", [
        (0, "0 с"), (0.4, "0 с"), (12, "12 с"), (59.4, "59 с"),
        (59.5, "1 мин 00 с"), (60, "1 мин 00 с"), (185, "3 мин 05 с"), (3599, "59 мин 59 с"),
        (3599.6, "1 ч 00 мин"), (3600, "1 ч 00 мин"), (3720, "1 ч 02 мин"),
        (4499, "1 ч 15 мин"),   # 1 ч 14 мин 59 с — до ближайшей минуты
        (12200.7, "3 ч 23 мин"),
    ])
    def test_formats(self, seconds, expected):
        assert format_duration(seconds) == expected

    def test_short_stages_are_not_shown_as_fractions_of_a_minute(self):
        # раньше: «Fork/Join завершён за 0.2 мин.»
        assert format_duration(12.3) == "12 с"


class TestParseDisplayedTime:
    """Поле показывает значение с округлением; нетронутое поле не должно «съезжать»."""

    def test_text_equal_to_the_displayed_value_returns_the_exact_value(self):
        assert seconds_to_hhmmss(12200.7) == "03:23:21"
        assert parse_displayed_time("03:23:21", 12200.7) == 12200.7    # а не 12201.0

    def test_other_text_is_parsed_as_typed(self):
        assert parse_displayed_time("00:10:00", 12200.7) == 600.0
        assert parse_displayed_time("00:00:30.5", 12200.7) == 30.5

    def test_text_may_match_any_of_the_known_values(self):
        assert parse_displayed_time("00:03:21", 0.0, 200.6) == 200.6    # 200.6 показано как 03:21

    def test_earlier_value_has_priority_when_several_match(self):
        assert parse_displayed_time("00:01:40", 100.2, 99.7) == 100.2

    def test_equivalent_spelling_of_the_same_time_matches_too(self):
        assert parse_displayed_time("0:10:00", 599.7) == 599.7
        assert parse_displayed_time(" 00:10:00 ", 599.7) == 599.7

    def test_no_known_values(self):
        assert parse_displayed_time("00:01:05") == 65.0

    @pytest.mark.parametrize("text", ["12:30", "", "aa:bb:cc"])
    def test_wrong_format_is_rejected(self, text):
        with pytest.raises(ValueError):
            parse_displayed_time(text, 1.0)

    def test_what_is_shown_always_parses_back_to_the_value(self):
        random.seed(5)
        for x in (random.uniform(0, 20000) for _ in range(2000)):
            assert parse_displayed_time(seconds_to_hhmmss(x), x) == x

