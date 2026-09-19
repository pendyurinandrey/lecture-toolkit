"""Время в удобном для человека виде: «ЧЧ:ММ:СС» и длительности для журнала.

Округление зависит от смысла величины, поэтому функций две:
- seconds_to_hhmmss — до БЛИЖАЙШЕЙ секунды: границы фрагментов для нарезки и длительности.
  Округление вниз обрезало бы до секунды звука в конце фрагмента; до ближайшей ошибка
  симметрична. Ровно .5 округляется вверх (а не «до чётного», как встроенный round).
- format_timestamp — вниз: метка «когда начинается речь» в транскрипте. Метка никогда не
  указывает позже начала фразы, так что перемотка по ней не пропустит начало слова.
"""

import math


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _hhmmss(total_seconds: int) -> str:
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def hhmmss_to_seconds(value: str) -> float:
    """«ЧЧ:ММ:СС» (секунды могут быть дробными) -> секунды. ValueError при неверном формате."""
    parts = value.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Неверный формат времени (ожидается HH:mm:ss): {value!r}")
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_displayed_time(text: str, *shown_values: float) -> float:
    """Разбирает время из поля ввода, в котором раньше было показано одно из shown_values.

    Поле показывает значения с округлением до секунды: «03:23:21» на экране может означать
    12200.7 с. Если введённое время — это то, как показано одно из shown_values, возвращается
    само значение с дробной частью (приоритет у более раннего в списке). Иначе нетронутое поле
    после разбора «съезжало» бы на долю секунды и, например, выходило за конец видео.
    ValueError при неверном формате."""
    parsed = hhmmss_to_seconds(text)
    for value in shown_values:
        if parsed == _round_half_up(value):
            return value
    return parsed


def seconds_to_hhmmss(seconds: float) -> str:
    """Секунды -> «ЧЧ:ММ:СС», округление до ближайшей секунды (см. описание модуля)."""
    return _hhmmss(_round_half_up(seconds))


def format_timestamp(seconds: float) -> str:
    """Секунды -> «ЧЧ:ММ:СС» с округлением ВНИЗ: метка времени в транскрипте."""
    return _hhmmss(int(seconds))


def format_duration(seconds: float) -> str:
    """Длительность этапа для журнала: «12 с», «3 мин 05 с», «1 ч 02 мин»."""
    total = _round_half_up(seconds)
    if total < 60:
        return f"{total} с"
    if total < 3600:
        minutes, secs = divmod(total, 60)
        return f"{minutes} мин {secs:02d} с"
    hours, minutes = divmod(_round_half_up(total / 60), 60)
    return f"{hours} ч {minutes:02d} мин"
