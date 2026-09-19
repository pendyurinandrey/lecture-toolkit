"""Чистая логика над интервалами фрагментов — без запуска внешних
процессов и чтения файлов (списки времён приходят из ffmpeg.py).

Используется автоопределением границ полезных фрагментов видео
(FragmentsPlayerDialog): сначала keep_intervals_between_silences() строит
интервалы между паузами, затем snap_intervals_to_keyframes() подгоняет их
под реальные точки нарезки.
"""

import bisect


def keep_intervals_between_silences(duration: float, silences: list, min_length: float = 1.0) -> list:
    """Возвращает интервалы МЕЖДУ паузами — то, что нужно сохранить, если
    каждую найденную паузу вырезать. Интервалы короче min_length секунд
    отбрасываются: если две паузы найдены близко друг к другу, между ними
    может остаться доля секунды звука (например, короткий кашель) — это не
    полезный фрагмент, а огрызок, с которым потом неудобно работать (в том
    числе почти невозможно попасть по нему кликом на таймлайне). Включает
    крайние интервалы [0, первая_пауза] и [последняя_пауза, duration],
    которые сами по себе паузами не являются (например, начало/конец
    записи до/после лекции) — предполагается, что их подрежут вручную.

    Границы здесь НЕ подгоняются под реальные точки нарезки (ключевые
    кадры) — этим занимается отдельно snap_intervals_to_keyframes,
    вызываемая следом в FragmentsPlayerDialog."""
    silences = sorted(silences)
    intervals = []
    cursor = 0.0
    for s_start, s_end in silences:
        if s_start - cursor >= min_length:
            intervals.append((cursor, s_start))
        cursor = max(cursor, s_end)
    if duration - cursor >= min_length:
        intervals.append((cursor, duration))
    return intervals


def snap_intervals_to_keyframes(
    intervals: list, keyframes: list, left_pad: float, duration: float,
) -> list:
    """Подгоняет границы фрагментов под реальные точки нарезки.

    fork_join режет видео через ffmpeg "-ss ... -c copy" (без перекодирования)
    — такая нарезка физически не может начаться иначе, чем с ключевого кадра:
    запрошенная точка молча "съезжает" на ближайший предыдущий keyframe.
    Чтобы preview в FragmentsPlayerDialog показывал то же самое, что получится
    на выходе, границы подгоняются под реальные ключевые кадры ДО показа
    пользователю:

    - левая граница — ближайший ключевой кадр СЛЕВА от неё. Если запас до
      найденной исходной границы меньше left_pad секунд — берётся ещё один
      кадр левее (ровно один шаг, не цикл: на редких GOP этого может не
      хватить впритык до left_pad, это осознанный компромисс). Если слева
      от границы кадров вообще нет (начало видео) — 00:00:00.
    - правая граница — ближайший ключевой кадр СПРАВА от неё, либо конец
      видео, если такого кадра нет.

    Если после этого границы двух соседних фрагментов пересеклись (правая
    "уехала" вправо, левая следующего — влево, к одному и тому же кадру
    или дальше) — такие фрагменты сливаются в один."""
    if not keyframes:
        return [list(iv) for iv in intervals]

    snapped = []
    for start, end in intervals:
        idx = bisect.bisect_right(keyframes, start) - 1
        if idx < 0:
            new_start = 0.0
        else:
            new_start = keyframes[idx]
            if start - new_start < left_pad and idx > 0:
                new_start = keyframes[idx - 1]

        idx_end = bisect.bisect_left(keyframes, end)
        new_end = keyframes[idx_end] if idx_end < len(keyframes) else duration

        snapped.append([new_start, new_end])

    snapped.sort(key=lambda iv: iv[0])
    merged = []
    for start, end in snapped:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
