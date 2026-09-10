#!/usr/bin/env python3
"""Поиск ключевых кадров видео и привязка к ним границ фрагментов.

fork_join режет видео через ffmpeg "-ss ... -c copy" (без перекодирования)
— такая нарезка физически не может начаться иначе, чем с ключевого кадра:
запрошенная точка молча "съезжает" на ближайший предыдущий keyframe.
Чтобы preview в FragmentsPlayerDialog показывал то же самое, что получится
на выходе, границы, найденные автоопределением пауз, подгоняются под
реальные ключевые кадры ДО показа пользователю — см.
snap_intervals_to_keyframes().

Список кадров достаётся через ffprobe на уровне пакетов контейнера (флаг
"K"), без декодирования — иначе на многочасовом видео это заняло бы
на порядок больше времени.
"""

import bisect
import subprocess


def get_keyframe_timestamps(path) -> list:
    """Отсортированный список времён (в секундах) всех ключевых кадров
    первого видеопотока файла."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    timestamps = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pts_str, flags = line.split(",", 1)
        if "K" in flags:
            timestamps.append(float(pts_str))
    timestamps.sort()
    return timestamps


def snap_intervals_to_keyframes(
    intervals: list, keyframes: list, left_pad: float, duration: float,
) -> list:
    """Подгоняет границы фрагментов под реальные точки нарезки:

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
