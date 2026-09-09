#!/usr/bin/env python3
"""Обнаружение пауз (тишины) в аудио/видео через ffmpeg silencedetect —
порог громкости, не нейросеть, быстрый разовый проход по файлу.

Используется и speech_to_text_gigaam (нарезка длинной лекции на куски
рядом с паузами), и pipeline_ui (автоопределение границ полезных
фрагментов видео — см. FragmentsPlayerDialog).
"""

import re
import subprocess

SILENCE_NOISE_DB = "-30dB"


def get_media_duration(path) -> float:
    """Длительность медиафайла (аудио или видео) в секундах."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def detect_silences(path, min_duration: float = 0.5, noise_db: str = SILENCE_NOISE_DB):
    """Возвращает список (start, end) пауз длиннее min_duration секунд.

    Работает по аудиодорожке файла — годится и для аудио, и для видео
    (декодирует только звук, "-vn" явно исключает видео, иначе ffmpeg зря
    декодировал бы и его перед сбросом в null)."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-vn",
         "-af", f"silencedetect=noise={noise_db}:d={min_duration}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    log = result.stderr
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", log)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", log)]
    return list(zip(starts, ends))


def keep_intervals_between_silences(
    duration: float, silences: list, min_length: float = 1.0, left_pad: float = 3.0,
) -> list:
    """Возвращает интервалы МЕЖДУ паузами — то, что нужно сохранить, если
    каждую найденную паузу вырезать. Интервалы короче min_length секунд
    отбрасываются: если две паузы найдены близко друг к другу, между ними
    может остаться доля секунды звука (например, короткий кашель) — это не
    полезный фрагмент, а огрызок, с которым потом неудобно работать (в том
    числе почти невозможно попасть по нему кликом на таймлайне). Включает
    крайние интервалы [0, первая_пауза] и [последняя_пауза, duration],
    которые сами по себе паузами не являются (например, начало/конец
    записи до/после лекции) — предполагается, что их подрежут вручную.

    К началу каждого интервала (кроме случая, когда оно и так упирается в
    0 или в конец предыдущего сохранённого интервала) добавляется left_pad
    секунд назад — граница silence_end от ffmpeg отмечает момент, когда
    звук возобновился, но начало самого первого слова может быть буквально
    впритык к этой границе, поэтому небольшой запас слева подстраховывает
    от потери начала фразы."""
    silences = sorted(silences)
    raw_intervals = []
    cursor = 0.0
    for s_start, s_end in silences:
        if s_start - cursor >= min_length:
            raw_intervals.append((cursor, s_start))
        cursor = max(cursor, s_end)
    if duration - cursor >= min_length:
        raw_intervals.append((cursor, duration))

    intervals = []
    prev_end = 0.0
    for start, end in raw_intervals:
        padded_start = max(prev_end, start - left_pad)
        intervals.append((padded_start, end))
        prev_end = end
    return intervals
