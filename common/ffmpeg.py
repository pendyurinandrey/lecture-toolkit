"""Обёртки над ffmpeg/ffprobe — единственное место в common, где
запускаются внешние программы.

Любая ошибка (программа не найдена, ненулевой код возврата) превращается в
FFmpegError с понятным текстом; вызывающий код не должен разбирать
CalledProcessError/FileNotFoundError.

Чистая логика над интервалами (без запуска процессов) лежит в intervals.py.
"""

import re
import subprocess
from pathlib import Path

SILENCE_NOISE_DB = "-30dB"
WAV_SAMPLE_RATE = 16000
_ERROR_TAIL_LINES = 15


class FFmpegError(RuntimeError):
    """ffmpeg/ffprobe не найден или завершился с ошибкой."""


def _run(command: list, action: str) -> subprocess.CompletedProcess:
    """Запускает команду, возвращает результат с захваченными stdout/stderr.
    Бросает FFmpegError, если программа не найдена или код возврата не нулевой."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as e:
        raise FFmpegError(
            f"Не найден {command[0]} в PATH. Установите ffmpeg, например:\n    brew install ffmpeg"
        ) from e
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-_ERROR_TAIL_LINES:])
        raise FFmpegError(f"{action}: {command[0]} завершился с ошибкой:\n{tail}")
    return result


def get_media_duration(path) -> float:
    """Длительность медиафайла (аудио или видео) в секундах."""
    result = _run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        f"Не удалось определить длительность {path}",
    )
    try:
        return float(result.stdout.strip())
    except ValueError as e:
        raise FFmpegError(f"Не удалось определить длительность {path}: ffprobe вернул {result.stdout.strip()!r}") from e


def detect_silences(path, min_duration: float = 0.5, noise_db: str = SILENCE_NOISE_DB) -> list:
    """Возвращает список (start, end) пауз длиннее min_duration секунд.

    Порог громкости, не нейросеть — быстрый разовый проход по файлу.
    Работает по аудиодорожке — годится и для аудио, и для видео (декодирует
    только звук, "-vn" явно исключает видео, иначе ffmpeg зря декодировал
    бы и его перед сбросом в null)."""
    result = _run(
        ["ffmpeg", "-i", str(path), "-vn",
         "-af", f"silencedetect=noise={noise_db}:d={min_duration}",
         "-f", "null", "-"],
        f"Не удалось найти паузы в {path}",
    )
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", result.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", result.stderr)]
    return list(zip(starts, ends))


def get_keyframe_timestamps(path) -> list:
    """Отсортированный список времён (в секундах) всех ключевых кадров
    первого видеопотока файла.

    Кадры достаются через ffprobe на уровне пакетов контейнера (флаг "K"),
    без декодирования — иначе на многочасовом видео это заняло бы на
    порядок больше времени."""
    result = _run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", str(path)],
        f"Не удалось получить ключевые кадры {path}",
    )
    timestamps = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pts_str, flags = line.split(",", 1)
        if "K" in flags:
            timestamps.append(float(pts_str))
    timestamps.sort()
    return timestamps


def convert_to_wav_16k_mono(src_path, dst_path) -> Path:
    """Конвертирует любой аудио/видео файл в WAV 16 кГц моно PCM16.

    Такой файл занимает около 110 МБ на час записи. Он нужен и GigaAM, и
    диаризации: на несжатом входе оба работают с заметно меньшим пиком памяти,
    чем на M4A."""
    dst_path = Path(dst_path)
    _run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src_path),
         "-vn", "-ac", "1", "-ar", str(WAV_SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst_path)],
        f"Не удалось конвертировать {src_path} в WAV",
    )
    return dst_path
