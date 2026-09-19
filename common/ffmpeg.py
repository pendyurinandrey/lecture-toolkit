"""Обёртки над ffmpeg/ffprobe — единственное место в common, где
запускаются внешние программы.

Любая ошибка (программа не найдена, ненулевой код возврата) превращается в
FFmpegError с понятным текстом; вызывающий код не должен разбирать
CalledProcessError/FileNotFoundError.

Чистая логика над интервалами (без запуска процессов) лежит в intervals.py.
"""

import re
import shutil
import subprocess
from pathlib import Path

SILENCE_NOISE_DB = "-30dB"
WAV_SAMPLE_RATE = 16000
_ERROR_TAIL_LINES = 15
# «ffmpeg version 9.0.1», «ffmpeg version n7.1», «ffmpeg version 4.4.2-0ubuntu0.22.04.1».
# Точка после мажорной версии обязательна: git-сборки («N-118000-g…», «2024-05-06-git-…») версию не несут.
_VERSION_RE = re.compile(r"ffmpeg version n?(\d+)\.\d")


class FFmpegError(RuntimeError):
    """ffmpeg/ffprobe не найден или завершился с ошибкой."""


def _not_found(tool: str) -> FFmpegError:
    return FFmpegError(f"Не найден {tool} в PATH. Установите ffmpeg, например:\n    brew install ffmpeg")


def check_ffmpeg() -> None:
    """Проверяет, что ffmpeg есть в PATH. Бросает FFmpegError с подсказкой по установке."""
    if shutil.which("ffmpeg") is None:
        raise _not_found("ffmpeg")


def _run(command: list, action: str, timeout: float = None) -> subprocess.CompletedProcess:
    """Запускает команду, возвращает результат с захваченными stdout/stderr.
    Бросает FFmpegError, если программа не найдена, не уложилась в timeout секунд
    или код возврата не нулевой."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
    except FileNotFoundError as e:
        raise _not_found(command[0]) from e
    except subprocess.TimeoutExpired as e:
        raise FFmpegError(f"{action}: {command[0]} не ответил за {timeout} с") from e
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-_ERROR_TAIL_LINES:])
        raise FFmpegError(f"{action}: {command[0]} завершился с ошибкой:\n{tail}")
    return result


def run_ffmpeg(args: list, action: str = "Не удалось выполнить команду ffmpeg") -> None:
    """Запускает ffmpeg с переданными аргументами (перезаписывает выходной файл,
    показывает только ошибки). action — что делали: попадёт в текст ошибки."""
    _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], action)


def get_ffmpeg_major_version() -> int:
    """Мажорная версия установленного ffmpeg (9 для «ffmpeg version 9.0.1»).
    Бросает FFmpegError, если ffmpeg не найден или версию не удалось разобрать
    (например, git-сборка)."""
    action = "Не удалось определить версию ffmpeg"
    result = _run(["ffmpeg", "-version"], action, timeout=10)
    match = _VERSION_RE.search(result.stdout)
    if match is None:
        first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        raise FFmpegError(f"{action}: {first_line!r}")
    return int(match.group(1))


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
    run_ffmpeg(
        ["-nostdin", "-i", str(src_path),
         "-vn", "-ac", "1", "-ar", str(WAV_SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst_path)],
        f"Не удалось конвертировать {src_path} в WAV",
    )
    return dst_path
