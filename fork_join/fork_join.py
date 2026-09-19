"""
Нарезает фрагменты из MP4-файлов по конфигурации, склеивает их в единый
MP4 и дополнительно сохраняет звуковую дорожку итогового файла в M4A —
оба шага копированием потоков, без перекодирования, без потери качества.

Формат конфигурации: см. README.md

Отдельного запуска из командной строки нет: модуль используется из
pipeline_ui.py. Функции validate_config()/process_config() бросают
ConfigError/FFmpegError, что позволяет UI показать ошибку пользователю.
"""

import os
import tempfile

from common.ffmpeg import check_ffmpeg, run_ffmpeg
from common.timecode import hhmmss_to_seconds


class ConfigError(ValueError):
    """Некорректная конфигурация fork_join."""


def validate_config(config: dict) -> None:
    """Проверяет структуру конфигурации. Бросает ConfigError при ошибке."""
    segments = config.get("segments")
    if not isinstance(segments, list) or not (1 <= len(segments) <= 20):
        raise ConfigError('Поле "segments" должно быть массивом из 1..20 объектов')

    for i, segment in enumerate(segments):
        path = segment.get("path")
        if not path or not isinstance(path, str):
            raise ConfigError(f'segments[{i}]: отсутствует или некорректно поле "path"')
        if not os.path.isfile(path):
            raise ConfigError(f'segments[{i}]: файл не найден: {path}')

        fragments = segment.get("fragments")
        if not isinstance(fragments, list) or not (1 <= len(fragments) <= 20):
            raise ConfigError(f'segments[{i}]: "fragments" должен быть массивом из 1..20 объектов')

        for j, fragment in enumerate(fragments):
            start = fragment.get("start")
            end = fragment.get("end")
            if not start or not end:
                raise ConfigError(f'segments[{i}].fragments[{j}]: нужны поля "start" и "end"')
            try:
                start_s = hhmmss_to_seconds(start)
                end_s = hhmmss_to_seconds(end)
            except ValueError as e:
                raise ConfigError(f'segments[{i}].fragments[{j}]: {e}') from e
            if end_s <= start_s:
                raise ConfigError(
                    f'segments[{i}].fragments[{j}]: "end" ({end}) должен быть больше "start" ({start})'
                )

    output = config.get("output", {})
    video_path = output.get("videoPath")
    audio_path = output.get("audioPath")
    if not video_path or not audio_path:
        raise ConfigError('Поле "output" должно содержать "videoPath" и "audioPath"')


def cut_fragment(source_path: str, start: str, end: str, out_path: str) -> None:
    duration = hhmmss_to_seconds(end) - hhmmss_to_seconds(start)
    run_ffmpeg(
        [
            "-ss", start,
            "-i", source_path,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            out_path,
        ],
        f"Не удалось вырезать фрагмент {start}–{end} из {source_path}",
    )


def escape_concat_path(path: str) -> str:
    # Формат concat-демультиплексора ffmpeg: file 'путь', одинарная
    # кавычка внутри пути экранируется как '\''
    return path.replace("'", "'\\''")


def build_video(segments: list, tmp_dir: str, video_path: str, log=lambda msg: None) -> None:
    fragment_paths = []
    for i, segment in enumerate(segments):
        source_path = segment["path"]
        for j, fragment in enumerate(segment["fragments"]):
            log(f"Нарезаю фрагмент {i + 1}.{j + 1} из {os.path.basename(source_path)}...")
            out_path = os.path.join(tmp_dir, f"fragment_{i:02d}_{j:02d}.mp4")
            cut_fragment(source_path, fragment["start"], fragment["end"], out_path)
            fragment_paths.append(out_path)

    concat_list_path = os.path.join(tmp_dir, "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for p in fragment_paths:
            f.write(f"file '{escape_concat_path(p)}'\n")

    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    log("Склеиваю фрагменты в итоговое видео...")
    run_ffmpeg(
        [
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            video_path,
        ],
        "Не удалось склеить фрагменты в итоговое видео",
    )


def extract_audio(video_path: str, audio_path: str) -> None:
    # Копируем AAC-поток как есть (без перекодирования) в M4A-контейнер —
    # исходный звук в MP4 уже сжат AAC, повторное сжатие в другой lossy-
    # формат (например MP3) даёт вторую генерацию потерь без необходимости.
    os.makedirs(os.path.dirname(audio_path) or ".", exist_ok=True)
    run_ffmpeg(
        [
            "-i", video_path,
            "-vn",
            "-acodec", "copy",
            audio_path,
        ],
        f"Не удалось извлечь звуковую дорожку из {video_path}",
    )


def process_config(config: dict, log=print) -> None:
    """Выполняет нарезку/склейку видео и извлечение аудио по готовой конфигурации.

    Бросает ConfigError при некорректной конфигурации и FFmpegError при
    ошибках ffmpeg (в т.ч. если ffmpeg не установлен).
    """
    validate_config(config)
    check_ffmpeg()

    video_path = config["output"]["videoPath"]
    audio_path = config["output"]["audioPath"]

    with tempfile.TemporaryDirectory(prefix="fork_join_") as tmp_dir:
        log("Нарезаю и склеиваю фрагменты...")
        build_video(config["segments"], tmp_dir, video_path, log=log)
        log(f"Видео сохранено: {video_path}")

        log("Извлекаю звуковую дорожку...")
        extract_audio(video_path, audio_path)
        log(f"Аудио сохранено: {audio_path}")
