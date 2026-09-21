"""
Нарезает фрагменты из MP4-файлов по конфигурации, склеивает их в единый
MP4 и дополнительно сохраняет звуковую дорожку итогового файла в M4A —
оба шага копированием потоков, без перекодирования, без потери качества.

Для записей без видео (mediaType = "audio") то же самое делается со звуком:
фрагменты M4A или MP3 нарезаются и склеиваются в один файл того же формата.

Перед склейкой проверяется, что параметры всех файлов одинаковы (см.
common/media_info.py): иначе ffmpeg склеил бы их молча и испортил результат.

Формат конфигурации: см. README.md

Отдельного запуска из командной строки нет: модуль используется из
pipeline_ui.py. Функции validate_config()/process_config() бросают
ConfigError/FFmpegError, что позволяет UI показать ошибку пользователю.
"""

import os
import tempfile
from pathlib import Path

from common.ffmpeg import check_ffmpeg, get_media_duration, probe_media, run_ffmpeg
from common.media_info import AUDIO_EXTENSIONS, audio_extension, incompatibilities
from common.timecode import hhmmss_to_seconds

MEDIA_TYPES = ("video", "audio")
# Время в конфигурации — целые секунды, а длительность файла дробная: конец «на всю запись»
# 5296.36 с записывается как 01:28:16 и отрезал бы хвост. Конец, отстоящий от конца файла не
# больше чем на это число секунд (меньше точности «ЧЧ:ММ:СС»), означает «до конца файла».
END_OF_FILE_TOLERANCE = 0.5


class ConfigError(ValueError):
    """Некорректная конфигурация fork_join."""


def media_type_of(config: dict) -> str:
    """"video" (по умолчанию) или "audio"."""
    return config.get("mediaType", "video")


def validate_config(config: dict) -> None:
    """Проверяет структуру конфигурации. Бросает ConfigError при ошибке."""
    media_type = media_type_of(config)
    if media_type not in MEDIA_TYPES:
        raise ConfigError(f'Поле "mediaType" должно быть "video" или "audio", а не {media_type!r}')

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
    if media_type == "audio":
        if not audio_path:
            raise ConfigError('Для mediaType "audio" поле "output" должно содержать "audioPath" ("videoPath" не нужен)')
        if Path(audio_path).suffix.lower() not in AUDIO_EXTENSIONS:
            raise ConfigError(f'"audioPath" должен заканчиваться на {" или ".join(AUDIO_EXTENSIONS)}: {audio_path}')
    elif not video_path or not audio_path:
        raise ConfigError('Поле "output" должно содержать "videoPath" и "audioPath"')


def explain_incompatibility(reference_path: str, other_path: str):
    """Почему other_path нельзя склеивать с reference_path без перекодирования (текст для
    пользователя) или None, если можно. Нужна и при добавлении файла в интерфейсе, и перед склейкой."""
    problems = incompatibilities(probe_media(reference_path), probe_media(other_path))
    if not problems:
        return None
    return f"{os.path.basename(other_path)}: " + "; ".join(problems)


def check_segments_compatible(config: dict) -> None:
    """Проверяет по содержимому файлов, что тип и параметры всех сегментов совпадают с типом
    конфигурации и друг с другом, а для аудио — что выходной файл имеет подходящее расширение.
    Бросает ConfigError. Без этой проверки ffmpeg склеил бы файлы молча и испортил результат."""
    media_type = media_type_of(config)
    paths = [segment["path"] for segment in config["segments"]]
    infos = [probe_media(path) for path in paths]

    for path, info in zip(paths, infos):
        if info.kind != media_type:
            found = "видео" if info.kind == "video" else "аудио"
            raise ConfigError(f"{os.path.basename(path)}: это {found}, а конфигурация для "
                              f"{'видео' if media_type == 'video' else 'аудио'}")

    problems = [f"{os.path.basename(path)}: " + "; ".join(found)
                for path, info in zip(paths[1:], infos[1:]) if (found := incompatibilities(infos[0], info))]
    if problems:
        raise ConfigError("Файлы нельзя склеить без перекодирования — параметры отличаются от первого файла:\n"
                          + "\n".join(problems))

    if media_type == "audio":
        expected = audio_extension(infos[0])
        if expected is None:
            raise ConfigError(f"Формат звука {infos[0].audio_codec!r} не поддерживается: "
                              f"нужен AAC (.m4a) или MP3 (.mp3)")
        actual = Path(config["output"]["audioPath"]).suffix.lower()
        if actual != expected:
            raise ConfigError(f'Звук в файлах {infos[0].audio_codec.upper()}, поэтому итоговый файл должен '
                              f'называться *{expected}, а не *{actual}')


def cut_fragment(source_path: str, start: str, end: str, out_path: str, media_type: str = "video",
                 source_duration: float = None) -> None:
    """Вырезает [start, end] без перекодирования. Если известна длительность файла
    (source_duration) и end почти у самого конца файла (см. END_OF_FILE_TOLERANCE), режет до
    конца файла — иначе округление секунд отрезало бы хвост записи."""
    end_s = hhmmss_to_seconds(end)
    to_end_of_file = source_duration is not None and source_duration - end_s <= END_OF_FILE_TOLERANCE
    run_ffmpeg(
        [
            "-ss", start,
            "-i", source_path,
            *([] if to_end_of_file else ["-t", str(end_s - hhmmss_to_seconds(start))]),
            # у аудио «-vn»: встроенная обложка (m4a/mp3) не нужна и мешала бы склейке
            *(["-vn"] if media_type == "audio" else []),
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


def join_fragments(segments: list, tmp_dir: str, output_path: str, media_type: str = "video",
                   log=lambda msg: None) -> None:
    """Нарезает фрагменты всех сегментов и склеивает их в output_path (видео или аудио)."""
    fragment_paths = []
    for i, segment in enumerate(segments):
        source_path = segment["path"]
        source_duration = get_media_duration(source_path)
        # расширение фрагмента = расширению исходника: контейнер должен уметь хранить его кодек как есть
        extension = ".mp4" if media_type == "video" else Path(source_path).suffix.lower()
        for j, fragment in enumerate(segment["fragments"]):
            log(f"Нарезаю фрагмент {i + 1}.{j + 1} из {os.path.basename(source_path)}...")
            out_path = os.path.join(tmp_dir, f"fragment_{i:02d}_{j:02d}{extension}")
            cut_fragment(source_path, fragment["start"], fragment["end"], out_path, media_type, source_duration)
            fragment_paths.append(out_path)

    concat_list_path = os.path.join(tmp_dir, "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for p in fragment_paths:
            f.write(f"file '{escape_concat_path(p)}'\n")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    target = "итоговое видео" if media_type == "video" else "итоговый аудиофайл"
    log(f"Склеиваю фрагменты в {target}...")
    run_ffmpeg(
        [
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_path,
        ],
        f"Не удалось склеить фрагменты в {target}",
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
    """Выполняет нарезку/склейку по готовой конфигурации: для видео — склеивает видео и
    извлекает из него звук, для аудио — склеивает звук.

    Бросает ConfigError при некорректной конфигурации или несовместимых файлах и
    FFmpegError при ошибках ffmpeg (в т.ч. если ffmpeg не установлен).
    """
    validate_config(config)
    check_ffmpeg()
    check_segments_compatible(config)

    media_type = media_type_of(config)
    audio_path = config["output"]["audioPath"]

    with tempfile.TemporaryDirectory(prefix="fork_join_") as tmp_dir:
        log("Нарезаю и склеиваю фрагменты...")
        if media_type == "audio":
            join_fragments(config["segments"], tmp_dir, audio_path, "audio", log=log)
            log(f"Аудио сохранено: {audio_path}")
            return

        video_path = config["output"]["videoPath"]
        join_fragments(config["segments"], tmp_dir, video_path, "video", log=log)
        log(f"Видео сохранено: {video_path}")

        log("Извлекаю звуковую дорожку...")
        extract_audio(video_path, audio_path)
        log(f"Аудио сохранено: {audio_path}")
