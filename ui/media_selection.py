"""Правила выбора файлов в интерфейсе: какой тип принимать, что показать в подписи,
подходит ли новый файл. Чистая логика без Qt — тестируется без PySide6.

Тип «проекта» задаёт первый добавленный файл (reference — его MediaInfo; None, пока
список пуст): дальше принимаются только файлы того же типа и формата, потому что
склейка идёт без перекодирования (см. common/media_info.py).
"""

from typing import Optional

from common.media_info import MediaInfo, audio_extension, incompatibilities

_ALL_FILES = "Все файлы (*)"
_AUDIO_NAMES = {".m4a": "M4A", ".mp3": "MP3"}

FILE_HINT = ("Все файлы в списке должны быть одного типа (видео или аудио) и с одинаковыми параметрами: "
             "склейка идёт без перекодирования.")


def is_audio(reference: Optional[MediaInfo]) -> bool:
    return reference is not None and reference.kind == "audio"


def open_file_filter(reference: Optional[MediaInfo]) -> str:
    """Фильтр диалога выбора файла: до первого файла — любые видео и аудио, потом только тот же тип."""
    if reference is None:
        return ("Видео и аудио (*.mp4 *.m4a *.mp3);;Видео MP4 (*.mp4);;"
                "Аудио M4A и MP3 (*.m4a *.mp3);;" + _ALL_FILES)
    if reference.kind == "video":
        return "Видео MP4 (*.mp4);;" + _ALL_FILES
    extension = audio_extension(reference) or ".m4a"
    return f"Аудио {_AUDIO_NAMES[extension]} (*{extension});;" + _ALL_FILES


def open_file_title(reference: Optional[MediaInfo]) -> str:
    if reference is None:
        return "Выберите видео или аудио файл"
    return "Выберите видеофайл" if reference.kind == "video" else "Выберите аудиофайл"


def _channels(count: Optional[int]) -> str:
    return {1: "моно", 2: "стерео"}.get(count, f"{count} кан.")


def source_caption(reference: Optional[MediaInfo]) -> str:
    """Подпись над списком файлов: какой тип определился и что от этого зависит."""
    if reference is None:
        return "Источник: не выбран — тип определит первый добавленный файл (видео или аудио)"
    if reference.kind == "video":
        return "Источник: видео. Добавлять можно только видео с такими же параметрами"
    codec = reference.audio_codec.upper()
    extension = audio_extension(reference)
    container = _AUDIO_NAMES.get(extension)
    fmt = f"{container}/{codec}" if container and container != codec else codec   # «M4A/AAC», «MP3»
    return (f"Источник: аудио ({fmt}, {reference.sample_rate / 1000:g} кГц, {_channels(reference.channels)}). "
            "Добавлять можно только аудио в таком же формате")


def check_new_file(reference: Optional[MediaInfo], new: MediaInfo, name: str) -> Optional[str]:
    """Текст ошибки, если файл name нельзя добавить, иначе None.

    Первый файл (reference None): для аудио нужен поддерживаемый формат (AAC/MP3).
    Следующие: тип и параметры должны совпадать с первым файлом."""
    if reference is None:
        if new.kind == "audio" and audio_extension(new) is None:
            return (f"{name}: формат звука {str(new.audio_codec).upper()} не поддерживается — "
                    "нужен AAC (.m4a) или MP3 (.mp3)")
        return None
    problems = incompatibilities(reference, new)
    return f"{name}: " + "; ".join(problems) if problems else None


def audio_output_extension(reference: Optional[MediaInfo]) -> str:
    """Расширение итогового аудиофайла: для аудио по формату источника, иначе .m4a
    (звук из видео извлекается в M4A)."""
    return (audio_extension(reference) if is_audio(reference) else None) or ".m4a"


def audio_output_filter(reference: Optional[MediaInfo]) -> str:
    extension = audio_output_extension(reference)
    return f"{_AUDIO_NAMES[extension]} аудио (*{extension})"


def audio_output_label(reference: Optional[MediaInfo]) -> str:
    return f"Аудио ({_AUDIO_NAMES[audio_output_extension(reference)]}):"


def video_output_visible(reference: Optional[MediaInfo]) -> bool:
    """Строка «Видео (MP4)» нужна всегда, кроме режима «только аудио»."""
    return not is_audio(reference)
