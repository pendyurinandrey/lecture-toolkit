"""Проверки окружения перед запуском: совместимость ffmpeg и токен Hugging Face.

Обе проверки нужны GigaAM: pyannote внутри него читает аудио через torchcodec
(поддерживает только FFmpeg 4-8) и скачивает gated VAD-модель
pyannote/segmentation-3.0 (нужен токен). Ничего тяжёлого не импортируется,
модуль безопасно подключать из GUI-процесса.
"""

import os

from common.ffmpeg import FFmpegError, get_ffmpeg_major_version

FFMPEG_MIN_SUPPORTED = 4
FFMPEG_MAX_SUPPORTED = 8


class EnvironmentCheckError(RuntimeError):
    """Окружение не готово для запуска (несовместимый ffmpeg или нет HF-токена)."""


def check_ffmpeg_compatibility() -> None:
    """Предупреждает, если установленный FFmpeg несовместим с torchcodec —
    не пытается ничего чинить сама (правильный способ зависит от ОС
    пользователя), только даёт понятную инструкцию."""
    if os.environ.get("DYLD_FALLBACK_LIBRARY_PATH") or os.environ.get("LD_LIBRARY_PATH"):
        return  # пользователь уже настроил сам (macOS/Linux) — доверяем

    try:
        major = get_ffmpeg_major_version()
    except FFmpegError:
        return  # версию не удалось однозначно определить — не блокируем запуск

    if FFMPEG_MIN_SUPPORTED <= major <= FFMPEG_MAX_SUPPORTED:
        return

    raise EnvironmentCheckError(
        f"Обнаружен FFmpeg версии {major} — torchcodec (через него pyannote.audio "
        f"внутри GigaAM читает аудио) поддерживает только версии "
        f"{FFMPEG_MIN_SUPPORTED}-{FFMPEG_MAX_SUPPORTED}.\n"
        "Установите совместимую версию FFmpeg рядом с текущей и укажите путь к её "
        "библиотекам переменной окружения DYLD_FALLBACK_LIBRARY_PATH, например:\n"
        '    export DYLD_FALLBACK_LIBRARY_PATH="/path/to/compatible/ffmpeg/lib"\n'
        "(на Linux аналогичная переменная называется LD_LIBRARY_PATH)."
    )


def ensure_hf_token() -> None:
    """Если HF_TOKEN не задан явно, подставляет токен, сохранённый локально
    командой `huggingface-cli login`. Бросает EnvironmentCheckError, если
    токена нет нигде."""
    if os.environ.get("HF_TOKEN"):
        return

    try:
        from huggingface_hub import get_token
        token = get_token()
    except ImportError:
        token = None

    if token:
        os.environ["HF_TOKEN"] = token
        return

    raise EnvironmentCheckError(
        "Не найден токен Hugging Face — он нужен для скачивания VAD-модели "
        "pyannote/segmentation-3.0.\n"
        "Выполните один раз:\n"
        "    hf auth login\n"
        "и примите условия использования на "
        "https://huggingface.co/pyannote/segmentation-3.0"
    )
