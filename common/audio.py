"""Конвертация аудио в формат, удобный для распознавания речи и диаризации."""

import subprocess
from pathlib import Path

WAV_SAMPLE_RATE = 16000


def convert_to_wav_16k_mono(src_path, dst_path) -> Path:
    """Конвертирует любой аудио/видео файл в WAV 16 кГц моно PCM16.

    Такой файл занимает около 110 МБ на час записи. Он нужен и GigaAM, и
    диаризации: на несжатом входе оба работают с заметно меньшим пиком памяти,
    чем на M4A. Бросает RuntimeError с выводом ffmpeg при сбое.
    """
    dst_path = Path(dst_path)
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src_path),
         "-vn", "-ac", "1", "-ar", str(WAV_SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Не удалось конвертировать аудио в WAV (ffmpeg):\n{result.stdout}")
    return dst_path
