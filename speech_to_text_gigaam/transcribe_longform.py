#!/usr/bin/env python3
"""
Полная транскрибация лекции через GigaAM-v3 (longform-режим, VAD от pyannote).

GigaAM (модель v3_e2e_rnnt) сам расставляет пунктуацию и регистр, но режет
текст по паузам в речи (VAD-сегменты) — получается много мелких абзацев,
часто обрывающихся посередине предложения. Этот скрипт дополнительно
перегруппировывает результат по границе предложения (после '.', '?' или '!'),
используя пословную интерполяцию таймкодов внутри каждого сегмента.

Нужны переменная окружения HF_TOKEN (для скачивания gated VAD-модели
pyannote/segmentation-3.0) и, если установленный в системе FFmpeg новее
версии 8, — DYLD_FALLBACK_LIBRARY_PATH на совместимую версию (torchcodec,
через который pyannote читает аудио, версии 9+ пока не поддерживает).
Обе проверки (check_ffmpeg_compatibility/ensure_hf_token) выполняются
автоматически при запуске — токен сам подхватится из кэша `huggingface-cli
login`, если уже не задан явно; для ffmpeg просто выводится понятная ошибка,
если версия несовместима, а переменная не выставлена вручную.

Использование:
    ./venv/bin/python transcribe_longform.py /path/to/lecture.mp3
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import gigaam

MODEL_NAME = "v3_e2e_rnnt"
SENTENCE_END_CHARS = ".?!"
PARAGRAPH_TARGET_CHARS = 500
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
        out = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True, timeout=10,
        ).stdout
        # Формат первой строки: "ffmpeg version 9.0.1 Copyright (c) ..."
        major = int(out.split()[2].split(".")[0])
    except Exception:
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


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def segment_words_with_timestamps(segment) -> list:
    """(word, timestamp) для одного сегмента — метка времени линейной
    интерполяцией между start и end сегмента (точных пословных таймкодов
    GigaAM в этом режиме не отдаёт)."""
    words = segment.text.split()
    if not words:
        return []
    start, end = segment.start, segment.end
    n = len(words)
    return [(w, start + (end - start) * i / n) for i, w in enumerate(words)]


def group_into_sentence_paragraphs(word_ts_pairs, target_chars=PARAGRAPH_TARGET_CHARS):
    """Абзацы, разрезанные только сразу после конца предложения — никогда
    посередине фразы."""
    paragraphs = []
    current_words = []
    current_start = None

    for word, ts in word_ts_pairs:
        if not current_words:
            current_start = ts
        current_words.append(word)
        if word and word[-1] in SENTENCE_END_CHARS and len(" ".join(current_words)) >= target_chars:
            paragraphs.append((current_start, " ".join(current_words)))
            current_words = []

    if current_words:
        paragraphs.append((current_start, " ".join(current_words)))

    return paragraphs


def main():
    parser = argparse.ArgumentParser(description="Транскрибация аудио через GigaAM-v3 (longform)")
    parser.add_argument("audio", type=Path, help="Путь к аудиофайлу")
    parser.add_argument("--output", type=Path, default=None, help="Путь к итоговому .txt")
    parser.add_argument("--save-json", action="store_true", help="Сохранить сырые VAD-сегменты в .json рядом с .txt")
    args = parser.parse_args()

    try:
        check_ffmpeg_compatibility()
        ensure_hf_token()
    except EnvironmentCheckError as e:
        raise SystemExit(str(e))

    audio_path = args.audio.expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Файл не найден: {audio_path}")

    output_path = args.output or audio_path.with_suffix(".gigaam.txt")

    print(f"Аудио:  {audio_path}")
    print(f"Модель: {MODEL_NAME} (CPU)")

    t0 = time.time()
    model = gigaam.load_model(MODEL_NAME, device="cpu", fp16_encoder=False)
    print(f"Модель загружена за {time.time() - t0:.1f} сек.")

    print("Начинаю транскрибацию (VAD-нарезка + распознавание)...")
    t0 = time.time()
    result = model.transcribe_longform(str(audio_path))
    elapsed = time.time() - t0
    print(f"Транскрибация завершена за {elapsed / 60:.1f} мин. Сегментов: {len(result)}")

    if args.save_json:
        json_path = output_path.with_suffix(".json")
        raw_segments = [{"start": s.start, "end": s.end, "text": s.text} for s in result]
        json_path.write_text(json.dumps(raw_segments, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Сырые сегменты сохранены: {json_path}")

    word_ts = []
    for segment in result:
        word_ts.extend(segment_words_with_timestamps(segment))
    paragraphs = group_into_sentence_paragraphs(word_ts)
    print(f"Абзацев после перегруппировки по предложениям: {len(paragraphs)}")

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"# {audio_path.name}\n")
        f.write(f"# Транскрибация: GigaAM {MODEL_NAME} (longform), перегруппировано по предложениям\n\n")
        for start, text in paragraphs:
            f.write(f"[{format_timestamp(start)}] {text}\n\n")

    print(f"Готово: {output_path}")


if __name__ == "__main__":
    main()
