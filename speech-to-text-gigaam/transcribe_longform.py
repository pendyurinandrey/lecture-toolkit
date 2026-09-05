#!/usr/bin/env python3
"""
Полная транскрибация лекции через GigaAM-v3 (longform-режим, VAD от pyannote).

GigaAM (модель v3_e2e_rnnt) сам расставляет пунктуацию и регистр, но режет
текст по паузам в речи (VAD-сегменты) — получается много мелких абзацев,
часто обрывающихся посередине предложения. Этот скрипт дополнительно
перегруппировывает результат по границе предложения (после '.', '?' или '!'),
используя пословную интерполяцию таймкодов внутри каждого сегмента.

Требует переменную окружения HF_TOKEN — она должна быть установлена в том же
терминале, где запускается этот скрипт; сам скрипт токен нигде не печатает
и никуда, кроме HF Hub, не передаёт.

Также требует DYLD_FALLBACK_LIBRARY_PATH на ffmpeg@8 (torchcodec, через который
pyannote.audio читает аудио, пока не поддерживает FFmpeg 9 — установленный в
системе через `brew install ffmpeg`; `brew install ffmpeg@8` ставит
совместимую версию рядом, не трогая основной ffmpeg в PATH).

Использование:
    export HF_TOKEN="..."
    DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@8/lib" \\
        ./venv/bin/python transcribe_longform.py /path/to/lecture.mp3
"""

import argparse
import json
import os
import time
from pathlib import Path

import gigaam

MODEL_NAME = "v3_e2e_rnnt"
SENTENCE_END_CHARS = ".?!"
PARAGRAPH_TARGET_CHARS = 500


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

    if not os.getenv("HF_TOKEN"):
        raise SystemExit(
            "Переменная окружения HF_TOKEN не установлена.\n"
            "Выполните в этом же терминале: export HF_TOKEN=\"ваш_токен\""
        )

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
