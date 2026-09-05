#!/usr/bin/env python3
"""
Транскрибация длинной лекции через GigaAM-v3 по частям — защита от роста
потребления памяти в transcribe_longform() на многочасовых файлах (см. README
эксперимента / историю утечки, из-за которой процесс убивала macOS).

Каждый кусок обрабатывается ОТДЕЛЬНЫМ процессом (subprocess к
transcribe_longform.py) — память гарантированно освобождается между кусками
независимо от причины её роста внутри GigaAM/pyannote.

Куски нарезаются НЕ по фиксированному времени, а по паузам в речи рядом
с целевой длительностью (ffmpeg silencedetect — лёгкий, не ML, разовый проход
по файлу) — чтобы не резать посередине слова.

Требует переменные окружения HF_TOKEN и DYLD_FALLBACK_LIBRARY_PATH (см.
transcribe_longform.py) — наследуются дочерними процессами автоматически.

Использование:
    export HF_TOKEN="..."
    DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@8/lib" \\
        ./venv/bin/python transcribe_longform_chunked.py lecture.mp3 --output lecture.gigaam.txt
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from filler_words import remove_fillers
from transcribe_longform import format_timestamp, group_into_sentence_paragraphs

CHUNK_TARGET_SEC = 30 * 60   # целевая длина куска
SEARCH_WINDOW_SEC = 5 * 60   # искать паузу в пределах +/- этого окна от цели
SILENCE_NOISE_DB = "-30dB"
SILENCE_MIN_DUR = 0.5        # секунд тишины, чтобы считать паузой


def get_audio_duration(audio_path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def detect_silences(audio_path: Path):
    """Возвращает список (start, end) пауз в аудио через ffmpeg silencedetect
    (порог громкости, не нейросеть — быстрый разовый проход)."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(audio_path),
         "-af", f"silencedetect=noise={SILENCE_NOISE_DB}:d={SILENCE_MIN_DUR}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    log = result.stderr
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", log)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", log)]
    return list(zip(starts, ends))


def pick_split_points(duration: float, silences: list, target: float, window: float):
    """Выбирает точки разреза рядом с кратными `target`, снапая к середине
    ближайшей найденной паузы в пределах `window`."""
    points = []
    target_time = target
    while target_time < duration - target / 2:
        candidates = [
            (s + e) / 2 for s, e in silences
            if target_time - window <= (s + e) / 2 <= target_time + window
        ]
        if candidates:
            split = min(candidates, key=lambda c: abs(c - target_time))
        else:
            split = target_time
            print(f"  предупреждение: пауза рядом с {format_timestamp(split)} не найдена, "
                  f"режу по фиксированному времени (риск разреза посередине слова)")
        points.append(split)
        target_time = split + target
    return points


def main():
    parser = argparse.ArgumentParser(description="GigaAM longform по частям (защита от роста памяти)")
    parser.add_argument("audio", type=Path, help="Путь к аудиофайлу")
    parser.add_argument("--output", type=Path, default=None, help="Путь к итоговому .txt")
    parser.add_argument("--chunk-minutes", type=float, default=CHUNK_TARGET_SEC / 60,
                         help="Целевая длина куска в минутах (по умолчанию 30)")
    parser.add_argument("--keep-fillers", action="store_true", help="Не удалять слова-паразиты (вот, ну и т.п.)")
    args = parser.parse_args()

    audio_path = args.audio.expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Файл не найден: {audio_path}")
    output_path = args.output or audio_path.with_suffix(".gigaam.txt")

    duration = get_audio_duration(audio_path)
    print(f"Длительность: {format_timestamp(duration)}")

    print("Ищу паузы в речи (ffmpeg silencedetect)...")
    silences = detect_silences(audio_path)
    print(f"Найдено пауз: {len(silences)}")

    target = args.chunk_minutes * 60
    split_points = pick_split_points(duration, silences, target=target, window=SEARCH_WINDOW_SEC)
    boundaries = [0.0] + split_points + [duration]
    print(f"Кусков: {len(boundaries) - 1}")
    for i in range(len(boundaries) - 1):
        print(f"  {i + 1}: {format_timestamp(boundaries[i])} - {format_timestamp(boundaries[i + 1])}")

    script_dir = Path(__file__).parent
    all_word_ts = []

    with tempfile.TemporaryDirectory(prefix="gigaam_chunks_") as tmp_dir:
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            chunk_audio = Path(tmp_dir) / f"chunk_{i:02d}.mp3"
            chunk_txt = Path(tmp_dir) / f"chunk_{i:02d}.txt"
            chunk_json = chunk_txt.with_suffix(".json")

            print(f"\n=== Кусок {i + 1}/{len(boundaries) - 1} "
                  f"({format_timestamp(start)}-{format_timestamp(end)}) ===")

            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path), "-ss", str(start), "-to", str(end),
                 "-c", "copy", str(chunk_audio), "-loglevel", "error"],
                check=True,
            )

            # Отдельный процесс на кусок — память освобождается полностью
            # при его завершении, независимо от причины роста внутри GigaAM.
            subprocess.run(
                [sys.executable, str(script_dir / "transcribe_longform.py"),
                 str(chunk_audio), "--output", str(chunk_txt), "--save-json"],
                check=True,
            )

            segments = json.loads(chunk_json.read_text(encoding="utf-8"))
            for seg in segments:
                words = seg["text"].split()
                if not words:
                    continue
                n = len(words)
                seg_start, seg_end = seg["start"] + start, seg["end"] + start
                for j, w in enumerate(words):
                    all_word_ts.append((w, seg_start + (seg_end - seg_start) * j / n))

    if not args.keep_fillers:
        before = len(all_word_ts)
        all_word_ts = remove_fillers(all_word_ts)
        print(f"Слов-паразитов удалено: {before - len(all_word_ts)}")

    paragraphs = group_into_sentence_paragraphs(all_word_ts)
    print(f"Всего абзацев после перегруппировки: {len(paragraphs)}")

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"# {audio_path.name}\n")
        f.write("# Транскрибация: GigaAM v3_e2e_rnnt (longform, по частям), "
                 "перегруппировано по предложениям")
        f.write(", слова-паразиты удалены\n\n" if not args.keep_fillers else "\n\n")
        for start, text in paragraphs:
            f.write(f"[{format_timestamp(start)}] {text}\n\n")

    print(f"Готово: {output_path}")


if __name__ == "__main__":
    main()
