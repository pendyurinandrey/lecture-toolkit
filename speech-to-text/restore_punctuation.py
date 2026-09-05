#!/usr/bin/env python3
"""
Восстановление пунктуации, регистра и (опционально) удаление слов-паразитов
в уже готовой транскрипции.

Берёт сырые сегменты (.json), сохранённые transcribe.py (--save-json),
и прогоняет их через kontur-ai/sbert_punc_case_ru — не требует повторного
распознавания речи, поэтому работает намного быстрее ASR-шага.

Абзацы в итоговом файле формируются ПОСЛЕ восстановления пунктуации и
разрезаются только по концу предложения — никогда посередине.

Использование:
    ./venv/bin/python restore_punctuation.py "Лекция.json"
    ./venv/bin/python restore_punctuation.py "Лекция.json" --keep-fillers
    ./venv/bin/python restore_punctuation.py "Лекция.json" --no-flag-uncertain
"""

import argparse
import json
import time
from pathlib import Path

from filler_words import remove_fillers
from lecture_common import (
    format_timestamp,
    group_into_sentence_paragraphs,
    group_segments_by_pause,
    segment_words_with_timestamps,
)
from punctuation_model import SbertPuncCase


def main():
    parser = argparse.ArgumentParser(description="Восстановление пунктуации (kontur-ai/sbert_punc_case_ru)")
    parser.add_argument("segments_json", type=Path, help="Путь к .json с сегментами (из transcribe.py --save-json)")
    parser.add_argument("--output", type=Path, default=None, help="Путь к итоговому .txt")
    parser.add_argument("--keep-fillers", action="store_true", help="Не удалять слова-паразиты (вот, ну и т.п.)")
    parser.add_argument(
        "--no-flag-uncertain", action="store_false", dest="flag_uncertain", default=True,
        help="Не помечать в тексте абзацы с низкой уверенностью распознавания (по умолчанию помечаются меткой [?]).",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=-0.5,
        help="Порог avg_logprob, ниже которого сегмент считается ненадёжным (по умолчанию -0.5).",
    )
    parser.add_argument(
        "--no-speech-threshold-flag", type=float, default=0.6,
        help="Порог no_speech_prob, выше которого сегмент считается ненадёжным (по умолчанию 0.6).",
    )
    args = parser.parse_args()

    json_path = args.segments_json.expanduser().resolve()
    if not json_path.exists():
        raise SystemExit(f"Файл не найден: {json_path}")

    output_path = args.output or json_path.with_name(json_path.stem + "_punctuated.txt")

    segments = json.loads(json_path.read_text(encoding="utf-8"))
    seg_chunks = group_segments_by_pause(segments)

    print("Загружаю модель восстановления пунктуации (kontur-ai/sbert_punc_case_ru)...")
    t0 = time.time()
    model = SbertPuncCase()
    print(f"Модель загружена за {time.time() - t0:.1f} сек.")

    all_restored = []
    t0 = time.time()
    for i, seg_chunk in enumerate(seg_chunks, 1):
        word_ts = []
        for seg in seg_chunk:
            word_ts.extend(segment_words_with_timestamps(
                seg,
                confidence_threshold=args.confidence_threshold,
                no_speech_threshold=args.no_speech_threshold_flag,
            ))

        if not args.keep_fillers:
            word_ts = remove_fillers(word_ts)

        if not word_ts:
            continue

        words = [w for w, _, _ in word_ts]
        restored_text = model.punctuate(" ".join(words))
        restored_words = restored_text.split()

        if len(restored_words) != len(word_ts):
            # Подстраховка на случай расхождения токенизации — не должна
            # случаться в норме (модель сохраняет 1:1 соответствие слов),
            # но лучше не уронить многочасовой прогон из-за одного куска.
            print(f"  предупреждение: несовпадение числа слов в куске {i} "
                  f"({len(restored_words)} vs {len(word_ts)})")
        n = min(len(restored_words), len(word_ts))
        for rw, (_, ts, uncertain) in zip(restored_words[:n], word_ts[:n]):
            all_restored.append((rw, ts, uncertain))

        print(f"[{i}/{len(seg_chunks)}] обработано")

    paragraphs = group_into_sentence_paragraphs(all_restored)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"# {json_path.stem}\n")
        f.write("# Пунктуация восстановлена: kontur-ai/sbert_punc_case_ru")
        f.write(", слова-паразиты удалены\n\n" if not args.keep_fillers else "\n\n")
        for start, text, uncertain in paragraphs:
            marker = "[?] " if (args.flag_uncertain and uncertain) else ""
            f.write(f"[{format_timestamp(start)}] {marker}{text}\n\n")

    print(f"\nГотово за {(time.time() - t0) / 60:.1f} мин: {output_path}")


if __name__ == "__main__":
    main()
