#!/usr/bin/env python3
"""
Локальная транскрибация лекции в текст через faster-whisper (CPU, beam search + VAD).

Использование:
    ./venv/bin/python transcribe.py input.mp3
    ./venv/bin/python transcribe.py input.mp3 --output out.txt --model medium
    ./venv/bin/python transcribe.py input.mp3 --initial-prompt "Вундт, интроспекция, психогностика"
    ./venv/bin/python transcribe.py input.mp3 --no-flag-uncertain
"""

import argparse
import json
import time
from pathlib import Path

from faster_whisper import WhisperModel

from speech_to_text_whisper.lecture_common import format_timestamp, group_into_paragraphs

DEFAULT_MODEL = "large-v3"


def main():
    parser = argparse.ArgumentParser(description="Транскрибация аудио в текст (faster-whisper)")
    parser.add_argument("audio", type=Path, help="Путь к аудиофайлу (mp3/wav/m4a и т.д.)")
    parser.add_argument("--output", type=Path, default=None, help="Путь к итоговому .txt (по умолчанию рядом с аудио)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Размер модели faster-whisper (large-v3, medium, ...)")
    parser.add_argument("--language", default="ru", help="Код языка для распознавания")
    parser.add_argument("--beam-size", type=int, default=5, help="Ширина beam search (больше = точнее, но медленнее)")
    parser.add_argument("--save-json", action="store_true", help="Сохранить сырые сегменты в .json рядом с .txt")
    parser.add_argument(
        "--initial-prompt", default=None,
        help="Подсказка модели с ожидаемыми именами/терминами лекции (например: 'Вундт, интроспекция'). "
             "Помогает распознавать редкие слова и имена собственные. По умолчанию не используется.",
    )
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

    audio_path = args.audio.expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Файл не найден: {audio_path}")

    output_path = args.output or audio_path.with_suffix(".txt")

    print(f"Аудио:  {audio_path}")
    print(f"Модель: {args.model}")
    print(f"Язык:   {args.language}")
    if args.initial_prompt:
        print(f"Подсказка: {args.initial_prompt}")
    print("Начинаю транскрибацию (может занять от 40 до 120 минут в зависимости от длины записи)...")

    t0 = time.time()
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    segments_gen, info = model.transcribe(
        str(audio_path),
        language=args.language,
        beam_size=args.beam_size,
        # Отфильтровывает тишину/шум перед распознаванием (Silero VAD) —
        # убирает главную причину галлюцинаций Whisper на пустом звуке.
        vad_filter=True,
        # Не давать модели опираться на текст предыдущего куска: если где-то
        # возникла ошибка, она не должна каскадом портить весь последующий текст.
        condition_on_previous_text=False,
        initial_prompt=args.initial_prompt,
    )

    segments = []
    for seg in segments_gen:
        print(f"[{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}] {seg.text}")
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            # Сырые метрики уверенности всегда сохраняются — это просто данные;
            # решение, показывать ли по ним предупреждения, принимается отдельно
            # (здесь и в restore_punctuation.py) и легко отключается флагом.
            "avg_logprob": seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
        })

    elapsed = time.time() - t0
    print(f"\nТранскрибация завершена за {elapsed / 60:.1f} мин.")

    if args.save_json:
        json_path = output_path.with_suffix(".json")
        json_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Сырые сегменты сохранены: {json_path}")

    paragraphs = group_into_paragraphs(
        segments,
        confidence_threshold=args.confidence_threshold,
        no_speech_threshold=args.no_speech_threshold_flag,
    )

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"# {audio_path.name}\n")
        f.write(f"# Транскрибация: faster-whisper {args.model}, язык: {args.language}, beam_size={args.beam_size}\n\n")
        for start, text, uncertain in paragraphs:
            marker = "[?] " if (args.flag_uncertain and uncertain) else ""
            f.write(f"[{format_timestamp(start)}] {marker}{text}\n\n")

    print(f"Готово: {output_path}")


if __name__ == "__main__":
    main()
