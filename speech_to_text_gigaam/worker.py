"""Процесс GigaAM: распознаёт речь в WAV и сохраняет сегменты в JSON.

ВНУТРЕННИЙ процесс, вручную не запускать: его запускает transcribe.py
отдельным процессом (`python -m speech_to_text_gigaam.worker ...`), чтобы
память, занятая моделью и VAD, освобождалась целиком по его завершении.

`import gigaam` (а с ним torch) — только внутри функций: модуль можно
импортировать и тестировать без GigaAM. HF-токен (для gated VAD-модели
pyannote/segmentation-3.0) процесс получает от родителя через переменную
окружения HF_TOKEN.

Формат результата — JSON-список сегментов:
    [{"start": 1.2, "end": 9.8, "text": "...",
      "words": [{"text": "...", "start": 1.2, "end": 1.6}, ...]}, ...]
Поле "words" есть только при --word-timestamps (точные таймкоды слов, медленнее).
"""

import argparse
import json
import time
from pathlib import Path

BATCH_SIZE = 16  # fr_batch_size по умолчанию в GigaAM.transcribe_longform
PROGRESS_STEP_PERCENT = 5


def install_progress_reporting(model, batch_size: int = BATCH_SIZE) -> None:
    """Печатает прогресс распознавания: число батчей заранее неизвестно,
    пока VAD не нарежет файл, поэтому перехватываем и VAD-этап, и forward()
    модели. Если внутреннее устройство GigaAM изменится, прогресс просто
    не будет выводиться — на результат это не влияет."""
    try:
        import gigaam.vad_utils as vad_utils
        original_segment = vad_utils.segment_audio_file
    except (ImportError, AttributeError):
        return

    state = {"total": 0, "done": 0, "bucket": -1}

    def segment_with_report(*args, **kwargs):
        print("Нарезка на сегменты речи (VAD)...", flush=True)
        t0 = time.time()
        segments, boundaries = original_segment(*args, **kwargs)
        state["total"] = -(-len(segments) // batch_size)
        print(f"VAD завершён за {(time.time() - t0) / 60:.1f} мин, сегментов речи: {len(segments)}", flush=True)
        return segments, boundaries

    original_forward = model.forward

    def forward_with_report(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        state["done"] += 1
        if state["total"]:
            bucket = 100 * state["done"] // state["total"] // PROGRESS_STEP_PERCENT
            if bucket > state["bucket"]:
                state["bucket"] = bucket
                print(f"Распознавание: {min(bucket * PROGRESS_STEP_PERCENT, 100)}% "
                      f"(пачка {state['done']}/{state['total']})", flush=True)
        return result

    vad_utils.segment_audio_file = segment_with_report
    model.forward = forward_with_report


def segments_to_json(result, word_timestamps: bool) -> list:
    """Сегменты результата GigaAM -> список словарей для JSON."""
    segments = []
    for s in result:
        raw = {"start": s.start, "end": s.end, "text": s.text}
        if word_timestamps:
            raw["words"] = [{"text": w.text, "start": w.start, "end": w.end} for w in s.words or []]
        segments.append(raw)
    return segments


def transcribe_to_json(wav_path: Path, output_path: Path, model_name: str, word_timestamps: bool) -> None:
    import gigaam

    print(f"Аудио:  {wav_path}")
    print(f"Модель: {model_name} (CPU)")

    t0 = time.time()
    model = gigaam.load_model(model_name, device="cpu", fp16_encoder=False)
    print(f"Модель загружена за {time.time() - t0:.1f} сек.", flush=True)
    install_progress_reporting(model)

    print("Начинаю транскрибацию (VAD-нарезка + распознавание)...", flush=True)
    t0 = time.time()
    result = model.transcribe_longform(str(wav_path), word_timestamps=word_timestamps)
    print(f"Транскрибация завершена за {(time.time() - t0) / 60:.1f} мин. Сегментов: {len(result)}", flush=True)

    Path(output_path).write_text(
        json.dumps(segments_to_json(result, word_timestamps), ensure_ascii=False, indent=2), encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Внутренний процесс GigaAM (запускается из transcribe.py)")
    parser.add_argument("wav", type=Path, help="WAV 16 кГц моно PCM16")
    parser.add_argument("--output", type=Path, required=True, help="Куда сохранить сегменты (JSON)")
    parser.add_argument("--model", required=True, help="Имя модели GigaAM, например v3_e2e_rnnt")
    parser.add_argument("--word-timestamps", action="store_true", help="Точные таймкоды слов (медленнее)")
    args = parser.parse_args()

    if not args.wav.exists():
        raise SystemExit(f"Файл не найден: {args.wav}")
    transcribe_to_json(args.wav, args.output, args.model, args.word_timestamps)


if __name__ == "__main__":
    main()
