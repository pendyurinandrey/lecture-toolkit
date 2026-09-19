#!/usr/bin/env python3
"""
Диаризация (кто и когда говорит) через pyannote community-1.

Вход: WAV 16 кГц моно PCM16 (см. common/audio.py). Файл целиком читается в
память и передаётся в pyannote как готовая волна — так не нужны ни torchcodec,
ни совместимая версия ffmpeg, а пик памяти втрое ниже, чем при передаче пути
к сжатому файлу.

Результат: JSON с репликами БЕЗ наложений (exclusive_speaker_diarization) —
их можно однозначно сопоставлять со словами транскрипта, см.
speaker_labels.py. Метки спикеров в файле — «сырые» (SPEAKER_00, ...),
номера «Спикер N» присваиваются позже, при склейке с текстом.

Устройство выбирается автоматически: cuda -> mps -> cpu. Результат на CPU и
MPS совпадает, отличается только скорость (MPS быстрее CPU примерно в 12 раз,
на CPU ожидайте время, сравнимое с длительностью записи).

Этот модуль подключается из GUI-процесса ради check_model_available(), поэтому
torch и pyannote импортируются только внутри run().

Использование:
    ./venv/bin/python -m diarization.diarize_pyannote lecture.wav --output lecture.diarization.json
"""

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

MODEL_ID = "pyannote/speaker-diarization-community-1"
MODEL_URL = f"https://huggingface.co/{MODEL_ID}"
_PROGRESS_STEP_PERCENT = 10


class DiarizationError(RuntimeError):
    """Диаризация невозможна (модель недоступна или входной файл не подходит)."""


def check_model_available() -> None:
    """Быстро проверяет, что модель уже есть в локальном кэше Hugging Face.

    Ничего не скачивает и сети не требует — чтобы не обнаружить проблему
    через час после запуска. Бросает DiarizationError с инструкцией.
    """
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, local_files_only=True)
    except Exception as e:  # noqa: BLE001 - любая причина означает "модели нет"
        raise DiarizationError(
            f"Модель диаризации {MODEL_ID} не найдена в локальном кэше.\n"
            f"1. Примите условия использования на {MODEL_URL}\n"
            "2. Выполните один раз:\n"
            "    hf auth login\n"
            f"    hf download {MODEL_ID}\n"
            f"({type(e).__name__})"
        ) from e


def read_wav_16k_mono(path):
    """Читает WAV 16 кГц моно PCM16 в float32-тензор формы (1, N)."""
    import numpy as np
    import torch

    with wave.open(str(path), "rb") as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        if (channels, width, rate) != (1, 2, 16000):
            raise DiarizationError(
                f"Нужен WAV 16 кГц моно PCM16, а в файле: {rate} Гц, {channels} кан., {width * 8} бит: {path}"
            )
        raw = w.readframes(w.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(samples).unsqueeze(0), rate


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _ProgressLogger:
    """Хук прогресса pyannote: короткие строки вместо progress bar'ов."""

    def __init__(self, log):
        self._log = log
        self._step = None
        self._last_bucket = -1

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if step_name != self._step:
            self._step, self._last_bucket = step_name, -1
        if not total:
            return
        bucket = int(100 * completed / total) // _PROGRESS_STEP_PERCENT
        if bucket > self._last_bucket:
            self._last_bucket = bucket
            self._log(f"  диаризация, этап «{step_name}»: {min(bucket * _PROGRESS_STEP_PERCENT, 100)}%")


def run(wav_path, output_path, device=None, log=print) -> Path:
    """Диаризует WAV и пишет реплики в output_path (JSON). Возвращает output_path."""
    wav_path, output_path = Path(wav_path), Path(output_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Файл не найден: {wav_path}")

    check_model_available()
    # Модель уже проверена в кэше — сеть и токен не нужны.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import warnings
    # pyannote предупреждает, что torchcodec не загрузился; нам он не нужен —
    # волна передаётся в памяти.
    warnings.filterwarnings("ignore", message=".*torchcodec.*")

    import torch
    from pyannote.audio import Pipeline

    device = device or pick_device()
    if device == "cpu":
        log("Внимание: диаризация на CPU (нет CUDA/MPS) — она займёт время порядка длительности записи.")

    t0 = time.time()
    waveform, rate = read_wav_16k_mono(wav_path)
    duration = waveform.shape[1] / rate
    log(f"Диаризация: {duration / 60:.1f} мин аудио, устройство: {device}")

    pipeline = Pipeline.from_pretrained(MODEL_ID)
    pipeline.to(torch.device(device))
    output = pipeline({"waveform": waveform, "sample_rate": rate}, hook=_ProgressLogger(log))

    turns = [
        {"start": round(seg.start, 3), "end": round(seg.end, 3), "speaker": speaker}
        for seg, _, speaker in output.exclusive_speaker_diarization.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: (t["start"], t["end"]))
    result = {
        "model": MODEL_ID,
        "device": device,
        "audio_duration": round(duration, 3),
        "speakers": len({t["speaker"] for t in turns}),
        "turns": turns,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"Диаризация завершена за {(time.time() - t0) / 60:.1f} мин: "
        f"спикеров {result['speakers']}, реплик {len(turns)}. Файл: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Диаризация через pyannote community-1")
    parser.add_argument("wav", type=Path, help="WAV 16 кГц моно PCM16")
    parser.add_argument("--output", type=Path, default=None, help="Путь к JSON (по умолчанию рядом с WAV)")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None,
                        help="Устройство (по умолчанию автоопределение)")
    args = parser.parse_args()
    output = args.output or args.wav.with_suffix(".diarization.json")
    try:
        run(args.wav, output, device=args.device, log=lambda m: print(m, flush=True))
    except (DiarizationError, FileNotFoundError) as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
