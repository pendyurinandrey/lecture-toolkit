#!/usr/bin/env python3
"""
Транскрибация лекции через GigaAM-v3 и (по желанию) определение говорящих.

Конвейер:
    1. Аудио (M4A) -> временный WAV 16 кГц моно PCM16 (110 МБ на час записи).
       На несжатом входе VAD-этап GigaAM тратит втрое меньше памяти, чем на
       M4A, — поэтому запись целиком обрабатывается за один вызов, без
       нарезки на куски (проверено на 3ч23м: пик памяти 2.8 ГБ).
    2. Если включена диаризация: pyannote community-1 (diarization/) —
       результат сохраняется в <транскрипт>.diarization.json.
    3. GigaAM v3_e2e_rnnt на всём WAV. При диаризации — с точными таймкодами
       слов; слова сохраняются в <транскрипт>.words.json.
    4. Итоговый TXT: перегруппировка по предложениям, удаление слов-паразитов
       (по желанию) и, при диаризации, метки «Спикер N» — спикер назначается
       каждому слову (diarization/speaker_labels.py), абзац рвётся при смене
       спикера.

Диаризация и GigaAM запускаются отдельными процессами (subprocess) — память
освобождается полностью после каждого этапа.

Модуль также предназначен для использования из pipeline_ui.py: функция run()
принимает колбэк log() вместо print и бросает исключения вместо sys.exit,
как и process_config()/validate_config() в fork_join.py.

Использование:
    ./venv/bin/python -m speech_to_text_gigaam.transcribe_lecture lecture.m4a --output lecture.txt
    ./venv/bin/python -m speech_to_text_gigaam.transcribe_lecture lecture.m4a --output lecture.txt --diarize
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import deque
from itertools import groupby
from pathlib import Path

from common.audio import convert_to_wav_16k_mono
from common.filler_words import remove_fillers
from common.silence import get_media_duration
from diarization import diarize_pyannote, speaker_labels
from speech_to_text_gigaam.transcribe_longform import (
    MODEL_NAME,
    EnvironmentCheckError,
    check_ffmpeg_compatibility,
    ensure_hf_token,
    format_timestamp,
    group_into_sentence_paragraphs,
)

DiarizationError = diarize_pyannote.DiarizationError
REPO_ROOT = Path(__file__).resolve().parent.parent


def diarization_path(transcript_path) -> Path:
    """lecture.txt -> lecture.diarization.json (рядом с транскриптом)."""
    return Path(transcript_path).with_suffix(".diarization.json")


def words_path(transcript_path) -> Path:
    """lecture.txt -> lecture.words.json (слова GigaAM с таймкодами)."""
    return Path(transcript_path).with_suffix(".words.json")


def _run_module(module: str, args: list, log) -> None:
    """Запускает `python -m module ...`, построчно передаёт вывод в log().
    Бросает RuntimeError с хвостом вывода, если процесс завершился с ошибкой."""
    # Корень репозитория — в PYTHONPATH: editable-установка (pip install -e .)
    # запоминает список пакетов на момент установки и не видит добавленных позже.
    python_path = os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", TQDM_DISABLE="1",
               PYTHONPATH=python_path)
    proc = subprocess.Popen(
        [sys.executable, "-m", module, *map(str, args)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env,
    )
    tail = deque(maxlen=15)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(line)
            tail.append(line)
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Сбой {module} (код {code}):\n" + "\n".join(tail))


def plain_paragraphs(segments: list, keep_fillers: bool, log) -> list:
    """Абзацы [(время, текст)] без спикеров. Таймкоды слов — линейная
    интерполяция внутри сегмента (точные при этом не запрашиваются: так
    быстрее, а для абзацев с временем в начале их точности достаточно)."""
    word_ts = []
    for seg in segments:
        words = seg["text"].split()
        n = len(words)
        for j, w in enumerate(words):
            word_ts.append((w, seg["start"] + (seg["end"] - seg["start"]) * j / n))

    if not keep_fillers:
        before = len(word_ts)
        word_ts = remove_fillers(word_ts)
        log(f"Слов-паразитов удалено: {before - len(word_ts)}")
    return group_into_sentence_paragraphs(word_ts)


def speaker_paragraphs(words: list, turns: list, keep_fillers: bool, log) -> list:
    """Абзацы [(время, текст, номер спикера)]: абзац рвётся при смене
    спикера, внутри одного спикера — по предложениям, как без диаризации."""
    smoothed = speaker_labels.smooth_turns(turns)
    log(f"Реплик диаризации: {len(turns)}, после сглаживания микрореплик: {len(smoothed)}")
    numbers = speaker_labels.label_words(words, smoothed)
    items = [(w["text"], w["start"], n) for w, n in zip(words, numbers)]

    if not keep_fillers:
        before = len(items)
        items = remove_fillers(items)
        log(f"Слов-паразитов удалено: {before - len(items)}")

    paragraphs = []
    for speaker, run in groupby(items, key=lambda item: item[2]):
        pairs = [(text, ts) for text, ts, _ in run]
        paragraphs.extend((start, text, speaker) for start, text in group_into_sentence_paragraphs(pairs))
    return paragraphs


def run(audio_path, output_path=None, keep_fillers: bool = False, diarize: bool = False, log=print) -> Path:
    """Программный вход — используется и CLI-обёрткой main(), и pipeline_ui.py.

    Бросает EnvironmentCheckError (HF-токен/ffmpeg), DiarizationError (нет
    модели диаризации), FileNotFoundError (нет аудио) или RuntimeError (сбой
    ffmpeg/GigaAM/диаризации) — вызывающий код сам решает, как показать
    их пользователю.
    """
    check_ffmpeg_compatibility()
    ensure_hf_token()
    if diarize:
        diarize_pyannote.check_model_available()  # до долгой работы, а не после

    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Файл не найден: {audio_path}")
    output_path = Path(output_path) if output_path else audio_path.with_suffix(".gigaam.txt")

    log(f"Длительность: {format_timestamp(get_media_duration(audio_path))}")

    with tempfile.TemporaryDirectory(prefix="lecture_") as tmp_dir:
        wav_path = Path(tmp_dir) / "audio_16k.wav"
        log("Конвертирую аудио в WAV 16 кГц моно...")
        convert_to_wav_16k_mono(audio_path, wav_path)

        if diarize:
            log("=== Определение говорящих ===")
            _run_module("diarization.diarize_pyannote",
                        [wav_path, "--output", diarization_path(output_path)], log)

        log("=== Распознавание речи (GigaAM) ===")
        gigaam_txt = Path(tmp_dir) / "gigaam.txt"  # .txt не создаётся, от него берётся имя .json
        gigaam_args = [wav_path, "--output", gigaam_txt, "--json-only"]
        if diarize:
            gigaam_args.append("--word-timestamps")
        _run_module("speech_to_text_gigaam.transcribe_longform", gigaam_args, log)
        segments = json.loads(gigaam_txt.with_suffix(".json").read_text(encoding="utf-8"))

    segments = [s for s in segments if s["text"].split()]

    if diarize:
        words = sorted((w for s in segments for w in s.get("words", [])), key=lambda w: (w["start"], w["end"]))
        words_file = words_path(output_path)
        words_file.write_text(
            json.dumps({"audio": audio_path.name, "model": MODEL_NAME, "words": words}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        log(f"Слова с таймкодами: {words_file}")
        paragraphs = speaker_paragraphs(
            words, speaker_labels.load_turns(diarization_path(output_path)), keep_fillers, log,
        )
        speakers = len({p[2] for p in paragraphs})
        log(f"Абзацев: {len(paragraphs)}, спикеров в тексте: {speakers}")
    else:
        paragraphs = plain_paragraphs(segments, keep_fillers, log)
        log(f"Всего абзацев после перегруппировки: {len(paragraphs)}")

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"# {audio_path.name}\n")
        f.write(f"# Транскрибация: GigaAM {MODEL_NAME} (longform), перегруппировано по предложениям")
        f.write(", слова-паразиты удалены\n" if not keep_fillers else "\n")
        if diarize:
            f.write(f"# Говорящие: {diarize_pyannote.MODEL_ID}; «{speaker_labels.SPEAKER_PREFIX} N» — "
                    "по порядку первого появления в записи\n")
        f.write("\n")
        for paragraph in paragraphs:
            start, text = paragraph[0], paragraph[1]
            label = f"{speaker_labels.format_speaker(paragraph[2])}: " if diarize else ""
            f.write(f"[{format_timestamp(start)}] {label}{text}\n\n")

    log(f"Готово: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="GigaAM: транскрибация лекции, по желанию с определением говорящих")
    parser.add_argument("audio", type=Path, help="Путь к аудиофайлу")
    parser.add_argument("--output", type=Path, default=None, help="Путь к итоговому .txt")
    parser.add_argument("--keep-fillers", action="store_true", help="Не удалять слова-паразиты (вот, ну и т.п.)")
    parser.add_argument("--diarize", action="store_true", help="Определять говорящих (медленнее)")
    args = parser.parse_args()

    try:
        run(args.audio, args.output, keep_fillers=args.keep_fillers, diarize=args.diarize)
    except (EnvironmentCheckError, DiarizationError, FileNotFoundError, RuntimeError) as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
