"""
Транскрибация лекции через GigaAM-v3 и (по желанию) определение говорящих —
входная точка пакета: run().

Конвейер:
    1. Аудио (M4A) -> временный WAV 16 кГц моно PCM16 (110 МБ на час записи).
       На несжатом входе VAD-этап GigaAM тратит втрое меньше памяти, чем на
       M4A, — поэтому запись целиком обрабатывается за один вызов, без
       нарезки на куски (проверено на 3ч23м: пик памяти 2.8 ГБ).
    2. Если включена диаризация: pyannote community-1 (diarization/) —
       результат сохраняется в <транскрипт>.diarization.json.
    3. GigaAM v3_e2e_rnnt на всём WAV (worker.py). При диаризации — с точными
       таймкодами слов; слова сохраняются в <транскрипт>.words.json.
    4. Итоговый TXT (transcript.py): перегруппировка по предложениям, удаление
       слов-паразитов (по желанию) и, при диаризации, метки «Спикер N» —
       спикер назначается каждому слову (diarization/speaker_labels.py), абзац
       рвётся при смене спикера.

Диаризация и GigaAM запускаются отдельными процессами (subprocess) — память
освобождается полностью после каждого этапа. Сам модуль ничего тяжёлого не
импортирует и вызывается из pipeline_ui.py: run() принимает колбэк log()
вместо print и бросает исключения вместо sys.exit, как process_config() в
fork_join.py. Отдельного запуска из командной строки нет.
"""

import json
import os
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

from common.ffmpeg import convert_to_wav_16k_mono, get_media_duration
from diarization import diarize_pyannote, speaker_labels
from speech_to_text_gigaam import transcript
from speech_to_text_gigaam.environment import check_ffmpeg_compatibility, ensure_hf_token

MODEL_NAME = "v3_e2e_rnnt"
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


def run(audio_path, output_path=None, keep_fillers: bool = False, diarize: bool = False, log=print) -> Path:
    """Транскрибирует аудио в output_path (по умолчанию <аудио>.gigaam.txt).

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

    log(f"Длительность видеофайла после обрезки и склейки: "
        f"{transcript.format_timestamp(get_media_duration(audio_path))}")

    with tempfile.TemporaryDirectory(prefix="lecture_") as tmp_dir:
        wav_path = Path(tmp_dir) / "audio_16k.wav"
        log("Конвертирую аудио в WAV 16 кГц моно...")
        convert_to_wav_16k_mono(audio_path, wav_path)

        if diarize:
            log("=== Определение говорящих ===")
            _run_module("diarization.diarize_pyannote",
                        [wav_path, "--output", diarization_path(output_path)], log)

        log("=== Распознавание речи (GigaAM) ===")
        segments_file = Path(tmp_dir) / "segments.json"
        worker_args = [wav_path, "--output", segments_file, "--model", MODEL_NAME]
        if diarize:
            worker_args.append("--word-timestamps")
        _run_module("speech_to_text_gigaam.worker", worker_args, log)
        segments = json.loads(segments_file.read_text(encoding="utf-8"))

    if diarize:
        words = sorted((w for s in segments for w in s.get("words", [])), key=lambda w: (w["start"], w["end"]))
        words_file = words_path(output_path)
        words_file.write_text(
            json.dumps({"audio": audio_path.name, "model": MODEL_NAME, "words": words}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        log(f"Слова с таймкодами: {words_file}")
        paragraphs = transcript.speaker_paragraphs(
            words, speaker_labels.load_turns(diarization_path(output_path)), keep_fillers, log,
        )
        speakers = len({p[2] for p in paragraphs})
        log(f"Абзацев: {len(paragraphs)}, спикеров в тексте: {speakers}")
    else:
        paragraphs = transcript.plain_paragraphs(segments, keep_fillers, log)
        log(f"Всего абзацев после перегруппировки: {len(paragraphs)}")

    transcript.write_transcript(output_path, audio_path.name, paragraphs, MODEL_NAME, keep_fillers, diarize)
    log(f"Готово: {output_path}")
    return output_path
