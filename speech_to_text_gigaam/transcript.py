"""Сборка итогового текста транскрипта из сегментов/слов GigaAM.

GigaAM сам расставляет пунктуацию и регистр, но режет текст по паузам в речи
(VAD-сегменты) — получается много мелких абзацев, часто обрывающихся
посередине предложения. Здесь результат перегруппировывается: абзацы рвутся
только после конца предложения ('.', '?' или '!'), а при диаризации ещё и при
смене говорящего.

Ничего тяжёлого не импортируется (ни gigaam, ни torch) — модуль целиком
тестируется без нейросетей.
"""

from itertools import groupby
from pathlib import Path

from common.filler_words import remove_fillers
from diarization import diarize_pyannote, speaker_labels

SENTENCE_END_CHARS = ".?!"
PARAGRAPH_TARGET_CHARS = 500


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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


def write_transcript(output_path, audio_name: str, paragraphs: list, model_name: str,
                     keep_fillers: bool, diarize: bool) -> None:
    """Пишет итоговый TXT: заголовок и абзацы «[ЧЧ:ММ:СС] текст».
    При diarize абзацы — тройки (время, текст, номер спикера) и получают метку «Спикер N: »."""
    with Path(output_path).open("w", encoding="utf-8") as f:
        f.write(f"# {audio_name}\n")
        f.write(f"# Транскрибация: GigaAM {model_name} (longform), перегруппировано по предложениям")
        f.write(", слова-паразиты удалены\n" if not keep_fillers else "\n")
        if diarize:
            f.write(f"# Говорящие: {diarize_pyannote.MODEL_ID}; «{speaker_labels.SPEAKER_PREFIX} N» — "
                    "по порядку первого появления в записи\n")
        f.write("\n")
        for paragraph in paragraphs:
            start, text = paragraph[0], paragraph[1]
            label = f"{speaker_labels.format_speaker(paragraph[2])}: " if diarize else ""
            f.write(f"[{format_timestamp(start)}] {label}{text}\n\n")
