"""Общие утилиты для transcribe.py и restore_punctuation.py."""

import re

MAX_PAUSE_SEC = 3.0        # пауза между сегментами, после которой начинается новый абзац/кусок
MAX_PARAGRAPH_CHARS = 800  # используется только в transcribe.py (сырой, ещё не пунктуированный текст)

# Whisper иногда сам расставляет часть пунктуации — убираем её перед подачей
# в модель восстановления, чтобы не смешивать два источника пунктуации.
_PUNCT_STRIP_RE = re.compile(r"[^\w\s]", re.UNICODE)


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def clean_words_only(text: str) -> str:
    """Убирает всю пунктуацию, оставляет только слова через пробел."""
    text = _PUNCT_STRIP_RE.sub(" ", text)
    return " ".join(text.split())


def is_uncertain(segment, confidence_threshold=-0.5, no_speech_threshold=0.6):
    """Похож ли сегмент на распознанный ненадёжно (стоит перепроверить на слух)."""
    avg_logprob = segment.get("avg_logprob")
    no_speech_prob = segment.get("no_speech_prob")
    if avg_logprob is not None and avg_logprob < confidence_threshold:
        return True
    if no_speech_prob is not None and no_speech_prob > no_speech_threshold:
        return True
    return False


def group_into_paragraphs(segments, max_pause=MAX_PAUSE_SEC, max_chars=MAX_PARAGRAPH_CHARS,
                           confidence_threshold=-0.5, no_speech_threshold=0.6):
    """Группировка сырых (ещё не пунктуированных) сегментов для transcribe.py.
    Возвращает список (start, text, uncertain) — uncertain=True, если абзац
    содержит хотя бы один сегмент с низкой уверенностью распознавания."""
    paragraphs = []
    current_texts = []
    current_start = None
    current_len = 0
    current_uncertain = False
    last_end = None

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue

        starts_new = current_texts and (
            (seg["start"] - last_end > max_pause) or (current_len + len(text) > max_chars)
        )
        if starts_new:
            paragraphs.append((current_start, " ".join(current_texts), current_uncertain))
            current_texts = []
            current_len = 0
            current_uncertain = False

        if not current_texts:
            current_start = seg["start"]

        current_texts.append(text)
        current_len += len(text)
        current_uncertain = current_uncertain or is_uncertain(seg, confidence_threshold, no_speech_threshold)
        last_end = seg["end"]

    if current_texts:
        paragraphs.append((current_start, " ".join(current_texts), current_uncertain))

    return paragraphs


def segment_words_with_timestamps(segment, confidence_threshold=-0.5, no_speech_threshold=0.6):
    """
    Разбивает сегмент на отдельные слова (без пунктуации), присваивая каждому
    слову метку времени линейной интерполяцией между start и end сегмента,
    и признак uncertain (унаследованный от всего сегмента — более точной,
    пословной оценки уверенности faster-whisper не даёт).
    Точных пословных таймкодов в сохранённом JSON нет, это приближение —
    для навигации по аудио с точностью до абзаца этого достаточно.
    Возвращает список (word, timestamp, uncertain).
    """
    words = clean_words_only(segment["text"]).split()
    if not words:
        return []
    start, end = segment["start"], segment["end"]
    n = len(words)
    uncertain = is_uncertain(segment, confidence_threshold, no_speech_threshold)
    return [(w, start + (end - start) * i / n, uncertain) for i, w in enumerate(words)]


def group_segments_by_pause(segments, max_pause=MAX_PAUSE_SEC):
    """Группирует сегменты в крупные куски только по паузам речи (без символьного
    лимита) — используется, чтобы подавать в модель пунктуации максимально
    цельные по смыслу куски."""
    chunks = []
    current = []
    last_end = None
    for seg in segments:
        if current and seg["start"] - last_end > max_pause:
            chunks.append(current)
            current = []
        current.append(seg)
        last_end = seg["end"]
    if current:
        chunks.append(current)
    return chunks


def group_into_sentence_paragraphs(word_ts_triples, target_chars=500):
    """
    Формирует абзацы итогового текста из уже пунктуированных слов.
    Разрез делается ТОЛЬКО сразу после конца предложения (после '.' или '?'),
    когда накопленная длина абзаца достигла target_chars — то есть абзац
    никогда не обрывается на середине предложения.
    Принимает (word, timestamp, uncertain), возвращает (start, text, uncertain) —
    uncertain=True, если в абзац попало хоть одно слово из сегмента с низкой
    уверенностью распознавания.
    """
    paragraphs = []
    current_words = []
    current_start = None
    current_uncertain = False

    for word, ts, uncertain in word_ts_triples:
        if not current_words:
            current_start = ts
        current_words.append(word)
        current_uncertain = current_uncertain or uncertain
        if word and word[-1] in ".?" and len(" ".join(current_words)) >= target_chars:
            paragraphs.append((current_start, " ".join(current_words), current_uncertain))
            current_words = []
            current_uncertain = False

    if current_words:
        paragraphs.append((current_start, " ".join(current_words), current_uncertain))

    return paragraphs
