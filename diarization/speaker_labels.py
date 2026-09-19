"""
Склейка слов транскрипта с результатом диаризации — общая для любых движков
диаризации (сейчас diarize_pyannote.py). Без тяжёлых импортов: модуль
подключается из GUI-процесса.

Контракт с движками диаризации — JSON-файл:
    {"model": "...", "turns": [{"start": 1.2, "end": 5.7, "speaker": "SPEAKER_00"}, ...]}
Реплики (turns) — БЕЗ наложений друг на друга (для pyannote это
exclusive_speaker_diarization); время в секундах от начала записи; метка
спикера — любая строка, уникальная для человека в пределах одной записи.

Слова — словари {"text": ..., "start": ..., "end": ...} (секунды), как их
отдаёт GigaAM при word_timestamps=True.
"""

import json
from bisect import bisect_right
from pathlib import Path

SPEAKER_PREFIX = "Спикер"
MICRO_TURN_SEC = 0.5      # реплики короче считаются шумом диаризации
MICRO_TURN_MAX_GAP = 0.5  # ...если рядом (не дальше этого) есть другая реплика


def load_turns(path) -> list:
    """Читает файл диаризации, возвращает реплики, отсортированные по началу."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return sorted(data["turns"], key=lambda t: (t["start"], t["end"]))


def smooth_turns(turns: list, min_duration: float = MICRO_TURN_SEC, max_gap: float = MICRO_TURN_MAX_GAP) -> list:
    """Убирает «дрожание» диаризации: микрореплики (короче min_duration),
    зажатые между другими репликами, вливаются в более длинного соседа.

    На стыке двух голосов pyannote иногда выдаёт цепочку реплик по 0.03-0.2 с,
    чередующихся между спикерами, — слова внутри такой цепочки разлетались бы
    по разным спикерам. Настоящая реплика (даже «Угу») длится дольше.
    Микрореплика, у которой нет соседа ближе max_gap (одинокая короткая
    реплика среди тишины), не трогается. Самые короткие обрабатываются первыми.
    Возвращает новый список, исходный не меняется."""
    turns = [dict(t) for t in sorted(turns, key=lambda t: (t["start"], t["end"]))]

    def length(t):
        return t["end"] - t["start"]

    while True:
        micro = sorted((i for i, t in enumerate(turns) if length(t) < min_duration), key=lambda i: length(turns[i]))
        for i in micro:
            t = turns[i]
            neighbours = [
                j for j in (i - 1, i + 1)
                if 0 <= j < len(turns)
                and (t["start"] - turns[j]["end"] if j < i else turns[j]["start"] - t["end"]) <= max_gap
            ]
            if neighbours:
                break
        else:
            return turns
        target = max(neighbours, key=lambda j: (length(turns[j]), j < i))  # при равенстве — предыдущий
        turns[target]["start"] = min(turns[target]["start"], t["start"])
        turns[target]["end"] = max(turns[target]["end"], t["end"])
        del turns[i]
        # слившиеся одинаковые спикеры рядом — в одну реплику
        merged = []
        for cur in turns:
            if merged and merged[-1]["speaker"] == cur["speaker"] and cur["start"] - merged[-1]["end"] <= max_gap:
                merged[-1]["end"] = max(merged[-1]["end"], cur["end"])
            else:
                merged.append(cur)
        turns = merged


def _distance(turn, moment: float) -> float:
    if turn["start"] <= moment <= turn["end"]:
        return 0.0
    return min(abs(moment - turn["start"]), abs(moment - turn["end"]))


def assign_speaker(word: dict, turns: list, starts: list):
    """Метка спикера для слова: с наибольшим перекрытием по времени; если
    слово целиком в паузе между репликами (или без длительности) — спикер
    ближайшей реплики. turns отсортированы по началу, starts — их начала."""
    w_start, w_end = word["start"], word["end"]

    overlap = {}
    # Реплики не пересекаются, поэтому назад от первой реплики, начавшейся
    # после конца слова, достаточно пройти, пока реплики ещё касаются слова.
    i = bisect_right(starts, w_end) - 1
    while i >= 0 and turns[i]["end"] > w_start:
        t = turns[i]
        common = min(w_end, t["end"]) - max(w_start, t["start"])
        if common > 0:
            overlap[t["speaker"]] = overlap.get(t["speaker"], 0.0) + common
        i -= 1
    if overlap:
        return max(overlap, key=overlap.get)

    middle = (w_start + w_end) / 2
    j = bisect_right(starts, middle)
    neighbours = turns[max(j - 1, 0):j + 1]
    return min(neighbours, key=lambda t: _distance(t, middle))["speaker"]


def label_words(words: list, turns: list) -> list:
    """Номера спикеров (1, 2, ...) для каждого слова, в порядке слов.
    Нумерация — по порядку первого появления в тексте с начала записи."""
    if not turns:
        return [1] * len(words)

    starts = [t["start"] for t in turns]
    numbers = {}
    result = []
    for word in words:
        raw = assign_speaker(word, turns, starts)
        result.append(numbers.setdefault(raw, len(numbers) + 1))
    return result


def format_speaker(number: int) -> str:
    return f"{SPEAKER_PREFIX} {number}"
