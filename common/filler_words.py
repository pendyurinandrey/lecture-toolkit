"""
Удаление слов-паразитов из потока слов — общий модуль для speech-to-text-whisper
и speech-to-text-gigaam.

Слова на входе могут быть как ещё без пунктуации (speech-to-text-whisper —
паразиты убираются до восстановления пунктуации), так и уже с пунктуацией
("вот,", "«ну", "ж." — speech-to-text-gigaam, где ASR сама расставляет знаки).
На чистом, ещё не пунктуированном тексте вся логика ниже, завязанная на
знаки препинания и заглавные буквы, просто не активируется — поведение
сводится к обычному удалению слова из списка. Поэтому одна реализация
корректно обслуживает оба сценария:

- концевая пунктуация переходит на предыдущее оставленное слово, заменяя
  его собственную, только если она "сильнее" (. ! ? сильнее ,) — иначе
  получился бы мусор вида "ребята,.";
- ведущая пунктуация (открывающая кавычка) переходит на следующее
  оставленное слово — иначе кавычка открылась бы и потерялась, оставив
  висящую закрывающую без пары;
- если перенос концевой пунктуации создаёт новую границу предложения, или
  само удаляемое слово было началом предложения (с большой буквы), эта
  роль (заглавная буква) переходит следующему оставшемуся слову.

Список слов намеренно консервативный: только однозначные слова-паразиты без
самостоятельного смысла. Многозначные слова вроде "значит" (может быть и
паразитом, и логической связкой "следовательно") или "просто" ("just") сюда
не включены — их автоматическое удаление слишком часто искажало бы смысл.
"""

import re

SIMPLE_FILLERS = {
    "вот",
    "ну",
}

PHRASE_FILLERS = [
    "ну что ж",
    "так сказать",
    "как бы",
    "это самое",
    "в общем-то",
]

SENTENCE_END_CHARS = ".!?"

_LEADING_PUNCT_RE = re.compile(r'^[«"\']+')
_TRAILING_PUNCT_RE = re.compile(r'[.,!?…»"\']+$')


def _leading_punct(word: str) -> str:
    m = _LEADING_PUNCT_RE.match(word)
    return m.group(0) if m else ""


def _trailing_punct(word: str) -> str:
    m = _TRAILING_PUNCT_RE.search(word)
    return m.group(0) if m else ""


def _strip_leading(word: str) -> str:
    return _LEADING_PUNCT_RE.sub("", word)


def _bare_lower(word: str) -> str:
    word = _LEADING_PUNCT_RE.sub("", word)
    word = _TRAILING_PUNCT_RE.sub("", word)
    return word.lower()


def _is_sentence_end(punct: str) -> bool:
    return any(c in SENTENCE_END_CHARS for c in punct)


def remove_fillers(word_ts_items):
    """Принимает список кортежей (word, ...любые доп. поля...), первый элемент
    всегда слово (возможно, с пунктуацией). Работает как с (word, ts), так
    и с (word, ts, uncertain)."""
    phrases = [p.split() for p in PHRASE_FILLERS]
    bare_lower = [_bare_lower(item[0]) for item in word_ts_items]
    n = len(word_ts_items)

    result = []
    new_sentence_boundary = False  # следующее добавленное слово нужно капитализировать
    pending_leading_punct = ""     # открывающая кавычка, потерявшая своё слово

    def carry_trailing_punct(removed_word: str) -> None:
        nonlocal new_sentence_boundary
        punct = _trailing_punct(removed_word)
        if not punct or not result:
            return
        word, *rest = result[-1]
        existing = _trailing_punct(word)
        if _is_sentence_end(punct) and not _is_sentence_end(existing):
            base = word[:len(word) - len(existing)] if existing else word
            result[-1] = (base + punct, *rest)
            new_sentence_boundary = True
        elif not existing:
            result[-1] = (word + punct, *rest)

    def carry_leading_punct(removed_word: str) -> None:
        nonlocal pending_leading_punct
        punct = _leading_punct(removed_word)
        if punct:
            pending_leading_punct += punct

    def mark_if_was_capitalized(first_word: str) -> None:
        # Если удаляемое слово само начинало предложение (с большой буквы),
        # эта роль переходит следующему оставшемуся слову.
        nonlocal new_sentence_boundary
        core = _strip_leading(first_word)
        if core[:1].isalpha() and core[:1].isupper():
            new_sentence_boundary = True

    def append(item) -> None:
        nonlocal new_sentence_boundary, pending_leading_punct
        word = item[0]
        if pending_leading_punct:
            word = pending_leading_punct + word
            pending_leading_punct = ""
        if new_sentence_boundary:
            for idx, ch in enumerate(word):
                if ch.isalpha():
                    if ch.islower():
                        word = word[:idx] + ch.upper() + word[idx + 1:]
                    break
        result.append((word, *item[1:]))
        new_sentence_boundary = False

    i = 0
    while i < n:
        matched_len = 0
        for phrase in phrases:
            length = len(phrase)
            if bare_lower[i:i + length] == phrase:
                matched_len = length
                break
        if matched_len:
            mark_if_was_capitalized(word_ts_items[i][0])
            carry_leading_punct(word_ts_items[i][0])
            carry_trailing_punct(word_ts_items[i + matched_len - 1][0])
            i += matched_len
            continue

        if bare_lower[i] in SIMPLE_FILLERS:
            mark_if_was_capitalized(word_ts_items[i][0])
            carry_leading_punct(word_ts_items[i][0])
            carry_trailing_punct(word_ts_items[i][0])
            i += 1
            continue

        append(word_ts_items[i])
        i += 1

    return result
