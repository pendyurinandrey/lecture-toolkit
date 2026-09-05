"""
Удаление слов-паразитов из потока слов ДО подачи в модель восстановления
пунктуации — так модель расставляет знаки препинания уже для чистого текста,
и не нужно потом чинить осиротевшие запятые на месте вырезанных слов.

Список намеренно консервативный: только однозначные слова-паразиты без
самостоятельного смысла. Многозначные слова вроде "значит" (может быть
и паразитом, и логической связкой "следовательно") или "просто" ("just")
сюда не включены — их автоматическое удаление слишком часто искажало бы
смысл. Список легко расширить под конкретную лекцию/лектора.
"""

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


def remove_fillers(word_ts_items):
    """Принимает список кортежей (word, ...любые доп. поля...), первый элемент
    всегда слово. Работает как с (word, ts), так и с (word, ts, uncertain)."""
    phrases = [p.split() for p in PHRASE_FILLERS]
    words_lower = [item[0].lower() for item in word_ts_items]
    n = len(word_ts_items)

    result = []
    i = 0
    while i < n:
        matched_len = 0
        for phrase in phrases:
            length = len(phrase)
            if words_lower[i:i + length] == phrase:
                matched_len = length
                break
        if matched_len:
            i += matched_len
            continue
        if words_lower[i] not in SIMPLE_FILLERS:
            result.append(word_ts_items[i])
        i += 1

    return result
