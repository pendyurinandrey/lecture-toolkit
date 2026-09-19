"""remove_fillers: слова-паразиты и перенос пунктуации/заглавных букв."""

from common.filler_words import remove_fillers


def cleaned(text: str) -> list:
    words = text.split()
    return [item[0] for item in remove_fillers([(w, float(i)) for i, w in enumerate(words)])]


def test_simple_fillers_are_removed():
    assert cleaned("Мы вот работаем ну хорошо") == ["Мы", "работаем", "хорошо"]


def test_text_without_fillers_is_unchanged():
    assert cleaned("Просто обычный текст") == ["Просто", "обычный", "текст"]


def test_empty_input():
    assert remove_fillers([]) == []


def test_ambiguous_words_are_kept():
    # «значит» и «просто» намеренно не в списке: могут нести смысл
    assert cleaned("Значит, так") == ["Значит,", "так"]


def test_phrase_fillers():
    assert cleaned("Мы как бы уже начали") == ["Мы", "уже", "начали"]
    assert cleaned("Это, так сказать, важно") == ["Это,", "важно"]


def test_sentence_end_punctuation_moves_to_previous_word():
    assert cleaned("Это важно, вот. Дальше идём") == ["Это", "важно.", "Дальше", "идём"]
    assert cleaned("Так вот? Да") == ["Так?", "Да"]


def test_capital_letter_moves_to_next_word():
    assert cleaned("Ну, давайте начнём. Вот, смотрите") == ["Давайте", "начнём.", "Смотрите"]


def test_opening_quote_moves_to_next_word():
    assert cleaned("Он сказал «ну давайте»") == ["Он", "сказал", "«давайте»"]


def test_extra_fields_are_preserved():
    # кортежи (слово, время, спикер): доп. поля не теряются
    result = remove_fillers([("Ну,", 1.0, 7), ("да", 2.0, 7)])
    assert result == [("Да", 2.0, 7)]
