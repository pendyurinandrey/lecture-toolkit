"""
Обёртка над kontur-ai/sbert_punc_case_ru — восстановление пунктуации и регистра
в тексте после распознавания речи. Веса скачиваются через transformers
(huggingface_hub), git-lfs не требуется.

Источник: https://huggingface.co/kontur-ai/sbert_punc_case_ru
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_REPO = "kontur-ai/sbert_punc_case_ru"

PUNK_MAPPING = {".": "PERIOD", ",": "COMMA", "?": "QUESTION"}
LABELS_CASE = ["LOWER", "UPPER", "UPPER_TOTAL"]
LABELS_PUNC = ["O"] + list(PUNK_MAPPING.values())

LABELS_list = []
for case in LABELS_CASE:
    for punc in LABELS_PUNC:
        LABELS_list.append(f"{case}_{punc}")
LABELS = {label: i + 1 for i, label in enumerate(LABELS_list)}
LABELS["O"] = -100
INVERSE_LABELS = {i: label for label, i in LABELS.items()}


def token_to_label(token, label):
    if isinstance(label, int):
        label = INVERSE_LABELS[label]
    if label == "LOWER_O":
        return token
    if label == "LOWER_PERIOD":
        return token + "."
    if label == "LOWER_COMMA":
        return token + ","
    if label == "LOWER_QUESTION":
        return token + "?"
    if label == "UPPER_O":
        return token.capitalize()
    if label == "UPPER_PERIOD":
        return token.capitalize() + "."
    if label == "UPPER_COMMA":
        return token.capitalize() + ","
    if label == "UPPER_QUESTION":
        return token.capitalize() + "?"
    if label == "UPPER_TOTAL_O":
        return token.upper()
    if label == "UPPER_TOTAL_PERIOD":
        return token.upper() + "."
    if label == "UPPER_TOTAL_COMMA":
        return token.upper() + ","
    if label == "UPPER_TOTAL_QUESTION":
        return token.upper() + "?"
    if label == "O":
        return token


def decode_label(label):
    return INVERSE_LABELS[label]


class SbertPuncCase(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, strip_accents=False)
        self.model = AutoModelForTokenClassification.from_pretrained(MODEL_REPO)
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def punctuate(self, text):
        text = text.strip().lower()
        words = text.split()
        if not words:
            return text

        tokenizer_output = self.tokenizer(words, is_split_into_words=True)

        if len(tokenizer_output.input_ids) > 512:
            mid = len(words) // 2
            return " ".join([
                self.punctuate(" ".join(words[:mid])),
                self.punctuate(" ".join(words[mid:])),
            ])

        with torch.no_grad():
            predictions = self(
                torch.tensor([tokenizer_output.input_ids], device=self.model.device),
                torch.tensor([tokenizer_output.attention_mask], device=self.model.device),
            ).logits.cpu().data.numpy()
        predictions = np.argmax(predictions, axis=2)

        splitted_text = []
        word_ids = tokenizer_output.word_ids()
        for i, word in enumerate(words):
            label_pos = word_ids.index(i)
            label_id = predictions[0][label_pos]
            label = decode_label(label_id)
            splitted_text.append(token_to_label(word, label))
        return " ".join(splitted_text)
