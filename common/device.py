"""Выбор устройства для torch: cuda -> mps -> cpu.

Нужен и диаризации, и VAD в GigaAM: на MPS/CUDA обе задачи считаются на
порядок быстрее, а результат совпадает с CPU (проверено на записи 3ч23м:
границы VAD-сегментов и реплики диаризации идентичны). torch импортируется
только внутри функций — модуль безопасно подключать из любого процесса.
"""

import os
import sys


def prepare_torch_environment() -> None:
    """Разрешает MPS падать на CPU для операций, которых на MPS нет.

    Переменная читается при загрузке torch, поэтому вызывать нужно ДО первого
    `import torch` в процессе. Уже заданное пользователем значение не трогается."""
    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device() -> str:
    """Лучшее доступное устройство: "cuda", "mps" или "cpu"."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
