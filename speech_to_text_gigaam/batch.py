"""
Пакетная обработка: список файлов -> расшифровка для каждого, одним запуском.

Для папки с независимыми лекциями — каждый файл целиком превращается в свой
собственный транскрипт, без нарезки и склейки (`fork_join` не используется):
`transcribe.run()` сам достаёт звук из любого файла, видео или аудио.

Обработка идёт строго последовательно (общий ресурс CPU/MPS и память). Модуль
ничего тяжёлого не импортирует на верхнем уровне — его безопасно подключать
из GUI-процесса (см. CLAUDE.md, "Тяжёлое — в отдельных процессах").
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from diarization import diarize_pyannote
from speech_to_text_gigaam import transcribe
from speech_to_text_gigaam.environment import check_ffmpeg_compatibility, ensure_hf_token

LOG_FILE_PREFIX = "lecture-toolkit-batch-"


@dataclass(frozen=True)
class BatchItem:
    """Один файл пачки. already_done — снимок на момент постройки плана (для
    пометки "(готово)" в списке ДО запуска); run_batch() перепроверяет заново
    прямо перед обработкой каждого файла, а не полагается на это поле."""
    path: Path
    transcript_path: Path
    already_done: bool


@dataclass
class BatchResult:
    processed: list = field(default_factory=list)  # list[Path]
    skipped: list = field(default_factory=list)     # list[Path] — уже был готов транскрипт
    failed: list = field(default_factory=list)      # list[tuple[Path, str]] — (путь, текст ошибки)
    stopped: bool = False                           # True, если прервано кнопкой "Остановить"


def transcript_path_for(path) -> Path:
    """<файл> -> <файл>.txt рядом с исходником."""
    return Path(path).with_suffix(".txt")


def sort_new_files(paths) -> list:
    """Новый набор файлов, отсортированный по имени (лексикографически, по
    возрастанию, обычное сравнение строк Python — без учёта локали). Порядок
    выбора в системном диалоге платформозависим и Qt его не гарантирует,
    поэтому не используется вовсе."""
    return sorted((Path(p) for p in paths), key=lambda p: p.name)


def add_files(existing: list, new_paths: list) -> list:
    """existing + новые файлы (отсортированные по имени), добавленные в конец.
    Порядок уже добавленных файлов не меняется. Файл, путь которого уже есть
    в existing (в том числе повторяющийся внутри самого new_paths), повторно
    не добавляется."""
    seen = set(existing)
    result = list(existing)
    for path in sort_new_files(new_paths):
        if path not in seen:
            result.append(path)
            seen.add(path)
    return result


def display_name(path, all_paths) -> str:
    """Имя файла для показа в списке. Если несколько путей в all_paths
    называются одинаково — добавляет родительскую папку в скобках, чтобы
    различить их (путь для обработки при этом всегда используется полный,
    отображение ни на что не влияет)."""
    path = Path(path)
    same_name = sum(1 for p in all_paths if Path(p).name == path.name)
    if same_name > 1:
        return f"{path.name} ({path.parent.name})"
    return path.name


def plan_batch(files) -> list:
    """Список файлов -> план (BatchItem на каждый), с текущим состоянием
    "уже готово" для отображения в списке до запуска."""
    items = []
    for path in files:
        path = Path(path)
        transcript_path = transcript_path_for(path)
        items.append(BatchItem(path=path, transcript_path=transcript_path, already_done=transcript_path.exists()))
    return items


def log_file_path(first_file, when: Optional[datetime] = None) -> Path:
    """Путь к лог-файлу пачки: рядом с первым файлом списка (на момент
    запуска), с меткой времени в имени — чтобы не перезаписать лог
    предыдущего запуска в той же папке."""
    timestamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return Path(first_file).parent / f"{LOG_FILE_PREFIX}{timestamp}.log"


@contextmanager
def batch_log_file(file_path, log: Callable[[str], None]):
    """Пока активен контекст, возвращённая log-функция одновременно вызывает
    исходную log() и дописывает строку в file_path — построчно, с немедленным
    flush, чтобы при краше приложения в файле остался весь прогресс до сбоя.
    Файл создаётся сразу при входе в контекст, даже если строк ещё не было."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        def combined(message: str) -> None:
            log(message)
            handle.write(message + "\n")
            handle.flush()

        yield combined


def check_environment(diarize: bool) -> None:
    """Проверки окружения перед стартом ВСЕЙ пачки, один раз, а не на каждый
    файл — fail fast. Бросает EnvironmentCheckError или DiarizationError."""
    check_ffmpeg_compatibility()
    ensure_hf_token()
    if diarize:
        diarize_pyannote.check_model_available()


def run_batch(items, keep_fillers: bool, diarize: bool, log=print,
              on_progress: Optional[Callable[[int, int, Path], None]] = None,
              should_stop: Optional[Callable[[], bool]] = None) -> BatchResult:
    """Обрабатывает items строго последовательно, с первого до последнего.

    should_stop() проверяется ПЕРЕД каждым файлом — начатый файл всегда
    дообрабатывается до конца, не прерывается на середине. Ошибка на одном
    файле не прерывает пачку: попадает в result.failed, обработка идёт дальше
    (как и остановка, это осознанное решение, не баг). "Уже готово"
    проверяется заново прямо перед файлом (а не берётся из BatchItem), на
    случай, если что-то изменилось между построением плана и этим моментом.
    """
    should_stop = should_stop or (lambda: False)
    total = len(items)
    result = BatchResult()

    for index, item in enumerate(items, start=1):
        if should_stop():
            result.stopped = True
            log("Остановлено пользователем.")
            break

        if on_progress:
            on_progress(index, total, item.path)

        if item.transcript_path.exists():
            log(f"Пропущено (уже обработано): {item.path.name}")
            result.skipped.append(item.path)
            continue

        log(f"=== Файл {index}/{total}: {item.path.name} ===")
        try:
            transcribe.run(item.path, item.transcript_path, keep_fillers=keep_fillers, diarize=diarize, log=log)
            result.processed.append(item.path)
        except Exception as e:  # noqa: BLE001 - ошибка одного файла не должна ронять всю пачку
            message = str(e)
            log(f"Ошибка: {item.path.name}: {message}")
            result.failed.append((item.path, message))

    return result
