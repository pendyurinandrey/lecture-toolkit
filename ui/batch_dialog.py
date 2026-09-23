"""
Диалог «Пакетная обработка»: список файлов -> расшифровка для каждого.

Независим от главного окна и его состояния (`PipelineUI.segments` и т.п.) —
свой список файлов, свои настройки, свои значения чекбоксов по умолчанию
(не наследуются от главного окна). `fork_join` не используется: каждый файл
транскрибируется целиком, без нарезки/склейки — см. speech_to_text_gigaam/batch.py.

Окно немодальное: открывается через .show(), а не .exec(). Закрытие крестиком
во время обработки не прерывает её — фоновый поток продолжает работать, окно
просто скрывается (стандартное поведение QDialog.closeEvent без
Qt::WA_DeleteOnClose); PipelineUI держит один и тот же экземпляр диалога и
повторно показывает его по пункту меню, так что прогресс/журнал никуда не
пропадают.
"""

import threading
from contextlib import nullcontext
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from diarization.diarize_pyannote import DiarizationError
from speech_to_text_gigaam import batch
from speech_to_text_gigaam.environment import EnvironmentCheckError
from ui import media_selection


class BatchDialog(QDialog):
    log_signal = Signal(str)
    progress_signal = Signal(int, int, str)  # (номер файла, всего файлов, имя файла)
    done_signal = Signal(object)             # несёт batch.BatchResult
    error_signal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Пакетная обработка")
        self.resize(640, 640)

        self.files = []  # list[Path], в порядке обработки
        self.worker_thread = None
        self._stop_requested = False

        self.log_signal.connect(self._log)
        self.progress_signal.connect(self._on_progress)
        self.done_signal.connect(self._on_done)
        self.error_signal.connect(self._on_error)

        self._build_widgets()
        self._refresh_list()

    # ------------------------------------------------------------------ UI

    def _build_widgets(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Файлы для распознавания (видео или аудио, каждый целиком — в свой транскрипт):"))

        self.file_list = QListWidget()
        layout.addWidget(self.file_list, stretch=1)

        list_buttons = QHBoxLayout()
        self.add_button = QPushButton("Добавить файлы")
        self.add_button.clicked.connect(self._add_files)
        list_buttons.addWidget(self.add_button)
        up_btn = QPushButton("Выше")
        up_btn.clicked.connect(lambda: self._move_selected(-1))
        list_buttons.addWidget(up_btn)
        down_btn = QPushButton("Ниже")
        down_btn.clicked.connect(lambda: self._move_selected(1))
        list_buttons.addWidget(down_btn)
        delete_btn = QPushButton("Удалить")
        delete_btn.clicked.connect(self._delete_selected)
        list_buttons.addWidget(delete_btn)
        list_buttons.addStretch()
        layout.addLayout(list_buttons)

        self.remove_fillers_check = QCheckBox("Убирать слова-паразиты (вот, ну и т.п.)")
        self.remove_fillers_check.setChecked(True)
        layout.addWidget(self.remove_fillers_check)

        self.diarize_check = QCheckBox("Определять говорящих (медленнее)")
        self.diarize_check.setToolTip(
            "Реплики помечаются «Спикер 1», «Спикер 2»… по порядку появления в записи.\n"
            "Без видеокарты NVIDIA (CUDA) или Apple Silicon (MPS) работает очень медленно."
        )
        layout.addWidget(self.diarize_check)

        self.keep_awake_check = QCheckBox("Не давать компьютеру уснуть во время обработки")
        self.keep_awake_check.setChecked(True)
        layout.addWidget(self.keep_awake_check)

        run_buttons = QHBoxLayout()
        run_buttons.addStretch()
        self.run_button = QPushButton("Запустить пакетную обработку")
        self.run_button.clicked.connect(self._start)
        run_buttons.addWidget(self.run_button)
        self.stop_button = QPushButton("Остановить")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        run_buttons.addWidget(self.stop_button)
        run_buttons.addStretch()
        layout.addLayout(run_buttons)

        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        layout.addWidget(QLabel("Журнал:"))
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text, stretch=1)

    # --------------------------------------------------------------- список

    def _refresh_list(self) -> None:
        self.file_list.clear()
        for path in self.files:
            text = batch.display_name(path, self.files)
            if batch.transcript_path_for(path).exists():
                text += "  (готово)"
            item = QListWidgetItem(text)
            item.setToolTip(str(path))
            self.file_list.addItem(item)
        self.run_button.setEnabled(bool(self.files))

    def _add_files(self) -> None:
        movies_dir = Path.home() / "Movies"
        default_dir = str(movies_dir) if movies_dir.is_dir() else ""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Добавить файлы", default_dir, media_selection.open_file_filter(None),
        )
        if not paths:
            return
        self.files = batch.add_files(self.files, [Path(p) for p in paths])
        self._refresh_list()

    def _move_selected(self, direction: int) -> None:
        row = self.file_list.currentRow()
        if row < 0:
            return
        new_row = row + direction
        if not (0 <= new_row < len(self.files)):
            return
        self.files[row], self.files[new_row] = self.files[new_row], self.files[row]
        self._refresh_list()
        self.file_list.setCurrentRow(new_row)

    def _delete_selected(self) -> None:
        row = self.file_list.currentRow()
        if row < 0:
            return
        del self.files[row]
        self._refresh_list()

    # ------------------------------------------------------------- запуск

    def _log(self, message: str) -> None:
        self.log_text.appendPlainText(message)

    def _on_progress(self, index: int, total: int, name: str) -> None:
        self.progress_label.setText(f"Файл {index} из {total}: {name}")
        self.progress.setValue(index)

    def _start(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        if not self.files:
            return

        diarize = self.diarize_check.isChecked()
        try:
            batch.check_environment(diarize)
        except (EnvironmentCheckError, DiarizationError) as e:
            QMessageBox.critical(self, "Окружение не готово", str(e))
            return

        # Уже готовые файлы (есть транскрипт) пропускаются молча — без подтверждения:
        # это и есть механизм восстановления после сбоя (см. batch.run_batch).
        items = batch.plan_batch(self.files)

        self._stop_requested = False
        self.progress.setRange(0, len(items))
        self.progress.setValue(0)
        self.progress_label.setText("")
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._log("Запуск пакетной обработки...")

        remove_fillers = self.remove_fillers_check.isChecked()
        keep_awake = self.keep_awake_check.isChecked()
        self.worker_thread = threading.Thread(
            target=self._worker, args=(items, remove_fillers, diarize, keep_awake), daemon=True,
        )
        self.worker_thread.start()

    def _stop(self) -> None:
        self._stop_requested = True
        self.stop_button.setEnabled(False)
        self._log("Останавливаю после текущего файла...")

    def _worker(self, items, remove_fillers, diarize, keep_awake) -> None:
        log = lambda msg: self.log_signal.emit(msg)  # noqa: E731

        if keep_awake:
            from wakepy import keep as wakepy_keep
            wake_cm = wakepy_keep.running(on_fail="pass")
        else:
            wake_cm = nullcontext()

        log_path = batch.log_file_path(items[0].path)

        try:
            with wake_cm as mode:
                if keep_awake and not getattr(mode, "active", True):
                    log("Предупреждение: не удалось предотвратить сон компьютера — "
                        "обработка продолжится в обычном режиме.")

                log(f"Лог-файл пачки: {log_path}")
                with batch.batch_log_file(log_path, log) as disk_log:
                    result = batch.run_batch(
                        items, keep_fillers=not remove_fillers, diarize=diarize, log=disk_log,
                        on_progress=lambda i, t, p: self.progress_signal.emit(i, t, p.name),
                        should_stop=lambda: self._stop_requested,
                    )
            self.done_signal.emit(result)
        except Exception as e:  # noqa: BLE001 - показать пользователю любую неожиданную ошибку
            self.error_signal.emit(f"Непредвиденная ошибка: {e}")

    def _on_done(self, result) -> None:
        self._refresh_list()  # обновить пометки "(готово)"
        lines = [f"Готово: обработано {len(result.processed)}, пропущено {len(result.skipped)}, "
                f"ошибок {len(result.failed)}."]
        if result.stopped:
            lines.append("Остановлено пользователем до конца списка.")
        for path, message in result.failed:
            lines.append(f"  {path.name}: {message}")
        summary = "\n".join(lines)
        self._log(summary)
        QMessageBox.information(self, "Пакетная обработка завершена", summary)
        self.run_button.setEnabled(bool(self.files))
        self.stop_button.setEnabled(False)

    def _on_error(self, message: str) -> None:
        self._log(f"Ошибка: {message}")
        QMessageBox.critical(self, "Ошибка", message)
        self.run_button.setEnabled(bool(self.files))
        self.stop_button.setEnabled(False)
