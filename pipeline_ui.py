#!/usr/bin/env python3
"""
Графический интерфейс для полного конвейера обработки лекции:
fork-join (обязательный шаг) + speech-to-text через GigaAM-v3 (опциональный).

Реализовано на PySide6 (Qt) — до этого использовался Tkinter, но его
управление окнами/фокусом на macOS регулярно приводило к тому, что клики по
кнопкам переставали регистрироваться после модальных диалогов (см. историю
разработки). Qt использует собственное, значительно более зрелое управление
окнами/модальностью, что должно снять эту категорию проблем.

Использование:
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
    python3 pipeline_ui.py
"""

import json
import os
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenuBar,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fork_join import fork_join
from speech_to_text_gigaam import transcribe_longform_chunked
from speech_to_text_gigaam.transcribe_longform import (
    EnvironmentCheckError,
    check_ffmpeg_compatibility,
    ensure_hf_token,
)

VIDEO_FILTER = "Видео MP4 (*.mp4);;Все файлы (*)"
AUDIO_FILTER = "M4A аудио (*.m4a)"
TEXT_FILTER = "Текстовый файл (*.txt)"


class FragmentDialog(QDialog):
    """Модальный диалог ввода start/end фрагмента в формате HH:mm:ss."""

    def __init__(self, parent=None, start: str = "00:00:00", end: str = "00:00:00"):
        super().__init__(parent)
        self.setWindowTitle("Фрагмент")
        self.result_value = None

        layout = QGridLayout(self)

        layout.addWidget(QLabel("Начало (ЧЧ:ММ:СС):"), 0, 0)
        self.start_edit = QLineEdit(start)
        layout.addWidget(self.start_edit, 0, 1)

        layout.addWidget(QLabel("Конец (ЧЧ:ММ:СС):"), 1, 0)
        self.end_edit = QLineEdit(end)
        layout.addWidget(self.end_edit, 1, 1)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_ok)
        buttons.addWidget(ok_btn)
        layout.addLayout(buttons, 2, 0, 1, 2)

        self.start_edit.setFocus()
        self.start_edit.selectAll()

    def _on_ok(self):
        start = self.start_edit.text().strip()
        end = self.end_edit.text().strip()
        try:
            start_s = fork_join.hhmmss_to_seconds(start)
            end_s = fork_join.hhmmss_to_seconds(end)
        except ValueError as e:
            QMessageBox.critical(self, "Некорректное время", str(e))
            return
        if end_s <= start_s:
            QMessageBox.critical(
                self, "Некорректный диапазон",
                f'"Конец" ({end}) должен быть больше "Начала" ({start})',
            )
            return
        self.result_value = {"start": start, "end": end}
        self.accept()


class PipelineUI(QWidget):
    # Сигналы для обновления UI из фонового потока — Qt автоматически
    # маршалит их на главный поток (аналог queue.Queue + polling в Tkinter,
    # но встроенный и надёжный).
    log_signal = Signal(str)
    done_signal = Signal()
    error_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lecture Pipeline")

        self.segments = []  # [{"path": str, "fragments": [{"start": str, "end": str}, ...]}]
        self.worker_thread = None

        self.log_signal.connect(self._log)
        self.done_signal.connect(self._on_done)
        self.error_signal.connect(self._on_error)

        self._build_widgets()
        self._refresh_tree()
        self._center_window(780, 760)

    def _center_window(self, width: int, height: int) -> None:
        self.resize(width, height)
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.x() + (screen.width() - width) // 2
        y = screen.y() + (screen.height() - height) // 2
        self.move(x, y)

    # ------------------------------------------------------------------ UI

    def _build_menu_bar(self) -> QMenuBar:
        menu_bar = QMenuBar(self)
        file_menu = menu_bar.addMenu("Файл")

        new_action = QAction("Новый", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._reset_all)
        file_menu.addAction(new_action)

        preview_action = QAction("Предпросмотр JSON", self)
        preview_action.triggered.connect(self._preview_json)
        file_menu.addAction(preview_action)

        file_menu.addSeparator()

        quit_action = QAction("Выход", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        return menu_bar

    def _build_widgets(self):
        layout = QVBoxLayout(self)
        layout.setMenuBar(self._build_menu_bar())

        title_label = QLabel("Видеофайлы и фрагменты")
        font = title_label.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        title_label.setFont(font)
        layout.addWidget(title_label)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Путь к файлу / фрагмент", "Начало", "Конец"])
        self.tree.setColumnWidth(0, 420)
        layout.addWidget(self.tree, stretch=1)

        tree_buttons = QHBoxLayout()
        add_seg_btn = QPushButton("Добавить видеофайл")
        add_seg_btn.clicked.connect(self._add_segment)
        tree_buttons.addWidget(add_seg_btn)
        add_frag_btn = QPushButton("Добавить фрагмент")
        add_frag_btn.clicked.connect(self._add_fragment)
        tree_buttons.addWidget(add_frag_btn)
        edit_btn = QPushButton("Изменить")
        edit_btn.clicked.connect(self._edit_selected)
        tree_buttons.addWidget(edit_btn)
        delete_btn = QPushButton("Удалить")
        delete_btn.clicked.connect(self._delete_selected)
        tree_buttons.addWidget(delete_btn)
        tree_buttons.addSpacing(12)
        up_btn = QPushButton("Вверх")
        up_btn.clicked.connect(lambda: self._move_selected(-1))
        tree_buttons.addWidget(up_btn)
        down_btn = QPushButton("Вниз")
        down_btn.clicked.connect(lambda: self._move_selected(1))
        tree_buttons.addWidget(down_btn)
        tree_buttons.addStretch()
        layout.addLayout(tree_buttons)

        output_group = QGroupBox("Результат fork-join")
        output_layout = QGridLayout(output_group)

        output_layout.addWidget(QLabel("Видео (MP4):"), 0, 0)
        self.video_path_edit = QLineEdit()
        output_layout.addWidget(self.video_path_edit, 0, 1)
        video_browse_btn = QPushButton("Обзор...")
        video_browse_btn.clicked.connect(self._browse_video_output)
        output_layout.addWidget(video_browse_btn, 0, 2)

        output_layout.addWidget(QLabel("Аудио (M4A):"), 1, 0)
        self.audio_path_edit = QLineEdit()
        output_layout.addWidget(self.audio_path_edit, 1, 1)
        audio_browse_btn = QPushButton("Обзор...")
        audio_browse_btn.clicked.connect(self._browse_audio_output)
        output_layout.addWidget(audio_browse_btn, 1, 2)

        layout.addWidget(output_group)

        # -------------------------------------------------- speech-to-text

        s2t_group = QGroupBox("Speech-to-text (GigaAM)")
        s2t_layout = QGridLayout(s2t_group)

        self.speech2text_check = QCheckBox("Выполнить распознавание речи (GigaAM)")
        self.speech2text_check.setChecked(True)
        self.speech2text_check.toggled.connect(self._on_speech2text_toggle)
        s2t_layout.addWidget(self.speech2text_check, 0, 0, 1, 3)

        self.transcript_label = QLabel("Транскрипция (TXT):")
        s2t_layout.addWidget(self.transcript_label, 1, 0)
        self.transcript_path_edit = QLineEdit()
        s2t_layout.addWidget(self.transcript_path_edit, 1, 1)
        self.transcript_browse_btn = QPushButton("Обзор...")
        self.transcript_browse_btn.clicked.connect(self._browse_transcript_output)
        s2t_layout.addWidget(self.transcript_browse_btn, 1, 2)

        self.remove_fillers_check = QCheckBox("Убирать слова-паразиты (вот, ну и т.п.)")
        self.remove_fillers_check.setChecked(True)
        s2t_layout.addWidget(self.remove_fillers_check, 2, 0, 1, 3)

        layout.addWidget(s2t_group)

        # --------------------------------------------------------- прочее

        self.keep_awake_check = QCheckBox("Не давать компьютеру уснуть во время обработки")
        self.keep_awake_check.setChecked(True)
        layout.addWidget(self.keep_awake_check)

        action_buttons = QHBoxLayout()
        action_buttons.addStretch()
        self.run_button = QPushButton("Запустить")
        self.run_button.setDefault(True)
        run_size = self.run_button.sizeHint()
        self.run_button.setFixedSize(run_size.width() * 2, run_size.height() * 2)
        self.run_button.clicked.connect(self._run)
        action_buttons.addWidget(self.run_button)
        action_buttons.addStretch()
        layout.addLayout(action_buttons)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        layout.addWidget(QLabel("Журнал:"))
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text, stretch=1)

    def _on_speech2text_toggle(self, checked: bool):
        self.transcript_label.setEnabled(checked)
        self.transcript_path_edit.setEnabled(checked)
        self.transcript_browse_btn.setEnabled(checked)
        self.remove_fillers_check.setEnabled(checked)

    # --------------------------------------------------------------- state

    def _selected_indices(self):
        """Возвращает (seg_idx, frag_idx) для текущего выделения дерева.

        frag_idx is None, если выбран видеофайл (а не фрагмент).
        """
        item = self.tree.currentItem()
        if item is None:
            return None, None
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(data, tuple):
            return data
        return data, None

    def _refresh_tree(self, select=None):
        self.tree.clear()
        select_item = None
        for i, segment in enumerate(self.segments):
            seg_item = QTreeWidgetItem([segment["path"], "", ""])
            seg_item.setData(0, Qt.ItemDataRole.UserRole, i)
            self.tree.addTopLevelItem(seg_item)
            seg_item.setExpanded(True)
            if select == i:
                select_item = seg_item
            for j, fragment in enumerate(segment["fragments"]):
                frag_item = QTreeWidgetItem(
                    [f"Фрагмент {j + 1}", fragment["start"], fragment["end"]]
                )
                frag_item.setData(0, Qt.ItemDataRole.UserRole, (i, j))
                seg_item.addChild(frag_item)
                if select == (i, j):
                    select_item = frag_item
        if select_item is not None:
            self.tree.setCurrentItem(select_item)

    # ------------------------------------------------------------- editing

    def _add_segment(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите видеофайл", "", VIDEO_FILTER)
        if not path:
            return
        self.segments.append({"path": path, "fragments": []})
        self._refresh_tree(select=len(self.segments) - 1)
        self._add_fragment()

    def _add_fragment(self):
        seg_idx, _ = self._selected_indices()
        if seg_idx is None:
            QMessageBox.information(self, "Добавить фрагмент", "Сначала выберите видеофайл.")
            return
        dialog = FragmentDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.segments[seg_idx]["fragments"].append(dialog.result_value)
        frag_idx = len(self.segments[seg_idx]["fragments"]) - 1
        self._refresh_tree(select=(seg_idx, frag_idx))

    def _edit_selected(self):
        seg_idx, frag_idx = self._selected_indices()
        if seg_idx is None:
            QMessageBox.information(self, "Изменить", "Выберите видеофайл или фрагмент.")
            return
        if frag_idx is None:
            new_path, _ = QFileDialog.getOpenFileName(
                self, "Выберите видеофайл", self.segments[seg_idx]["path"], VIDEO_FILTER,
            )
            if not new_path:
                return
            self.segments[seg_idx]["path"] = new_path
            self._refresh_tree(select=seg_idx)
        else:
            fragment = self.segments[seg_idx]["fragments"][frag_idx]
            dialog = FragmentDialog(self, start=fragment["start"], end=fragment["end"])
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            self.segments[seg_idx]["fragments"][frag_idx] = dialog.result_value
            self._refresh_tree(select=(seg_idx, frag_idx))

    def _delete_selected(self):
        seg_idx, frag_idx = self._selected_indices()
        if seg_idx is None:
            QMessageBox.information(self, "Удалить", "Выберите видеофайл или фрагмент для удаления.")
            return
        if frag_idx is None:
            reply = QMessageBox.question(
                self, "Удалить видеофайл",
                "Удалить выбранный видеофайл вместе со всеми его фрагментами?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            del self.segments[seg_idx]
            self._refresh_tree()
        else:
            del self.segments[seg_idx]["fragments"][frag_idx]
            self._refresh_tree(select=seg_idx)

    def _move_selected(self, direction: int):
        seg_idx, frag_idx = self._selected_indices()
        if seg_idx is None:
            return
        if frag_idx is None:
            new_idx = seg_idx + direction
            if not (0 <= new_idx < len(self.segments)):
                return
            self.segments[seg_idx], self.segments[new_idx] = (
                self.segments[new_idx],
                self.segments[seg_idx],
            )
            self._refresh_tree(select=new_idx)
        else:
            fragments = self.segments[seg_idx]["fragments"]
            new_idx = frag_idx + direction
            if not (0 <= new_idx < len(fragments)):
                return
            fragments[frag_idx], fragments[new_idx] = fragments[new_idx], fragments[frag_idx]
            self._refresh_tree(select=(seg_idx, new_idx))

    def _reset_all(self):
        if self.segments or self.video_path_edit.text() or self.audio_path_edit.text():
            reply = QMessageBox.question(
                self, "Новый", "Очистить текущую конфигурацию?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.segments = []
        self.video_path_edit.setText("")
        self.audio_path_edit.setText("")
        self._refresh_tree()

    # -------------------------------------------------------------- output

    @staticmethod
    def _ensure_extension(path: str, extension: str) -> str:
        return path if path.endswith(extension) else path + extension

    def _browse_video_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Куда сохранить итоговое видео", "", "MP4 видео (*.mp4)")
        if not path:
            return
        path = self._ensure_extension(path, ".mp4")
        self.video_path_edit.setText(path)

        video_path = Path(path)
        if not self.audio_path_edit.text().strip():
            self.audio_path_edit.setText(str(video_path.with_suffix(".m4a")))
        if self.speech2text_check.isChecked() and not self.transcript_path_edit.text().strip():
            self.transcript_path_edit.setText(str(video_path.with_suffix(".txt")))

    def _browse_audio_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Куда сохранить аудио", "", AUDIO_FILTER)
        if path:
            self.audio_path_edit.setText(self._ensure_extension(path, ".m4a"))

    def _browse_transcript_output(self):
        audio_path = self.audio_path_edit.text().strip()
        initial = f"{Path(audio_path).stem}_gigaam.txt" if audio_path else "transcript.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Куда сохранить транскрипцию", initial, TEXT_FILTER)
        if path:
            self.transcript_path_edit.setText(self._ensure_extension(path, ".txt"))

    # ------------------------------------------------------------- config

    def _build_config(self) -> dict:
        return {
            "segments": self.segments,
            "output": {
                "videoPath": self.video_path_edit.text().strip(),
                "audioPath": self.audio_path_edit.text().strip(),
            },
        }

    def _validated_config(self):
        config = self._build_config()
        try:
            fork_join.validate_config(config)
        except fork_join.ConfigError as e:
            QMessageBox.critical(self, "Некорректная конфигурация", str(e))
            return None
        return config

    def _preview_json(self):
        config = self._validated_config()
        if config is None:
            return
        text = json.dumps(config, ensure_ascii=False, indent=2)

        dialog = QDialog(self)
        dialog.setWindowTitle("Предпросмотр JSON")
        dialog.resize(600, 500)
        dialog_layout = QVBoxLayout(dialog)
        widget = QPlainTextEdit()
        widget.setPlainText(text)
        widget.setReadOnly(True)
        widget.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        dialog_layout.addWidget(widget)
        dialog.exec()

    # ------------------------------------------------------------- running

    def _log(self, message: str):
        self.log_text.appendPlainText(message)

    def _run(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return

        config = self._validated_config()
        if config is None:
            return

        try:
            fork_join.check_ffmpeg()
        except fork_join.FFmpegError as e:
            QMessageBox.critical(self, "ffmpeg не найден", str(e))
            return

        run_speech2text = self.speech2text_check.isChecked()
        transcript_path = self.transcript_path_edit.text().strip()
        remove_fillers = self.remove_fillers_check.isChecked()
        keep_awake = self.keep_awake_check.isChecked()

        if run_speech2text:
            if not transcript_path:
                QMessageBox.critical(
                    self, "Транскрипция",
                    "Укажите путь к файлу транскрипции или отключите шаг Speech-to-text.",
                )
                return
            try:
                check_ffmpeg_compatibility()
                ensure_hf_token()
            except EnvironmentCheckError as e:
                QMessageBox.critical(self, "Окружение не готово", str(e))
                return

        video_path = config["output"]["videoPath"]
        audio_path = config["output"]["audioPath"]
        check_paths = [video_path, audio_path]
        if run_speech2text:
            check_paths.append(transcript_path)
        existing = [p for p in check_paths if p and os.path.exists(p)]
        if existing:
            names = "\n".join(existing)
            reply = QMessageBox.question(
                self, "Файл уже существует",
                f"Следующие файлы будут перезаписаны:\n{names}\n\nПродолжить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.run_button.setEnabled(False)
        self.progress.setRange(0, 0)  # индикатор "занято" (busy/marquee)
        self._log("Запуск обработки...")

        self.worker_thread = threading.Thread(
            target=self._worker,
            args=(config, run_speech2text, transcript_path, remove_fillers, keep_awake),
            daemon=True,
        )
        self.worker_thread.start()

    def _worker(self, config, run_speech2text, transcript_path, remove_fillers, keep_awake):
        log = lambda msg: self.log_signal.emit(msg)  # noqa: E731

        if keep_awake:
            from wakepy import keep as wakepy_keep
            wake_cm = wakepy_keep.running(on_fail="pass")
        else:
            wake_cm = nullcontext()

        try:
            with wake_cm as mode:
                if keep_awake and not getattr(mode, "active", True):
                    log("Предупреждение: не удалось предотвратить сон компьютера — "
                        "обработка продолжится в обычном режиме.")

                t0 = time.time()
                fork_join.process_config(config, log=log)
                log(f"Fork/Join завершён за {(time.time() - t0) / 60:.1f} мин.")

                if run_speech2text:
                    t0 = time.time()
                    transcribe_longform_chunked.run(
                        config["output"]["audioPath"],
                        transcript_path,
                        keep_fillers=not remove_fillers,
                        log=log,
                    )
                    log(f"Speech-to-text завершён за {(time.time() - t0) / 60:.1f} мин.")

            self.done_signal.emit()
        except (fork_join.ConfigError, fork_join.FFmpegError, EnvironmentCheckError) as e:
            self.error_signal.emit(str(e))
        except Exception as e:  # noqa: BLE001 - показать пользователю любую неожиданную ошибку
            self.error_signal.emit(f"Непредвиденная ошибка: {e}")

    def _on_done(self):
        self._log("Готово.")
        QMessageBox.information(self, "Готово", "Обработка успешно завершена.")
        self.progress.setRange(0, 1)
        self.run_button.setEnabled(True)

    def _on_error(self, message: str):
        self._log(f"Ошибка: {message}")
        QMessageBox.critical(self, "Ошибка", message)
        self.progress.setRange(0, 1)
        self.run_button.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    window = PipelineUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
