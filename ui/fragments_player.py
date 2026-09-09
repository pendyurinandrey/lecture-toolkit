#!/usr/bin/env python3
"""Диалог редактирования фрагментов видео со встроенным плеером.

Вынесено из pipeline_ui.py, чтобы не перегружать основной файл: здесь и
FragmentsTimeline (таймлайн с зонами-фрагментами, только просмотр/выбор —
границы редактируются числовыми полями в диалоге, не перетаскиванием
мышью по таймлайну, т.к. это слишком легко сдвинуть случайно), и
FragmentsPlayerDialog (плеер + таймлайн + автоопределение пауз через
ffmpeg silencedetect).
"""

import threading
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from common.silence import detect_silences, keep_intervals_between_silences
from fork_join import fork_join


class _CenteredMessageBox(QMessageBox):
    """QMessageBox, центрирующийся над parent непосредственно перед показом.

    На macOS QMessageBox по умолчанию рендерится через нативный NSAlert —
    в этом случае move()/showEvent() на самом Qt-виджете вообще не влияют
    на позицию видимого окна (оно нативное, Qt-геометрия для него не
    применяется), поэтому сначала явно отключаем нативный рендеринг через
    DontUseNativeDialog. showEvent(), а не adjustSize()+move() до первого
    show(), — потому что до показа adjustSize() даёт размер меньше
    реального (особенно с нативными иконками/кнопками)."""

    def showEvent(self, event) -> None:
        super().showEvent(event)
        parent = self.parentWidget()
        if parent is not None:
            parent_geo = parent.frameGeometry()
            self.move(
                parent_geo.center().x() - self.width() // 2,
                parent_geo.center().y() - self.height() // 2,
            )


def _centered_message_box(
    parent, icon: QMessageBox.Icon, title: str, text: str,
    buttons: QMessageBox.StandardButton = QMessageBox.StandardButton.Ok,
) -> QMessageBox.StandardButton:
    """QMessageBox, явно отцентрированный над parent — статические
    QMessageBox.information/question/critical на этом диалоге почему-то
    открывались в левом верхнем углу экрана вместо центра родителя."""
    box = _CenteredMessageBox(icon, title, text, buttons, parent)
    box.setOption(QMessageBox.Option.DontUseNativeDialog, True)
    return box.exec()


def _update_multi_selection(current: list, index: int, extend: bool) -> list:
    """Обновляет список выбранных индексов по правилам обычного
    multi-select (максимум 2 одновременно, старые в начале списка):
    - обычный клик (extend=False): сбрасывает выбор к [index]
    - Cmd/Ctrl+клик по уже выбранному индексу: снимает с него выделение
    - Cmd/Ctrl+клик по новому индексу: добавляет его; если выбранных уже
      было 2, самый старый вытесняется — всегда остаются 2 последних клика
    Используется и таймлайном, и списком фрагментов, чтобы оба источника
    клика вели себя одинаково."""
    if not extend:
        return [index]
    if index in current:
        return [i for i in current if i != index]
    updated = current + [index]
    return updated[-2:]


class FragmentsTimeline(QWidget):
    """Горизонтальная шкала [0, duration] с цветными зонами — фрагментами,
    которые останутся после склейки (всё остальное будет вырезано). Клик по
    зоне её выделяет (Cmd/Ctrl+клик — выбрать вторую зону для объединения,
    не более 2 одновременно); редактирование границ — числовыми полями в
    диалоге, а не мышью прямо на таймлайне (слишком легко случайно
    сдвинуть)."""

    intervalsChanged = Signal()
    selectionChanged = Signal(list)  # индексы выбранных фрагментов (0, 1 или 2)

    def __init__(self, duration: float, parent=None):
        super().__init__(parent)
        self.duration = max(duration, 0.001)
        self.intervals: list = []  # [[start, end], ...] в секундах, отсортировано по start
        self.selected: list = []  # индексы выбранных фрагментов, максимум 2
        self.playhead = 0.0
        self.setMinimumHeight(48)

    def set_intervals(self, intervals) -> None:
        self.intervals = sorted([list(iv) for iv in intervals], key=lambda iv: iv[0])
        self.selected = []
        self.selectionChanged.emit([])
        self.intervalsChanged.emit()
        self.update()

    def set_playhead(self, seconds: float) -> None:
        self.playhead = seconds
        self.update()

    def remove_selected(self) -> None:
        """Удаляет все выбранные фрагменты (при двух выбранных — оба)."""
        if not self.selected:
            return
        for idx in sorted(self.selected, reverse=True):
            del self.intervals[idx]
        self.selected = []
        self.selectionChanged.emit([])
        self.intervalsChanged.emit()
        self.update()

    def merge_selected(self) -> None:
        """Объединяет 2 выбранных фрагмента в один: левая граница — от
        геометрически левого, правая — от геометрически правого (порядок
        клика не важен). Всё, что было между ними (включая сами выбранные
        фрагменты), удаляется и заменяется одним новым."""
        if len(self.selected) != 2:
            return
        i1, i2 = sorted(self.selected)  # self.intervals всегда отсортирован по start
        new_start = self.intervals[i1][0]
        new_end = self.intervals[i2][1]
        del self.intervals[i1:i2 + 1]
        self.intervals.insert(i1, [new_start, new_end])
        self.selected = [i1]
        self.selectionChanged.emit(list(self.selected))
        self.intervalsChanged.emit()
        self.update()

    def add_interval_near(self, time: float, default_length: float = 30.0):
        """Добавляет новый фрагмент рядом с `time` в ближайший свободный
        промежуток. Возвращает индекс нового фрагмента, либо None, если
        рядом нет места хотя бы на секунду."""
        start = max(0.0, min(time, self.duration))
        for s, e in self.intervals:
            if s <= start < e:
                start = e
                break
        upper = self.duration
        for s, e in self.intervals:
            if s > start:
                upper = min(upper, s)
                break
        end = min(start + default_length, upper)
        if end - start < 1.0:
            return None
        self.intervals.append([start, end])
        self.intervals.sort(key=lambda iv: iv[0])
        idx = self.intervals.index([start, end])
        self.selected = [idx]
        self.selectionChanged.emit(list(self.selected))
        self.intervalsChanged.emit()
        self.update()
        return idx

    def sizeHint(self) -> QSize:
        return QSize(600, 48)

    def _time_to_x(self, t: float) -> int:
        return int((t / self.duration) * self.width())

    def _x_to_time(self, x: int) -> float:
        w = max(self.width(), 1)
        return max(0.0, min(self.duration, (x / w) * self.duration))

    def _hit_test_band(self, x: int):
        t = self._x_to_time(x)
        for i, (s, e) in enumerate(self.intervals):
            if s <= t <= e:
                return i
        return None

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QColor("#3a3a3a"))
        for i, (s, e) in enumerate(self.intervals):
            x0 = self._time_to_x(s)
            x1 = self._time_to_x(e)
            color = QColor("#f5a623") if i in self.selected else QColor("#4a90d9")
            painter.fillRect(x0, 0, max(x1 - x0, 2), rect.height(), color)
        px = self._time_to_x(self.playhead)
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(px, 0, px, rect.height())

    def mousePressEvent(self, event) -> None:
        x = int(event.position().x())
        band = self._hit_test_band(x)
        extend = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if band is None:
            if not extend:
                self.selected = []
            # Cmd/Ctrl+клик по пустому месту не трогает текущий выбор
        else:
            self.selected = _update_multi_selection(self.selected, band, extend)
        self.selectionChanged.emit(list(self.selected))
        self.update()


class FragmentListWidget(QListWidget):
    """QListWidget с тем же правилом выбора (клик/Cmd-Ctrl+клик, максимум 2
    одновременно), что и у FragmentsTimeline — чтобы фрагмент, который
    невозможно выбрать на таймлайне (слишком тонкая полоска), всё равно
    можно было включить в объединение через список."""

    selectionRequested = Signal(int, bool)  # (row, extend); row=-1 — клик по пустому месту

    def mousePressEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        row = self.row(item) if item is not None else -1
        extend = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        self.selectionRequested.emit(row, extend)
        event.accept()


class FragmentsPlayerDialog(QDialog):
    """Диалог просмотра видео с редактированием сразу нескольких фрагментов
    (полезных интервалов) на одном таймлайне — заменяет ручной перенос
    таймкодов из QuickTime Player. Поддерживает автоопределение пауз через
    ffmpeg silencedetect (например, чтобы вырезать перемену)."""

    detect_done_signal = Signal(list)
    detect_error_signal = Signal(str)

    def __init__(self, parent, video_path: str, duration: float, fragments: list):
        super().__init__(parent)
        self.setWindowTitle(f"Фрагменты — {Path(video_path).name}")
        self.video_path = video_path
        self.duration = duration
        self.result_fragments = None

        self.detect_done_signal.connect(self._on_detect_done)
        self.detect_error_signal.connect(self._on_detect_error)

        self._build_widgets()
        self.finished.connect(lambda _: self.media_player.stop())

        initial = [
            [fork_join.hhmmss_to_seconds(f["start"]), fork_join.hhmmss_to_seconds(f["end"])]
            for f in fragments
        ] or [[0.0, self.duration]]
        self.timeline.set_intervals(initial)

        self.media_player.setSource(QUrl.fromLocalFile(video_path))
        self.resize(960, 560)

    # ------------------------------------------------------------------ UI

    def _build_widgets(self) -> None:
        layout = QVBoxLayout(self)

        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(300)
        self.media_player.setVideoOutput(self.video_widget)
        layout.addWidget(self.video_widget, stretch=1)

        controls = QHBoxLayout()
        self.play_button = QPushButton("▶")
        self.play_button.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_button)
        back5_btn = QPushButton("−5с")
        back5_btn.clicked.connect(lambda: self._seek_relative(-5))
        controls.addWidget(back5_btn)
        fwd5_btn = QPushButton("+5с")
        fwd5_btn.clicked.connect(lambda: self._seek_relative(5))
        controls.addWidget(fwd5_btn)
        controls.addStretch()
        self.position_label = QLabel(f"00:00:00 / {fork_join.seconds_to_hhmmss(self.duration)}")
        controls.addWidget(self.position_label)
        layout.addLayout(controls)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, int(self.duration * 1000))
        self.position_slider.sliderMoved.connect(self._on_slider_moved)
        layout.addWidget(self.position_slider)

        self.timeline = FragmentsTimeline(self.duration)
        self.timeline.selectionChanged.connect(self._refresh_numeric_fields)
        self.timeline.selectionChanged.connect(self._sync_list_selection)
        self.timeline.selectionChanged.connect(self._seek_to_selected_start)
        self.timeline.selectionChanged.connect(self._update_merge_button)
        self.timeline.intervalsChanged.connect(self._refresh_numeric_fields)
        self.timeline.intervalsChanged.connect(self._refresh_fragment_list)
        layout.addWidget(self.timeline)

        # Список фрагментов текстом — основной способ их выбора. Полоска на
        # таймлайне бывает слишком тонкой, чтобы по ней попасть кликом
        # (особенно на многочасовом видео), список работает всегда. Правило
        # выбора (Cmd/Ctrl+клик — до 2 одновременно) то же, что у таймлайна.
        self.fragment_list = FragmentListWidget()
        # MultiSelection нужен, чтобы item.setSelected() визуально подсвечивал
        # строку (NoSelection блокирует это программно, не только по клику) —
        # но клики Qt сам не обрабатывает: mousePressEvent переопределён ниже
        # и не вызывает super(), так что реальный выбор всегда только наш.
        self.fragment_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.fragment_list.setMaximumHeight(100)
        self.fragment_list.selectionRequested.connect(self._on_list_selection_requested)
        layout.addWidget(self.fragment_list)

        numeric_row = QHBoxLayout()
        numeric_row.addWidget(QLabel("Начало (ЧЧ:ММ:СС):"))
        self.start_edit = QLineEdit()
        self.start_edit.setMinimumWidth(90)
        self.start_edit.editingFinished.connect(self._on_numeric_edit)
        numeric_row.addWidget(self.start_edit)
        numeric_row.addWidget(QLabel("Конец (ЧЧ:ММ:СС):"))
        self.end_edit = QLineEdit()
        self.end_edit.setMinimumWidth(90)
        self.end_edit.editingFinished.connect(self._on_numeric_edit)
        numeric_row.addWidget(self.end_edit)
        add_frag_btn = QPushButton("+ Добавить фрагмент")
        add_frag_btn.clicked.connect(self._on_add_fragment)
        numeric_row.addWidget(add_frag_btn)
        remove_frag_btn = QPushButton("Удалить фрагмент")
        remove_frag_btn.clicked.connect(self.timeline.remove_selected)
        numeric_row.addWidget(remove_frag_btn)
        self.merge_button = QPushButton("Объединить")
        self.merge_button.setEnabled(False)
        self.merge_button.setToolTip(
            "Выберите 2 фрагмента (Cmd/Ctrl+клик на таймлайне или в списке), "
            "чтобы объединить их в один — вместе со всем, что было между ними."
        )
        self.merge_button.clicked.connect(self._on_merge_fragments)
        numeric_row.addWidget(self.merge_button)
        layout.addLayout(numeric_row)

        detect_group = QGroupBox("Автоопределение пауз")
        detect_layout = QHBoxLayout(detect_group)
        detect_layout.addWidget(QLabel("Определить паузы длиннее чем"))
        self.min_pause_spin = QSpinBox()
        self.min_pause_spin.setRange(1, 3600)
        self.min_pause_spin.setValue(60)
        self.min_pause_spin.setSuffix(" с")
        detect_layout.addWidget(self.min_pause_spin)
        self.detect_button = QPushButton("Определить")
        self.detect_button.clicked.connect(self._on_detect_clicked)
        detect_layout.addWidget(self.detect_button)
        detect_layout.addStretch()
        layout.addWidget(detect_group)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_ok)
        buttons.addWidget(ok_btn)
        layout.addLayout(buttons)

        self.media_player.positionChanged.connect(self._on_player_position_changed)
        self.media_player.playbackStateChanged.connect(self._on_playback_state_changed)

        self._refresh_numeric_fields()
        self._refresh_fragment_list()

    # ------------------------------------------------------------- playback

    def _toggle_play(self) -> None:
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def _on_playback_state_changed(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_button.setText("⏸" if playing else "▶")

    def _seek_relative(self, delta_seconds: float) -> None:
        new_pos_ms = self.media_player.position() + int(delta_seconds * 1000)
        new_pos_ms = max(0, min(int(self.duration * 1000), new_pos_ms))
        self.media_player.setPosition(new_pos_ms)

    def _on_slider_moved(self, value: int) -> None:
        self.media_player.setPosition(value)

    def _seek_to_selected_start(self, selected: list) -> None:
        """При выборе фрагмента (клик по таймлайну или по строке списка)
        переносит позицию воспроизведения на его левую границу — на
        последний выбранный, если выбрано сразу два. Если воспроизведение
        уже идёт — продолжается с новой позиции, если нет — просто
        сдвигается указатель (setPosition не меняет play/pause)."""
        if not selected:
            return
        index = selected[-1]
        if index >= len(self.timeline.intervals):
            return
        start_s = self.timeline.intervals[index][0]
        self.media_player.setPosition(int(start_s * 1000))

    def _on_player_position_changed(self, position_ms: int) -> None:
        self.position_slider.setValue(position_ms)
        self.timeline.set_playhead(position_ms / 1000.0)
        current = fork_join.seconds_to_hhmmss(position_ms / 1000.0)
        total = fork_join.seconds_to_hhmmss(self.duration)
        self.position_label.setText(f"{current} / {total}")

    # ------------------------------------------------------------ fragments

    def _refresh_fragment_list(self) -> None:
        self.fragment_list.blockSignals(True)
        self.fragment_list.clear()
        for i, (s, e) in enumerate(self.timeline.intervals):
            self.fragment_list.addItem(
                f"{i + 1}. {fork_join.seconds_to_hhmmss(s)}–{fork_join.seconds_to_hhmmss(e)}"
            )
        self.fragment_list.blockSignals(False)
        self._sync_list_selection(self.timeline.selected)

    def _sync_list_selection(self, selected: list) -> None:
        self.fragment_list.blockSignals(True)
        for i in range(self.fragment_list.count()):
            self.fragment_list.item(i).setSelected(i in selected)
        self.fragment_list.blockSignals(False)

    def _on_list_selection_requested(self, row: int, extend: bool) -> None:
        if row < 0:
            if not extend:
                self.timeline.selected = []
            # Cmd/Ctrl+клик по пустому месту не трогает текущий выбор
        else:
            self.timeline.selected = _update_multi_selection(self.timeline.selected, row, extend)
        self.timeline.selectionChanged.emit(list(self.timeline.selected))
        self.timeline.update()

    def _update_merge_button(self, selected: list) -> None:
        self.merge_button.setEnabled(len(selected) == 2)

    def _on_merge_fragments(self) -> None:
        self.timeline.merge_selected()

    def _refresh_numeric_fields(self) -> None:
        # Читает self.timeline.selected напрямую, а не через аргумент
        # сигнала — метод подписан и на selectionChanged (несёт list), и на
        # intervalsChanged (без аргументов), сигнатуры разные.
        selected = self.timeline.selected
        if len(selected) != 1:
            self.start_edit.clear()
            self.end_edit.clear()
            self.start_edit.setEnabled(False)
            self.end_edit.setEnabled(False)
            return
        idx = selected[0]
        self.start_edit.setEnabled(True)
        self.end_edit.setEnabled(True)
        s, e = self.timeline.intervals[idx]
        if not self.start_edit.hasFocus():
            self.start_edit.setText(fork_join.seconds_to_hhmmss(s))
        if not self.end_edit.hasFocus():
            self.end_edit.setText(fork_join.seconds_to_hhmmss(e))

    def _on_numeric_edit(self) -> None:
        if len(self.timeline.selected) != 1:
            return
        idx = self.timeline.selected[0]
        try:
            start_s = fork_join.hhmmss_to_seconds(self.start_edit.text())
            end_s = fork_join.hhmmss_to_seconds(self.end_edit.text())
        except ValueError as e:
            _centered_message_box(self, QMessageBox.Icon.Critical, "Некорректное время", str(e))
            self._refresh_numeric_fields()
            return
        lower = self.timeline.intervals[idx - 1][1] if idx > 0 else 0.0
        upper = (
            self.timeline.intervals[idx + 1][0]
            if idx + 1 < len(self.timeline.intervals)
            else self.duration
        )
        if not (lower <= start_s < end_s <= upper):
            _centered_message_box(
                self, QMessageBox.Icon.Critical, "Некорректный диапазон",
                f"Фрагмент должен быть в пределах {fork_join.seconds_to_hhmmss(lower)}"
                f"–{fork_join.seconds_to_hhmmss(upper)}, и начало должно быть меньше конца.",
            )
            self._refresh_numeric_fields()
            return
        self.timeline.intervals[idx] = [start_s, end_s]
        self.timeline.update()

    def _on_add_fragment(self) -> None:
        idx = self.timeline.add_interval_near(self.timeline.playhead)
        if idx is None:
            _centered_message_box(
                self, QMessageBox.Icon.Information, "Добавить фрагмент",
                "Недостаточно свободного места рядом с текущей позицией плеера.",
            )

    # ----------------------------------------------------- автоопределение

    def _on_detect_clicked(self) -> None:
        if self.timeline.intervals:
            reply = _centered_message_box(
                self, QMessageBox.Icon.Question, "Определить паузы",
                "Текущие фрагменты будут заменены результатом автоопределения. Продолжить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.detect_button.setEnabled(False)
        self.detect_button.setText("Определяю...")
        min_pause = float(self.min_pause_spin.value())
        threading.Thread(target=self._detect_worker, args=(min_pause,), daemon=True).start()

    def _detect_worker(self, min_pause: float) -> None:
        try:
            silences = detect_silences(self.video_path, min_duration=min_pause)
            intervals = keep_intervals_between_silences(self.duration, silences)
            self.detect_done_signal.emit(intervals)
        except Exception as e:  # noqa: BLE001 - показать пользователю любую ошибку ffmpeg
            self.detect_error_signal.emit(str(e))

    def _on_detect_done(self, intervals: list) -> None:
        self.detect_button.setEnabled(True)
        self.detect_button.setText("Определить")
        if not intervals:
            _centered_message_box(
                self, QMessageBox.Icon.Information, "Определить паузы",
                "Пауз длиннее заданного порога не найдено.",
            )
            return
        self.timeline.set_intervals(intervals)

    def _on_detect_error(self, message: str) -> None:
        self.detect_button.setEnabled(True)
        self.detect_button.setText("Определить")
        _centered_message_box(self, QMessageBox.Icon.Critical, "Ошибка определения пауз", message)

    # ------------------------------------------------------------- готово

    def _on_ok(self) -> None:
        if not self.timeline.intervals:
            _centered_message_box(
                self, QMessageBox.Icon.Critical, "Нет фрагментов", "Добавьте хотя бы один фрагмент.",
            )
            return
        self.result_fragments = [
            {"start": fork_join.seconds_to_hhmmss(s), "end": fork_join.seconds_to_hhmmss(e)}
            for s, e in self.timeline.intervals
        ]
        self.accept()
