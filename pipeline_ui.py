#!/usr/bin/env python3
"""
Графический интерфейс для полного конвейера обработки лекции:
fork-join (обязательный шаг) + speech-to-text через GigaAM-v3 (опциональный).

Наследуется от fork_join_ui.ForkJoinUI — переиспользует редактор дерева
файлов/фрагментов как есть, добавляя блок Speech-to-text и чекбокс защиты
от сна компьютера во время обработки.

Использование:
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    python3 pipeline_ui.py
"""

import os
import threading
import time
import tkinter as tk
from contextlib import nullcontext
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from fork_join import fork_join
from fork_join import fork_join_ui
from speech_to_text_gigaam import transcribe_longform_chunked
from speech_to_text_gigaam.transcribe_longform import (
    EnvironmentCheckError,
    check_ffmpeg_compatibility,
    ensure_hf_token,
)

TEXT_FILETYPES = [("Текстовый файл", "*.txt")]


class PipelineUI(fork_join_ui.ForkJoinUI):
    def __init__(self):
        super().__init__()
        self.title("Lecture Pipeline")
        self._center_window(780, 760)

    def _center_window(self, width: int, height: int) -> None:
        x = (self.winfo_screenwidth() - width) // 2
        y = (self.winfo_screenheight() - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

    # ------------------------------------------------------------------ UI

    def _build_widgets(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)

        ttk.Label(root, text="Видеофайлы и фрагменты", font=("", 11, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        tree_frame = ttk.Frame(root)
        tree_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 8))
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            tree_frame, columns=("start", "end"), selectmode="browse"
        )
        self.tree.heading("#0", text="Путь к файлу / фрагмент")
        self.tree.heading("start", text="Начало")
        self.tree.heading("end", text="Конец")
        self.tree.column("#0", width=440)
        self.tree.column("start", width=100, anchor="center")
        self.tree.column("end", width=100, anchor="center")
        self.tree.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")

        tree_buttons = ttk.Frame(root)
        tree_buttons.grid(row=2, column=0, sticky="w", pady=(0, 10))
        ttk.Button(tree_buttons, text="Добавить видеофайл", command=self._add_segment).pack(
            side="left"
        )
        ttk.Button(tree_buttons, text="Добавить фрагмент", command=self._add_fragment).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(tree_buttons, text="Изменить", command=self._edit_selected).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(tree_buttons, text="Удалить", command=self._delete_selected).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(tree_buttons, text="Вверх", command=lambda: self._move_selected(-1)).pack(
            side="left", padx=(12, 0)
        )
        ttk.Button(tree_buttons, text="Вниз", command=lambda: self._move_selected(1)).pack(
            side="left", padx=(6, 0)
        )

        output_frame = ttk.LabelFrame(root, text="Результат fork-join", padding=10)
        output_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        output_frame.columnconfigure(1, weight=1)

        ttk.Label(output_frame, text="Видео (MP4):").grid(row=0, column=0, sticky="w")
        self.video_path_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.video_path_var).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(output_frame, text="Обзор...", command=self._browse_video_output).grid(
            row=0, column=2
        )

        ttk.Label(output_frame, text="Аудио (MP3):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.audio_path_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.audio_path_var).grid(
            row=1, column=1, sticky="ew", padx=6, pady=(6, 0)
        )
        ttk.Button(output_frame, text="Обзор...", command=self._browse_audio_output).grid(
            row=1, column=2, pady=(6, 0)
        )

        # -------------------------------------------------- speech-to-text

        s2t_frame = ttk.LabelFrame(root, text="Speech-to-text (GigaAM)", padding=10)
        s2t_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        s2t_frame.columnconfigure(1, weight=1)

        self.speech2text_var = tk.BooleanVar(value=True)
        self.speech2text_check = ttk.Checkbutton(
            s2t_frame, text="Выполнить распознавание речи (GigaAM)",
            variable=self.speech2text_var, command=self._on_speech2text_toggle,
        )
        self.speech2text_check.grid(row=0, column=0, columnspan=3, sticky="w")

        self.transcript_label = ttk.Label(s2t_frame, text="Транскрипция (TXT):")
        self.transcript_label.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.transcript_path_var = tk.StringVar()
        self.transcript_entry = ttk.Entry(s2t_frame, textvariable=self.transcript_path_var)
        self.transcript_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        self.transcript_browse_btn = ttk.Button(
            s2t_frame, text="Обзор...", command=self._browse_transcript_output
        )
        self.transcript_browse_btn.grid(row=1, column=2, pady=(6, 0))

        self.remove_fillers_var = tk.BooleanVar(value=True)
        self.remove_fillers_check = ttk.Checkbutton(
            s2t_frame, text="Убирать слова-паразиты (вот, ну и т.п.)",
            variable=self.remove_fillers_var,
        )
        self.remove_fillers_check.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        # --------------------------------------------------------- прочее

        self.keep_awake_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            root, text="Не давать компьютеру уснуть во время обработки",
            variable=self.keep_awake_var,
        ).grid(row=5, column=0, sticky="w", pady=(0, 10))

        action_buttons = ttk.Frame(root)
        action_buttons.grid(row=6, column=0, sticky="w", pady=(0, 8))
        ttk.Button(action_buttons, text="Новый", command=self._reset_all).pack(side="left")
        ttk.Button(action_buttons, text="Предпросмотр JSON", command=self._preview_json).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(action_buttons, text="Сохранить JSON...", command=self._save_json).pack(
            side="left", padx=(6, 0)
        )
        self.run_button = ttk.Button(
            action_buttons, text="Запустить", command=self._run
        )
        self.run_button.pack(side="left", padx=(6, 0))

        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.grid(row=7, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(root, text="Журнал:").grid(row=8, column=0, sticky="w")
        self.log_text = scrolledtext.ScrolledText(root, height=14, state="disabled", wrap="word")
        self.log_text.grid(row=9, column=0, sticky="nsew")
        root.rowconfigure(9, weight=1)

    def _on_speech2text_toggle(self):
        state = "normal" if self.speech2text_var.get() else "disabled"
        self.transcript_label.configure(state=state)
        self.transcript_entry.configure(state=state)
        self.transcript_browse_btn.configure(state=state)
        self.remove_fillers_check.configure(state=state)

    # -------------------------------------------------------------- output

    def _browse_transcript_output(self):
        audio_path = self.audio_path_var.get().strip()
        initial = f"{Path(audio_path).stem}_gigaam.txt" if audio_path else "transcript.txt"
        path = filedialog.asksaveasfilename(
            title="Куда сохранить транскрипцию",
            defaultextension=".txt",
            initialfile=initial,
            filetypes=TEXT_FILETYPES,
        )
        if path:
            self.transcript_path_var.set(path)

    # ------------------------------------------------------------- running

    def _run(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return

        config = self._validated_config()
        if config is None:
            return

        try:
            fork_join.check_ffmpeg()
        except fork_join.FFmpegError as e:
            messagebox.showerror("ffmpeg не найден", str(e))
            return

        run_speech2text = self.speech2text_var.get()
        transcript_path = self.transcript_path_var.get().strip()
        remove_fillers = self.remove_fillers_var.get()
        keep_awake = self.keep_awake_var.get()

        if run_speech2text:
            if not transcript_path:
                messagebox.showerror(
                    "Транскрипция",
                    "Укажите путь к файлу транскрипции или отключите шаг Speech-to-text.",
                )
                return
            try:
                check_ffmpeg_compatibility()
                ensure_hf_token()
            except EnvironmentCheckError as e:
                messagebox.showerror("Окружение не готово", str(e))
                return

        video_path = config["output"]["videoPath"]
        audio_path = config["output"]["audioPath"]
        check_paths = [video_path, audio_path]
        if run_speech2text:
            check_paths.append(transcript_path)
        existing = [p for p in check_paths if p and os.path.exists(p)]
        if existing:
            names = "\n".join(existing)
            if not messagebox.askyesno(
                "Файл уже существует",
                f"Следующие файлы будут перезаписаны:\n{names}\n\nПродолжить?",
            ):
                return

        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self._log("Запуск обработки...")

        self.worker_thread = threading.Thread(
            target=self._worker,
            args=(config, run_speech2text, transcript_path, remove_fillers, keep_awake),
            daemon=True,
        )
        self.worker_thread.start()
        self.after(100, self._poll_log_queue)

    def _worker(self, config, run_speech2text, transcript_path, remove_fillers, keep_awake):
        log = lambda msg: self.log_queue.put(("log", msg))  # noqa: E731

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

            self.log_queue.put(("done", None))
        except (fork_join.ConfigError, fork_join.FFmpegError, EnvironmentCheckError) as e:
            self.log_queue.put(("error", str(e)))
        except Exception as e:  # noqa: BLE001 - показать пользователю любую неожиданную ошибку
            self.log_queue.put(("error", f"Непредвиденная ошибка: {e}"))


def main():
    app = PipelineUI()
    # На macOS окно Tkinter, запущенное не из полноценного .app-бандла
    # (например, из PyCharm или терминала), не получает фокус автоматически
    # и может оказаться позади запустившей его программы. Кратковременный
    # -topmost поднимает окно поверх остальных один раз при старте, не делая
    # его залипающим "всегда сверху" на постоянной основе.
    app.lift()
    app.attributes("-topmost", True)
    app.after_idle(app.attributes, "-topmost", False)
    app.focus_force()
    app.mainloop()


if __name__ == "__main__":
    main()
