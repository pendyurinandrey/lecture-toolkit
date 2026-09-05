#!/usr/bin/env python3
"""
Графический интерфейс (tkinter) для сборки конфигурации fork_join.py
без ручного редактирования JSON.

Формат конфигурации: см. fork_join_format.txt
Обработка видео выполняется функциями модуля fork_join.py.

Использование:
    python3 fork_join_ui.py
"""

import json
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from fork_join import fork_join


VIDEO_FILETYPES = [("Видео MP4", "*.mp4"), ("Все файлы", "*.*")]


class FragmentDialog(tk.Toplevel):
    """Модальный диалог ввода start/end фрагмента в формате HH:mm:ss."""

    def __init__(self, parent, start: str = "00:00:00", end: str = "00:00:00"):
        super().__init__(parent)
        self.title("Фрагмент")
        self.resizable(False, False)
        self.transient(parent)
        self.result = None

        form = ttk.Frame(self, padding=12)
        form.grid(row=0, column=0, sticky="nsew")

        ttk.Label(form, text="Начало (ЧЧ:ММ:СС):").grid(row=0, column=0, sticky="w", pady=4)
        self.start_var = tk.StringVar(value=start)
        start_entry = ttk.Entry(form, textvariable=self.start_var, width=12)
        start_entry.grid(row=0, column=1, pady=4, padx=(8, 0))

        ttk.Label(form, text="Конец (ЧЧ:ММ:СС):").grid(row=1, column=0, sticky="w", pady=4)
        self.end_var = tk.StringVar(value=end)
        end_entry = ttk.Entry(form, textvariable=self.end_var, width=12)
        end_entry.grid(row=1, column=1, pady=4, padx=(8, 0))

        btns = ttk.Frame(form)
        btns.grid(row=2, column=0, columnspan=2, pady=(12, 0), sticky="e")
        ttk.Button(btns, text="Отмена", command=self._cancel).pack(side="right", padx=(6, 0))
        ttk.Button(btns, text="OK", command=self._ok).pack(side="right")

        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self._cancel())

        start_entry.focus_set()
        self.grab_set()
        self.wait_window(self)

    def _ok(self):
        start = self.start_var.get().strip()
        end = self.end_var.get().strip()
        try:
            start_s = fork_join.hhmmss_to_seconds(start)
            end_s = fork_join.hhmmss_to_seconds(end)
        except ValueError as e:
            messagebox.showerror("Некорректное время", str(e), parent=self)
            return
        if end_s <= start_s:
            messagebox.showerror(
                "Некорректный диапазон",
                f'"Конец" ({end}) должен быть больше "Начала" ({start})',
                parent=self,
            )
            return
        self.result = {"start": start, "end": end}
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


class ForkJoinUI(tk.Tk):
    SEG_PREFIX = "seg"

    def __init__(self):
        super().__init__()
        self.title("Fork/Join — сборка конфигурации")
        self.geometry("760x620")
        self.minsize(640, 480)

        self.segments = []  # [{"path": str, "fragments": [{"start": str, "end": str}, ...]}]
        self.log_queue = queue.Queue()
        self.worker_thread = None

        self._build_widgets()
        self._refresh_tree()

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

        output_frame = ttk.LabelFrame(root, text="Результат", padding=10)
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

        action_buttons = ttk.Frame(root)
        action_buttons.grid(row=4, column=0, sticky="w", pady=(0, 8))
        ttk.Button(action_buttons, text="Новый", command=self._reset_all).pack(side="left")
        ttk.Button(action_buttons, text="Предпросмотр JSON", command=self._preview_json).pack(
            side="left", padx=(6, 0)
        )
        ttk.Button(action_buttons, text="Сохранить JSON...", command=self._save_json).pack(
            side="left", padx=(6, 0)
        )
        self.run_button = ttk.Button(
            action_buttons, text="Проверить и запустить", command=self._run
        )
        self.run_button.pack(side="left", padx=(6, 0))

        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.grid(row=5, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(root, text="Журнал:").grid(row=6, column=0, sticky="w")
        self.log_text = scrolledtext.ScrolledText(root, height=10, state="disabled", wrap="word")
        self.log_text.grid(row=7, column=0, sticky="nsew")
        root.rowconfigure(7, weight=1)

    # --------------------------------------------------------------- state

    def _selected_indices(self):
        """Возвращает (seg_idx, frag_idx) для текущего выделения дерева.

        frag_idx is None, если выбран видеофайл (а не фрагмент).
        """
        selection = self.tree.selection()
        if not selection:
            return None, None
        iid = selection[0]
        parts = iid.split("_")
        seg_idx = int(parts[0][len(self.SEG_PREFIX):])
        if len(parts) == 1:
            return seg_idx, None
        frag_idx = int(parts[1][len("frag"):])
        return seg_idx, frag_idx

    def _refresh_tree(self, select_iid: str = None):
        self.tree.delete(*self.tree.get_children())
        for i, segment in enumerate(self.segments):
            seg_iid = f"{self.SEG_PREFIX}{i}"
            self.tree.insert("", "end", iid=seg_iid, text=segment["path"], open=True)
            for j, fragment in enumerate(segment["fragments"]):
                frag_iid = f"{seg_iid}_frag{j}"
                self.tree.insert(
                    seg_iid,
                    "end",
                    iid=frag_iid,
                    text=f"Фрагмент {j + 1}",
                    values=(fragment["start"], fragment["end"]),
                )
        if select_iid and self.tree.exists(select_iid):
            self.tree.selection_set(select_iid)

    # ------------------------------------------------------------- editing

    def _add_segment(self):
        path = filedialog.askopenfilename(title="Выберите видеофайл", filetypes=VIDEO_FILETYPES)
        if not path:
            return
        self.segments.append({"path": path, "fragments": []})
        seg_iid = f"{self.SEG_PREFIX}{len(self.segments) - 1}"
        self._refresh_tree(select_iid=seg_iid)
        self._add_fragment()

    def _add_fragment(self):
        seg_idx, _ = self._selected_indices()
        if seg_idx is None:
            messagebox.showinfo("Добавить фрагмент", "Сначала выберите видеофайл.")
            return
        dialog = FragmentDialog(self)
        if dialog.result is None:
            return
        self.segments[seg_idx]["fragments"].append(dialog.result)
        frag_idx = len(self.segments[seg_idx]["fragments"]) - 1
        self._refresh_tree(select_iid=f"{self.SEG_PREFIX}{seg_idx}_frag{frag_idx}")

    def _edit_selected(self):
        seg_idx, frag_idx = self._selected_indices()
        if seg_idx is None:
            messagebox.showinfo("Изменить", "Выберите видеофайл или фрагмент.")
            return
        if frag_idx is None:
            new_path = filedialog.askopenfilename(
                title="Выберите видеофайл",
                initialfile=os.path.basename(self.segments[seg_idx]["path"]),
                filetypes=VIDEO_FILETYPES,
            )
            if not new_path:
                return
            self.segments[seg_idx]["path"] = new_path
            self._refresh_tree(select_iid=f"{self.SEG_PREFIX}{seg_idx}")
        else:
            fragment = self.segments[seg_idx]["fragments"][frag_idx]
            dialog = FragmentDialog(self, start=fragment["start"], end=fragment["end"])
            if dialog.result is None:
                return
            self.segments[seg_idx]["fragments"][frag_idx] = dialog.result
            self._refresh_tree(select_iid=f"{self.SEG_PREFIX}{seg_idx}_frag{frag_idx}")

    def _delete_selected(self):
        seg_idx, frag_idx = self._selected_indices()
        if seg_idx is None:
            messagebox.showinfo("Удалить", "Выберите видеофайл или фрагмент для удаления.")
            return
        if frag_idx is None:
            if not messagebox.askyesno(
                "Удалить видеофайл",
                "Удалить выбранный видеофайл вместе со всеми его фрагментами?",
            ):
                return
            del self.segments[seg_idx]
            self._refresh_tree()
        else:
            del self.segments[seg_idx]["fragments"][frag_idx]
            self._refresh_tree(select_iid=f"{self.SEG_PREFIX}{seg_idx}")

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
            self._refresh_tree(select_iid=f"{self.SEG_PREFIX}{new_idx}")
        else:
            fragments = self.segments[seg_idx]["fragments"]
            new_idx = frag_idx + direction
            if not (0 <= new_idx < len(fragments)):
                return
            fragments[frag_idx], fragments[new_idx] = fragments[new_idx], fragments[frag_idx]
            self._refresh_tree(select_iid=f"{self.SEG_PREFIX}{seg_idx}_frag{new_idx}")

    def _reset_all(self):
        if self.segments or self.video_path_var.get() or self.audio_path_var.get():
            if not messagebox.askyesno("Новый", "Очистить текущую конфигурацию?"):
                return
        self.segments = []
        self.video_path_var.set("")
        self.audio_path_var.set("")
        self._refresh_tree()

    # -------------------------------------------------------------- output

    def _browse_video_output(self):
        path = filedialog.asksaveasfilename(
            title="Куда сохранить итоговое видео",
            defaultextension=".mp4",
            filetypes=[("MP4 видео", "*.mp4")],
        )
        if path:
            self.video_path_var.set(path)

    def _browse_audio_output(self):
        path = filedialog.asksaveasfilename(
            title="Куда сохранить аудио",
            defaultextension=".mp3",
            filetypes=[("MP3 аудио", "*.mp3")],
        )
        if path:
            self.audio_path_var.set(path)

    # ------------------------------------------------------------- config

    def _build_config(self) -> dict:
        return {
            "segments": self.segments,
            "output": {
                "videoPath": self.video_path_var.get().strip(),
                "audioPath": self.audio_path_var.get().strip(),
            },
        }

    def _validated_config(self):
        config = self._build_config()
        try:
            fork_join.validate_config(config)
        except fork_join.ConfigError as e:
            messagebox.showerror("Некорректная конфигурация", str(e))
            return None
        return config

    def _preview_json(self):
        config = self._validated_config()
        if config is None:
            return
        text = json.dumps(config, ensure_ascii=False, indent=2)

        preview = tk.Toplevel(self)
        preview.title("Предпросмотр JSON")
        preview.geometry("600x500")
        widget = scrolledtext.ScrolledText(preview, wrap="none")
        widget.pack(fill="both", expand=True)
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _save_json(self):
        config = self._validated_config()
        if config is None:
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить конфигурацию",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        self._log(f"Конфигурация сохранена: {path}")
        messagebox.showinfo("Сохранено", f"Конфигурация сохранена:\n{path}")

    # ------------------------------------------------------------- running

    def _log(self, message: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

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

        video_path = config["output"]["videoPath"]
        audio_path = config["output"]["audioPath"]
        existing = [p for p in (video_path, audio_path) if os.path.exists(p)]
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
            target=self._worker, args=(config,), daemon=True
        )
        self.worker_thread.start()
        self.after(100, self._poll_log_queue)

    def _worker(self, config: dict):
        try:
            fork_join.process_config(config, log=lambda msg: self.log_queue.put(("log", msg)))
            self.log_queue.put(("done", None))
        except (fork_join.ConfigError, fork_join.FFmpegError) as e:
            self.log_queue.put(("error", str(e)))
        except Exception as e:  # noqa: BLE001 - показать пользователю любую неожиданную ошибку
            self.log_queue.put(("error", f"Непредвиденная ошибка: {e}"))

    def _poll_log_queue(self):
        finished = False
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "done":
                    self._log("Готово.")
                    messagebox.showinfo("Готово", "Обработка успешно завершена.")
                    finished = True
                elif kind == "error":
                    self._log(f"Ошибка: {payload}")
                    messagebox.showerror("Ошибка", payload)
                    finished = True
        except queue.Empty:
            pass

        if finished:
            self.progress.stop()
            self.run_button.configure(state="normal")
        else:
            self.after(100, self._poll_log_queue)


def main():
    app = ForkJoinUI()
    app.mainloop()


if __name__ == "__main__":
    main()
