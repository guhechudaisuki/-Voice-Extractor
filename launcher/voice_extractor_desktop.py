from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from desktop_bootstrap import (
    BootstrapContext,
    CUDA_OFFICIAL_URL,
    PYTORCH_OFFICIAL_URL,
    finalize_inspected_context,
    install_python_runtime,
    inspect_installation,
    load_cached_context,
    prepare_context,
    suggested_install_root,
)


CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
AUDIO_TYPES = (
    ("音频文件", "*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.wma"),
    ("全部文件", "*.*"),
)
MEDIA_TYPES = (
    (
        "音频或视频",
        "*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.wma *.mp4 *.mkv *.avi *.mov *.webm *.m4v",
    ),
    ("全部文件", "*.*"),
)


class VoiceExtractorDesktop:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Voice Extractor")
        self.root.geometry("1240x820")
        self.root.minsize(1040, 700)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.context: BootstrapContext | None = None
        self.worker_process: subprocess.Popen[str] | None = None
        self.boot_running = False
        self.scan_running = False
        self.running_job = False
        self.last_result: dict[str, object] = {}
        self.last_scan_result: dict[str, object] = {}
        self.job_log_path: Path | None = None
        self.runtime_dialog: tk.Toplevel | None = None

        self.references: list[str] = []
        self.targets: list[str] = []
        self.negative_roles: list[list[str]] = []

        initial_install_root = suggested_install_root()
        try:
            cached_context = load_cached_context(initial_install_root)
        except Exception:
            cached_context = None

        self.install_root_var = tk.StringVar(value=str(initial_install_root))
        self.python_var = tk.StringVar(
            value=str(cached_context.python) if cached_context else "等待完整扫描……"
        )
        self.setup_summary_var = tk.StringVar(value="正在扫描本地环境……")
        self.setup_status_var = tk.StringVar(value="准备检查")
        self.main_status_var = tk.StringVar(value="等待任务")
        self.output_root_var = tk.StringVar()
        self.threshold_var = tk.DoubleVar(value=0.68)
        self.silence_min_var = tk.DoubleVar(value=0.20)
        self.silence_max_var = tk.DoubleVar(value=0.85)
        self.overlap_var = tk.BooleanVar(value=True)
        self.singing_var = tk.BooleanVar(value=True)
        self.all_sentences_var = tk.BooleanVar(value=False)
        self.video_segments_var = tk.BooleanVar(value=False)

        self._configure_style()
        self.container = ttk.Frame(self.root, padding=0)
        self.container.pack(fill="both", expand=True)
        self._build_setup_page()
        self._build_main_page()
        self.main_page.pack_forget()
        self._append_setup_log("Voice Extractor 原生桌面版")
        if cached_context is not None:
            self._show_main(cached_context, from_cache=True)
        else:
            self.root.after(150, self._start_scan)
        self.root.after(100, self._poll_events)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        bg = "#f5f7fb"
        card = "#ffffff"
        accent = "#5b5bd6"
        text = "#202331"
        muted = "#666b7a"
        self.root.configure(bg=bg)
        style.configure("TFrame", background=bg)
        style.configure("Card.TFrame", background=card)
        style.configure(
            "Title.TLabel",
            background=bg,
            foreground=text,
            font=("Microsoft YaHei UI", 22, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=bg,
            foreground=muted,
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "CardTitle.TLabel",
            background=card,
            foreground=text,
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        style.configure(
            "CardText.TLabel",
            background=card,
            foreground=muted,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Accent.TButton",
            font=("Microsoft YaHei UI", 10, "bold"),
            foreground="white",
            background=accent,
            borderwidth=0,
            padding=(18, 9),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#4747c2"), ("disabled", "#aaaad4")],
        )
        style.configure("TButton", font=("Microsoft YaHei UI", 9), padding=(10, 6))
        style.configure("TCheckbutton", background=card, font=("Microsoft YaHei UI", 9))
        style.configure("Treeview", font=("Microsoft YaHei UI", 9), rowheight=28)
        style.configure(
            "Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold")
        )
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#e7e9f2",
            background=accent,
            lightcolor=accent,
            darkcolor=accent,
        )

    def _clear_container(self, frame: ttk.Frame) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def _build_setup_page(self) -> None:
        self.setup_page = ttk.Frame(self.container, padding=(46, 34))
        self.setup_page.pack(fill="both", expand=True)

        ttk.Label(
            self.setup_page,
            text="安装与环境检查",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            self.setup_page,
            text="只需选择安装位置。程序会完整检测 Python、CUDA/PyTorch、依赖、模型和后端资源。",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 22))

        card = ttk.Frame(self.setup_page, style="Card.TFrame", padding=24)
        card.pack(fill="both", expand=True)
        card.columnconfigure(1, weight=1)
        card.rowconfigure(6, weight=1)

        ttk.Label(card, text="安装目录", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 7)
        )
        install_entry = ttk.Entry(card, textvariable=self.install_root_var)
        install_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 10))
        ttk.Button(card, text="选择目录", command=self._browse_install_root).grid(
            row=1, column=2, sticky="e"
        )

        ttk.Label(
            card,
            text="Python 运行环境（自动检测）",
            style="CardTitle.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=(18, 7))
        python_entry = ttk.Entry(card, textvariable=self.python_var, state="readonly")
        python_entry.grid(row=3, column=0, columnspan=3, sticky="ew")
        ttk.Label(
            card,
            text=(
                "自动检查环境变量、PATH、py.exe、Windows 注册表、Conda/venv、"
                "常见安装目录和 GPT-SoVITS runtime；不要求填写 python.exe 路径。"
            ),
            style="CardText.TLabel",
            wraplength=950,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(5, 0))

        summary = ttk.Label(
            card,
            textvariable=self.setup_summary_var,
            style="CardText.TLabel",
            justify="left",
            wraplength=950,
        )
        summary.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(18, 10))

        log_frame = ttk.Frame(card, style="Card.TFrame")
        log_frame.grid(row=6, column=0, columnspan=3, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.setup_log = tk.Text(
            log_frame,
            height=13,
            relief="flat",
            background="#11131a",
            foreground="#d9dbea",
            insertbackground="white",
            font=("Cascadia Mono", 9),
            padx=12,
            pady=10,
            state="disabled",
        )
        setup_scroll = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.setup_log.yview
        )
        self.setup_log.configure(yscrollcommand=setup_scroll.set)
        self.setup_log.grid(row=0, column=0, sticky="nsew")
        setup_scroll.grid(row=0, column=1, sticky="ns")

        self.setup_progress = ttk.Progressbar(card, maximum=100)
        self.setup_progress.grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=(15, 4)
        )
        ttk.Label(
            card, textvariable=self.setup_status_var, style="CardText.TLabel"
        ).grid(row=8, column=0, sticky="w")
        button_row = ttk.Frame(card, style="Card.TFrame")
        button_row.grid(row=8, column=1, columnspan=2, sticky="e")
        self.rescan_button = ttk.Button(
            button_row, text="重新扫描", command=self._start_scan
        )
        self.rescan_button.pack(side="left", padx=(0, 8))
        self.runtime_help_button = ttk.Button(
            button_row,
            text="PyTorch / CUDA 官网",
            command=self._show_runtime_requirement_dialog,
            state="disabled",
        )
        self.runtime_help_button.pack(side="left", padx=(0, 8))
        ttk.Button(
            button_row,
            text="停止安装",
            command=self.root.destroy,
        ).pack(side="left", padx=(0, 8))
        self.install_button = ttk.Button(
            button_row,
            text="等待扫描",
            style="Accent.TButton",
            command=self._continue_setup,
            state="disabled",
        )
        self.install_button.pack(side="left")

    def _build_main_page(self) -> None:
        self.main_page = ttk.Frame(self.container, padding=(28, 22))

        header = ttk.Frame(self.main_page)
        header.pack(fill="x", pady=(0, 16))
        left_header = ttk.Frame(header)
        left_header.pack(side="left", fill="x", expand=True)
        ttk.Label(
            left_header, text="参考音色句子提取器", style="Title.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            left_header,
            text="从音频或视频中提取指定人物的完整讲话，并生成对应 STT。",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        ttk.Button(header, text="打开输出目录", command=self._open_output_root).pack(
            side="right"
        )

        body = ttk.Panedwindow(self.main_page, orient="horizontal")
        body.pack(fill="both", expand=True)
        controls = ttk.Frame(body, style="Card.TFrame", padding=16)
        results = ttk.Frame(body, style="Card.TFrame", padding=16)
        body.add(controls, weight=3)
        body.add(results, weight=4)

        self._build_input_controls(controls)
        self._build_results_panel(results)

    def _list_card(
        self,
        parent: ttk.Frame,
        title: str,
        add_command,
        remove_command,
        clear_command,
    ) -> tuple[tk.Listbox, ttk.Frame]:
        frame = ttk.Frame(parent, style="Card.TFrame")
        title_row = ttk.Frame(frame, style="Card.TFrame")
        title_row.pack(fill="x", pady=(0, 6))
        ttk.Label(title_row, text=title, style="CardTitle.TLabel").pack(side="left")
        ttk.Button(title_row, text="添加", command=add_command).pack(side="right")
        ttk.Button(title_row, text="移除", command=remove_command).pack(
            side="right", padx=5
        )
        ttk.Button(title_row, text="清空", command=clear_command).pack(side="right")
        listbox = tk.Listbox(
            frame,
            height=4,
            selectmode="extended",
            relief="solid",
            borderwidth=1,
            font=("Microsoft YaHei UI", 9),
            activestyle="none",
        )
        listbox.pack(fill="both", expand=True)
        return listbox, frame

    def _build_input_controls(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)
        self.reference_list, ref_frame = self._list_card(
            parent,
            "参考音频（可多选）",
            self._add_references,
            lambda: self._remove_selected(self.reference_list, self.references),
            lambda: self._clear_paths(self.reference_list, self.references),
        )
        ref_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
        self.target_list, target_frame = self._list_card(
            parent,
            "待提取音频或视频（可多选）",
            self._add_targets,
            lambda: self._remove_selected(self.target_list, self.targets),
            lambda: self._clear_paths(self.target_list, self.targets),
        )
        target_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 12))

        exclusion = ttk.LabelFrame(parent, text="排除人物（可选）", padding=10)
        exclusion.grid(row=2, column=0, sticky="nsew", pady=(0, 12))
        exclusion.columnconfigure(0, weight=1)
        exclusion.columnconfigure(1, weight=2)
        exclusion.rowconfigure(1, weight=1)
        role_buttons = ttk.Frame(exclusion, style="Card.TFrame")
        role_buttons.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Button(role_buttons, text="添加人物", command=self._add_negative_role).pack(
            side="left"
        )
        ttk.Button(
            role_buttons, text="删除人物", command=self._delete_negative_role
        ).pack(side="left", padx=6)
        ttk.Button(
            role_buttons, text="添加该人物音频", command=self._add_negative_files
        ).pack(side="left")
        ttk.Button(
            role_buttons, text="移除选中音频", command=self._remove_negative_files
        ).pack(side="left", padx=6)
        self.role_list = tk.Listbox(
            exclusion,
            height=4,
            exportselection=False,
            font=("Microsoft YaHei UI", 9),
        )
        self.role_list.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.role_list.bind("<<ListboxSelect>>", lambda _event: self._refresh_role_files())
        self.role_file_list = tk.Listbox(
            exclusion,
            height=4,
            selectmode="extended",
            exportselection=False,
            font=("Microsoft YaHei UI", 9),
        )
        self.role_file_list.grid(row=1, column=1, sticky="nsew")

        options = ttk.LabelFrame(parent, text="提取设置", padding=10)
        options.grid(row=3, column=0, sticky="nsew")
        for column in range(4):
            options.columnconfigure(column, weight=1)
        ttk.Checkbutton(
            options, text="过滤多人同时说话", variable=self.overlap_var
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(
            options, text="过滤有人唱歌", variable=self.singing_var
        ).grid(row=0, column=2, columnspan=2, sticky="w")
        ttk.Label(options, text="声纹阈值").grid(row=1, column=0, sticky="w", pady=(8, 2))
        ttk.Spinbox(
            options,
            from_=0.50,
            to=0.90,
            increment=0.01,
            textvariable=self.threshold_var,
            width=8,
        ).grid(row=2, column=0, sticky="ew", padx=(0, 8))
        ttk.Label(options, text="静音下限（秒）").grid(
            row=1, column=1, sticky="w", pady=(8, 2)
        )
        ttk.Spinbox(
            options,
            from_=0.0,
            to=3.0,
            increment=0.05,
            textvariable=self.silence_min_var,
            width=8,
        ).grid(row=2, column=1, sticky="ew", padx=(0, 8))
        ttk.Label(options, text="静音上限（秒）").grid(
            row=1, column=2, sticky="w", pady=(8, 2)
        )
        ttk.Spinbox(
            options,
            from_=0.0,
            to=3.0,
            increment=0.05,
            textvariable=self.silence_max_var,
            width=8,
        ).grid(row=2, column=2, sticky="ew", padx=(0, 8))
        ttk.Checkbutton(
            options, text="输出全部单句", variable=self.all_sentences_var
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            options, text="同时输出视频片段", variable=self.video_segments_var
        ).grid(row=3, column=2, columnspan=2, sticky="w", pady=(8, 0))

        output_frame = ttk.Frame(options, style="Card.TFrame")
        output_frame.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        output_frame.columnconfigure(0, weight=1)
        ttk.Label(output_frame, text="结果保存位置（默认 output）").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 3)
        )
        ttk.Entry(output_frame, textvariable=self.output_root_var).grid(
            row=1, column=0, sticky="ew", padx=(0, 8)
        )
        ttk.Button(
            output_frame, text="选择", command=self._browse_output_root
        ).grid(row=1, column=1)
        ttk.Button(
            output_frame, text="恢复默认", command=self._reset_output_root
        ).grid(row=1, column=2, padx=(8, 0))

    def _build_results_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        ttk.Label(parent, text="提取结果", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        columns = ("target", "text", "language", "duration", "score")
        self.result_tree = ttk.Treeview(parent, columns=columns, show="headings")
        headings = {
            "target": "目标文件",
            "text": "STT 文本",
            "language": "语言",
            "duration": "时长",
            "score": "声纹",
        }
        widths = {"target": 150, "text": 360, "language": 55, "duration": 65, "score": 65}
        for column in columns:
            self.result_tree.heading(column, text=headings[column])
            self.result_tree.column(column, width=widths[column], anchor="w")
        result_scroll = ttk.Scrollbar(
            parent, orient="vertical", command=self.result_tree.yview
        )
        self.result_tree.configure(yscrollcommand=result_scroll.set)
        self.result_tree.grid(row=1, column=0, sticky="nsew")
        result_scroll.grid(row=1, column=1, sticky="ns")
        self.result_tree.bind("<Double-1>", self._open_selected_result)

        status_box = ttk.Frame(parent, style="Card.TFrame")
        status_box.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        status_box.columnconfigure(0, weight=1)
        self.job_progress = ttk.Progressbar(status_box, maximum=100)
        self.job_progress.grid(row=0, column=0, columnspan=4, sticky="ew")
        ttk.Label(
            status_box, textvariable=self.main_status_var, style="CardText.TLabel"
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 8))
        self.run_button = ttk.Button(
            status_box,
            text="开始提取",
            style="Accent.TButton",
            command=self._start_job,
        )
        self.run_button.grid(row=2, column=0, sticky="w")
        self.cancel_button = ttk.Button(
            status_box, text="中止任务", command=self._cancel_job, state="disabled"
        )
        self.cancel_button.grid(row=2, column=1, sticky="w", padx=8)
        self.open_batch_button = ttk.Button(
            status_box, text="打开本次结果", command=self._open_batch, state="disabled"
        )
        self.open_batch_button.grid(row=2, column=2, sticky="e", padx=8)
        self.open_zip_button = ttk.Button(
            status_box, text="查看 ZIP", command=self._open_archive, state="disabled"
        )
        self.open_zip_button.grid(row=2, column=3, sticky="e")

        ttk.Label(parent, text="运行日志", style="CardTitle.TLabel").grid(
            row=3, column=0, sticky="w", pady=(14, 6)
        )
        self.job_log = tk.Text(
            parent,
            height=8,
            relief="flat",
            background="#11131a",
            foreground="#d9dbea",
            insertbackground="white",
            font=("Cascadia Mono", 9),
            padx=10,
            pady=8,
            state="disabled",
        )
        self.job_log.grid(row=4, column=0, columnspan=2, sticky="ew")

    def _append_text(self, widget: tk.Text, message: str) -> None:
        widget.configure(state="normal")
        widget.insert("end", message.rstrip() + "\n")
        widget.see("end")
        widget.configure(state="disabled")

    def _append_setup_log(self, message: str) -> None:
        self._append_text(self.setup_log, message)
        self._write_log("desktop.log", message)

    def _append_job_log(self, message: str) -> None:
        self._append_text(self.job_log, message)
        self._write_log("desktop.log", message)
        if self.job_log_path is not None:
            self._write_log(self.job_log_path.name, message)

    def _write_log(self, filename: str, message: str) -> None:
        try:
            if self.context is not None:
                log_root = self.context.layout.logs
            else:
                install_text = self.install_root_var.get().strip()
                if not install_text:
                    return
                install_root = Path(install_text).expanduser()
                if not install_root.exists():
                    return
                log_root = install_root / "logs"
            log_root.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with (log_root / filename).open("a", encoding="utf-8") as handle:
                handle.write(f"[{timestamp}] {message.rstrip()}\n")
        except OSError:
            pass

    def _browse_install_root(self) -> None:
        chosen = filedialog.askdirectory(
            title="选择 Voice Extractor 安装目录",
            initialdir=self.install_root_var.get() or str(Path.home()),
        )
        if chosen:
            self.install_root_var.set(chosen)
            self._start_scan()

    def _browse_output_root(self) -> None:
        chosen = filedialog.askdirectory(
            title="选择结果保存目录",
            initialdir=self.output_root_var.get().strip()
            or str(self._default_output_root()),
        )
        if chosen:
            self.output_root_var.set(chosen)

    def _default_output_root(self) -> Path:
        if self.context is not None:
            return self.context.layout.output
        install_root = self.install_root_var.get().strip()
        if install_root:
            return Path(install_root).expanduser() / "output"
        return suggested_install_root() / "output"

    def _reset_output_root(self) -> None:
        self.output_root_var.set(str(self._default_output_root()))

    def _start_scan(self) -> None:
        if self.scan_running or self.boot_running:
            return
        if self.runtime_dialog is not None and self.runtime_dialog.winfo_exists():
            self.runtime_dialog.destroy()
        self.scan_running = True
        self.last_scan_result = {}
        self.install_button.configure(state="disabled")
        self.install_button.configure(text="等待扫描")
        self.runtime_help_button.configure(state="disabled")
        self.rescan_button.configure(state="disabled")
        self.setup_summary_var.set("正在完整扫描，请稍候。扫描结束前不会要求做决定。")
        self.setup_status_var.set("正在扫描本地运行环境、依赖和全部模型……")
        self.setup_progress.configure(value=3)
        install_text = self.install_root_var.get().strip()
        if not install_text:
            self.scan_running = False
            self.rescan_button.configure(state="normal")
            messagebox.showerror("安装目录无效", "请选择安装目录。")
            return
        install_root = Path(install_text).expanduser()

        def reporter(value: float, message: str) -> None:
            self.events.put(("scan_progress", (value, message)))

        def worker() -> None:
            try:
                result = inspect_installation(install_root, reporter=reporter)
                self.events.put(("scan_complete", result))
            except Exception as exc:
                self.events.put(("scan_error", str(exc)))

        threading.Thread(target=worker, name="desktop-scan", daemon=True).start()

    def _continue_setup(self) -> None:
        result = self.last_scan_result
        if not result.get("scan_complete"):
            self._start_scan()
            return
        if result.get("ready"):
            self._start_finalize_inspection()
            return
        if result.get("python_install_required"):
            self._start_python_install()
            return
        if not result.get("runtime_ready"):
            self._show_runtime_requirement_dialog()
            return
        self._start_bootstrap()

    def _start_python_install(self) -> None:
        if self.boot_running:
            return
        self.boot_running = True
        self.install_button.configure(state="disabled")
        self.rescan_button.configure(state="disabled")
        self.runtime_help_button.configure(state="disabled")
        self.setup_status_var.set("正在安装官方 Python……")
        install_root = Path(self.install_root_var.get()).expanduser()

        def reporter(value: float, message: str) -> None:
            self.events.put(("boot_progress", (value, message)))

        def worker() -> None:
            try:
                python = install_python_runtime(install_root, reporter)
                self.events.put(("python_install_complete", python))
            except Exception as exc:
                self.events.put(("boot_error", str(exc)))

        threading.Thread(
            target=worker,
            name="desktop-python-install",
            daemon=True,
        ).start()

    def _start_bootstrap(self) -> None:
        if self.boot_running:
            return
        result = self.last_scan_result
        preferred_value = result.get("python")
        if not preferred_value or not result.get("runtime_ready"):
            self._show_runtime_requirement_dialog()
            return
        preferred = Path(str(preferred_value))
        self.boot_running = True
        self.install_button.configure(state="disabled")
        self.rescan_button.configure(state="disabled")
        self.runtime_help_button.configure(state="disabled")
        self.setup_status_var.set("正在检查并安装……")
        install_root = Path(self.install_root_var.get()).expanduser()

        def reporter(value: float, message: str) -> None:
            self.events.put(("boot_progress", (value, message)))

        def worker() -> None:
            try:
                context = prepare_context(install_root, preferred, reporter)
                self.events.put(("boot_complete", context))
            except Exception as exc:
                self.events.put(("boot_error", str(exc)))

        threading.Thread(target=worker, name="desktop-bootstrap", daemon=True).start()

    def _start_finalize_inspection(self) -> None:
        if self.boot_running:
            return
        result = self.last_scan_result
        if not result.get("ready"):
            return
        self.boot_running = True
        self.install_button.configure(state="disabled")
        self.rescan_button.configure(state="disabled")
        self.runtime_help_button.configure(state="disabled")
        self.setup_status_var.set("全部依赖已满足，正在保存永久缓存……")
        install_root = Path(self.install_root_var.get()).expanduser()

        def reporter(value: float, message: str) -> None:
            self.events.put(("boot_progress", (value, message)))

        def worker() -> None:
            try:
                context = finalize_inspected_context(
                    install_root,
                    result,
                    reporter,
                )
                self.events.put(("boot_complete", context))
            except Exception as exc:
                self.events.put(("boot_error", str(exc)))

        threading.Thread(
            target=worker,
            name="desktop-cache-finalize",
            daemon=True,
        ).start()

    def _prompt_install_missing(self) -> None:
        result = self.last_scan_result
        if not result.get("scan_complete") or result.get("ready"):
            return
        missing = [
            getattr(item, "label", str(item))
            for item in result.get("missing", [])
        ]
        missing.extend(str(value) for value in result.get("supplementary_missing", []))
        detail = "、".join(missing) if missing else "部分运行依赖"
        size = int(result.get("missing_mb", 0))
        message = f"检测到缺失依赖：{detail}。"
        if size:
            message += f"\n预计下载约 {size / 1024:.2f} GB。"
        message += "\n\n是否现在安装缺失项？"
        if messagebox.askyesno("安装缺失依赖", message):
            self._continue_setup()
        else:
            self.setup_status_var.set("已取消安装；可稍后点击“安装缺失项”")

    def _show_runtime_requirement_dialog(self) -> None:
        result = self.last_scan_result
        if not result.get("scan_complete"):
            return
        if self.runtime_dialog is not None and self.runtime_dialog.winfo_exists():
            self.runtime_dialog.lift()
            self.runtime_dialog.focus_force()
            return

        runtime = result.get("runtime")
        states = result.get("python_states", [])
        if not isinstance(runtime, dict):
            runtime = next(
                (
                    item
                    for item in states
                    if isinstance(item, dict) and item.get("runnable")
                ),
                {},
            )
        if result.get("python_install_required"):
            reason = (
                "完整扫描已结束：没有找到可运行的 Python。\n"
                "可以自动安装官方 Python 3.9.13；安装后会再次完整扫描。"
            )
        else:
            missing = []
            if not runtime.get("torch"):
                missing.append("PyTorch")
            if not runtime.get("torchaudio"):
                missing.append("torchaudio")
            if not runtime.get("cuda_available"):
                missing.append("可用 CUDA")
            reason = (
                "完整扫描已结束，但当前运行环境缺少："
                + "、".join(missing or ["可用的 CUDA/PyTorch 组合"])
                + "。\n本工具不会自动下载或替换 CUDA/PyTorch。"
            )

        dialog = tk.Toplevel(self.root)
        self.runtime_dialog = dialog
        dialog.title("CUDA / PyTorch 环境未就绪")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        frame = ttk.Frame(dialog, padding=22)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text="运行环境需要处理",
            font=("Microsoft YaHei UI", 14, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text=reason,
            justify="left",
            wraplength=610,
        ).pack(anchor="w", pady=(10, 16))
        ttk.Label(
            frame,
            text="模型和其他依赖也已经全部扫描，详细结果仍保留在主安装页。",
            foreground="#666b7a",
        ).pack(anchor="w", pady=(0, 16))
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")

        def open_url(url: str) -> None:
            webbrowser.open(url, new=2)

        if result.get("python_install_required"):
            ttk.Button(
                buttons,
                text="自动安装 Python",
                style="Accent.TButton",
                command=lambda: (dialog.destroy(), self._start_python_install()),
            ).pack(side="left", padx=(0, 8))
        ttk.Button(
            buttons,
            text="PyTorch 官网",
            command=lambda: open_url(PYTORCH_OFFICIAL_URL),
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            buttons,
            text="CUDA 官网",
            command=lambda: open_url(CUDA_OFFICIAL_URL),
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            buttons,
            text="停止安装",
            command=self.root.destroy,
        ).pack(side="right")
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
        dialog.grab_set()

    def _show_main(
        self,
        context: BootstrapContext,
        from_cache: bool = False,
    ) -> None:
        self.context = context
        self.output_root_var.set(str(context.layout.output))
        self.setup_page.pack_forget()
        self.main_page.pack(fill="both", expand=True)
        self.main_status_var.set(
            "已读取永久依赖缓存，直接启动"
            if from_cache
            else "本地模型与运行环境已就绪"
        )
        self._append_job_log(f"安装目录：{context.layout.root}")
        self._append_job_log(f"本地 Python：{context.python}")
        if from_cache:
            self._append_job_log("已跳过重复依赖检查并直接进入主界面")

    def _add_paths(self, values: list[str], listbox: tk.Listbox, filetypes) -> None:
        selected = filedialog.askopenfilenames(filetypes=filetypes)
        for path in selected:
            normal = str(Path(path).resolve())
            if normal not in values:
                values.append(normal)
                listbox.insert("end", normal)

    def _add_references(self) -> None:
        self._add_paths(self.references, self.reference_list, AUDIO_TYPES)

    def _add_targets(self) -> None:
        self._add_paths(self.targets, self.target_list, MEDIA_TYPES)

    @staticmethod
    def _remove_selected(listbox: tk.Listbox, values: list[str]) -> None:
        for index in reversed(listbox.curselection()):
            listbox.delete(index)
            del values[index]

    @staticmethod
    def _clear_paths(listbox: tk.Listbox, values: list[str]) -> None:
        listbox.delete(0, "end")
        values.clear()

    def _add_negative_role(self) -> None:
        self.negative_roles.append([])
        self.role_list.insert("end", f"排除人物 {len(self.negative_roles)}")
        self.role_list.selection_clear(0, "end")
        self.role_list.selection_set(len(self.negative_roles) - 1)
        self._refresh_role_files()

    def _selected_role_index(self) -> int | None:
        selection = self.role_list.curselection()
        return int(selection[0]) if selection else None

    def _delete_negative_role(self) -> None:
        index = self._selected_role_index()
        if index is None:
            return
        del self.negative_roles[index]
        self.role_list.delete(0, "end")
        for row in range(len(self.negative_roles)):
            self.role_list.insert("end", f"排除人物 {row + 1}")
        if self.negative_roles:
            next_index = min(index, len(self.negative_roles) - 1)
            self.role_list.selection_set(next_index)
            self.role_list.activate(next_index)
        self._refresh_role_files()

    def _add_negative_files(self) -> None:
        index = self._selected_role_index()
        if index is None:
            self._add_negative_role()
            index = self._selected_role_index()
        if index is None:
            return
        selected = filedialog.askopenfilenames(filetypes=AUDIO_TYPES)
        for path in selected:
            normal = str(Path(path).resolve())
            if normal not in self.negative_roles[index]:
                self.negative_roles[index].append(normal)
        self._refresh_role_files()

    def _remove_negative_files(self) -> None:
        index = self._selected_role_index()
        if index is None:
            return
        for file_index in reversed(self.role_file_list.curselection()):
            del self.negative_roles[index][file_index]
        self._refresh_role_files()

    def _refresh_role_files(self) -> None:
        self.role_file_list.delete(0, "end")
        index = self._selected_role_index()
        if index is None or index >= len(self.negative_roles):
            return
        for path in self.negative_roles[index]:
            self.role_file_list.insert("end", path)

    def _validate_job(self) -> dict[str, object] | None:
        if not self.references:
            messagebox.showwarning("缺少参考音频", "请至少添加一段参考音频。")
            return None
        if not self.targets:
            messagebox.showwarning("缺少目标文件", "请至少添加一个待提取音频或视频。")
            return None
        try:
            threshold = float(self.threshold_var.get())
            silence_min = float(self.silence_min_var.get())
            silence_max = float(self.silence_max_var.get())
        except (TypeError, ValueError):
            messagebox.showerror("设置错误", "声纹阈值和静音范围必须是数字。")
            return None
        if not 0.50 <= threshold <= 0.90:
            messagebox.showerror("设置错误", "声纹阈值应在 0.50 到 0.90 之间。")
            return None
        if silence_min < 0 or silence_max < silence_min:
            messagebox.showerror("设置错误", "静音范围无效，下限不能大于上限。")
            return None
        output_text = self.output_root_var.get().strip()
        output_root = (
            Path(output_text).expanduser()
            if output_text
            else self._default_output_root()
        )
        try:
            output_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("保存位置不可用", str(exc))
            return None
        output_root = output_root.resolve()
        self.output_root_var.set(str(output_root))
        return {
            "references": list(self.references),
            "targets": list(self.targets),
            "negative_groups": [list(group) for group in self.negative_roles if group],
            "output_root": str(output_root),
            "options": {
                "speaker_threshold": threshold,
                "silence_min_seconds": silence_min,
                "silence_max_seconds": silence_max,
                "use_overlap_detector": bool(self.overlap_var.get()),
                "use_singing_detector": bool(self.singing_var.get()),
                "export_all_sentences": bool(self.all_sentences_var.get()),
                "export_video_clips": bool(self.video_segments_var.get()),
            },
        }

    def _start_job(self) -> None:
        if self.running_job or self.context is None:
            return
        request = self._validate_job()
        if request is None:
            return
        self.job_log_path = self.context.layout.logs / (
            f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.log"
        )
        self.last_result = {}
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        request_path = (
            self.context.layout.requests
            / f"request_{int(time.time())}_{uuid.uuid4().hex[:8]}.json"
        )
        request_path.write_text(
            json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        env = self.context.backend_environment(Path(str(request["output_root"])))
        command = [
            str(self.context.python),
            "-u",
            str(self.context.source_root / "desktop_worker.py"),
            str(request_path),
        ]
        try:
            self.worker_process = subprocess.Popen(
                command,
                cwd=str(self.context.source_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
            )
        except OSError as exc:
            messagebox.showerror("无法启动任务", str(exc))
            return
        self.running_job = True
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.open_batch_button.configure(state="disabled")
        self.open_zip_button.configure(state="disabled")
        self.job_progress.configure(value=0)
        self.main_status_var.set("任务已启动")
        self._append_job_log("开始处理……")

        def reader() -> None:
            assert self.worker_process is not None
            assert self.worker_process.stdout is not None
            for line in self.worker_process.stdout:
                value = line.strip()
                if not value:
                    continue
                try:
                    event = json.loads(value)
                except ValueError:
                    self.events.put(("job_log", value))
                else:
                    self.events.put(("job_event", event))
            code = self.worker_process.wait()
            self.events.put(("job_exit", code))

        threading.Thread(target=reader, name="desktop-job-reader", daemon=True).start()

    def _cancel_job(self) -> None:
        process = self.worker_process
        if process is None or process.poll() is not None:
            return
        if not messagebox.askyesno("中止任务", "确定要停止当前提取任务吗？"):
            return
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                creationflags=CREATE_NO_WINDOW,
                check=False,
                capture_output=True,
            )
        except OSError:
            process.terminate()
        self.main_status_var.set("正在停止任务……")

    def _handle_job_event(self, event: dict[str, object]) -> None:
        event_type = event.get("type")
        if event_type == "progress":
            value = float(event.get("value", 0.0))
            message = str(event.get("message", "正在处理"))
            self.job_progress.configure(value=value * 100)
            self.main_status_var.set(f"{value * 100:.1f}%  {message}")
            self._append_job_log(f"[{value * 100:5.1f}%] {message}")
        elif event_type == "complete":
            self.last_result = event
            for row in event.get("rows", []):
                if not isinstance(row, dict):
                    continue
                self.result_tree.insert(
                    "",
                    "end",
                    values=(
                        row.get("target", ""),
                        row.get("text", ""),
                        row.get("language", ""),
                        f"{float(row.get('duration', 0.0)):.2f}s",
                        f"{float(row.get('speaker_score', 0.0)):.3f}",
                    ),
                    tags=(str(row.get("audio_file", "")),),
                )
            accepted = int(event.get("accepted_count", 0))
            rejected = int(event.get("rejected_count", 0))
            self.job_progress.configure(value=100)
            self.main_status_var.set(f"完成：保留 {accepted} 句，舍弃 {rejected} 句")
            self.open_batch_button.configure(state="normal")
            self.open_zip_button.configure(state="normal")
            self._append_job_log(f"批次完成：{event.get('output_dir', '')}")
        elif event_type == "error":
            message = str(event.get("message", "未知错误"))
            trace = str(event.get("traceback", ""))
            self.main_status_var.set("任务失败")
            self._append_job_log(message)
            if trace:
                self._append_job_log(trace)
            messagebox.showerror("提取失败", message)

    def _finish_job(self, return_code: int) -> None:
        if not self.running_job:
            return
        self.running_job = False
        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        if return_code and not self.last_result:
            self.main_status_var.set(f"任务已停止或失败（退出代码 {return_code}）")
        self.worker_process = None

    def _open_path(self, value: object, select: bool = False) -> None:
        if not value:
            return
        path = Path(str(value))
        try:
            if select and path.is_file():
                subprocess.Popen(["explorer", "/select,", str(path)])
            elif path.exists():
                os.startfile(str(path))
        except OSError as exc:
            messagebox.showerror("无法打开", str(exc))

    def _open_output_root(self) -> None:
        value = self.output_root_var.get().strip()
        self._open_path(value or self._default_output_root())

    def _open_batch(self) -> None:
        self._open_path(self.last_result.get("output_dir"))

    def _open_archive(self) -> None:
        self._open_path(self.last_result.get("archive_path"), select=True)

    def _open_selected_result(self, _event=None) -> None:
        selected = self.result_tree.selection()
        if not selected:
            return
        tags = self.result_tree.item(selected[0], "tags")
        if tags:
            self._open_path(tags[0])

    def _format_scan_summary(self, result: dict[str, object]) -> str:
        runtime = result.get("runtime")
        any_python = result.get("any_python")
        if isinstance(runtime, dict):
            device_names = runtime.get("devices") or []
            device = "、".join(str(value) for value in device_names) or "CUDA 不可用"
            python_line = (
                f"Python：{runtime.get('version', '?')}  |  "
                f"PyTorch {runtime.get('torch_version', '?')}  |  "
                f"CUDA {runtime.get('cuda_build') or '不可用'}  |  {device}"
            )
        elif any_python:
            python_line = f"Python：已找到 {any_python}，但 CUDA/PyTorch 组合未就绪"
        else:
            python_line = "Python：未找到，将由用户确认后自动安装官方 Python 3.9.13"

        nvidia = result.get("nvidia") if isinstance(result.get("nvidia"), dict) else {}
        if nvidia.get("driver_available"):
            gpu_names = "、".join(str(value) for value in nvidia.get("gpus", []))
            driver_line = f"NVIDIA：{gpu_names}（驱动 {nvidia.get('driver_version', '?')}）"
        else:
            driver_line = "NVIDIA：未检测到可用驱动或显卡"

        missing = result.get("missing", [])
        found_models = len(result.get("assets", {}))
        model_line = f"模型：已找到 {found_models}/{found_models + len(missing)}"
        if missing:
            model_line += "；缺失 " + "、".join(
                getattr(item, "label", str(item)) for item in missing
            )

        missing_modules = result.get("missing_modules", [])
        module_line = (
            "Python 依赖：全部可用"
            if not missing_modules
            else "Python 依赖缺失：" + "、".join(str(value) for value in missing_modules)
        )
        tools = result.get("tools") if isinstance(result.get("tools"), dict) else {}
        missing_tools = tools.get("missing", []) if tools else []
        tool_line = (
            "媒体与后端：全部可用"
            if not missing_tools
            else "媒体与后端缺失：" + "、".join(str(value) for value in missing_tools)
        )
        download_mb = int(result.get("missing_mb", 0))
        download_line = (
            "无需下载模型或 Python。"
            if download_mb == 0
            else f"预计缺失下载量约 {download_mb / 1024:.2f} GB（已有资源不会重复下载）。"
        )
        return "\n".join(
            (python_line, driver_line, module_line, model_line, tool_line, download_line)
        )

    def _log_scan_result(self, result: dict[str, object]) -> None:
        self._append_setup_log("—— 完整扫描报告 ——")
        for state in result.get("python_states", []):
            if not isinstance(state, dict):
                continue
            candidate = state.get("candidate") or state.get("executable")
            if not state.get("runnable"):
                self._append_setup_log(f"[Python 不可用] {candidate}")
                continue
            self._append_setup_log(
                "[Python] "
                f"{candidate} | {state.get('version', '?')} | "
                f"torch={'是' if state.get('torch') else '否'} | "
                f"torchaudio={'是' if state.get('torchaudio') else '否'} | "
                f"CUDA={'是' if state.get('cuda_available') else '否'}"
            )
        for row in result.get("component_rows", []):
            if not isinstance(row, dict):
                continue
            if row.get("ready"):
                self._append_setup_log(f"[模型可用] {row.get('label')}：{row.get('path')}")
            else:
                self._append_setup_log(
                    f"[模型缺失] {row.get('label')}（约 {row.get('download_mb', 0)} MB）"
                )
        tools = result.get("tools") if isinstance(result.get("tools"), dict) else {}
        for label, path in tools.get("rows", {}).items():
            self._append_setup_log(
                f"[{'可用' if path else '缺失'}] {label}"
                + (f"：{path}" if path else "")
            )
        self._append_setup_log("—— 扫描已全部完成，现在可以决定下一步 ——")

    def _poll_events(self) -> None:
        try:
            while True:
                event_type, payload = self.events.get_nowait()
                if event_type == "scan_progress":
                    value, message = payload
                    self.setup_progress.configure(value=float(value) * 100)
                    self.setup_status_var.set(str(message))
                    self._append_setup_log(str(message))
                elif event_type == "scan_complete":
                    self.scan_running = False
                    self.rescan_button.configure(state="normal")
                    result = payload if isinstance(payload, dict) else {}
                    self.last_scan_result = result
                    python = result.get("python") or result.get("any_python")
                    self.python_var.set(str(python) if python else "未找到 Python")
                    self.setup_summary_var.set(self._format_scan_summary(result))
                    self.setup_progress.configure(value=100)
                    self._log_scan_result(result)
                    if result.get("python_install_required"):
                        self.setup_status_var.set("扫描完成：请选择自动安装 Python、打开官网或停止安装")
                        self.install_button.configure(text="自动安装 Python", state="normal")
                        self.runtime_help_button.configure(state="normal")
                        self.root.after(150, self._show_runtime_requirement_dialog)
                    elif not result.get("runtime_ready"):
                        self.setup_status_var.set("扫描完成：CUDA/PyTorch 环境未就绪")
                        self.install_button.configure(text="环境未就绪", state="disabled")
                        self.runtime_help_button.configure(state="normal")
                        self.root.after(150, self._show_runtime_requirement_dialog)
                    elif result.get("ready"):
                        self.setup_status_var.set("扫描完成：全部依赖满足，正在直接进入主界面")
                        self.install_button.configure(text="正在进入", state="disabled")
                        self.runtime_help_button.configure(state="disabled")
                        self.root.after(50, self._start_finalize_inspection)
                    else:
                        self.setup_status_var.set("扫描完成：发现缺失依赖")
                        self.install_button.configure(text="安装缺失项", state="normal")
                        self.runtime_help_button.configure(state="disabled")
                        self.root.after(150, self._prompt_install_missing)
                elif event_type == "scan_error":
                    self.scan_running = False
                    self.rescan_button.configure(state="normal")
                    self.install_button.configure(text="等待扫描", state="disabled")
                    self.setup_status_var.set("扫描失败")
                    self._append_setup_log(str(payload))
                elif event_type == "boot_progress":
                    value, message = payload
                    self.setup_progress.configure(value=float(value) * 100)
                    self.setup_status_var.set(str(message))
                    self._append_setup_log(str(message))
                elif event_type == "boot_complete":
                    self.boot_running = False
                    self._show_main(payload)
                elif event_type == "python_install_complete":
                    self.boot_running = False
                    self.python_var.set(str(payload))
                    self.setup_status_var.set("Python 安装完成，正在重新完整扫描……")
                    self._append_setup_log(f"Python 安装完成：{payload}")
                    self.root.after(100, self._start_scan)
                elif event_type == "boot_error":
                    self.boot_running = False
                    self.rescan_button.configure(state="normal")
                    if self.last_scan_result.get("ready"):
                        self.install_button.configure(text="重试进入", state="normal")
                    elif self.last_scan_result.get("python_install_required"):
                        self.install_button.configure(text="自动安装 Python", state="normal")
                    elif self.last_scan_result.get("runtime_ready"):
                        self.install_button.configure(text="安装缺失项", state="normal")
                    else:
                        self.install_button.configure(text="环境未就绪", state="disabled")
                        self.runtime_help_button.configure(state="normal")
                    self.setup_status_var.set("安装或检查失败")
                    self._append_setup_log(str(payload))
                    messagebox.showerror("安装失败", str(payload))
                elif event_type == "job_event":
                    if isinstance(payload, dict):
                        self._handle_job_event(payload)
                elif event_type == "job_log":
                    self._append_job_log(str(payload))
                elif event_type == "job_exit":
                    self._finish_job(int(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _on_close(self) -> None:
        if self.running_job:
            if not messagebox.askyesno("退出", "任务仍在运行，确定停止任务并退出吗？"):
                return
            process = self.worker_process
            if process and process.poll() is None:
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        creationflags=CREATE_NO_WINDOW,
                        check=False,
                        capture_output=True,
                    )
                except OSError:
                    process.terminate()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    VoiceExtractorDesktop(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
