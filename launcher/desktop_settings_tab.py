from __future__ import annotations

import json
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

BASE_DIR = Path(os.environ.get("VOICE_EXTRACT_DATA_ROOT", Path(__file__).resolve().parent.parent))
LAUNCHER_DIR = Path(os.environ.get("VOICE_EXTRACT_SOURCE_ROOT", BASE_DIR)) / "launcher"
SETTINGS_PATH = BASE_DIR / "voice_extractor_settings.json"
WEBUI_SERVER_PATH = LAUNCHER_DIR / "webui_server.py"


def load_app_settings() -> dict:
    default_settings = {
        "ui_mode": "desktop",
        "webui_port": 7860,
        "auto_next_on_rate": True,
        "blind_mode": True,
    }
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                default_settings.update(data)
        except Exception:
            pass
    return default_settings


def save_app_settings(settings: dict) -> None:
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


class DesktopSettingsTab:
    """
    桌面端系统设置与界面切换标签页
    - 支持在 Desktop 原生模式与 WebUI 现代网页模式之间自由切换
    - 切换为 WebUI 时，自动拉起 WebUI 前端并自动关闭当前 Desktop 桌面程序
    """

    def __init__(self, parent_notebook: ttk.Notebook, root: tk.Tk, close_callback=None) -> None:
        self.root = root
        self.close_callback = close_callback or root.destroy
        self.notebook = parent_notebook
        self.settings = load_app_settings()

        self.tab_frame = ttk.Frame(self.notebook, padding=16)
        self.notebook.add(self.tab_frame, text=" ⚙️ 界面与系统设置 ")

        self.ui_mode_var = tk.StringVar(value=self.settings.get("ui_mode", "desktop"))
        self.port_var = tk.IntVar(value=self.settings.get("webui_port", 7860))
        self.blind_mode_var = tk.BooleanVar(value=self.settings.get("blind_mode", True))
        self.auto_next_var = tk.BooleanVar(value=self.settings.get("auto_next_on_rate", True))

        self._build_ui()

    def _build_ui(self) -> None:
        self.tab_frame.columnconfigure(0, weight=1)

        header = ttk.Frame(self.tab_frame, style="Card.TFrame", padding=14)
        header.pack(fill="x", pady=(0, 14))
        ttk.Label(header, text="⚙️ 系统偏好与前端模式切换", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(
            header,
            text="切换模式时将自动启动目标界面，并自动关闭当前界面。全局只有一个启动器。",
            style="CardText.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        # 模式切换卡片
        mode_box = ttk.LabelFrame(self.tab_frame, text=" 前端呈现模式 (UI Mode) ", padding=14)
        mode_box.pack(fill="x", pady=(0, 14))

        rb_desktop = ttk.Radiobutton(
            mode_box,
            text="💻 Desktop 原生桌面模式 (当前已处于该模式)",
            variable=self.ui_mode_var,
            value="desktop",
        )
        rb_desktop.pack(anchor="w", pady=(0, 8))

        rb_webui = ttk.Radiobutton(
            mode_box,
            text="🌐 WebUI 现代声学网页模式 (高质感暗黑流线、实时音频频谱动画、全触控卡片)",
            variable=self.ui_mode_var,
            value="webui",
        )
        rb_webui.pack(anchor="w", pady=(0, 12))

        # 切换动作按钮
        btn_bar = ttk.Frame(mode_box)
        btn_bar.pack(anchor="w", pady=(4, 0))
        ttk.Button(
            btn_bar,
            text="🚀 切换为 WebUI (将自动打开浏览器并关闭当前桌面窗口)",
            style="Accent.TButton",
            command=self.switch_to_webui_and_close,
        ).pack(side="left")

        ttk.Label(
            mode_box,
            text="切换后当前桌面窗口会自动关闭；在网页端点击「切回桌面版」可再切回来。",
            style="CardText.TLabel",
        ).pack(anchor="w", pady=(8, 0))

        # 服务端口与标注参数设置
        params_box = ttk.LabelFrame(self.tab_frame, text=" 标注与服务配置 ", padding=14)
        params_box.pack(fill="x", pady=(0, 14))

        port_row = ttk.Frame(params_box)
        port_row.pack(fill="x", pady=(0, 8))
        ttk.Label(port_row, text="Web 服务监听端口: ").pack(side="left")
        ttk.Spinbox(port_row, from_=1024, to=65535, textvariable=self.port_var, width=8).pack(side="left", padx=8)

        ttk.Checkbutton(
            params_box,
            text="默认开启盲听审查 (隐藏置信度与模型判定路由，防主观偏见)",
            variable=self.blind_mode_var,
        ).pack(anchor="w", pady=(0, 6))

        ttk.Checkbutton(
            params_box,
            text="试听标注点击后默认自动切换到下一句",
            variable=self.auto_next_var,
        ).pack(anchor="w", pady=(0, 6))

        # 保存按钮
        action_bar = ttk.Frame(self.tab_frame)
        action_bar.pack(fill="x", pady=(10, 0))
        ttk.Button(action_bar, text="保存偏好", width=14, command=self.save_settings).pack(side="right")

    def save_settings(self, silent: bool = False) -> None:
        self.settings["ui_mode"] = self.ui_mode_var.get()
        self.settings["webui_port"] = self.port_var.get()
        self.settings["blind_mode"] = self.blind_mode_var.get()
        self.settings["auto_next_on_rate"] = self.auto_next_var.get()
        save_app_settings(self.settings)
        if not silent:
            messagebox.showinfo("成功", "设置已成功保存！")

    def switch_to_webui_and_close(self) -> None:
        self.ui_mode_var.set("webui")
        self.save_settings(silent=True)
        port = self.port_var.get()
        url = f"http://127.0.0.1:{port}"

        try:
            from launcher.webview_shell import start_server
            start_server(port)
            if not webbrowser.open(url):
                raise RuntimeError('无法打开默认浏览器')
        except Exception as e:
            messagebox.showerror("启动失败", f"启动 WebUI 失败: {e}")
            return

        # 优雅销毁并退出当前桌面程序
        self.root.after(300, self._exit_desktop)

    def _exit_desktop(self) -> None:
        self.close_callback()
