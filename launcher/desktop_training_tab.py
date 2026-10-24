"""Native file dialogs and controls for local material-based training."""
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from launcher.training_session import TrainingSession
from launcher.desktop_widgets import ScrollableForm


class DesktopTrainingTab:
    def __init__(self, notebook, root, python_provider=lambda: sys.executable, environment_provider=lambda: {}):
        self.root = root
        self.python_provider = python_provider
        self.environment_provider = environment_provider
        self.session = TrainingSession()
        self.files = []
        self.events = queue.Queue()
        self.busy = False
        self.tab_frame = ttk.Frame(notebook, padding=12)
        notebook.add(self.tab_frame, text=' 模型强化 ')
        self.form = ScrollableForm(self.tab_frame, padding=22)
        self.form.pack(fill='both', expand=True)
        frame = self.form.body
        ttk.Label(frame, text='从人工标注强化人物识别', style='Title.TLabel').pack(anchor='w')
        ttk.Label(frame, text='选择 annotations 目录中的训练 ZIP，无需解压。ZIP 内含标注、片段音频和目标参考音频，可汇总不同集、不同电脑的 ZIP。',
                  wraplength=920).pack(anchor='w', pady=(8, 20))
        controls = ttk.Frame(frame)
        controls.pack(fill='x')
        self.choose_button = ttk.Button(controls, text='选择训练 ZIP…', command=self.choose_files)
        self.choose_button.pack(side='left')
        self.file_count = tk.StringVar(value='未选择原料')
        ttk.Label(controls, textvariable=self.file_count).pack(side='left', padx=16)
        self.file_list = tk.Listbox(frame, height=5, font=('Microsoft YaHei UI', 10), exportselection=False)
        self.file_list.pack(fill='x', pady=12)
        ttk.Label(frame, text='每次最多 30 份，单份最多 4 GB、合计最多 8 GB。首次强化至少需要 2 段目标匹配和 2 段整句非目标。\n模型记住训练成功的 ZIP 哈希，重复 ZIP 自动跳过；重新打包后哈希不同则按新 ZIP 处理。',
                  wraplength=920).pack(anchor='w', pady=(0, 16))
        buttons = ttk.Frame(frame)
        buttons.pack(fill='x')
        self.start_button = ttk.Button(buttons, text='开始强化', style='Accent.TButton', command=self.start)
        self.start_button.pack(side='left')
        self.clear_button = ttk.Button(buttons, text='清理临时文件', command=self.discard, state='disabled')
        self.clear_button.pack(side='left', padx=12)
        self.status = tk.StringVar()
        self.storage = tk.StringVar()
        ttk.Label(frame, textvariable=self.status, wraplength=920).pack(anchor='w', pady=(18, 8))
        ttk.Label(frame, textvariable=self.storage, wraplength=920, style='Subtitle.TLabel').pack(anchor='w')
        self.report = tk.Text(frame, height=7, font=('Consolas', 11), state='disabled', wrap='word',
                              background='#101522', foreground='#e0e7ff', relief='flat', padx=16, pady=14)
        self.report.pack(fill='both', expand=True, pady=16)
        ttk.Label(frame, text='直接在现用模型上继续训练，成功后自动保存回原位置，下次启动继续使用。\n清理或关闭只删除训练临时文件，已强化的模型会保留。识别效果以实际语音验证为准。',
                  wraplength=920).pack(anchor='w')
        self._last_report = None
        self.root.after(100, self._poll)

    def choose_files(self):
        selected = filedialog.askopenfilenames(parent=self.root, title='选择标注与音频 ZIP',
            initialdir=str(Path(__file__).resolve().parents[1] / 'annotations'), filetypes=[('训练 ZIP', '*.zip')])
        if selected:
            self.files = [Path(path) for path in selected]
            self.file_list.delete(0, 'end')
            for path in self.files:
                self.file_list.insert('end', str(path))
            self.file_count.set(f'已选择 {len(self.files)} 份原料')

    def start(self):
        try:
            self.session.start(self.files, str(self.python_provider()), environment=self.environment_provider())
        except (OSError, ValueError) as exc:
            messagebox.showerror('无法开始强化', str(exc), parent=self.root)

    def _background(self, action, message):
        self.busy = True
        def run():
            try:
                action()
                self.events.put((True, message))
            except Exception as exc:
                self.events.put((False, str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def discard(self):
        self._background(self.session.discard, '本次训练临时文件已清理')

    def _poll(self):
        while not self.events.empty():
            ok, message = self.events.get_nowait()
            self.busy = False
            if not ok:
                messagebox.showerror('操作失败', message, parent=self.root)
            else:
                messagebox.showinfo('已完成', message, parent=self.root)
        snapshot = self.session.snapshot()
        self.status.set(snapshot['message'])
        self.storage.set('临时磁盘：' + snapshot['temporary_path'])
        self.start_button.configure(state='normal' if snapshot['state'] == 'idle' and not self.busy else 'disabled')
        self.choose_button.configure(state='normal' if snapshot['state'] == 'idle' and not self.busy else 'disabled')
        self.clear_button.configure(state='normal' if snapshot['state'] != 'idle' and not self.busy else 'disabled')
        report = snapshot['report']
        if report != self._last_report:
            self.report.configure(state='normal')
            self.report.delete('1.0', 'end')
            if report:
                state = '全部 ZIP 已处理，模型未改动' if report['status'] == 'already_applied' else '已强化并自动保存'
                text = (f"ZIP：新增 {report['new_packages']} 份，跳过重复 {report['duplicate_packages']} 份\n"
                        f"新 ZIP 来源：{report['source_count']} 集／文件，{report['reference_sets']} 组参考\n"
                        f"参与训练 {report['training_clips']} 段，留出验证 {report['held_out_clips']} 段\n"
                        f"状态：{state}\n现用模型：{report['model_path']}\n")
                for name, metrics in report['metrics'].items():
                    text += f"{'训练数据' if name == 'training' else '独立来源留出'}：{metrics['sources']} 个来源，{metrics['pairs']} 对参考与片段\n"
                self.report.insert('end', text)
            self.report.configure(state='disabled')
            self._last_report = report
        self.root.after(400, self._poll)
