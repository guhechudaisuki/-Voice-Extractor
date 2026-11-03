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
        ttk.Label(frame, text='每次最多 30 份，单份最多 4 GB、合计最多 8 GB。经验池累计至少 3 个独立来源，且训练集和留出集均有正负样本后，才可通过验证更新权重；不足时先保存特征。\n重复 ZIP 和已入库音频自动去重，新样本与适量历史样本共同训练。',
                  wraplength=920).pack(anchor='w', pady=(0, 16))
        buttons = ttk.Frame(frame)
        buttons.pack(fill='x')
        self.start_button = ttk.Button(buttons, text='开始强化', style='Accent.TButton', command=self.start)
        self.start_button.pack(side='left')
        self.clear_button = ttk.Button(buttons, text='清理临时文件', command=self.discard, state='disabled')
        self.clear_button.pack(side='left', padx=12)
        self.rollback_available = False
        self.rollback_button = ttk.Button(buttons, text='回滚至上一次模型备份', command=self.rollback, state='disabled')
        self.rollback_button.pack(side='left')
        self.status = tk.StringVar()
        self.storage = tk.StringVar()
        ttk.Label(frame, textvariable=self.status, wraplength=920).pack(anchor='w', pady=(18, 8))
        ttk.Label(frame, textvariable=self.storage, wraplength=920, style='Subtitle.TLabel').pack(anchor='w')
        self.report = tk.Text(frame, height=7, font=('Consolas', 11), state='disabled', wrap='word',
                              background='#101522', foreground='#e0e7ff', relief='flat', padx=16, pady=14)
        self.report.pack(fill='both', expand=True, pady=16)
        ttk.Label(frame, text='全局模型通过独立留出集验证后保存，后续提取自动使用。\n清理或关闭只删除训练临时文件，已强化的模型会保留。识别效果以实际语音验证为准。',
                  wraplength=920).pack(anchor='w')
        self._last_report = None
        self.refresh_rollback()
        self.root.after(100, self._poll)

    def _model_command(self, operation):
        import os
        import json
        import subprocess
        if getattr(sys, 'frozen', False):
            from launcher.corpus_worker import invoke
            return invoke(operation, {})
        environment = {**os.environ, **self.environment_provider()}
        source = environment.get('VOICE_EXTRACT_SOURCE_ROOT', str(Path(__file__).resolve().parents[1]))
        program = ("import sys,json; sys.path.insert(0,sys.argv[1]); "
                   "from launcher.corpus_worker import execute; "
                   "print(json.dumps(execute(sys.argv[2],{}),ensure_ascii=False))")
        command = [str(self.python_provider()), '-B', '-X', 'utf8', '-c', program, source, operation]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace',
            env=environment, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=600)
        if result.returncode:
            raise ValueError(result.stderr[-1500:])
        return json.loads(result.stdout.strip().splitlines()[-1])

    def refresh_rollback(self):
        def run():
            try:
                self.events.put(('rollback_status', self._model_command('rollback_status')))
            except Exception as exc:
                self.events.put(('rollback_error', str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def rollback(self):
        if not self.busy and self.session.snapshot()['state'] != 'running':
            self._background(lambda: self._model_command('rollback'), '已回滚至上一次模型备份，下次提取生效')

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
        if self.busy:
            return
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
            if ok == 'rollback_status':
                self.rollback_available = bool(message['available'])
                continue
            if ok == 'rollback_error':
                self.rollback_available = False
                continue
            self.busy = False
            if not ok:
                messagebox.showerror('操作失败', message, parent=self.root)
            else:
                messagebox.showinfo('已完成', message, parent=self.root)
                self.refresh_rollback()
        snapshot = self.session.snapshot()
        from launcher.reinforcement_history import completion_message as training_completion_message
        self.status.set(training_completion_message(snapshot['report']) if snapshot['report'] else snapshot['message'])
        self.storage.set('临时磁盘：' + snapshot['temporary_path'])
        self.start_button.configure(state='normal' if snapshot['state'] == 'idle' and not self.busy else 'disabled')
        self.choose_button.configure(state='normal' if snapshot['state'] == 'idle' and not self.busy else 'disabled')
        self.clear_button.configure(state='normal' if snapshot['state'] != 'idle' and not self.busy else 'disabled')
        self.rollback_button.configure(state='normal' if self.rollback_available and snapshot['state'] != 'running' and not self.busy else 'disabled')
        report = snapshot['report']
        if report != self._last_report:
            self.report.configure(state='normal')
            self.report.delete('1.0', 'end')
            if report:
                state = training_completion_message(report)
                self.refresh_rollback()
                text = (f"ZIP：新增 {report['new_packages']} 份，跳过重复 {report['duplicate_packages']} 份\n"
                        f"新 ZIP 来源：{report['source_count']} 集／文件，{report['reference_sets']} 组参考\n"
                        f"参与训练 {report['training_clips']} 段，留出验证 {report['held_out_clips']} 段\n"
                        f"状态：{state}\n现用模型：{report['model_path']}\n")
                for name, metrics in report['metrics'].items():
                    text += f"{'训练数据' if name == 'training' else '独立来源留出'}：{metrics['sources']} 个来源，{metrics['pairs']} 对参考与片段\n"
                updates = report.get('channel_updates', {})
                if updates:
                    labels = {'primary': 'ERes2NetV2', 'secondary': 'CAM++', 'wavlm': 'WavLM',
                              'anime_char': '动漫角色声纹', 'anime_va': '动漫声优声纹',
                              'local_classifier': '局部分类器'}
                    states = {'weights_updated': '权重已更新', 'memory_updated': '错人记忆已更新',
                              'memory_preserved': '保留已有记忆', 'memory_unavailable': '暂无记忆',
                              'recalibrated': '已重新校准', 'unchanged': '未改动', 'failed': '更新失败'}
                    text += '六通道状态：\n'
                    for channel, label in labels.items():
                        state = updates.get(channel, 'unchanged')
                        text += f"  {label}：{states.get(state, state)}\n"
                self.report.insert('end', text)
            self.report.configure(state='disabled')
            self._last_report = report
        self.root.after(400, self._poll)
