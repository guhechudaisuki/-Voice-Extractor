"""Desktop controls for cached rescoring and multi-batch corpus export."""
import json
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class DesktopCorpusTools:
    def __init__(self, review, parent):
        self.review, self.events, self.busy = review, queue.Queue(), False
        row = ttk.Frame(parent); row.pack(fill='x', pady=5)
        self.rescore = ttk.Button(row, text='用最新强化模型重评本批次', command=self.start_rescore)
        self.rescore.pack(side='left')
        self.merge = ttk.Button(row, text='合并多个已复核批次', command=self.choose_batches)
        self.merge.pack(side='left', padx=6)
        review.root.after(250, self.poll)

    def work(self, operation, callback):
        if self.busy:
            return
        self.busy = True
        self.rescore.configure(state='disabled'); self.merge.configure(state='disabled')
        def run():
            try:
                self.events.put((callback, operation(), None))
            except Exception as exc:
                self.events.put((callback, None, str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def poll(self):
        while not self.events.empty():
            callback, result, error = self.events.get_nowait()
            self.busy = False
            self.rescore.configure(state='normal'); self.merge.configure(state='normal')
            if error:
                messagebox.showerror('批次操作失败', error, parent=self.review.root)
            else:
                callback(result)
        self.review.root.after(250, self.poll)

    def start_rescore(self):
        batch = self.review.current_batch_path
        if not batch or not self.review.save_review(show_prompt=False):
            return
        from launcher.role_rescore import rescore_batch
        self.review.current_status_var.set('重评中；首次缺失项从已有干声建立特征缓存，不重跑分离或转写')
        def completed(result):
            if self.review.current_batch_path == batch:
                self.review._on_batch_selected()
            self.review.current_status_var.set(f"重评完成：晋升 {result['promoted']} 段，降级 {result['demoted']} 段，用时 {result['seconds']} 秒")
        self.work(lambda: rescore_batch(batch), completed)

    def choose_batches(self):
        if self.busy or (self.review.current_batch_path and not self.review.save_review(show_prompt=False)):
            return
        from launcher.review_store import existing_review_path, read_review
        from launcher.role_rescore import manifest_path
        from launcher.corpus_worker import execute, invoke
        dialog = tk.Toplevel(self.review.root); dialog.title('多集数据集总装'); dialog.transient(self.review.root)
        ttk.Label(dialog, text='Ctrl / Shift 多选，仅导出已标注的干净片段').pack(padx=12, pady=8)
        choices = [p for p in self.review.batches if existing_review_path(p).is_file()]
        listing = tk.Listbox(dialog, selectmode='extended', width=65, height=14, exportselection=False)
        listing.pack(padx=12, fill='both', expand=True)
        for index, path in enumerate(choices):
            listing.insert('end', path.name)
            if path == self.review.current_batch_path:
                listing.selection_set(index)
        row = ttk.Frame(dialog); row.pack(pady=10)
        lufs, peak = tk.StringVar(value='-23'), tk.StringVar(value='-1')
        for title, value in (('目标 LUFS', lufs), ('真峰值 dBTP', peak)):
            ttk.Label(row, text=title).pack(side='left')
            ttk.Entry(row, textvariable=value, width=8).pack(side='left', padx=5)
        def export():
            selected = [choices[i] for i in listing.curselection()]
            if not selected:
                return
            try:
                target, ceiling = float(lufs.get()), float(peak.get())
            except ValueError:
                messagebox.showerror('参数错误', '响度和峰值需为数字', parent=dialog); return
            destination = filedialog.asksaveasfilename(parent=dialog, defaultextension='.zip', initialfile='reviewed_corpus.zip')
            if not destination:
                return
            try:
                for path in selected:
                    json.loads(manifest_path(path).read_text(encoding='utf-8'))
                    read_review(path)
            except (ValueError, OSError) as exc:
                messagebox.showerror('读取失败', str(exc), parent=dialog); return
            speaker, language = self.review.export_speaker.get(), self.review.export_language.get()
            quality = {'Gold': 'Gold', 'Silver': 'Silver'}.get(self.review.quality_var.get())
            dialog.destroy(); self.review.current_status_var.set('正在合并、去重并统一响度…')
            payload = {'batches': [str(path.resolve()) for path in selected], 'destination': destination,
                       'options': dict(speaker=speaker, language=language, target_lufs=target, true_peak=ceiling,
                                       min_quality_grade=quality)}
            import sys
            operation = invoke if getattr(sys, 'frozen', False) else execute
            self.work(lambda: operation('merge', payload),
                lambda data: messagebox.showinfo('总装完成', f"导出 {data['valid_count']} 段，去重 {data['duplicate_count']} 段\n{destination}", parent=self.review.root))
        ttk.Button(dialog, text='去重、统一响度并导出', command=export).pack(pady=10)
