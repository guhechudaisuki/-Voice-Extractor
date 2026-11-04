from __future__ import annotations

import json
import os
import copy
import queue
import threading
import time
import wave
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from launcher.training_material import (reviewable, validated_rows, clean_reviews,
    review_rank, information_rank, high_confidence_borderline, discover_batch_speakers, export_batch_seeds)
from launcher.review_export import export_review_zip
from launcher.review_store import read_review, write_review, review_path
from launcher.desktop_widgets import ScrollableForm

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

class DesktopReviewTab:
    """批次音频试听、人工标注与训练素材导出。"""

    def __init__(self, parent_notebook: ttk.Notebook, root: tk.Tk, base_output_dir: Path, set_references=None) -> None:
        self.root = root
        self.notebook = parent_notebook
        self.base_output_dir = base_output_dir
        self.set_references = set_references
        self.playing_until = 0.0
        self.speaker_events = queue.Queue()
        self.speaker_busy = False

        self.tab_frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.tab_frame, text=" 🎧 试听标注与复核 ")

        self.batches: list[Path] = []
        self.current_batch_path: Path | None = None
        self.segments: list[dict] = []
        self.current_index: int = -1
        self.reviews: dict[str, dict] = {}
        self.filter_var = tk.StringVar(value="全部可审片段")
        self.gpt_list_var = tk.BooleanVar(value=True)
        self.quality_var = tk.StringVar(value='全部质量')
        self.blind_mode_var = tk.BooleanVar(value=True)
        self.auto_next_var = tk.BooleanVar(value=True)
        self.current_status_var = tk.StringVar(value="请选择一个输出批次开始试听")
        self.cluster_summary_var = tk.StringVar(value="簇级判决：未载入")
        self.current_audio_file: Path | None = None
        self.manifest = {}
        self.include_material_var = tk.BooleanVar(value=False)
        self.material_state = tk.StringVar(value='已有标注即可导出训练原料')
        self.export_busy = False
        self.export_events = queue.Queue()

        self._build_ui()
        self._bind_review_keys(self.tab_frame)
        self.root.bind('<KeyPress>', self._on_key, add='+')
        self.notebook.bind('<<NotebookTabChanged>>',
            lambda _event: self.stop_playback() if self.notebook.select() != str(self.tab_frame) else None, add='+')
        self.refresh_batches()
        self.root.after(250, self._poll_export)

    def _build_ui(self) -> None:
        self.tab_frame.columnconfigure(1, weight=1)
        self.tab_frame.rowconfigure(0, weight=1)

        left_card = ttk.Frame(self.tab_frame, style="Card.TFrame", padding=10)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_card.columnconfigure(0, weight=1)
        left_card.rowconfigure(4, weight=1)

        batch_header = ttk.Frame(left_card, style="Card.TFrame")
        batch_header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(batch_header, text="选择输出批次:", font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Button(batch_header, text="刷新", width=5, command=self.refresh_batches).pack(side="right")

        self.batch_combo = ttk.Combobox(left_card, state="readonly", width=32)
        self.batch_combo.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.batch_combo.bind("<<ComboboxSelected>>", self._on_batch_selected)

        filter_row = ttk.Frame(left_card, style="Card.TFrame")
        filter_row.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self.filter_combo = ttk.Combobox(
            filter_row, state="readonly", textvariable=self.filter_var, width=32,
            values=("全部可审片段", "高信息量"))
        self.filter_combo.pack(side="left", fill="x", expand=True)
        self.filter_combo.bind("<<ComboboxSelected>>", lambda _e: self._render_tree())
        ttk.Button(filter_row, text="一键信任主输出", width=14,
                   command=self.trust_main_output).pack(side="right")
        self.borderline_button = ttk.Button(filter_row, text="采纳高分候选",
                                            command=self.trust_borderline)

        progress_row = ttk.Frame(left_card, style="Card.TFrame")
        progress_row.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        self.progress_label = ttk.Label(progress_row, text="进度: 0 / 0 已审", style="CardText.TLabel")
        self.progress_label.pack(side="left")

        columns = ("ordinal", "status", "duration", "text")
        self.tree = ttk.Treeview(left_card, columns=columns, show="headings", height=20)
        self.tree.heading("ordinal", text="#")
        self.tree.heading("status", text="状态")
        self.tree.heading("duration", text="时长")
        self.tree.heading("text", text="STT 识别文本")

        self.tree.column("ordinal", width=36, anchor="center")
        self.tree.column("status", width=70, anchor="center")
        self.tree.column("duration", width=50, anchor="center")
        self.tree.column("text", width=180, anchor="w")

        tree_scroll = ttk.Scrollbar(left_card, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.grid(row=4, column=0, sticky="nsew")
        tree_scroll.grid(row=4, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        self.review_form = ScrollableForm(self.tab_frame)
        self.review_form.grid(row=0, column=1, sticky="nsew")
        right_card = self.review_form.body
        right_card.columnconfigure(0, weight=1)

        top_meta = ttk.Frame(right_card, style="Card.TFrame")
        top_meta.pack(fill="x", pady=(0, 10))
        from launcher.desktop_corpus import DesktopCorpusTools
        self.corpus_tools = DesktopCorpusTools(self, top_meta)
        ttk.Label(top_meta, textvariable=self.cluster_summary_var, wraplength=680).pack(anchor="w")
        ttk.Button(top_meta, text="🔍 角色簇探索 / 导出角色参考种子", command=self.open_speakers).pack(anchor="w", pady=6)
        self.current_title_label = ttk.Label(top_meta, text="未选择片段", font=("Segoe UI", 12, "bold"))
        self.current_title_label.pack(anchor='w', pady=(4, 8))

        blind_cb = ttk.Checkbutton(
            top_meta,
            text="盲听模式 (隐藏分数/层级防偏见)",
            variable=self.blind_mode_var,
            command=self._on_blind_toggle,
        )
        blind_cb.pack(anchor='w')

        auto_next_cb = ttk.Checkbutton(
            top_meta,
            text="点击标注后自动下一句",
            variable=self.auto_next_var,
        )
        auto_next_cb.pack(anchor='w')

        self.diag_frame = ttk.Frame(right_card, style="Card.TFrame")
        self.diag_frame.pack(fill="x", pady=(0, 8))
        self.diag_label = ttk.Label(self.diag_frame, text="", foreground="#b9a0ff")
        self.diag_label.pack(anchor="w")

        stt_box = ttk.LabelFrame(right_card, text=" STT 字幕识别 ", padding=10)
        stt_box.pack(fill="x", pady=(0, 12))
        self.stt_var = tk.StringVar(value="暂无文本")
        self.stt_display = ttk.Entry(stt_box, textvariable=self.stt_var, font=("Segoe UI", 11), state='readonly')
        self.stt_display.pack(fill='x')
        self.stt_display.bind('<KeyRelease>', self._revise_transcript)
        ttk.Label(stt_box, text='标为目标匹配后可直接修改台词，自动保存').pack(anchor='w')

        play_card = ttk.LabelFrame(right_card, text=" 音频试听控制 ", padding=12)
        play_card.pack(fill="x", pady=(0, 12))
        btn_bar = ttk.Frame(play_card)
        btn_bar.pack(anchor="center")

        ttk.Button(btn_bar, text="◀ 上一句 [↑/W]", width=10, command=self.prev_segment).pack(side="left", padx=6)
        ttk.Button(btn_bar, text="▶ 播放/停止 [Space]", width=14, style="Accent.TButton", command=self.toggle_play).pack(side="left", padx=8)
        ttk.Button(btn_bar, text="↺ 重播 [R]", width=12, command=self.replay_current).pack(side="left", padx=6)
        ttk.Button(btn_bar, text="下一句 [↓/S] ▶", width=10, command=self.next_segment).pack(side="left", padx=6)
        self.keep_play_button = ttk.Button(btn_bar, text="▶ 试听保留段", width=12,
                                           command=self.play_keep_segment, state="disabled")
        self.keep_play_button.pack(side="left", padx=6)

        decision_box = ttk.LabelFrame(right_card, text=" 人工试听裁决 (点击即标注并记录) ", padding=12)
        decision_box.pack(fill="x", pady=(0, 12))
        decision_box.columnconfigure((0, 1, 2), weight=1)

        self.btn_correct = tk.Button(
            decision_box,
            text="✔ 目标匹配 [1]\n完全纯净，直接入训练集",
            font=("Segoe UI", 9, "bold"),
            bg="#ecfdf5",
            fg="#047857",
            activebackground="#a7f3d0",
            relief="groove",
            height=2,
            command=lambda: self.apply_decision("correct"),
        )
        self.btn_correct.grid(row=0, column=0, sticky="ew", padx=4, pady=4)

        self.btn_wrong = tk.Button(
            decision_box,
            text="✖ 整句非目标 [2]\n错人/完全非目标角色",
            font=("Segoe UI", 9, "bold"),
            bg="#fff1f2",
            fg="#be123c",
            activebackground="#fecdd3",
            relief="groove",
            height=2,
            command=lambda: self.apply_decision("wrong"),
        )
        self.btn_wrong.grid(row=0, column=1, sticky="ew", padx=4, pady=4)

        self.btn_partial = tk.Button(
            decision_box,
            text="✂ 部分非目标 [3]\n前后夹带他人声音",
            font=("Segoe UI", 9, "bold"),
            bg="#fffbeb",
            fg="#b45309",
            activebackground="#fde68a",
            relief="groove",
            height=2,
            command=lambda: self.apply_decision("partial"),
        )
        self.btn_partial.grid(row=0, column=2, sticky="ew", padx=4, pady=4)

        self.btn_singing = tk.Button(
            decision_box,
            text="♫ 歌声混入 [4]\n伴奏人声/唱歌段落",
            font=("Segoe UI", 9),
            bg="#f5f3ff",
            fg="#6d28d9",
            relief="groove",
            height=2,
            command=lambda: self.apply_decision("singing"),
        )
        self.btn_singing.grid(row=1, column=0, sticky="ew", padx=4, pady=4)

        self.btn_overlap = tk.Button(
            decision_box,
            text="👥 重叠杂音 [5]\n多人同时发音无法纯净切分",
            font=("Segoe UI", 9),
            bg="#eff6ff",
            fg="#1d4ed8",
            relief="groove",
            height=2,
            command=lambda: self.apply_decision("overlap"),
        )
        self.btn_overlap.grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        self.btn_skip = tk.Button(
            decision_box,
            text="⊘ 存疑 / 跳过 [6]\n暂不确定，保留待复查",
            font=("Segoe UI", 9),
            bg="#f3f4f6",
            fg="#4b5563",
            relief="groove",
            height=2,
            command=lambda: self.apply_decision("skip"),
        )
        self.btn_skip.grid(row=1, column=2, sticky="ew", padx=4, pady=4)

        bottom_bar = ttk.Frame(right_card, style="Card.TFrame")
        bottom_bar.pack(fill="x", side="bottom", pady=(10, 0))

        ttk.Label(bottom_bar, textvariable=self.current_status_var, style="CardText.TLabel", wraplength=620).pack(anchor='w')
        ttk.Label(bottom_bar, textvariable=self.material_state, wraplength=620).pack(anchor='w', pady=5)
        raw_row = ttk.Frame(bottom_bar)
        raw_row.pack(fill='x', pady=5)
        self.material_check = ttk.Checkbutton(raw_row, text='附带训练原料（已标注子集和音频，ZIP 可直接强化）', variable=self.include_material_var, state='disabled')
        self.material_check.pack(side='left')
        speaker_row = ttk.Frame(bottom_bar)
        speaker_row.pack(fill='x', pady=4)
        self.export_speaker = tk.StringVar(value='target')
        self.export_language = tk.StringVar(value='JA')
        ttk.Label(speaker_row, text='说话人').pack(side='left')
        ttk.Entry(speaker_row, textvariable=self.export_speaker, width=20).pack(side='left', padx=6)
        ttk.Label(speaker_row, text='未知语言回退值').pack(side='left')
        ttk.Entry(speaker_row, textvariable=self.export_language, width=6).pack(side='left', padx=6)
        ttk.Checkbutton(raw_row, text='附带 GPT-SoVITS train.list', variable=self.gpt_list_var).pack(side='left', padx=12)
        ttk.Combobox(bottom_bar, textvariable=self.quality_var, state="readonly",
                     values=("全部质量", "Silver", "Gold"), width=16).pack(anchor="w", pady=4)
        self.export_button = ttk.Button(bottom_bar, text="导出 ZIP", command=self.export_verified_zip)
        self.export_button.pack(side="right", padx=6, pady=6)
        ttk.Button(bottom_bar, text="保存标注进度", command=self.save_review).pack(side="right", padx=6, pady=6)
        ttk.Button(bottom_bar, text="打开标注目录", command=self.open_annotations).pack(side="right", padx=6, pady=6)

    def refresh_batches(self) -> None:
        if not self.base_output_dir.exists():
            return
        self.batches = [
            p for p in sorted(self.base_output_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            if p.is_dir() and ((p / "batch_manifest.json").exists() or (p / "manifest.json").exists() or list(p.glob("*.wav")))
        ]
        items = [b.name for b in self.batches]
        self.batch_combo["values"] = items
        if items and not self.batch_combo.get():
            self.batch_combo.current(0)
            self._on_batch_selected()

    def _on_batch_selected(self, _event=None) -> None:
        selected_name = self.batch_combo.get()
        if not selected_name:
            return
        for b in self.batches:
            if b.name == selected_name:
                self.current_batch_path = b
                break
        self.load_current_batch()

    def load_current_batch(self) -> None:
        if not self.current_batch_path:
            return
        manifest_file = self.current_batch_path / "batch_manifest.json"
        if not manifest_file.is_file():
            manifest_file = self.current_batch_path / "manifest.json"

        manifest = {}
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                manifest = {}

        self.reviews = {}
        from launcher.role_rescore import apply_rescore
        manifest = apply_rescore(self.current_batch_path, manifest)
        summaries = []
        for report_file in sorted(self.current_batch_path.rglob("cluster_report.json")):
            try:
                report = json.loads(report_file.read_text(encoding="utf-8"))
                target = next((c for c in report.get("clusters", []) if c.get("is_target")), None)
                summaries.append(f"{report.get('cluster_count', 0)} 个簇 · " +
                                 (f"目标 #{target['cluster_id']}（{target['member_count']} 轮）" if target else "未找到目标簇"))
            except (OSError, ValueError, KeyError):
                summaries.append("簇报告无法读取")
        self.cluster_summary_var.set("；".join(summaries) or "旧批次：没有簇报告")
        try:
            self.reviews = read_review(self.current_batch_path).get('reviews', {})
        except (OSError, ValueError) as exc:
            self.current_status_var.set(f'标注读取失败：{exc}')

        wavs = sorted(list(self.current_batch_path.glob("*.wav")), key=lambda p: p.name)
        self.segments = []

        manifest_items = {}
        if "items" in manifest and isinstance(manifest["items"], list):
            for it in manifest["items"]:
                fname = it.get("filename") or it.get("audio_file") or (Path(it.get("path", "")).name if "path" in it else None)
                if fname:
                    manifest_items[fname] = it

        for idx, wav in enumerate(wavs, start=1):
            txt_path = wav.with_suffix(".txt")
            stt = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
            m_info = manifest_items.get(wav.name, {})
            duration = m_info.get('duration') or m_info.get('audio_duration') or 0.0
            if not duration:
                try:
                    with wave.open(str(wav), 'rb') as sound:
                        duration = sound.getnframes() / sound.getframerate()
                except (OSError, EOFError, wave.Error):
                    duration = 0.0

            self.segments.append({
                "ordinal": idx,
                "file": wav,
                "filename": wav.name,
                "duration": duration,
                "stt_text": stt,
                "tier": m_info.get("tier", "normal"),
                "pathway": m_info.get("pathway") or m_info.get("route") or "standard",
                "score": m_info.get("score") or m_info.get("target_similarity"),
            })

        if isinstance(manifest.get("sentences"), list):
            self.segments = []
            for ordinal, row, path, _relative in reviewable(self.current_batch_path, manifest):
                diagnostics = row.get("diagnostics") or {}
                cluster_tier = diagnostics.get("cluster_tier")
                if row.get("accepted"):
                    view = "A"
                elif cluster_tier == "B":
                    view = "B"
                else:
                    view = "removed"
                prefix = {"A": "", "B": "[⚡边缘] ", "removed": "[已撤下] "}[view]
                self.segments.append({"ordinal": ordinal, "file": path, "filename": path.name,
                    "duration": max(0, float(row.get("end") or 0)-float(row.get("start") or 0)),
                    "stt_text": prefix + (row.get("text") or row.get("whisper_text", "")),
                    "row": row,
                    "view": view,
                    "tier": diagnostics.get("speaker_tier", ""),
                    "pathway": "簇级判决" if diagnostics.get("cluster_identity") else "原判决",
                    "score": row.get("speaker_score")})
        else:
            manifest = {'sentences': [{'accepted': True, 'audio_file': seg['filename'], 'start': 0,
                                      'end': seg['duration'], 'text': seg['stt_text']} for seg in self.segments]}
        self.manifest = manifest
        views = {self._segment_view(seg) for seg in self.segments}
        filters = ["全部可审片段", "高信息量"]
        if len(views) > 1 and "A" in views:
            filters.append("主输出")
        if "B" in views:
            filters.append("已保存候选")
            self.borderline_button.pack(side="right")
        else:
            self.borderline_button.pack_forget()
        if "removed" in views:
            filters.append("已撤下片段")
        self.filter_combo.configure(values=filters)
        if self.filter_var.get() not in filters:
            self.filter_var.set("全部可审片段")
        self.current_index = -1
        self.include_material_var.set(False)
        self._render_tree()
        if self.tree.get_children():
            self.select_segment(int(self.tree.get_children()[0]))

    def _segment_view(self, seg: dict) -> str:
        # 旧批次（无 manifest 清单）只含已接受片段，全部视作 Tier A。
        return seg.get("view") or "A"

    def _render_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        label_map = {
            "correct": "✔ 目标",
            "wrong": "✖ 非目标",
            "partial": "✂ 部分",
            "singing": "♫ 歌声",
            "overlap": "👥 重叠",
            "skip": "⊘ 跳过",
        }
        view = self.filter_var.get()
        view_filter = {"主输出": "A", "已保存候选": "B", "已撤下片段": "removed"}.get(view)
        indexed = list(enumerate(self.segments))
        if view == '高信息量':
            indexed.sort(key=lambda item: information_rank(item[1].get('row',{})),reverse=True)
        elif view_filter == "B":
            indexed.sort(key=lambda item: review_rank(item[1].get("row", {})), reverse=True)
        for idx, seg in indexed:
            if view_filter and self._segment_view(seg) != view_filter:
                continue
            rev = self.reviews.get(str(seg["ordinal"]), {})
            if view == "高信息量" and rev.get('label'):
                continue
            status_text = label_map.get(rev.get("label", ""), "待审")
            if seg.get('row', {}).get('diagnostics', {}).get('role_rescore', {}).get('promoted'):
                status_text = '[重评救回] ' + status_text
            dur = f"{seg['duration']:.2f}s" if seg["duration"] else ""
            self.tree.insert("", "end", iid=str(idx), values=(seg["ordinal"], status_text, dur, seg["stt_text"]))

        visible = self.tree.get_children()
        if self.current_index >= 0 and visible and str(self.current_index) not in visible:
            self.select_segment(int(visible[0]))
        elif str(self.current_index) in visible and self.tree.selection() != (str(self.current_index),):
            self.tree.selection_set(str(self.current_index))
        elif not visible:
            self.current_index = -1
            self.current_audio_file = None
            self.stop_playback()
        reviewed_cnt = len(self.reviews)
        self.progress_label.config(text=f"进度: {reviewed_cnt} / {len(self.segments)} 已审")
        try:
            if not self.current_batch_path:
                raise ValueError('请选择输出批次')
            validated_rows(self.current_batch_path, self.manifest, self.reviews)
        except ValueError as exc:
            self.material_check.configure(state='disabled')
            self.include_material_var.set(False)
            self.material_state.set(str(exc))
        else:
            self.material_check.configure(state='normal')
            self.material_state.set('可导出当前已标注子集，未标注片段自动跳过')

    def trust_main_output(self) -> None:
        """将未标注的 Tier A 片段批量标为目标匹配。"""
        if not self.segments:
            return
        pending = [seg for seg in self.segments
                   if self._segment_view(seg) == "A"
                   and not self.reviews.get(str(seg["ordinal"]), {}).get("label")]
        if not pending:
            messagebox.showinfo('一键信任主输出', '没有待标注的 Tier A 片段。', parent=self.root)
            return
        if not messagebox.askyesno('一键信任主输出',
                                   f'将把 {len(pending)} 段未标注的 Tier A 主输出全部标为“目标匹配”，继续？',
                                   parent=self.root):
            return
        for seg in pending:
            self.reviews[str(seg["ordinal"])] = {
                "ordinal": seg["ordinal"],
                "filename": seg["filename"],
                "label": "correct",
            }
        self._render_tree()
        self.save_review(show_prompt=False)
        self.current_status_var.set(f'已将 {len(pending)} 段输出标为目标匹配')

    def trust_borderline(self):
        pending = [seg for seg in self.segments
                   if high_confidence_borderline(seg.get('row') or {})
                   and not self.reviews.get(str(seg['ordinal']), {}).get('label')]
        for seg in pending:
            self.reviews[str(seg['ordinal'])] = dict(ordinal=seg['ordinal'],
                filename=seg['filename'], label='correct')
        self._render_tree()
        if self.save_review(show_prompt=False):
            seconds = sum(seg['duration'] for seg in pending)
            messagebox.showinfo('高分边缘句', f'已救回 {len(pending)} 句，共 {seconds:.2f} 秒', parent=self.root)

    def open_speakers(self):
        if not self.current_batch_path or self.speaker_busy:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title('角色簇探索 / 导出角色参考种子')
        dialog.geometry('780x520')
        status = tk.StringVar(value='正在扫描角色簇…')
        ttk.Label(dialog, textvariable=status, wraplength=750).pack(fill='x', padx=12, pady=8)
        form = ScrollableForm(dialog)
        form.pack(fill='both', expand=True)
        batch = self.current_batch_path

        def render(speakers):
            status.set(f'发现 {len(speakers)} 个角色簇；每簇导出最多 4 段参考 WAV')
            for speaker in speakers:
                card = ttk.LabelFrame(form.body, text=f"{speaker['episode']} · 簇 #{speaker['cluster_id']}", padding=10)
                card.pack(fill='x', padx=12, pady=5)
                summary = (f"{speaker['total_speech_seconds']:.1f}s · {speaker['member_count']} 段 · "
                           f"纯度 {speaker['purity_score']:.3f}\n" + ' / '.join(speaker.get('sample_texts', ())))
                ttk.Label(card, text=summary, wraplength=700).pack(anchor='w')
                ttk.Button(card, text='导出 Top-4 并设为目标参考音频',
                           command=lambda key=speaker['key']: export(key)).pack(anchor='w', pady=5)

        def exported(result):
            paths = result['paths']
            status.set('\n'.join(paths) if paths else '没有可导出的纯净种子；原轨及对应切片可能已删除')
            if paths and self.set_references:
                self.set_references(paths)

        def work(fn, callback):
            if self.speaker_busy:
                return
            self.speaker_busy = True
            status.set('正在处理…')
            def run():
                try:
                    self.speaker_events.put((dialog, status, callback, fn(), None))
                except Exception as exc:
                    self.speaker_events.put((dialog, status, callback, None, str(exc)))
            threading.Thread(target=run, daemon=True).start()

        def export(key):
            work(lambda: export_batch_seeds(batch, [key]), exported)

        work(lambda: discover_batch_speakers(batch), render)

    def play_keep_segment(self) -> None:
        """试听 partial 标注裁掉错人区间后的干净保留段。"""
        if self.current_index < 0:
            return
        seg = self.segments[self.current_index]
        review = self.reviews.get(str(seg["ordinal"]), {})
        if review.get("label") != "partial":
            return
        from launcher.training_material import partial_keep_segments
        keep = [piece for piece in partial_keep_segments(
            {"start": 0, "end": seg["duration"]}, review) if piece[1] - piece[0] >= .2]
        if not keep:
            messagebox.showinfo('试听保留段', '错人区间覆盖了整句，没有可保留的干净段。', parent=self.root)
            return
        import tempfile
        from launcher.audio_export import write_sequence
        handle, name = tempfile.mkstemp(prefix='voice_keep_', suffix='.wav')
        os.close(handle)
        try:
            duration = write_sequence(seg['file'],name,keep)
            if HAS_WINSOUND:
                self.stop_playback()
                winsound.PlaySound(name, winsound.SND_FILENAME | winsound.SND_ASYNC)
                self.playing_until = time.monotonic() + duration
                self.current_status_var.set(
                    f"正在依次播放 {len(keep)} 个保留段（共 {sum(b-a for a,b in keep):.2f}s）")
        except (ValueError, OSError, RuntimeError) as exc:
            self.current_status_var.set(f'保留段播放失败: {exc}')
        finally:
            # 播放是异步的，立即删除会被 winsound 拒绝；延迟清理。
            def _cleanup(path=name):
                try:
                    os.unlink(path)
                except OSError:
                    pass
            self.root.after(max(30000, round((seg['duration'] + len(keep)) * 1000)), _cleanup)

    def _on_tree_select(self, _event=None) -> None:
        sel = self.tree.selection()
        if sel:
            idx = int(sel[0])
            if idx != self.current_index:
                self.select_segment(idx)

    def select_segment(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.segments):
            return
        self.stop_playback()
        self.current_index = idx
        seg = self.segments[idx]
        self.current_audio_file = seg["file"]

        if self.tree.exists(str(idx)):
            # 重复设置同一选中项也会排入 TreeviewSelect，需避免事件循环。
            if self.tree.selection() != (str(idx),):
                self.tree.selection_set(str(idx))
            self.tree.see(str(idx))

        dur_text = f"{seg['duration']:.2f}s" if seg["duration"] else ""
        self.current_title_label.config(text=f"片段 #{seg['ordinal']} - {seg['filename']} ({dur_text})")
        review = self.reviews.get(str(seg['ordinal']), {})
        row = seg.get('row', {})
        self.stt_var.set(review.get('revised_text') if review.get('revised_text') is not None
                         else row.get('text') or row.get('whisper_text') or seg.get('stt_text', ''))
        self.stt_display.configure(state='normal' if review.get('label') == 'correct' else 'readonly')

        if not self.blind_mode_var.get():
            score_s = f"{seg['score']:.3f}" if seg.get("score") is not None else "无"
            diagnostics = (seg.get('row') or {}).get('diagnostics') or {}
            qc = diagnostics.get('acoustic_qc') or {}
            grade = qc.get('quality_grade', '—')
            badge = {'Gold': '🏅 Gold', 'Silver': '🥈 Silver', 'Review': '⚠️ Review'}.get(grade, '—')
            self.diag_label.config(text=f"{badge} | SNR: {qc.get('snr_db', '—')} dB | 融合分: {diagnostics.get('calibrated_fusion_score', '—')}\n层级: {seg['tier']} | 相似度: {score_s}")
            self.diag_frame.pack(fill="x", pady=(0, 8))
        else:
            self.diag_frame.pack_forget()

        review = self.reviews.get(str(seg["ordinal"]), {})
        if review.get("label") == "partial" and (review.get('partial_intervals') or
                review.get("partial_end", 0) > (review.get("partial_start") or 0)):
            self.keep_play_button.configure(state="normal")
        else:
            self.keep_play_button.configure(state="disabled")

        self.play_current()

    def play_current(self) -> None:
        if not self.current_audio_file or not self.current_audio_file.exists():
            return
        if HAS_WINSOUND:
            try:
                winsound.PlaySound(str(self.current_audio_file), winsound.SND_FILENAME | winsound.SND_ASYNC)
                duration = self.segments[self.current_index]['duration']
                self.playing_until = time.monotonic() + duration if duration > 0 else float('inf')
                self.current_status_var.set(f"正在播放 #{self.segments[self.current_index]['ordinal']}")
            except Exception as e:
                self.current_status_var.set(f"播放失败: {e}")

    def stop_playback(self):
        if HAS_WINSOUND:
            winsound.PlaySound(None, 0)
        self.playing_until = 0.0

    def toggle_play(self):
        if time.monotonic() < self.playing_until:
            self.stop_playback()
        else:
            self.play_current()

    def replay_current(self) -> None:
        if self.current_index >= 0 and self.reviews.get(str(self.segments[self.current_index]['ordinal']), {}).get('label') == 'partial':
            self.play_keep_segment()
        else:
            self.play_current()

    def _step(self, delta):
        visible = list(self.tree.get_children())
        current = str(self.current_index)
        if current in visible:
            position = visible.index(current) + delta
            if 0 <= position < len(visible):
                self.select_segment(int(visible[position]))

    def prev_segment(self) -> None:
        self._step(-1)

    def next_segment(self) -> None:
        self._step(1)

    def _on_key(self, event):
        if self.notebook.select() != str(self.tab_frame) or event.widget.winfo_toplevel() != self.root:
            return
        if event.widget.winfo_class() in {'Entry', 'TEntry', 'Text', 'TCombobox', 'Spinbox', 'TSpinbox'} or event.state & 0x2000C:
            return
        key = event.keysym.lower()
        labels = dict(zip('123456', ('correct', 'wrong', 'partial', 'singing', 'overlap', 'skip')))
        actions = {'space': self.toggle_play, 'r': self.replay_current,
                   'up': self.prev_segment, 'w': self.prev_segment,
                   'down': self.next_segment, 's': self.next_segment}
        if key in labels:
            self.apply_decision(labels[key])
        elif key in actions:
            actions[key]()
        else:
            return
        return 'break'

    def _bind_review_keys(self, widget):
        # 先于控件类处理快捷键，避免空格和方向键再次触发默认操作。
        widget.bind('<KeyPress>', self._on_key, add='+')
        for child in widget.winfo_children():
            self._bind_review_keys(child)

    def apply_decision(self, label: str) -> None:
        if self.current_index < 0 or self.current_index >= len(self.segments):
            return
        seg = self.segments[self.current_index]
        ordinal_str = str(seg["ordinal"])
        partial = {}
        if label == 'partial':
            previous = self.reviews.get(ordinal_str, {})
            ranges = previous.get('partial_intervals') or (
                [[previous['partial_start'], previous['partial_end']]]
                if previous.get('partial_end') is not None else [])
            value = simpledialog.askstring('污染区间',
                '输入一个或多个区间，单位秒，例如 0.5-1.2; 2.8-3.4', parent=self.root,
                initialvalue='; '.join(f'{a:.2f}-{b:.2f}' for a,b in ranges))
            if value is None:
                return
            try:
                import re
                intervals = []
                for part in re.split('[;；]', value):
                    match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*', part)
                    if not match:
                        raise ValueError('区间格式无效')
                    intervals.append([float(match[1]), float(match[2])])
                from launcher.training_material import partial_intervals
                partial = {'partial_intervals': intervals,
                           'partial_start': intervals[0][0], 'partial_end': intervals[0][1]}
                partial_intervals({'start': 0., 'end': seg['duration']}, partial)
                text = simpledialog.askstring('保留段文本',
                    '填写保留音频对应文本；多个保留段请按顺序用 ;; 分隔。', parent=self.root,
                    initialvalue=previous.get('partial_text', ''))
                if text is None:
                    return
                partial['partial_text'] = text
            except ValueError as exc:
                messagebox.showerror('污染区间无效', str(exc), parent=self.root)
                return
        self.reviews[ordinal_str] = {
            'revised_text': self.reviews.get(ordinal_str, {}).get('revised_text'),
            "ordinal": seg["ordinal"],
            "filename": seg["filename"],
            "label": label,
            **partial,
        }

        self._render_tree()
        current_review = (self.reviews.get(str(self.segments[self.current_index]['ordinal']), {})
                          if self.current_index >= 0 else {})
        self.stt_display.configure(state='normal' if current_review.get('label') == 'correct' else 'readonly')
        if not self.save_review(show_prompt=False):
            return

        if self.auto_next_var.get() and label != "partial":
            self.next_segment()
        elif label == "partial":
            self.keep_play_button.configure(state="normal")

    def _revise_transcript(self, _event=None):
        if not 0 <= self.current_index < len(self.segments):
            return
        review = self.reviews.get(str(self.segments[self.current_index]['ordinal']))
        if review and review.get('label') == 'correct':
            review['revised_text'] = self.stt_var.get()
            self.save_review(show_prompt=False)

    def _on_blind_toggle(self) -> None:
        if self.current_index >= 0:
            self.select_segment(self.current_index)

    def save_review(self, show_prompt: bool = True) -> bool:
        if not self.current_batch_path:
            return False
        try:
            existing = read_review(self.current_batch_path)
            save_data = {
                **existing,
                "version": "2.0",
                "batch_id": self.current_batch_path.name,
                "reviewer": "desktop_user",
                "updated_at": time.time(),
                "reviews": self.reviews,
            }
            review_file = write_review(self.current_batch_path, save_data)
        except (OSError, ValueError) as exc:
            self.current_status_var.set(f'标注保存失败：{exc}')
            messagebox.showerror('标注未保存', f'{exc}\n已保留原文件，请检查后重试。', parent=self.root)
            return False
        self.current_status_var.set(f"标注已保存：{review_file}")
        if show_prompt:
            messagebox.showinfo("成功", f"标注已成功保存至:\n{review_file}")
        return True

    def open_annotations(self):
        batch = self.current_batch_path or (self.base_output_dir / 'batch')
        directory = review_path(batch).parent
        directory.mkdir(parents=True, exist_ok=True)
        os.startfile(str(directory))

    def export_verified_zip(self) -> None:
        if not self.current_batch_path or self.export_busy:
            return
        include_material = self.include_material_var.get()
        try:
            if not self.save_review(show_prompt=False):
                return
            if include_material:
                validated_rows(self.current_batch_path, self.manifest, self.reviews)
        except (OSError, ValueError) as exc:
            messagebox.showerror('无法导出', str(exc), parent=self.root)
            return
        annotation_file = review_path(self.current_batch_path)
        if include_material:
            destination = str(annotation_file.with_suffix('.zip'))
        else:
            destination = filedialog.asksaveasfilename(parent=self.root, title='导出合格音频 ZIP',
                initialdir=str(annotation_file.parent), defaultextension='.zip',
                initialfile=f'{self.current_batch_path.name}_reviewed.zip', filetypes=[('音频 ZIP', '*.zip')])
        if not destination:
            return
        batch = self.current_batch_path
        manifest, reviews = copy.deepcopy(self.manifest), copy.deepcopy(self.reviews)
        # 在主线程读取 Tk 变量，再将值传给导出线程。
        include_gpt_list = self.gpt_list_var.get() and not include_material
        min_quality = self.quality_var.get() if self.quality_var.get() != "全部质量" else None
        speaker, language = self.export_speaker.get(), self.export_language.get()
        self.export_busy = True
        self.export_button.configure(state='disabled')
        self.current_status_var.set('正在打包标注与音频…' if include_material else '正在导出合格音频…')
        def run():
            try:
                stats = {}
                count = export_review_zip(batch, manifest, reviews, Path(destination), include_material,
                                          gpt_sovits_list=include_gpt_list, min_quality_grade=min_quality,
                                          speaker=speaker, language=language,
                                          stats=stats)
                message = '已保存训练 ZIP（标注、已标片段和参考音频），可直接用于模型强化' if include_material else f'已导出 {count} 段合格音频'
                message += (f"\n有效训练条数：{stats['valid_count']}；总净时长：{stats['total_duration_seconds']:.2f} 秒"
                            f"\n清单（ZIP 内）：\n{stats['slicer_opt_path']}\n{stats['metadata_path']}")
                self.export_events.put((True, message + f'\n{destination}'))
            except Exception as exc:
                self.export_events.put((False, str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def _poll_export(self):
        while not self.speaker_events.empty():
            dialog, status, callback, result, error = self.speaker_events.get_nowait()
            self.speaker_busy = False
            if dialog.winfo_exists():
                if error:
                    status.set('角色探索失败：' + error)
                else:
                    callback(result)
        while not self.export_events.empty():
            ok, message = self.export_events.get_nowait()
            self.export_busy = False
            self.export_button.configure(state='normal')
            self.current_status_var.set('导出完成' if ok else '导出失败')
            (messagebox.showinfo if ok else messagebox.showerror)('导出结果', message, parent=self.root)
        self.root.after(250, self._poll_export)
