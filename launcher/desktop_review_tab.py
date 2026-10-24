from __future__ import annotations

import json
import os
import copy
import queue
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from launcher.training_material import reviewable, validated_rows, clean_reviews
from launcher.review_export import export_review_zip
from launcher.review_store import read_review, write_review, review_path
from launcher.desktop_widgets import ScrollableForm

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

class DesktopReviewTab:
    """
    计划方案 Phase 1.2: 桌面试听标注标签页
    - 无快捷键依赖，全鼠标直觉点选
    - 盲听模式（默认开启，隐藏分数和层级，消除偏见）
    - 原生音频播放（标准库 winsound，低延迟、零新依赖）
    - 快速决策大按钮（正确、整句错、部分错、歌声、重叠、跳过）
    - 标注后自动下一句
    - 独立 annotations 目录保存标注与仅确认正确段的 ZIP 导出
    """

    def __init__(self, parent_notebook: ttk.Notebook, root: tk.Tk, base_output_dir: Path) -> None:
        self.root = root
        self.notebook = parent_notebook
        self.base_output_dir = base_output_dir

        self.tab_frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.tab_frame, text=" 🎧 试听标注与复核 ")

        self.batches: list[Path] = []
        self.current_batch_path: Path | None = None
        self.segments: list[dict] = []
        self.current_index: int = -1
        self.reviews: dict[str, dict] = {}
        self.filter_var = tk.StringVar(value="全部可审片段")
        self.gpt_list_var = tk.BooleanVar(value=True)
        self.blind_mode_var = tk.BooleanVar(value=True)
        self.auto_next_var = tk.BooleanVar(value=True)
        self.current_status_var = tk.StringVar(value="请选择一个输出批次开始试听")
        self.cluster_summary_var = tk.StringVar(value="簇级判决：未载入")
        self.current_audio_file: Path | None = None
        self.manifest = {}
        self.include_material_var = tk.BooleanVar(value=False)
        self.material_state = tk.StringVar(value='全部片段打标后可附带训练原料')
        self.export_busy = False
        self.export_events = queue.Queue()

        self._build_ui()
        self.refresh_batches()
        self.root.after(250, self._poll_export)

    def _build_ui(self) -> None:
        self.tab_frame.columnconfigure(1, weight=1)
        self.tab_frame.rowconfigure(0, weight=1)

        # 左侧批次与片段列表面板
        left_card = ttk.Frame(self.tab_frame, style="Card.TFrame", padding=10)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_card.columnconfigure(0, weight=1)
        left_card.rowconfigure(4, weight=1)

        # 批次下拉
        batch_header = ttk.Frame(left_card, style="Card.TFrame")
        batch_header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(batch_header, text="选择输出批次:", font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Button(batch_header, text="刷新", width=5, command=self.refresh_batches).pack(side="right")

        self.batch_combo = ttk.Combobox(left_card, state="readonly", width=32)
        self.batch_combo.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.batch_combo.bind("<<ComboboxSelected>>", self._on_batch_selected)

        # 筛选视图：Tier A 主输出 / Tier B 边缘捡漏 / 已撤下
        filter_row = ttk.Frame(left_card, style="Card.TFrame")
        filter_row.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self.filter_combo = ttk.Combobox(
            filter_row, state="readonly", textvariable=self.filter_var, width=32,
            values=("全部可审片段", "✅ 主输出 Tier A（高确信）",
                    "⚡ 边缘待捡漏 Tier B", "🚫 已撤下片段"))
        self.filter_combo.pack(side="left", fill="x", expand=True)
        self.filter_combo.bind("<<ComboboxSelected>>", lambda _e: self._render_tree())
        ttk.Button(filter_row, text="一键信任主输出", width=14,
                   command=self.trust_main_output).pack(side="right")

        # 过滤与进度条
        progress_row = ttk.Frame(left_card, style="Card.TFrame")
        progress_row.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        self.progress_label = ttk.Label(progress_row, text="进度: 0 / 0 已审", style="CardText.TLabel")
        self.progress_label.pack(side="left")

        # 列表视图
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

        # 右侧操作工作台
        self.review_form = ScrollableForm(self.tab_frame)
        self.review_form.grid(row=0, column=1, sticky="nsew")
        right_card = self.review_form.body
        right_card.columnconfigure(0, weight=1)

        # 顶部音频元数据与盲听切换
        top_meta = ttk.Frame(right_card, style="Card.TFrame")
        top_meta.pack(fill="x", pady=(0, 10))
        ttk.Label(top_meta, textvariable=self.cluster_summary_var, wraplength=680).pack(anchor="w")
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

        # 诊断信息 (非盲听时展示)
        self.diag_frame = ttk.Frame(right_card, style="Card.TFrame")
        self.diag_frame.pack(fill="x", pady=(0, 8))
        self.diag_label = ttk.Label(self.diag_frame, text="", foreground="#b9a0ff")
        self.diag_label.pack(anchor="w")

        # STT 字幕展示卡片
        stt_box = ttk.LabelFrame(right_card, text=" STT 字幕识别 ", padding=10)
        stt_box.pack(fill="x", pady=(0, 12))
        self.stt_var = tk.StringVar(value="暂无文本")
        self.stt_display = ttk.Label(stt_box, textvariable=self.stt_var, font=("Segoe UI", 11), wraplength=520)
        self.stt_display.pack(anchor="w")

        # 播放控制大按钮区 (零快捷键、大按钮点选)
        play_card = ttk.LabelFrame(right_card, text=" 音频试听控制 ", padding=12)
        play_card.pack(fill="x", pady=(0, 12))
        btn_bar = ttk.Frame(play_card)
        btn_bar.pack(anchor="center")

        ttk.Button(btn_bar, text="◀ 上一句", width=10, command=self.prev_segment).pack(side="left", padx=6)
        ttk.Button(btn_bar, text="▶ 播放当前句", width=14, style="Accent.TButton", command=self.play_current).pack(side="left", padx=8)
        ttk.Button(btn_bar, text="↺ 重新播放", width=12, command=self.replay_current).pack(side="left", padx=6)
        ttk.Button(btn_bar, text="下一句 ▶", width=10, command=self.next_segment).pack(side="left", padx=6)
        # partial（部分错人）保留段试听：直接听裁掉错人区间后的干净部分。
        self.keep_play_button = ttk.Button(btn_bar, text="▶ 试听保留段", width=12,
                                           command=self.play_keep_segment, state="disabled")
        self.keep_play_button.pack(side="left", padx=6)

        # 核心裁决大按钮点选区
        decision_box = ttk.LabelFrame(right_card, text=" 人工试听裁决 (点击即标注并记录) ", padding=12)
        decision_box.pack(fill="x", pady=(0, 12))
        decision_box.columnconfigure((0, 1, 2), weight=1)

        self.btn_correct = tk.Button(
            decision_box,
            text="✔ 目标匹配 (Correct)\n完全纯净，直接入训练集",
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
            text="✖ 整句非目标 (Wrong)\n错人/完全非目标角色",
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
            text="✂ 部分非目标 (Partial)\n前后夹带他人声音",
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
            text="♫ 歌声混入 (Singing)\n伴奏人声/唱歌段落",
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
            text="👥 重叠杂音 (Overlap)\n多人同时发音无法纯净切分",
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
            text="⊘ 存疑 / 跳过 (Skip)\n暂不确定，保留待复查",
            font=("Segoe UI", 9),
            bg="#f3f4f6",
            fg="#4b5563",
            relief="groove",
            height=2,
            command=lambda: self.apply_decision("skip"),
        )
        self.btn_skip.grid(row=1, column=2, sticky="ew", padx=4, pady=4)

        # 底部状态与导出操作卡片
        bottom_bar = ttk.Frame(right_card, style="Card.TFrame")
        bottom_bar.pack(fill="x", side="bottom", pady=(10, 0))

        ttk.Label(bottom_bar, textvariable=self.current_status_var, style="CardText.TLabel", wraplength=620).pack(anchor='w')
        ttk.Label(bottom_bar, textvariable=self.material_state, wraplength=620).pack(anchor='w', pady=5)
        raw_row = ttk.Frame(bottom_bar)
        raw_row.pack(fill='x', pady=5)
        self.material_check = ttk.Checkbutton(raw_row, text='附带训练原料（全部标注和音频，ZIP 可直接强化）', variable=self.include_material_var, state='disabled')
        self.material_check.pack(side='left')
        ttk.Checkbutton(raw_row, text='附带 GPT-SoVITS train.list', variable=self.gpt_list_var).pack(side='left', padx=12)
        self.export_button = ttk.Button(bottom_bar, text="导出 ZIP", command=self.export_verified_zip)
        self.export_button.pack(side="right", padx=6, pady=6)
        ttk.Button(bottom_bar, text="保存标注进度", command=self.save_review).pack(side="right", padx=6, pady=6)
        ttk.Button(bottom_bar, text="打开标注目录", command=self.open_annotations).pack(side="right", padx=6, pady=6)

    def refresh_batches(self) -> None:
        if not self.base_output_dir.exists():
            return
        self.batches = [
            p for p in sorted(self.base_output_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            if p.is_dir() and (p / "batch_manifest.json").exists() or list(p.glob("*.wav"))
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

        manifest = {}
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                manifest = {}

        self.reviews = {}
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

            self.segments.append({
                "ordinal": idx,
                "file": wav,
                "filename": wav.name,
                "duration": m_info.get("duration") or m_info.get("audio_duration") or 0.0,
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
                    "view": view,
                    "tier": diagnostics.get("speaker_tier", ""),
                    "pathway": "簇级判决" if diagnostics.get("cluster_identity") else "原判决",
                    "score": row.get("speaker_score")})
        else:
            manifest = {'sentences': [{'accepted': True, 'audio_file': seg['filename'], 'start': 0,
                                      'end': seg['duration']} for seg in self.segments]}
        self.manifest = manifest
        self.current_index = -1
        self.include_material_var.set(False)
        self._render_tree()
        if self.segments:
            self.select_segment(0)

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
        view_filter = {"✅ 主输出 Tier A（高确信）": "A",
                       "⚡ 边缘待捡漏 Tier B": "B",
                       "🚫 已撤下片段": "removed"}.get(view)
        for idx, seg in enumerate(self.segments):
            if view_filter and self._segment_view(seg) != view_filter:
                continue
            rev = self.reviews.get(str(seg["ordinal"]), {})
            status_text = label_map.get(rev.get("label", ""), "待审")
            dur = f"{seg['duration']:.2f}s" if seg["duration"] else ""
            self.tree.insert("", "end", iid=str(idx), values=(seg["ordinal"], status_text, dur, seg["stt_text"]))

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
            self.material_state.set('已全部打标；勾选后保存到 annotations/批次名.zip，模型强化直接读取')

    def trust_main_output(self) -> None:
        """一键信任主输出：未标注的 Tier A 片段全部标为 correct。

        主输出准确率 >= 95% 时无需逐句点几十次"正确"；只需听 Tier B 边缘池
        把漏网好句救回。
        """
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
        self.current_status_var.set(f'已信任 {len(pending)} 段主输出；请切到「⚡ 边缘待捡漏 Tier B」继续捡漏')

    def play_keep_segment(self) -> None:
        """试听 partial 标注裁掉错人区间后的干净保留段。"""
        if self.current_index < 0:
            return
        seg = self.segments[self.current_index]
        review = self.reviews.get(str(seg["ordinal"]), {})
        if review.get("label") != "partial":
            return
        from launcher.training_material import partial_keep_segments
        keep = max(partial_keep_segments(
            {"start": 0, "end": seg["duration"]}, review),
            key=lambda piece: piece[1] - piece[0], default=None)
        if keep is None or keep[1] - keep[0] < 0.2:
            messagebox.showinfo('试听保留段', '错人区间覆盖了整句，没有可保留的干净段。', parent=self.root)
            return
        import tempfile
        from extractor.audio import write_clip
        handle, name = tempfile.mkstemp(prefix='voice_keep_', suffix='.wav')
        os.close(handle)
        try:
            write_clip(seg["file"], Path(name), keep[0], keep[1], normalize_level=True)
            if HAS_WINSOUND:
                winsound.PlaySound(name, winsound.SND_FILENAME | winsound.SND_ASYNC)
                self.current_status_var.set(
                    f"正在播放保留段 [{keep[0]:.2f}s – {keep[1]:.2f}s]（共 {seg['duration']:.2f}s）")
        except (ValueError, OSError, RuntimeError) as exc:
            self.current_status_var.set(f'保留段播放失败: {exc}')
        finally:
            # 播放是异步的，立即删除会被 winsound 拒绝；延迟清理。
            def _cleanup(path=name):
                try:
                    os.unlink(path)
                except OSError:
                    pass
            self.root.after(30000, _cleanup)

    def _on_tree_select(self, _event=None) -> None:
        sel = self.tree.selection()
        if sel:
            idx = int(sel[0])
            if idx != self.current_index:
                self.select_segment(idx)

    def select_segment(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.segments):
            return
        self.current_index = idx
        seg = self.segments[idx]
        self.current_audio_file = seg["file"]

        # 选中高亮
        if self.tree.exists(str(idx)):
            # ttk queues TreeviewSelect even when selection_set repeats the
            # same row. Requeueing here starves window painting and timers.
            if self.tree.selection() != (str(idx),):
                self.tree.selection_set(str(idx))
            self.tree.see(str(idx))

        dur_text = f"{seg['duration']:.2f}s" if seg["duration"] else ""
        self.current_title_label.config(text=f"片段 #{seg['ordinal']} - {seg['filename']} ({dur_text})")
        self.stt_var.set(seg["stt_text"] if seg["stt_text"] else "（无识别字幕）")

        # 诊断信息
        if not self.blind_mode_var.get():
            score_s = f"{seg['score']:.3f}" if seg.get("score") is not None else "无"
            self.diag_label.config(text=f"[诊断模式] 层级: {seg['tier']} | 路径: {seg['pathway']} | 相似度: {score_s}")
            self.diag_frame.pack(fill="x", pady=(0, 8))
        else:
            self.diag_frame.pack_forget()

        # partial 标注的片段提供保留段试听
        review = self.reviews.get(str(seg["ordinal"]), {})
        if review.get("label") == "partial" and review.get("partial_end", 0) > (review.get("partial_start") or 0):
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
                self.current_status_var.set(f"正在播放 #{self.segments[self.current_index]['ordinal']}")
            except Exception as e:
                self.current_status_var.set(f"播放失败: {e}")

    def replay_current(self) -> None:
        self.play_current()

    def prev_segment(self) -> None:
        if self.current_index > 0:
            self.select_segment(self.current_index - 1)

    def next_segment(self) -> None:
        if self.current_index < len(self.segments) - 1:
            self.select_segment(self.current_index + 1)

    def apply_decision(self, label: str) -> None:
        if self.current_index < 0 or self.current_index >= len(self.segments):
            return
        seg = self.segments[self.current_index]
        ordinal_str = str(seg["ordinal"])
        partial = {}
        if label == 'partial':
            start = simpledialog.askfloat('错人区间', '非目标声音从第几秒开始？', parent=self.root, minvalue=0, maxvalue=seg['duration'])
            if start is None:
                return
            end = simpledialog.askfloat('错人区间', '非目标声音到第几秒结束？', parent=self.root, minvalue=start, maxvalue=seg['duration'])
            if end is None or end <= start:
                return
            partial = {'partial_start': start, 'partial_end': end}
        self.reviews[ordinal_str] = {
            "ordinal": seg["ordinal"],
            "filename": seg["filename"],
            "label": label,
            **partial,
        }

        self._render_tree()
        if not self.save_review(show_prompt=False):
            return

        if self.auto_next_var.get() and self.current_index < len(self.segments) - 1:
            self.root.after(200, lambda: self.select_segment(self.current_index + 1))

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
        # Tk 变量只能在主线程读取；线程内访问会触发
        # "main thread is not in main loop"。
        include_gpt_list = self.gpt_list_var.get() and not include_material
        self.export_busy = True
        self.export_button.configure(state='disabled')
        self.current_status_var.set('正在打包标注与音频…' if include_material else '正在导出合格音频…')
        def run():
            try:
                count = export_review_zip(batch, manifest, reviews, Path(destination), include_material,
                                          gpt_sovits_list=include_gpt_list)
                message = '已保存训练 ZIP（标注、全部片段和参考音频），可直接用于模型强化' if include_material else f'已导出 {count} 段合格音频'
                self.export_events.put((True, message + f'\n{destination}'))
            except Exception as exc:
                self.export_events.put((False, str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def _poll_export(self):
        while not self.export_events.empty():
            ok, message = self.export_events.get_nowait()
            self.export_busy = False
            self.export_button.configure(state='normal')
            self.current_status_var.set('导出完成' if ok else '导出失败')
            (messagebox.showinfo if ok else messagebox.showerror)('导出结果', message, parent=self.root)
        self.root.after(250, self._poll_export)
