"""Voice Extractor WebUI 后端。

输出目录结构（由 extractor/pipeline.py 的 _write_outputs 决定）：

    output/<batch_id>/<目标目录>/audio/0001_xxx.wav
    output/<batch_id>/<目标目录>/text/0001_xxx.txt
    output/<batch_id>/<目标目录>/stage_audit.json
    output/<batch_id>/batch_manifest.json      ← sentences[] 是权威索引

因此批次 ≠ 单个目标目录：一个批次可以包含多个目标目录。片段由
batch_manifest.json 的 sentences[] 定义，音频路径由每条记录的 audio_file
字段给出（相对批次根，反斜杠分隔）。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import psutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    # Embedded Python omits cwd/script paths on standalone BAT launches.
    sys.path.insert(0, str(BASE_DIR))
from launcher.training_workspace import Workspace
from launcher.training_material import clean_reviews
from launcher.training_upload import MaterialUploadLimit
from launcher.review_store import read_review, write_review, review_path, existing_review_path
from launcher.reinforcement_history import completion_message
from launcher.review_export import export_review_zip
OUTPUT_DIR = BASE_DIR / "output"
# 簇判决报告由 evaluation/cluster_report.py 写到 docs/ 下
ROOT_DOCS = BASE_DIR / "docs"
SETTINGS_PATH = BASE_DIR / "voice_extractor_settings.json"
STATIC_DIR = Path(__file__).resolve().parent / "webui_static"
LAUNCH_BAT = BASE_DIR / "启动当前版本.bat"
DESKTOP_PY = Path(__file__).resolve().parent / "voice_extractor_desktop.py"

# 与 extractor PipelineOptions 默认值保持一致；UI 只是回显，不作为真值来源。
DEFAULT_THRESHOLD = 0.68
DEFAULT_SILENCE_MIN = 0.20
DEFAULT_SILENCE_MAX = 0.85


def _app_version() -> str:
    """从 desktop_bootstrap.py 里读版本号，避免两处各写一份。"""
    source = Path(__file__).resolve().parent / "desktop_bootstrap.py"
    if source.is_file():
        match = re.search(
            r'^APP_VERSION\s*=\s*["\']([^"\']+)["\']',
            source.read_text(encoding="utf-8", errors="replace"),
            re.MULTILINE,
        )
        if match:
            return match.group(1)
    return "2.01"


APP_VERSION = _app_version()

@asynccontextmanager
async def lifespan(app):
    yield
    cleanup_application()


app = FastAPI(title="Voice Extractor Studio", version=APP_VERSION, lifespan=lifespan)
app.add_middleware(MaterialUploadLimit)
_review_lock = threading.RLock()
_desktop_switch_lock = threading.Lock()
TRAINING_WORKSPACE = Workspace()
TRAINING_JOBS: dict[str, dict] = {}
DOWNLOADS: dict[str, tuple[Path, str]] = {}
_training_lock = threading.RLock()
_artifact_lock = threading.RLock()
TRAINING_PROCESSES: dict[str, subprocess.Popen] = {}
_closing = threading.Event()
_retain_browser_server = False
ACTIVE_PORT = 7860


def load_settings() -> dict:
    settings = {
        "ui_mode": "desktop",
        "webui_port": 7860,
        "auto_next_on_rate": True,
        "blind_mode": True,
        "speaker_threshold": DEFAULT_THRESHOLD,
        "silence_min_seconds": DEFAULT_SILENCE_MIN,
        "silence_max_seconds": DEFAULT_SILENCE_MAX,
        "use_overlap_detector": True,
        "use_singing_detector": True,
        "export_video_clips": False,
    }
    if SETTINGS_PATH.exists():
        try:
            stored = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                settings.update(stored)
        except (OSError, json.JSONDecodeError):
            pass
    return settings


def save_settings(settings: dict) -> None:
    SETTINGS_PATH.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8",
    )


class SettingsPayload(BaseModel):
    ui_mode: Optional[str] = None
    webui_port: Optional[int] = None
    auto_next_on_rate: Optional[bool] = None
    blind_mode: Optional[bool] = None
    speaker_threshold: Optional[float] = None
    silence_min_seconds: Optional[float] = None
    silence_max_seconds: Optional[float] = None
    use_overlap_detector: Optional[bool] = None
    use_singing_detector: Optional[bool] = None
    export_video_clips: Optional[bool] = None


class ItemReview(BaseModel):
    ordinal: int
    filename: str
    label: str
    partial_start: Optional[float] = None
    partial_end: Optional[float] = None
    notes: Optional[str] = ""


class BatchReviewPayload(BaseModel):
    batch_id: str
    reviews: Dict[str, ItemReview]
    reviewer: Optional[str] = "user"


class VerifiedExportPayload(BaseModel):
    include_material: bool = False
    gpt_sovits_list: bool = False


def _batch_dir(batch_id: str) -> Path:
    """解析批次目录，拒绝任何越出 OUTPUT_DIR 的路径。"""
    candidate = (OUTPUT_DIR / batch_id).resolve()
    root = OUTPUT_DIR.resolve()
    if root not in candidate.parents or not candidate.is_dir():
        raise HTTPException(status_code=404, detail="批次不存在")
    return candidate


def _clip_relative(row: dict) -> str:
    """取单条记录的批次内相对路径，统一成正斜杠。"""
    raw = row.get("audio_file") or ""
    return str(raw).replace("\\", "/").lstrip("/")


def _attach_cluster_review_audio(data: dict, batch_dir: Path) -> None:
    """Expose saved veto clips for review, verifying their interval sidecars."""
    target_rows, sentence_rows = data.get("targets"), data.get("sentences")
    if not isinstance(target_rows, list) or not isinstance(sentence_rows, list):
        return
    targets: dict[str, list[str]] = {}
    for target in target_rows:
        if isinstance(target, dict) and isinstance(target.get("target"), str) and isinstance(target.get("output_dir"), str):
            targets.setdefault(target["target"], []).append(target["output_dir"])
    counts: dict[str, int] = {}
    root = batch_dir.resolve()
    for row in sentence_rows:
        if not isinstance(row, dict):
            continue
        diagnostics = row.get("diagnostics")
        if not isinstance(diagnostics, dict) or not diagnostics.get("cluster_removed"):
            continue
        target = row.get("target", "")
        folders = targets.get(target, [])
        if len(folders) != 1 or not folders[0]:
            continue
        counts[target] = counts.get(target, 0) + 1
        if row.get("audio_file"):
            continue
        clip = (root / folders[0] / "cluster_outliers" / f"{counts[target]:04d}.wav").resolve()
        if root not in clip.parents or not clip.is_file():
            continue
        try:
            span = clip.with_suffix(".txt").read_text(encoding="utf-8").splitlines()[0]
            start, end = map(float, span.split("–"))
            matches = abs(start - float(row["start"])) <= .001 and abs(end - float(row["end"])) <= .001
        except (OSError, ValueError, KeyError, IndexError):
            continue
        if matches:
            row["audio_file"] = clip.relative_to(root).as_posix()


def _read_manifest(batch_dir: Path) -> dict:
    manifest_file = batch_dir / "batch_manifest.json"
    if not manifest_file.is_file():
        return {}
    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    _attach_cluster_review_audio(data, batch_dir)
    return data


@app.get("/api/settings")
def get_settings():
    return {**load_settings(), "version": APP_VERSION}


@app.post("/api/settings")
def update_settings(payload: SettingsPayload):
    updated = load_settings()
    updated.update(payload.model_dump(exclude_unset=True, exclude_none=True))
    save_settings(updated)
    return {"status": "ok", "settings": updated}


@app.post("/api/switch_to_desktop")
def switch_to_desktop():
    """Only report success after the native window has mapped and handled events."""
    global _retain_browser_server
    if not _desktop_switch_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="桌面版正在启动，请稍候")
    scheduled = False
    try:
        with RUN.lock:
            if RUN.process is not None:
                raise HTTPException(status_code=409, detail="提取任务运行中，请完成或停止后再切换界面")
        from launcher.desktop_handoff import launch_desktop
        try:
            ready = launch_desktop(DESKTOP_PY, BASE_DIR, port=ACTIVE_PORT)
            _retain_browser_server = False
            settings = load_settings()
            settings["ui_mode"] = "desktop"
            save_settings(settings)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"切回桌面端失败：{exc}") from exc

        if not ready.get("retain_server"):
            def kill_server():
                time.sleep(1.2)
                shutdown_application()
            threading.Thread(target=kill_server, daemon=True).start()
        else:
            from launcher.edge_app import visible_window
            def watch_desktop():
                while not _closing.wait(.5):
                    if visible_window({ready["pid"]}) is None:
                        if not _retain_browser_server:
                            shutdown_application()
                        break
            threading.Thread(target=watch_desktop, daemon=True).start()
        scheduled = True
        return {"status": "ok", "message": "桌面窗口已显示", **ready}
    finally:
        if not scheduled or (scheduled and ready.get("retain_server")):
            _desktop_switch_lock.release()


@app.get("/api/batches")
def list_batches():
    if not OUTPUT_DIR.is_dir():
        return []
    batches: List[dict] = []
    for item in OUTPUT_DIR.iterdir():
        if not item.is_dir():
            continue
        manifest = _read_manifest(item)
        sentences = manifest.get("sentences")
        sentences = sentences if isinstance(sentences, list) else []
        accepted = [row for row in sentences if row.get("accepted")]
        review_file = existing_review_path(item)
        try:
            label_count = len(read_review(item).get('reviews', {}))
        except (OSError, ValueError):
            label_count = 0
        target_dirs = sorted(
            {_clip_relative(row).split("/")[0] for row in accepted if _clip_relative(row)}
        )
        batches.append({
            "id": item.name,
            "path": str(item),
            "created_time": item.stat().st_mtime,
            "has_manifest": bool(sentences),
            "accepted_count": len(accepted),
            "total_count": len(sentences),
            "target_count": len(target_dirs),
            "targets": target_dirs,
            "labeled_count": label_count,
            "is_reviewed": review_file.is_file(),
        })
    return sorted(batches, key=lambda row: row["created_time"], reverse=True)


@app.get("/api/batches/{batch_id}")
def get_batch_details(batch_id: str):
    batch_dir = _batch_dir(batch_id)
    manifest = _read_manifest(batch_dir)

    existing = _read_review(batch_dir).get('reviews', {})

    sentences = manifest.get("sentences")
    if not isinstance(sentences, list):
        sentences = []

    segments: List[dict] = []
    for ordinal, row in enumerate(sentences, start=1):
        if not isinstance(row, dict):
            continue
        relative = _clip_relative(row)
        # 音频必须真实存在，否则前端会显示一个点了没声音的空行。
        if not relative or not (batch_dir / relative).is_file():
            continue
        diagnostics = row.get("diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        start = float(row.get("start") or 0.0)
        end = float(row.get("end") or 0.0)
        segments.append({
            "ordinal": ordinal,
            "filename": Path(relative).name,
            "relative_path": relative,
            "target_dir": relative.split("/")[0],
            "stt_text": str(row.get("text") or row.get("whisper_text") or ""),
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(max(0.0, end - start), 3),
            "accepted": bool(row.get("accepted")),
            "cluster_removed": bool(diagnostics.get("cluster_removed")),
            "cluster_tier": diagnostics.get("cluster_tier") or "",
            "reject_reason": str(row.get("reject_reason") or ""),
            "score": row.get("speaker_score"),
            "tier": diagnostics.get("speaker_tier") or "",
            "route": _route_of(diagnostics),
            "exclusion_margin": diagnostics.get("exclusion_margin"),
            "review": existing.get(str(ordinal)) or {},
        })

    return {
        "batch_id": batch_id,
        "total_segments": len(segments),
        "segments": segments,
        "reviewed_count": sum(1 for seg in segments if seg["review"]),
        "settings": load_settings(),
    }


@app.get("/api/batches/{batch_id}/stage_turns")
def get_stage_turns(batch_id: str):
    """返回 stage_audit.json 里的说话轮次与最终输出段。

    簇判决报告里的成员是**说话轮次序号**（254 个），而界面上试听的是**最终
    输出片段**（28 段）。两者不是一回事，前端要靠时间段重叠把它们对上。
    这个接口把两套时间段都给出来，好在浏览器里做重叠匹配。
    """
    batch_dir = _batch_dir(batch_id)
    audit_file = next(iter(sorted(batch_dir.rglob("stage_audit.json"))), None)
    if audit_file is None:
        return {"available": False, "reason": "这一集没有 stage_audit.json"}
    try:
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": f"无法解析：{exc}"}

    stages = audit.get("stages", {})
    return {
        "available": True,
        "source_duration_seconds": audit.get("source_duration_seconds", 0.0),
        "turns": stages.get("speaker_turns", {}).get("spans", []),
        "funnel": [
            {"stage": name, "count": stages.get(name, {}).get("count", 0)}
            for name in (
                "raw_initial_vad", "clean_speech_islands", "speaker_turns",
                "identity_accepted_before_stt", "final_accepted",
            )
            if name in stages
        ],
    }


def _route_of(diagnostics: dict) -> str:
    """把诊断标记映射成人类可读的判定路线，用于盲听关闭后的展示。"""
    ordered = (
        ("cluster_removed", "簇级撤下"),
        ("post_target_silence_merge", "静音合并"),
        ("target_locator_recovery", "定位恢复"),
        ("cluster_identity", "簇级判决"),
        ("global_target_cluster", "全局簇恢复"),
        ("classifier_target_rescue", "分类器救援"),
        ("subtitle_completion", "字幕补全"),
        ("island_child_rescue", "孤岛子段"),
        ("multi_model_target_subcluster", "多模型子簇"),
        ("final_identity_boundary_crop", "边界裁切"),
        ("narration", "旁白"),
    )
    for key, label in ordered:
        if diagnostics.get(key):
            return label
    return "标准路径"


def _cluster_reports(batch_dir: Path):
    reports = []
    for path in sorted(batch_dir.rglob("cluster_report.json")):
        if batch_dir.resolve() not in path.resolve().parents:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("clusters"), list):
            reports.append((path, data, str(path.parent.relative_to(batch_dir)).replace("\\", "/")))
    if reports:
        return reports
    # Historical development reports are bound by full batch ID, never filename suffix.
    for path in sorted(ROOT_DOCS.glob("cluster*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("meta", {}).get("batch_id") == batch_dir.name and isinstance(data.get("clusters"), list):
            targets = _read_manifest(batch_dir).get("targets", [])
            target = targets[0].get("output_dir", ".") if len(targets) == 1 else "."
            return [(path, data, target)]
    return []


def _read_review(batch_dir):
    try:
        return read_review(batch_dir)
    except (OSError, ValueError) as exc:
        raise HTTPException(409, '标注文件无法读取，已保留原文件') from exc


def _write_review(batch_dir, data):
    try:
        return write_review(batch_dir, data)
    except (OSError, ValueError) as exc:
        raise HTTPException(409, '标注保存失败，已保留原文件，请检查后重试') from exc


@app.get("/api/batches/{batch_id}/clusters")
def get_batch_clusters(batch_id: str, target_dir: Optional[str] = None):
    import hashlib
    directory = _batch_dir(batch_id)
    reports = _cluster_reports(directory)
    if not reports:
        return {"available": False, "reason": "这一批次没有簇判决报告"}
    selected = next((r for r in reports if r[2] == target_dir), None) if target_dir else reports[0]
    if selected is None:
        raise HTTPException(404, "目标文件没有簇报告")
    path, report, target = selected
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    override = _read_review(directory).get("cluster_override", {}).get(target)
    if override and override.get("report_sha256") != digest:
        override = None
    clusters = report.get("clusters", [])
    check = report.get("review_check", {})
    return {
        "available": True, "source": path.name, "target_dir": target,
        "targets": [r[2] for r in reports], "report_sha256": digest,
        "batch_id": batch_id, "reason": report.get("reason", ""),
        "mode": report.get("mode", "offline"), "override": override,
        "target_cluster_id": report.get("target_cluster_id"),
        "cluster_count": len(clusters),
        "flagged": {
            "whole_wrong": check.get("whole_wrong_count", check.get("reviewed_wrong_count", 0)),
            "inside_target": check.get("whole_wrong_inside_target", check.get("wrong_inside_target_cluster", 0)),
            "elsewhere": check.get("whole_wrong_elsewhere", check.get("wrong_outside_target_cluster", 0)),
            "caught": check.get("whole_wrong_caught_by_outlier_rule", 0),
            "withheld": check.get("whole_wrong_withheld_no_target", 0),
            "mixed_total": check.get("mixed_clips_total", 0),
            "mixed_in_target": check.get("mixed_clips_in_target", 0),
            "verdict": check.get("verdict", ""),
        },
        "clusters": [{**item, "cohesion": item.get("primary_cohesion", 0),
                      "member_indexes": [m["index"] for m in item.get("members", [])]}
                     for item in clusters],
    }


class ClusterOverridePayload(BaseModel):
    cluster_id: Optional[int] = None
    target_dir: str
    report_sha256: str


@app.post("/api/batches/{batch_id}/cluster_override")
def save_cluster_override(batch_id: str, payload: ClusterOverridePayload):
    directory = _batch_dir(batch_id)
    report = get_batch_clusters(batch_id, payload.target_dir)
    if not report.get("available") or report["report_sha256"] != payload.report_sha256:
        raise HTTPException(409, "簇报告已改变，请刷新后重新选择")
    if payload.cluster_id is not None and payload.cluster_id not in {c["cluster_id"] for c in report["clusters"]}:
        raise HTTPException(400, "簇不存在")
    with _review_lock:
        data = _read_review(directory)
        overrides = data.setdefault("cluster_override", {})
        if payload.cluster_id is None:
            overrides.pop(payload.target_dir, None)
        else:
            overrides[payload.target_dir] = {"cluster_id": payload.cluster_id,
                "report_sha256": payload.report_sha256, "updated_at": time.time(),
                "scope": "review_annotation_only"}
        data.update(version="2.0", batch_id=batch_id, updated_at=time.time())
        _write_review(directory, data)
    return {"status": "ok", "message": "目标簇标注已保存"}


@app.get("/api/audio/{batch_id}/{ordinal}")
def stream_audio(batch_id: str, ordinal: int):
    """按序号取音频：序号是稳定索引，比在 URL 里传文件名可靠。"""
    batch_dir = _batch_dir(batch_id)
    sentences = _read_manifest(batch_dir).get("sentences")
    if not isinstance(sentences, list) or not 1 <= ordinal <= len(sentences):
        raise HTTPException(status_code=404, detail="片段不存在")
    relative = _clip_relative(sentences[ordinal - 1])
    clip = (batch_dir / relative).resolve()
    if not clip.is_file() or batch_dir.resolve() not in clip.parents:
        raise HTTPException(status_code=404, detail="音频文件缺失")
    return FileResponse(clip, media_type="audio/wav")


@app.post("/api/review/{batch_id}")
def save_batch_review(batch_id: str, payload: BatchReviewPayload):
    batch_dir = _batch_dir(batch_id)
    with _review_lock:
        save_data = {
            **_read_review(batch_dir),
            "version": "2.0",
            "batch_id": batch_id,
            "reviewer": payload.reviewer,
            "updated_at": time.time(),
            "reviews": {key: item.model_dump() for key, item in payload.reviews.items()},
        }
        _write_review(batch_dir, save_data)
    return {
        "status": "ok",
        "message": "标注已保存",
        "total_reviewed": len(payload.reviews),
        "annotation_path": str(review_path(batch_dir)),
    }


@app.post("/api/export_verified/{batch_id}")
def export_verified_segments(batch_id: str, payload: Optional[VerifiedExportPayload] = None):
    with _artifact_lock:
        if _closing.is_set():
            raise HTTPException(409, "程序正在关闭")
        return _build_verified_export(batch_id, payload)


def _build_verified_export(batch_id: str, payload: Optional[VerifiedExportPayload]):
    batch_dir = _batch_dir(batch_id)
    if not existing_review_path(batch_dir).is_file():
        raise HTTPException(400, '尚未保存任何标注')
    reviews = _read_review(batch_dir).get('reviews', {})

    manifest = _read_manifest(batch_dir)
    include_material = payload is not None and payload.include_material
    gpt_sovits_list = payload is not None and payload.gpt_sovits_list
    zip_name = f"{batch_id}.zip" if include_material else f"{batch_id}_verified_only.zip"
    zip_path = review_path(batch_dir).with_suffix('.zip') if include_material else TRAINING_WORKSPACE.root / f"{uuid.uuid4().hex}.zip"
    try:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        verified_count = export_review_zip(batch_dir, manifest, reviews, zip_path, include_material,
                                           gpt_sovits_list=gpt_sovits_list and not include_material)
    except (OSError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    token = uuid.uuid4().hex
    # Download cleanup owns only this temporary copy. The user's annotation
    # ZIP stays on disk, and a later re-export cannot change an active download.
    download_path = zip_path
    if include_material:
        download_path = TRAINING_WORKSPACE.root / f'{token}.zip'
        try:
            shutil.copyfile(zip_path, download_path)
        except OSError as exc:
            download_path.unlink(missing_ok=True)
            raise HTTPException(500, 'ZIP 已保存到标注目录，但下载副本创建失败') from exc
    DOWNLOADS[token] = (download_path, zip_name)

    return {
        "status": "ok",
        "verified_count": verified_count,
        "zip_path": str(zip_path),
        "zip_name": zip_name,
        "download_url": f"/api/download/{token}",
        "includes_material": include_material,
    }


@app.get("/api/download/{token}")
def download_artifact(token: str):
    with _artifact_lock:
        data = DOWNLOADS.pop(token, None)
    if data is None or not data[0].is_file():
        raise HTTPException(404, "临时导出文件已清理，请重新导出")
    def clean():
        data[0].unlink(missing_ok=True)
        DOWNLOADS.pop(token, None)
    return FileResponse(data[0], media_type="application/zip", filename=data[1], background=BackgroundTask(clean))


def _training_worker(job_id: str, process: subprocess.Popen, destination: Path):
    try:
        for line in process.stdout:
            with _training_lock:
                if job_id in TRAINING_JOBS:
                    TRAINING_JOBS[job_id]["message"] = line.strip()
        process.wait()
        with _training_lock:
            if job_id not in TRAINING_JOBS or _closing.is_set():
                return
            if process.returncode:
                TRAINING_JOBS[job_id]["state"] = "failed"
                shutil.rmtree(destination)
                return
            report = json.loads((destination / "training_report.json").read_text(encoding="utf-8"))
            TRAINING_JOBS[job_id].update(state="complete", message=completion_message(report),
                                         report=report)
    except Exception as exc:
        with _training_lock:
            if job_id in TRAINING_JOBS:
                TRAINING_JOBS[job_id].update(state="failed", message=str(exc))
    finally:
        with _training_lock:
            TRAINING_PROCESSES.pop(job_id, None)
            if TRAINING_JOBS.get(job_id, {}).get("state") == "failed" and destination.exists():
                try:
                    shutil.rmtree(destination)
                except OSError as exc:
                    TRAINING_JOBS[job_id]["message"] += f"；临时文件清理失败，请点击清理：{exc}"
        if process.stdout:
            process.stdout.close()


@app.post("/api/training/start")
def start_training(files: List[UploadFile] = File(...)):
    if not 1 <= len(files) <= 30:
        raise HTTPException(400, "请选择 1 至 30 个含标注和音频的 ZIP 文件")
    if sum(uploaded.size or 0 for uploaded in files) > 8 * 1024**3:
        raise HTTPException(400, "原料文件总大小超过 8 GB")
    with _training_lock:
        if _closing.is_set():
            raise HTTPException(409, "程序正在关闭")
        if TRAINING_JOBS:
            raise HTTPException(409, "请先清理本次训练的临时文件，再开始新的强化任务")
        job_id = uuid.uuid4().hex
        destination = TRAINING_WORKSPACE.job(job_id)
        paths = []
        try:
            total_size = 0
            for index, uploaded in enumerate(files):
                if not (uploaded.filename or "").lower().endswith(".zip"):
                    raise ValueError("请选择含标注和音频的 ZIP 文件，无需解压")
                path = destination / f"upload_{index}.zip"
                size = 0
                with path.open("wb") as output:
                    for block in iter(lambda: uploaded.file.read(1024 * 1024), b""):
                        size += len(block)
                        total_size += len(block)
                        if size > 4 * 1024**3:
                            raise ValueError("单个原料文件超过 4 GB")
                        if total_size > 8 * 1024**3:
                            raise ValueError("原料文件总大小超过 8 GB")
                        output.write(block)
                paths.append(path)
        except (OSError, ValueError) as exc:
            shutil.rmtree(destination)
            raise HTTPException(400, str(exc)) from exc
        TRAINING_JOBS[job_id] = {"state": "running", "message": "正在核对原料文件",
                                 "temporary_path": str(destination), "report": None}
        process = None
        try:
            process = subprocess.Popen([sys.executable, "-B", str(BASE_DIR / "launcher/model_reinforcement.py"), str(destination)],
                                       cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            TRAINING_PROCESSES[job_id] = process
            TRAINING_WORKSPACE.add_worker(process.pid)
        except (OSError, psutil.Error) as exc:
            if process is not None:
                _stop_process(process)
                if process.stdout:
                    process.stdout.close()
            TRAINING_PROCESSES.pop(job_id, None)
            TRAINING_JOBS.pop(job_id, None)
            shutil.rmtree(destination)
            raise HTTPException(500, f"模型强化任务启动失败：{exc}") from exc
    threading.Thread(target=_training_worker, args=(job_id, process, destination), daemon=True).start()
    return {"job_id": job_id, "state": "running"}


@app.get("/api/training/status/{job_id}")
def training_status(job_id: str):
    with _training_lock:
        job = TRAINING_JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "训练任务不存在或程序已关闭")
        data = dict(job)
    path = Path(data["temporary_path"])
    size = 0
    for file in path.rglob("*"):
        try:
            if file.is_file():
                size += file.stat().st_size
        except FileNotFoundError:
            pass  # A concurrent clear may remove a file after directory enumeration.
    data["temporary_bytes"] = size
    return data


@app.get("/api/training/current")
def current_training():
    with _training_lock:
        return {"job_id": next(reversed(TRAINING_JOBS), None)}


@app.delete("/api/training/{job_id}")
def discard_training(job_id: str):
    with _training_lock:
        job = TRAINING_JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "训练任务不存在")
        process = TRAINING_PROCESSES.pop(job_id, None)
        if process is not None:
            _stop_process(process)
        directory = Path(job["temporary_path"])
        if directory.is_dir():
            shutil.rmtree(directory)
        del TRAINING_JOBS[job_id]
    return {"status": "cleared"}


@app.post("/api/close")
def close_application():
    try:
        cleanup_application()
    except (OSError, subprocess.TimeoutExpired) as exc:
        _closing.clear()
        raise HTTPException(409, f"临时文件尚未清理完成，请重试关闭：{exc}") from exc
    def stop():
        time.sleep(.5)
        os._exit(0)
    threading.Thread(target=stop, daemon=True).start()
    return {"status": "cleaned"}


def _stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def cleanup_application():
    _closing.set()
    with _training_lock:
        processes = list(TRAINING_PROCESSES.values())
        with RUN.lock:
            if RUN.process is not None:
                processes.append(RUN.process)
        for process in processes:
            _stop_process(process)
        with _artifact_lock:
            TRAINING_WORKSPACE.close()
            DOWNLOADS.clear()
        TRAINING_PROCESSES.clear()
        TRAINING_JOBS.clear()


def shutdown_application():
    cleanup_application()
    os._exit(0)


class ExtractPayload(BaseModel):
    targets: List[str] = []
    references: List[str] = []
    negative_groups: List[List[str]] = []
    subtitles: Dict[str, str] = {}
    options: Dict[str, Any] = {}


class _Run:
    """当前提取任务的内存状态。WebUI 与桌面端一样一次只跑一个任务。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None
        self.lines: List[dict] = []
        self.state = "idle"
        self.batch_id = ""
        self.finished_at = 0.0


RUN = _Run()


def _pump(process: subprocess.Popen) -> None:
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = {"type": "log", "message": line}
        with RUN.lock:
            RUN.lines.append(event)
            if event.get("type") == "complete":
                RUN.batch_id = str(event.get("batch_id") or "")
    code = process.wait()
    with RUN.lock:
        RUN.state = "done" if code == 0 else "failed"
        RUN.finished_at = time.time()
        RUN.process = None


@app.post("/api/extract")
def start_extract(payload: ExtractPayload):
    with RUN.lock:
        if _closing.is_set():
            raise HTTPException(409, "程序正在关闭")
        if _desktop_switch_lock.locked():
            raise HTTPException(status_code=409, detail="正在切换桌面版，请在桌面窗口中开始提取")
        if RUN.process is not None:
            raise HTTPException(status_code=409, detail="已有提取任务在运行")

    targets = [Path(item).expanduser() for item in payload.targets if item.strip()]
    references = [Path(item).expanduser() for item in payload.references if item.strip()]
    if not targets:
        raise HTTPException(status_code=400, detail="请填写待提取的音频或视频路径")
    if not references:
        raise HTTPException(status_code=400, detail="请填写参考音频路径")
    missing = [str(p) for p in [*targets, *references] if not p.exists()]
    if missing:
        raise HTTPException(status_code=400, detail=f"路径不存在: {missing[0]}")

    subtitles = {}
    if payload.subtitles:
        from extractor.subtitles import validate_subtitle_bindings
        try:
            subtitles = {str(target): str(subtitle) for target, subtitle in
                validate_subtitle_bindings(targets, {
                    Path(target).expanduser(): Path(subtitle).expanduser()
                    for target, subtitle in payload.subtitles.items()}).items()}
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"字幕绑定无效：{exc}") from exc

    request = {
        "output_root": str(OUTPUT_DIR),
        "targets": [str(p) for p in targets],
        "references": [str(p) for p in references],
        "negative_groups": [
            [str(Path(item).expanduser()) for item in group if item.strip()]
            for group in payload.negative_groups
            if any(item.strip() for item in group)
        ],
        "subtitles": subtitles,
        "options": {**load_settings(), **payload.options},
    }

    requests_dir = BASE_DIR / "work" / "webui_requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    request_path = requests_dir / f"request_{int(time.time() * 1000)}.json"
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    worker = BASE_DIR / "desktop_worker.py"
    if not worker.is_file():
        raise HTTPException(status_code=500, detail="找不到 desktop_worker.py")

    with RUN.lock:
        # Validation may have overlapped a desktop switch or another request.
        if _closing.is_set():
            raise HTTPException(409, "程序正在关闭")
        if _desktop_switch_lock.locked():
            raise HTTPException(status_code=409, detail="正在切换桌面版，请在桌面窗口中开始提取")
        if RUN.process is not None:
            raise HTTPException(status_code=409, detail="已有提取任务在运行")
        process = subprocess.Popen(
            [sys.executable, str(worker), str(request_path)],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        RUN.process = process
        RUN.lines = []
        RUN.state = "running"
        RUN.batch_id = ""
    threading.Thread(target=_pump, args=(process,), daemon=True).start()
    return {"status": "ok", "request": request_path.name}


@app.get("/api/extract/status")
def extract_status(cursor: int = 0):
    """增量返回终端输出。cursor 是调用方已消费的行数。"""
    with RUN.lock:
        lines = RUN.lines[max(0, cursor):]
        return {
            "state": RUN.state,
            "cursor": len(RUN.lines),
            "batch_id": RUN.batch_id,
            "lines": lines,
        }


@app.post("/api/extract/stop")
def stop_extract():
    with RUN.lock:
        process = RUN.process
    if process is None:
        return {"status": "ok", "message": "当前没有任务在运行"}
    process.terminate()
    with RUN.lock:
        RUN.state = "stopped"
        RUN.lines.append({"type": "log", "message": "任务已被手动停止"})
    return {"status": "ok", "message": "已请求停止"}


@app.post("/api/switch_to_webui")
def switch_to_webui():
    """桌面端调用此接口切到网页端：保存偏好、拉起 WebUI、然后由调用方退出自身。"""
    global _retain_browser_server
    import webbrowser
    settings = load_settings()
    settings["ui_mode"] = "webui"
    save_settings(settings)
    port = ACTIVE_PORT
    _retain_browser_server = True
    if not webbrowser.open(f"http://127.0.0.1:{port}/"):
        _retain_browser_server = False
        raise HTTPException(500, "无法打开默认浏览器")
    return {"status": "ok", "port": port, "url": f"http://127.0.0.1:{port}/"}


@app.get("/api/lifecycle")
def lifecycle_status():
    return {"retain_server": _retain_browser_server}


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    index_html = STATIC_DIR / "index.html"
    if index_html.is_file():
        return HTMLResponse(index_html.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Voice Extractor Studio</h1>")


def run_webui(port: int = 7860, host: str = "127.0.0.1") -> None:
    import uvicorn
    global ACTIVE_PORT
    ACTIVE_PORT = port

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Voice Extractor WebUI 服务")
    parser.add_argument("--port", type=int, default=int(load_settings().get("webui_port", 7860)))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("端口必须在 1–65535 之间")
    run_webui(port=args.port)
