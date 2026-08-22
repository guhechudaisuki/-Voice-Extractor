from __future__ import annotations

import logging
import os
import queue
import socket
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.pipeline import ExtractionPipeline, PipelineOptions


LOG_PATH = ROOT / "output" / "app.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
LOGGER = logging.getLogger("voice-extract")


def _paths(value) -> list[Path]:
    """Normalize Gradio File values across versions and upload modes."""
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    result: list[Path] = []
    for item in values:
        if isinstance(item, dict):
            item = item.get("path") or item.get("name")
        else:
            item = getattr(item, "path", item)
        if item:
            result.append(Path(str(item)))
    return result


def run_job(
    references,
    targets,
    strictness: str,
    use_overlap: bool,
    use_singing: bool,
):
    refs = _paths(references)
    target_paths = _paths(targets)
    if not refs:
        raise gr.Error("请上传至少一段参考音频")
    if not target_paths:
        raise gr.Error("请上传至少一段待提取音频")

    thresholds = {"标准": 0.68, "严格": 0.70, "极严格": 0.72}
    options = PipelineOptions(
        speaker_threshold=thresholds.get(strictness, 0.70),
        use_overlap_detector=bool(use_overlap),
        use_singing_detector=bool(use_singing),
    )

    events: queue.Queue[tuple[float, str]] = queue.Queue()
    outcome: dict[str, object] = {}
    started = time.monotonic()

    def report(value: float, message: str) -> None:
        events.put((max(0.0, min(1.0, value)), message))

    def worker() -> None:
        try:
            outcome["batch"] = ExtractionPipeline(options=options).run_many(
                refs,
                target_paths,
                progress=report,
            )
        except Exception as exc:
            outcome["error"] = exc
            outcome["traceback"] = traceback.format_exc()

    thread = threading.Thread(target=worker, name="voice-extraction-job", daemon=True)
    thread.start()
    current_value = 0.0
    current_message = "任务已启动"
    while thread.is_alive() or not events.empty():
        changed = False
        try:
            event_value, event_message = events.get(timeout=2.0 if thread.is_alive() else 0.0)
            current_value = max(current_value, event_value)
            current_message = event_message
            changed = True
            while True:
                event_value, event_message = events.get_nowait()
                current_value = max(current_value, event_value)
                current_message = event_message
        except queue.Empty:
            pass
        elapsed = int(time.monotonic() - started)
        activity = (
            "正在处理"
            if changed
            else "仍在处理（当前阶段暂时没有新的进度回报）"
        )
        status_text = (
            f"总体进度：{current_value * 100:.1f}%\n"
            f"当前阶段：{current_message}\n"
            f"运行状态：{activity}\n"
            f"已运行：{elapsed // 60:02d}:{elapsed % 60:02d}"
        )
        yield status_text, round(current_value * 100, 1), [], None, None, None
    thread.join()
    if "error" in outcome:
        LOGGER.error("任务失败\n%s", outcome.get("traceback", ""))
        raise gr.Error(f"任务失败：{outcome['error']}")
    batch = outcome["batch"]

    rows = []
    for target_path, result in zip(target_paths, batch.results):
        for sentence in result.accepted:
            # Gradio Dataframe outputs are positional rows. Returning dicts
            # raises a Pydantic tuple validation error in Gradio 4.44.x.
            rows.append(
                [
                    target_path.name,
                    sentence.text or sentence.whisper_text,
                    sentence.language,
                    round(sentence.start, 3),
                    round(sentence.end, 3),
                    round(sentence.speaker_score, 4),
                    round(sentence.singing_score, 4),
                    sentence.audio_file,
                ]
            )
    accepted_count = sum(len(result.accepted) for result in batch.results)
    rejected_count = sum(len(result.rejected) for result in batch.results)
    summary = (
        f"批次完成：目标文件 {len(batch.results)} 个；保留 {accepted_count} 句；舍弃 {rejected_count} 句\n"
        f"批次目录：{batch.output_dir}\n"
        f"批次压缩包：{batch.archive_path}\n"
        f"处理明细：" + "；".join(
            f"{target_path.name}（保留 {len(result.accepted)}，舍弃 {len(result.rejected)}）"
            for target_path, result in zip(target_paths, batch.results)
        )
    )
    yield (
        summary,
        100.0,
        rows,
        str(batch.archive_path),
        str(batch.manifest_path),
        str(batch.transcript_path),
    )


with gr.Blocks(title="参考音色句子提取", analytics_enabled=False) as demo:
    gr.Markdown("# 参考音色句子提取")
    gr.Markdown("上传同一人的多段参考讲话和一个或多个目标音频，工具会逐个目标文件处理，只保留匹配音色的完整讲话句子，并输出纯人声和文本。")
    with gr.Row():
        references = gr.File(
            label="参考音频（可多选）",
            file_count="multiple",
            file_types=["audio"],
            type="filepath",
        )
        targets = gr.File(
            label="待提取音频（可多选）",
            file_count="multiple",
            file_types=["audio"],
            type="filepath",
        )
    with gr.Row():
        strictness = gr.Radio(["标准", "严格", "极严格"], value="严格", label="匹配严格度")
        use_overlap = gr.Checkbox(value=True, label="过滤多人同时说话")
        use_singing = gr.Checkbox(value=True, label="过滤唱歌/歌声")
    run_button = gr.Button("开始提取", variant="primary")
    live_progress = gr.Number(value=0, label="总体进度（0-100%）", interactive=False)
    status = gr.Textbox(label="任务状态", lines=5)
    table = gr.Dataframe(
        headers=["目标文件", "文本", "语言", "起始", "结束", "声纹", "歌声", "音频"],
        datatype=["str", "str", "str", "number", "number", "number", "number", "str"],
        label="已保留句子（按目标文件汇总）",
        interactive=False,
    )
    with gr.Row():
        archive = gr.File(label="批次结果 ZIP")
        manifest = gr.File(label="批次 JSON 清单")
        transcript = gr.File(label="批次 SRT 文本")
    run_button.click(
        run_job,
        inputs=[references, targets, strictness, use_overlap, use_singing],
        outputs=[status, live_progress, table, archive, manifest, transcript],
        show_progress="hidden",
    )


if __name__ == "__main__":
    port = int(os.environ.get("VOICE_EXTRACT_PORT", "7865"))
    url = f"http://127.0.0.1:{port}"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        occupied = probe.connect_ex(("127.0.0.1", port)) == 0
    if occupied:
        try:
            with urllib.request.urlopen(f"{url}/config", timeout=2.0) as response:
                existing_config = response.read().decode("utf-8", errors="ignore")
        except Exception:
            existing_config = ""
        if "参考音色句子提取" in existing_config:
            print(f"提取工具已经在运行：{url}")
            webbrowser.open(url)
            raise SystemExit(0)
        raise OSError(f"端口 {port} 已被其他程序占用，请设置 VOICE_EXTRACT_PORT 后重试")

    demo.queue(max_size=2).launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=True,
    )
