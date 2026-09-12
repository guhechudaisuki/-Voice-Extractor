"""Exercise subtitle bindings in a disposable UI without scanning/installing."""

import sys
from pathlib import Path
from unittest.mock import patch
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "launcher"))
import voice_extractor_desktop as desktop


def main():
    root = tk.Tk()
    try:
        with patch.object(desktop, "load_cached_context", return_value=None), patch.object(desktop.VoiceExtractorDesktop, "_start_scan", lambda self: None):
            app = desktop.VoiceExtractorDesktop(root)
            app.setup_page.pack_forget()
            app.main_page.pack(fill="both", expand=True)
            app.targets = [str(ROOT / "episode01.wav"), str(ROOT / "episode02.mkv")]
            app._refresh_target_list()
            app.target_list.selection_set(0)
            with patch.object(desktop.filedialog, "askopenfilename", return_value=str(ROOT / "episode01.sc.ass")):
                app._bind_subtitle()
            assert app.subtitles == {app.targets[0]: str(ROOT / "episode01.sc.ass")}
            app.target_list.selection_clear(0, "end")
            app.target_list.selection_set(1)
            assert "未绑定" in (app._refresh_subtitle_status() or app.subtitle_status_var.get())
            app.target_list.selection_clear(0, "end")
            app.target_list.selection_set(0)
            app._refresh_subtitle_status()
            from PIL import ImageGrab
            for width, height in ((1240, 820), (1040, 700)):
                root.geometry(f"{width}x{height}+10+10")
                root.attributes("-topmost", True)
                root.lift()
                root.update()
                root.after(250)
                left, top = root.winfo_rootx(), root.winfo_rooty()
                ImageGrab.grab(bbox=(left, top, left + root.winfo_width(), top + root.winfo_height())).save(ROOT / f"work/subtitle_ui_{width}.png")
                assert app.target_list.winfo_ismapped()
                assert app.target_list.winfo_height() >= 35
                app.controls_canvas.yview_moveto(1)
                root.update()
                root.after(250)
                ImageGrab.grab(bbox=(left, top, left + root.winfo_width(), top + root.winfo_height())).save(ROOT / f"work/subtitle_ui_{width}_scrolled.png")
                app.controls_canvas.yview_moveto(0)
            app._remove_subtitles()
            assert not app.subtitles
            app.subtitles[app.targets[0]] = str(ROOT / "episode01.sc.ass")
            app._remove_targets()
            assert len(app.targets) == 1 and not app.subtitles
            app._clear_targets()
            assert not app.targets and not app.subtitles
            print("Desktop subtitle bind/select/remove/delete/clear and two viewport screenshots passed")
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
