"""Exercise real ttk selection events in a child so a loop cannot hang tests."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DesktopSelectionTests(unittest.TestCase):
    def test_programmatic_and_mouse_selection_do_not_requeue_themselves(self):
        child = r'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / 'launcher'))
import tkinter as tk
from tkinter import ttk
from unittest.mock import patch
from desktop_review_tab import DesktopReviewTab
root = tk.Tk()
root.withdraw()
try:
    notebook = ttk.Notebook(root)
    with patch.object(DesktopReviewTab, 'refresh_batches'), patch.object(DesktopReviewTab, 'play_current') as play:
        tab = DesktopReviewTab(notebook, root, Path.cwd() / 'output')
        tab.segments = [dict(ordinal=i+1, file=Path('unused.wav'), filename='unused.wav', duration=2,
                            stt_text=str(i), tier='', pathway='', score=None) for i in range(2)]
        tab._render_tree()
        tab.select_segment(0)
        events = []
        tab.tree.bind('<<TreeviewSelect>>', lambda event: events.append(tab.tree.selection()), add='+')
        root.after(150, lambda: tab.tree.selection_set('1'))
        root.after(300, root.quit)
        root.mainloop()
        assert tab.current_index == 1
        assert play.call_count == 2, play.call_count
        assert len(events) == 2, events
        print('selection events settled; timer fired; two selections played once each', flush=True)
finally:
    root.destroy()
'''
        try:
            result = subprocess.run([sys.executable, '-B', '-c', child], cwd=ROOT,
                capture_output=True, text=True, timeout=5,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired:
            self.fail('Treeview selection loop prevented the Tk timer from firing')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('selection events settled', result.stdout)


if __name__ == '__main__':
    unittest.main()
