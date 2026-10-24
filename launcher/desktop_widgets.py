"""Scrollable native form shared by review and training tabs."""
import tkinter as tk
from tkinter import ttk


class ScrollableForm(ttk.Frame):
    def __init__(self, parent, padding=14):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, background='#192338')
        bar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.body = ttk.Frame(self.canvas, style='Card.TFrame', padding=padding)
        window = self.canvas.create_window((0, 0), window=self.body, anchor='nw')
        self.body.bind('<Configure>', lambda event: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>', lambda event: self.canvas.itemconfigure(window, width=event.width))
        self.winfo_toplevel().bind('<MouseWheel>', self._scroll, add='+')

    def _scroll(self, event):
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if isinstance(widget, (tk.Listbox, tk.Text, ttk.Spinbox)):
                return
            if widget is self:
                if self.body.winfo_height() > self.canvas.winfo_height():
                    self.canvas.yview_scroll(-int(event.delta / 120), 'units')
                return
            widget = getattr(widget, 'master', None)
