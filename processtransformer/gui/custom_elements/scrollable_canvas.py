
import logging
import tkinter as tk
from tkinter import ttk

from processtransformer.gui.custom_elements.common_settings import common_padding_x


class ScrollableCanvas(ttk.Frame):
    def __init__(self, parent, inner_cls, **inner_kwargs):
        super().__init__(parent)

        self.canvas = tk.Canvas(self, borderwidth=0)
        inner_kwargs['borderwidth'] = 0
        inner_kwargs['padding'] = common_padding_x
        if 'resize_grid_callback' in inner_kwargs.keys():
            # Add own resize as well
            old_callback = inner_kwargs['resize_grid_callback']
            inner_kwargs['resize_grid_callback'] = lambda: (
                old_callback(),
                self._configure_canvas(),
            )

        self.tab = inner_cls(self.canvas, **inner_kwargs)

        self.ver_scrollbar = ttk.Scrollbar(self.canvas, command=self.canvas.yview)
        self.hor_scrollbar = ttk.Scrollbar(self.canvas, command=self.canvas.xview, orient='horizontal')
        self.canvas.configure(yscrollcommand=self.ver_scrollbar.set)
        self.canvas.configure(xscrollcommand=self.hor_scrollbar.set)
        self.canvas.create_window((0, 0), window=self.tab, anchor="nw", tags="self.tab")

        self.canvas.pack(side='top', fill='both', expand=True, anchor='nw')
        self.ver_scrollbar.pack(side='right', fill='y', expand=False)
        self.hor_scrollbar.pack(side='bottom', fill='x', expand=False)

        self.canvas.bind("<Configure>", self._frame_configure)
        self._set_mousewheel(self.canvas)

    def _frame_configure(self, event):
        self._configure_canvas()

    def _configure_canvas(self):
        logging.debug('Scrollable canvas resize')
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    # https://stackoverflow.com/a/63242272/9523044
    def _set_mousewheel(self, widget):
        """Activate / deactivate mousewheel scrolling when
        cursor is over / not over the widget respectively."""
        widget.bind("<Enter>", lambda _: widget.bind_all('<MouseWheel>', self._on_mousewheel))
        widget.bind("<Leave>", lambda _: widget.unbind_all('<MouseWheel>'))

    # https://stackoverflow.com/a/42831106/9523044
    def _on_mousewheel(self, event):
        shift = (event.state & 0x1) != 0
        scroll_amount = 1
        scroll = -scroll_amount if event.delta > 0 else scroll_amount
        if shift:
            self.canvas.xview_scroll(scroll, "units")
        else:
            self.canvas.yview_scroll(scroll, "units")
