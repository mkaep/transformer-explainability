
import logging
import tkinter as tk
from tkinter import ttk

from processtransformer.gui.custom_elements.common_settings import min_col_width
from processtransformer.gui.main_tabs.explainer_tabs.explainer_tab import ExplainerTab
from processtransformer.gui.custom_elements.scrollable_canvas import ScrollableCanvas
from processtransformer.util.subclassing import ExplainerSubclasses


class XaiTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.columnconfigure(0, weight=1, minsize=min_col_width)
        self.columnconfigure(1, weight=2, minsize=min_col_width)
        self.columnconfigure(2, weight=6, minsize=min_col_width)

        # See https://stackoverflow.com/a/51106148/9523044
        self.canvas = tk.Canvas(self, borderwidth=0)
        self.notebook = ttk.Notebook(self.canvas)

        self.canvas.create_window((0, 0), window=self.notebook, anchor="nw", tags="frame")

        self.canvas.pack(side='left', fill='both', expand=True, anchor='nw')
        self.notebook.pack(side='left', fill='both', expand=True, anchor='nw')

        self._fill_notebook_with_tabs()

        self.canvas.bind("<Configure>", self._frame_configure)

    def _fill_notebook_with_tabs(self):
        explainer_subclasses = ExplainerSubclasses.get_all_subclasses()
        # noinspection PyUnresolvedReferences
        import processtransformer.gui.main_tabs.explainer_tabs
        explainer_tab_subclasses = ExplainerTab.__subclasses__()

        # Try to find suiting tab for every explainer, otherwise, default.
        for explainer in explainer_subclasses:
            tab_cls = None
            for possible_tab in explainer_tab_subclasses:
                if possible_tab.get_explainer_cls() is explainer:
                    tab_cls = possible_tab
                    break

            if tab_cls is None:
                continue

            tab_wrapper = ScrollableCanvas(self, tab_cls,
                                           **{'explainer': explainer,
                                              'resize_grid_callback': self._configure_canvas})
            self.notebook.add(tab_wrapper, text=explainer.get_name())

    def _frame_configure(self, event):
        self._configure_canvas()

    def _configure_canvas(self):
        logging.debug('XAI canvas resize')
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
