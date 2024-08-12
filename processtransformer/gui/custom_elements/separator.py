

import tkinter as tk
from tkinter import ttk

from processtransformer.gui.custom_elements.common_settings import common_padding_x, common_padding_y


class Separator(ttk.Frame):
    def __init__(self, parent, row, column_span):
        super().__init__(parent)

        self.sep = ttk.Separator(parent, orient='horizontal')
        self.sep.grid(column=0, columnspan=column_span, row=row, sticky=tk.EW,
                      padx=common_padding_x, pady=common_padding_y)
