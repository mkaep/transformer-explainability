

import tkinter as tk
from tkinter import ttk

from processtransformer.gui.custom_elements.common_settings import common_padding_x, common_padding_y


class TextContainer(ttk.Frame):
    def __init__(self, parent, row: int, column_span: int, text: str):
        super().__init__(parent)

        self.text = tk.Text(parent, height=len(text.splitlines()))
        self.text.insert(tk.END, text)
        self.text.configure(state='disabled')
        self.text.grid(column=0, columnspan=column_span, row=row, sticky=tk.EW,
                       padx=common_padding_x, pady=common_padding_y)

    def destroy(self) -> None:
        super().destroy()
        self.text.destroy()
