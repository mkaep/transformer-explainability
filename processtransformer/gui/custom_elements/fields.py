
import tkinter as tk
from tkinter import ttk
from typing import Callable

from processtransformer.gui.custom_elements.common_settings import common_padding_y, common_padding_x, check_float, \
    check_int


class Field(ttk.Frame):
    def __init__(self, parent, row: int, label_text: str):
        super().__init__(parent)

        self.label = ttk.Label(parent, text=label_text)
        self.entry = ttk.Entry(parent)
        self.label.grid(column=0, row=row, sticky=tk.W,
                        padx=common_padding_x, pady=common_padding_y)
        self.entry.grid(column=1, row=row, sticky=tk.W,
                        padx=common_padding_x, pady=common_padding_y)

    def get(self):
        return self.entry.get()

    def disable(self):
        self.entry.config(state='disabled')


class ReadOnlyField(Field):

    def __init__(self, parent, row: int, label_text: str, entry_text_ro: str):
        super().__init__(parent, row, label_text)
        self.entry.grid_remove()
        self.second_label = ttk.Label(parent, text=entry_text_ro)
        self.second_label.grid(column=1, row=row, sticky=tk.W,
                               padx=common_padding_x, pady=common_padding_y)


class NumericField(Field):
    def __init__(self, parent, row: int, label_text: str, check_func: Callable[[str], bool], cast_func: Callable,
                 lower_bound: float = 0.0, upper_bound: float = 1.0, preset: float = None):
        super().__init__(parent, row, label_text)

        self.cast_func = cast_func
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        # Configure check on entry
        self.entry['validate'] = 'key'
        val_cmd = parent.register(check_func)
        self.entry['validatecommand'] = (val_cmd, '%P')
        if preset is not None:
            self.entry.insert(0, str(preset))

    def get(self):
        num = self.cast_func(self.entry.get())
        assert self.lower_bound <= num <= self.upper_bound
        return num


class FloatField(NumericField):

    def __init__(self, parent, row: int, label_text: str,
                 lower_bound: float = 0.0, upper_bound: float = 1.0, preset: float = None):
        super().__init__(parent, row, label_text, check_float, float, lower_bound, upper_bound, preset)


class IntField(NumericField):

    def __init__(self, parent, row: int, label_text: str,
                 lower_bound: int = 0, upper_bound: int = 10, preset: float = None):
        super().__init__(parent, row, label_text, check_int, int, lower_bound, upper_bound, preset)
