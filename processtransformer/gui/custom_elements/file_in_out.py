

import abc
import tkinter as tk
from tkinter import ttk, filedialog as filedialog
from tkinter.ttk import Frame
from typing import Callable, Optional, Any

from processtransformer.gui.custom_elements.common_settings import common_padding_x, common_padding_y, EMPTY


class GeneralInputOutput(Frame, abc.ABC):
    def __init__(self, parent, row: int, label_text: str, button_text: str,
                 on_set_callback: Optional[Callable[[str], Any]] = None,
                 empty_value_raises_error=True,
                 preset_dir: str = None,
                 ):
        super().__init__(parent)

        self.input_label = ttk.Label(parent, text=label_text)
        self.input_label.grid(column=0, row=row, sticky=tk.W, padx=common_padding_x, pady=common_padding_y)

        self.input_button = ttk.Button(parent, text=button_text, command=self._open_path, takefocus=True)
        self.input_button.bind('<Return>', lambda x: self._open_path())
        self.input_button.grid(column=1, row=row, sticky=tk.W, padx=common_padding_x, pady=common_padding_y)

        self.dir_var = tk.StringVar(value=EMPTY)
        self.text_label = ttk.Label(parent, textvariable=self.dir_var)
        self.text_label.grid(column=2, row=row, sticky=tk.W, padx=common_padding_x, pady=common_padding_y)

        self.on_set_callback = on_set_callback
        self.empty_value_raises_error = empty_value_raises_error

        self.preset_dir = preset_dir

    def _open_path(self):
        path_by_user: str = self._get_path()
        if path_by_user == '':
            path_by_user = EMPTY

        self.dir_var.set(path_by_user)
        if self.on_set_callback is not None:
            self.on_set_callback(self.dir_var.get())

    @abc.abstractmethod
    def _get_path(self):
        pass

    def get(self):
        path_var = self.dir_var.get()
        if path_var == EMPTY:
            if self.empty_value_raises_error:
                raise ValueError('No path (file or directory) chosen')
            else:
                return None
        return path_var

    def set(self, path_text: str):
        self.dir_var.set(path_text)

    def disable(self):
        self.input_button.config(state='disabled')


class Output(GeneralInputOutput):
    def __init__(self, parent, row: int, label_text: str = None, button_text: str = None,
                 on_set_callback: Optional[Callable[[str], Any]] = None,
                 empty_value_raises_error=True,
                 preset_dir: str = None,
                 ):
        if label_text is None:
            label_text = 'Output directory'
        if button_text is None:
            button_text = 'Click to choose directory'
        super().__init__(parent, row, label_text, button_text, on_set_callback,
                         empty_value_raises_error, preset_dir)

    def _get_path(self):
        dir_name = filedialog.askdirectory(initialdir=self.preset_dir)
        if dir_name is None:
            return EMPTY
        return dir_name


class FileInput(GeneralInputOutput):
    def __init__(self, parent, row: int, label_text: str, button_text: str,
                 on_set_callback: Optional[Callable[[str], Any]] = None,
                 empty_value_raises_error=True,
                 preset_dir: str = None,
                 ):
        super().__init__(parent, row, label_text, button_text, on_set_callback,
                         empty_value_raises_error, preset_dir)

    @classmethod
    @abc.abstractmethod
    def _get_accepted_files(cls):
        pass

    def _get_path(self):
        result = filedialog.askopenfile(mode='r', filetypes=self._get_accepted_files(), initialdir=self.preset_dir)
        if result is None:
            return EMPTY
        return result.name


class XesInput(FileInput):
    def __init__(self, parent, row: int, label_text: str = None, button_text: str = None,
                 on_set_callback: Optional[Callable[[str], Any]] = None,
                 empty_value_raises_error=True,
                 preset_dir: str = None,
                 ):
        if label_text is None:
            label_text = 'Open XES file'
        if button_text is None:
            button_text = 'Click to open XES file'
        super().__init__(parent, row, label_text, button_text, on_set_callback,
                         empty_value_raises_error, preset_dir)

    @classmethod
    def _get_accepted_files(cls):
        return [('XES-logs', '*.xes')]


class BpmnInput(FileInput):

    def __init__(self, parent, row: int, label_text: str = None, button_text: str = None,
                 on_set_callback: Optional[Callable[[str], Any]] = None,
                 empty_value_raises_error=True,
                 preset_dir: str = None,
                 ):
        if label_text is None:
            label_text = 'Open BPMN file'
        if button_text is None:
            button_text = 'Click to open BPMN file'
        super().__init__(parent, row, label_text, button_text, on_set_callback,
                         empty_value_raises_error, preset_dir)

    @classmethod
    def _get_accepted_files(cls):
        return [('BPMN-model', '*.bpmn')]


class DirInput(GeneralInputOutput):

    def __init__(self, parent, row: int, label_text: str = None, button_text: str = None,
                 on_set_callback: Optional[Callable[[str], Any]] = None,
                 empty_value_raises_error=True,
                 preset_dir: str = None,
                 ):
        if label_text is None:
            label_text = 'Open directory'
        if button_text is None:
            button_text = 'Click to open directory'
        super().__init__(parent, row, label_text, button_text, on_set_callback,
                         empty_value_raises_error, preset_dir)

    def _get_path(self):
        dir_name = filedialog.askdirectory(initialdir=self.preset_dir)
        if dir_name is None:
            return EMPTY
        return dir_name
