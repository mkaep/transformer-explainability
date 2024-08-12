
import tkinter as tk
import typing
from tkinter import ttk

from processtransformer.gui.custom_elements.common_settings import min_col_width
from processtransformer.gui.custom_elements.fields import ReadOnlyField, FloatField
from processtransformer.xai.event_eval import Evaluator


class EvaluatorTab(ttk.Frame):
    def __init__(self, parent, evaluator_cls: typing.Type[Evaluator], **eval_kwargs):
        super().__init__(parent)

        self.evaluator_cls = evaluator_cls
        self.eval_kwargs = eval_kwargs

        self.columnconfigure(0, weight=1, minsize=min_col_width)
        self.columnconfigure(1, weight=3, minsize=min_col_width)
        self.columnconfigure(2, weight=3, minsize=min_col_width)
        self.columns = 3
        self.num_rows = 0

        self.name_label = ReadOnlyField(self, self.num_rows, evaluator_cls.get_name(), '')
        self.num_rows += 1

        self.state_button_var = tk.BooleanVar()
        self.state_button = ttk.Checkbutton(self, text='Enabled', offvalue=False, onvalue=True,
                                            state=tk.ACTIVE, variable=self.state_button_var)
        self.state_button_var.set(True)
        self.state_button.grid(row=self.num_rows, column=0)
        self.num_rows += 1

        self.allowed_pred_delta_field = FloatField(self, self.num_rows, 'Allowed prediction delta', 0.0, 1.0, 0.25)
        self.num_rows += 1

    def get_evaluator_cls(self):
        return self.evaluator_cls

    def get_evaluator_kwargs(self) -> typing.Union[typing.Dict, bool]:
        """Returns dict, if enabled, otherwise returns False."""
        is_enabled = self.state_button_var.get()
        if is_enabled:
            return {'allowed_pred_delta': self.allowed_pred_delta_field.get()}
        return False
