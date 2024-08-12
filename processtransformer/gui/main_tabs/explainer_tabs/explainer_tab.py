
import abc
import tkinter as tk
import typing
from abc import ABCMeta, abstractmethod
from threading import Thread
from tkinter import ttk
from tkinter.messagebox import showinfo, showerror

from processtransformer.data_models.explaining_model import ExplainingModel
from processtransformer.gui.custom_elements.common_settings import min_col_width, common_padding_x, common_padding_y, \
    check_and_set_path, disable_button, enable_button, training_dir, preprocessed_dir, explaining_dir
from processtransformer.gui.custom_elements.fields import Field, ReadOnlyField
from processtransformer.gui.custom_elements.file_in_out import XesInput, DirInput, Output
from processtransformer.gui.custom_elements.image_container import ImageContainer
from processtransformer.gui.custom_elements.separator import Separator
from processtransformer.gui.custom_elements.text_container import TextContainer
from processtransformer.xai.explainer import Explainer
from processtransformer.xai.main import explain
from processtransformer.xai.visualization.viz_funcs.base_viz import FigureOutput, TextOutput


class ExplainerTab(ttk.Frame, metaclass=ABCMeta):
    dialog_message = 'Message from XAI tab'

    def __init__(self, notebook, explainer: Explainer, resize_grid_callback, **kwargs):
        super().__init__(notebook, **kwargs)

        self.resize_grid_callback = resize_grid_callback

        self.columnconfigure(0, weight=1, minsize=min_col_width)
        self.columnconfigure(1, weight=2, minsize=min_col_width)
        self.columnconfigure(2, weight=6, minsize=min_col_width)
        self.columns = 3
        self.num_rows = 0

        self.name_label = ReadOnlyField(self, row=self.num_rows, label_text=explainer.get_name(),
                                        entry_text_ro='')
        self.num_rows += 1

        Separator(self, self.num_rows, self.columns)
        self.num_rows += 1

        ReadOnlyField(self, self.num_rows, 'Inputs', '')
        self.num_rows += 1

        self.input = DirInput(self, row=self.num_rows, label_text='Open training directory',
                              on_set_callback=self._find_dirs, preset_dir=training_dir)
        self.input.grid(column=0, row=self.num_rows, columnspan=2, sticky=tk.W,
                        padx=common_padding_x, pady=common_padding_y)
        self.num_rows += 1

        self.nn_model_input = DirInput(self, row=self.num_rows, label_text='Open NN-directory')
        self.nn_model_input.grid(column=0, row=self.num_rows, columnspan=2, sticky=tk.W,
                                 padx=common_padding_x, pady=common_padding_y)
        self.num_rows += 1

        self.nn_dicts_input = DirInput(self, row=self.num_rows, label_text='Open NN-dictionaries')
        self.nn_dicts_input.grid(column=0, row=self.num_rows, columnspan=2, sticky=tk.W,
                                 padx=common_padding_x, pady=common_padding_y)
        self.num_rows += 1

        Separator(self, self.num_rows, self.columns)
        self.num_rows += 1

        ReadOnlyField(self, self.num_rows, 'Result-name and traces', '')
        self.num_rows += 1

        self.result_name_field = Field(self, self.num_rows, 'Name of resulting XAI-directory')
        self.num_rows += 1

        trace_support = explainer.get_trace_support()

        self.event_log_input = XesInput(self, row=self.num_rows, label_text='Open Event Log',
                                        preset_dir=preprocessed_dir)
        self.event_log_input.grid(column=0, row=self.num_rows, columnspan=2, sticky=tk.W,
                                  padx=common_padding_x, pady=common_padding_y)
        if not (trace_support.multi_trace or trace_support.multi_with_single_trace):
            self.event_log_input.disable()
        self.num_rows += 1

        self.single_trace_field = Field(self, self.num_rows, 'Enter single trace')
        if not (trace_support.single_trace or trace_support.multi_with_single_trace):
            self.single_trace_field.disable()
        self.num_rows += 1

        Separator(self, self.num_rows, self.columns)
        self.num_rows += 1

        # Elements of child classes go here:
        # Start of child elements
        self._layout_child_inputs()
        # End of child elements

        ReadOnlyField(self, self.num_rows, 'Output', '')
        self.num_rows += 1

        self.result_dir_path = Output(self, row=self.num_rows, preset_dir=explaining_dir)
        self.result_dir_path.grid(column=0, row=self.num_rows, columnspan=2, sticky=tk.W,
                                  padx=common_padding_x, pady=common_padding_y)
        self.num_rows += 1

        Separator(self, self.num_rows, self.columns)
        self.num_rows += 1

        self.explaining_button = ttk.Button(self, text='Run XAI', command=self._on_explaining_button_click,
                                            takefocus=True)
        self.explaining_button.bind('<Return>', lambda x: self._on_explaining_button_click())
        self.explaining_button.grid(column=0, row=self.num_rows, sticky=tk.W, columnspan=2,
                                    padx=common_padding_x, pady=common_padding_y)
        self.num_rows += 1

        self.xai_results = []

    def _get_common_args(self) -> typing.Tuple[str, str, str, str]:
        """Returns nn_model_dir, nn_dicts_dir, result_name, result_dir"""
        nn_model_dir = self.nn_model_input.get()
        nn_dicts_dir = self.nn_dicts_input.get()
        result_name = self.result_name_field.get()
        result_dir = self.result_dir_path.get()
        return nn_model_dir, nn_dicts_dir, result_name, result_dir

    def _find_dirs(self, parent_dir: str):
        check_and_set_path(parent_dir, 'model', self.nn_model_input)
        check_and_set_path(parent_dir, 'dicts', self.nn_dicts_input)

    @classmethod
    @abc.abstractmethod
    def get_explainer_cls(cls):
        pass

    @abc.abstractmethod
    def _layout_child_inputs(self):
        pass

    def _on_explaining_button_click(self):
        self._clear_previous_results()

        try:
            # Common parameters
            nn_model_dir, nn_dicts_dir, result_name, result_dir = self._get_common_args()
            event_log_path = self.event_log_input.get()
            single_trace = self.single_trace_field.get()
            single_trace = single_trace.split(',')
            single_trace = [event.strip() for event in single_trace]

            xai_model = ExplainingModel(result_name, event_log_path, single_trace,
                                        nn_model_dir, nn_dicts_dir, result_dir,
                                        self.get_explainer_cls(),
                                        dict())
            # Specialized parameters
            self._core_on_explaining_button_click(xai_model)
        except Exception as e:
            showerror(title=self.dialog_message, message=f'Input invalid. Message: {e}')
            return

        self._start_xai_computation(xai_model)

    def _clear_previous_results(self):
        for previous_result in self.xai_results:
            previous_result.grid_remove()
            previous_result.destroy()
        self.num_rows -= len(self.xai_results)
        self.xai_results = []
        self._update_grid()

    def _update_grid(self):
        self.resize_grid_callback()

    def _display_xai_outputs(self, viz_outputs):
        for out in viz_outputs:
            container = None

            if isinstance(out, FigureOutput):
                container = ImageContainer(self, self.num_rows, out.path_to_figure, column_span=self.columns)
            elif isinstance(out, TextOutput):
                container = TextContainer(self, self.num_rows, self.columns, out.text)

            if container is None:
                continue
            self.xai_results.append(container)
            self.num_rows += 1

        self._update_grid()

    def _start_explaining(self, xai_model: ExplainingModel):
        try:
            viz_outputs = explain(xai_model)
            self._display_xai_outputs(viz_outputs)
            showinfo(title=self.dialog_message, message='XAI finished successfully')
        except Exception as e:
            showerror(title=self.dialog_message, message=f'XAI failed. Message: {e}')
        finally:
            enable_button(self.explaining_button)

    @abstractmethod
    def _core_on_explaining_button_click(self, xai_model: ExplainingModel):
        pass

    def _start_xai_computation(self, xai_model: ExplainingModel):
        disable_button(self.explaining_button)
        thread = Thread(target=self._start_explaining, args=(xai_model,))
        thread.daemon = True
        thread.start()
