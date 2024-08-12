
import tkinter as tk
from threading import Thread
from tkinter import ttk
from tkinter.messagebox import showinfo, showerror

from processtransformer.gui.custom_elements.common_settings import disable_button, common_padding_x, common_padding_y, \
    enable_button, min_col_width
from processtransformer.gui.custom_elements.fields import FloatField, Field, ReadOnlyField
from processtransformer.gui.custom_elements.file_in_out import XesInput, Output
from processtransformer.gui.custom_elements.separator import Separator
from processtransformer.ml.core.model import AugmentationExperiment, Dataset, SplitterConfiguration
from processtransformer.ml.prepare.splitter import AbstractSplitter
from processtransformer.ml.preprocess.data_preprocessing import preprocess


class PreProcessingTab(ttk.Frame):
    dialog_message = 'Preprocessing-Tab Message'

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.columnconfigure(0, weight=1, minsize=min_col_width)
        self.columnconfigure(1, weight=2, minsize=min_col_width)
        self.columnconfigure(2, weight=6, minsize=min_col_width)
        self.columns = 3
        self.num_rows = 0

        ReadOnlyField(self, self.num_rows, 'Inputs', '')
        self.num_rows += 1

        self.input = XesInput(self, row=self.num_rows, label_text='XES-File')
        self.input.grid(column=0, row=self.num_rows, columnspan=2, sticky=tk.W, padx=common_padding_x,
                        pady=common_padding_y)
        self.num_rows += 1

        Separator(self, self.num_rows, self.columns)
        self.num_rows += 1

        ReadOnlyField(self, self.num_rows, 'Configuration', '')
        self.num_rows += 1

        self.result_name_field = Field(self, self.num_rows, 'Name of resulting logs')
        self.num_rows += 1

        self.training_field = FloatField(self, self.num_rows, 'Training size [0.01, .., 0.99]', 0.01, 0.99)
        self.num_rows += 1

        self.splitter_label = ttk.Label(self, text="Select Splitter")
        self.splitter_label.grid(column=0, row=self.num_rows, sticky=tk.W, padx=common_padding_x, pady=common_padding_y)
        available_splitters = {splitter.format_id(): splitter for splitter in AbstractSplitter.__subclasses__()}
        self.splitter_var = tk.StringVar()
        self.splitter_var.set(list(available_splitters.keys())[0])
        self.splitter_combobox = ttk.Combobox(self, textvariable=self.splitter_var,
                                              values=list(available_splitters.keys()),
                                              state='readonly')
        self.splitter_combobox.config(width=30)
        self.splitter_combobox.grid(column=1, row=self.num_rows, sticky=tk.W, padx=common_padding_x,
                                    pady=common_padding_y)
        self.num_rows += 1

        Separator(self, self.num_rows, self.columns)
        self.num_rows += 1

        ReadOnlyField(self, self.num_rows, 'Output', '')
        self.num_rows += 1

        self.output = Output(self, self.num_rows)
        self.output.grid(columnspan=2, column=0, row=self.num_rows, sticky=tk.W, padx=common_padding_x,
                         pady=common_padding_y)
        self.num_rows += 1

        Separator(self, self.num_rows, self.columns)
        self.num_rows += 1

        self.preprocess_button = ttk.Button(self, text='Pre-process', command=self.on_preprocess_button_click,
                                            takefocus=True)
        self.preprocess_button.bind('<Return>', lambda x: self.on_preprocess_button_click())
        self.preprocess_button.grid(column=0, row=self.num_rows, sticky=tk.W, columnspan=2,
                                    padx=common_padding_x, pady=common_padding_y)
        self.num_rows += 1

    def on_preprocess_button_click(self):
        try:
            exp_name = self.result_name_field.get()
            assert exp_name is not None and exp_name != ''
            input_file = self.input.get()
            assert exp_name is not None and exp_name != ''
            splitter_name = self.splitter_combobox.get()
            assert exp_name is not None and exp_name != ''
            training_ratio = float(self.training_field.get())
        except Exception as e:
            showerror(title=self.dialog_message, message=f'Input invalid. Message: {e}')
            return

        exp = AugmentationExperiment(name=exp_name,
                                     data_dir='',
                                     run_dir='',
                                     evaluation_dir='',
                                     event_logs=[
                                         Dataset(name=exp_name,
                                                 file_path=input_file)
                                     ],
                                     splitter_configuration=SplitterConfiguration(
                                         name=splitter_name,
                                         training_size=training_ratio,
                                         by='first',
                                         seeds=[42, 567, 789],
                                         repetitions=3,
                                         folds=3
                                     ),
                                     min_pref_length=1,
                                     approaches=[],
                                     augmentation_strategies=[])

        disable_button(self.preprocess_button)
        thread = Thread(target=self._start_preprocessing, args=(exp,))
        thread.daemon = True
        thread.start()

    def _start_preprocessing(self, exp: AugmentationExperiment):
        try:
            preprocess(exp, True)
            showinfo(title=self.dialog_message, message='Pre-processed successfully')
        except Exception as e:
            showerror(title=self.dialog_message, message=f'Pre-processing failed. Message: {e}')
        finally:
            enable_button(self.preprocess_button)
