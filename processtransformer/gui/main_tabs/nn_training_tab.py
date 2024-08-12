

import tkinter as tk
from threading import Thread
from tkinter import ttk
from tkinter.messagebox import showinfo, showerror

import processtransformer.constants
from main import prepare_and_train
from processtransformer.data_models.training_configuration import TrainingConfiguration, DataSource, \
    TransformerParameters
from processtransformer.gui.custom_elements.common_settings import min_col_width, common_padding_x, common_padding_y, \
    disable_button, enable_button, check_and_set_path
from processtransformer.gui.custom_elements.fields import Field, FloatField, IntField, ReadOnlyField
from processtransformer.gui.custom_elements.file_in_out import XesInput, DirInput, Output
from processtransformer.gui.custom_elements.separator import Separator


class NNTrainingTab(ttk.Frame):
    dialog_message = 'Training-Tab Message'

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.columnconfigure(0, weight=1, minsize=min_col_width)
        self.columnconfigure(1, weight=2, minsize=min_col_width)
        self.columnconfigure(2, weight=6, minsize=min_col_width)
        self.columns = 3
        self.num_rows = 0

        ReadOnlyField(self, self.num_rows, 'Inputs', '')
        self.num_rows += 1

        self.input = DirInput(self, row=self.num_rows, label_text='Open pre-processing directory',
                              on_set_callback=self._find_train_within_dir)
        self.input.grid(column=0, row=self.num_rows, columnspan=2, sticky=tk.W,
                        padx=common_padding_x, pady=common_padding_y)
        self.num_rows += 1

        self.preprocessed_xes_input = XesInput(self, row=self.num_rows, label_text='Open preprocessed.xes')
        self.preprocessed_xes_input.grid(column=0, row=self.num_rows, columnspan=2, sticky=tk.W,
                                         padx=common_padding_x, pady=common_padding_y)
        self.num_rows += 1

        self.train_xes_input = XesInput(self, row=self.num_rows, label_text='Open train.xes')
        self.train_xes_input.grid(column=0, row=self.num_rows, columnspan=2, sticky=tk.W,
                                  padx=common_padding_x, pady=common_padding_y)
        self.num_rows += 1

        self.test_pref_input = XesInput(self, row=self.num_rows, label_text='Open test_pref.xes')
        self.test_pref_input.grid(column=0, row=self.num_rows, columnspan=2, sticky=tk.W,
                                  padx=common_padding_x, pady=common_padding_y)
        self.num_rows += 1

        Separator(self, self.num_rows, self.columns)
        self.num_rows += 1

        ReadOnlyField(self, self.num_rows, 'Configuration', '')
        self.num_rows += 1

        self.task_field = ReadOnlyField(self, self.num_rows, 'Task', 'Next Activity')
        self.num_rows += 1

        self.result_name_field = Field(self, self.num_rows, 'Name of resulting training-dir')
        self.num_rows += 1

        self.epochs_field = IntField(self, self.num_rows, 'Number of epochs [1, .., 1e9]', 1, int(1e9), preset=10)
        self.num_rows += 1

        self.batch_size_field = IntField(self, self.num_rows, 'Batch size [1, .., 1e6]', 1, int(1e6), preset=12)
        self.num_rows += 1

        self.learning_rate_field = FloatField(self, self.num_rows, 'Learning rate [1e-6, .., 1.0]', 1e-6, 1.0,
                                              preset=0.001)
        self.num_rows += 1

        self.gpu_id_field = IntField(self, self.num_rows, 'GPU ID [0, .., 1e6]', 0, int(1e6), preset=0)
        self.num_rows += 1

        Separator(self, self.num_rows, self.columns)
        self.num_rows += 1

        ReadOnlyField(self, self.num_rows, 'Output', '')
        self.num_rows += 1

        self.output = Output(self, self.num_rows)
        self.output.grid(columnspan=2, column=0, row=self.num_rows, sticky=tk.W,
                         padx=common_padding_x, pady=common_padding_y)
        self.num_rows += 1

        Separator(self, self.num_rows, self.columns)
        self.num_rows += 1

        self.training_button = ttk.Button(self, text='Train NN', command=self._on_training_button_click,
                                          takefocus=True)
        self.training_button.bind('<Return>', lambda x: self._on_training_button_click())
        self.training_button.grid(column=0, row=self.num_rows, sticky=tk.W, columnspan=2,
                                  padx=common_padding_x, pady=common_padding_y)
        self.num_rows += 1

    def _find_train_within_dir(self, train_dir: str):
        check_and_set_path(train_dir, 'preprocessed.xes', self.preprocessed_xes_input)
        check_and_set_path(train_dir, 'train.xes', self.train_xes_input)
        check_and_set_path(train_dir, 'test_pref.xes', self.test_pref_input)

    def _on_training_button_click(self):
        try:
            name = self.result_name_field.get()
            original_data = self.preprocessed_xes_input.get()
            train_data = self.train_xes_input.get()
            test_data = self.test_pref_input.get()
            result_dir = self.output.get()

            epochs = int(self.epochs_field.get())
            batch_size = int(self.batch_size_field.get())
            learning_rate = float(self.learning_rate_field.get())
            gpu_id = int(self.gpu_id_field.get())
        except Exception as e:
            showerror(title=self.dialog_message, message=f'Input invalid. Message: {e}')
            return

        train_config = TrainingConfiguration(name,
                                             None,
                                             DataSource(
                                                 original_data,
                                                 train_data,
                                                 test_data,
                                                 result_dir
                                             ),
                                             TransformerParameters(
                                                 processtransformer.constants.Task.NEXT_ACTIVITY,
                                                 epochs,
                                                 batch_size,
                                                 learning_rate,
                                                 gpu_id
                                             ))
        disable_button(self.training_button)
        thread = Thread(target=self._start_training, args=(train_config,))
        thread.daemon = True
        thread.start()

    def _start_training(self, train_config: TrainingConfiguration):
        try:
            prepare_and_train(train_config)
            showinfo(title=self.dialog_message, message='Trained successfully')
        except Exception as e:
            showerror(title=self.dialog_message, message=f'Training failed. Message: {e}')
        finally:
            enable_button(self.training_button)
