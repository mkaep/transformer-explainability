

"""
Copied from https://github.com/formazione/Azure-ttk-theme/blob/main/azure/example.py

"""
import argparse
import logging
import sys
import tkinter as tk
import ttkbootstrap as ttk
# from tkinter import ttk

from processtransformer.gui.custom_elements.common_settings import rel_padding, rel_widget_height, rel_widget_width
from processtransformer.gui.main_tabs.nn_training_tab import NNTrainingTab
from processtransformer.gui.main_tabs.pre_processing_tab import PreProcessingTab
from processtransformer.gui.custom_elements.scrollable_canvas import ScrollableCanvas
from processtransformer.gui.main_tabs.xai_tab import XaiTab


def center_screen():
    """ gets the coordinates of the center of the screen """
    global screen_height, screen_width, x_cordinate, y_cordinate

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_cordinate = int((screen_width / 2) - (window_width / 2))
    y_cordinate = int((screen_height / 2) - (window_height / 2))
    root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))


def main():
    parser = argparse.ArgumentParser(description="Main XAI-GUI parser")
    parser.add_argument("--log_level", type=str, required=False, help='Log level.',
                        choices=['debug', 'info', 'warn', 'error'])
    args = parser.parse_args()

    global root, window_width, window_height
    root = tk.Tk()
    root.title('XAI-Tool for PPM-Transformers')
    window_scale = 0.6
    window_width = int(root.winfo_screenwidth() * window_scale)
    window_height = int(root.winfo_screenheight() * window_scale)
    center_screen()

    notebook = ttk.Notebook(root)
    preprocessing_tab_wrapper = ScrollableCanvas(notebook, PreProcessingTab, borderwidth=0)
    preprocessing_tab_wrapper.place(relheight=rel_widget_height, relwidth=rel_widget_width)
    notebook.add(preprocessing_tab_wrapper, text='Pre-Processing')
    nn_training_tab_wrapper = ScrollableCanvas(notebook, NNTrainingTab)
    nn_training_tab_wrapper.place(relheight=rel_widget_height, relwidth=rel_widget_width)
    notebook.add(nn_training_tab_wrapper, text='Training')
    xai_notebook = XaiTab(notebook)
    xai_notebook.place(relheight=rel_widget_height, relwidth=rel_widget_width)
    notebook.add(xai_notebook, text='XAI')
    notebook.place(relx=rel_padding, rely=rel_padding, relwidth=rel_widget_width, relheight=rel_widget_height)

    log_level = args.log_level
    if log_level is None:
        get_trace = getattr(sys, 'gettrace', None)
        if get_trace():
            # In debugging mode -> set level to debug
            logging.basicConfig(level=logging.DEBUG)
    else:
        # Log level provided: use it
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {log_level}')
        logging.basicConfig(level=numeric_level)
    root.mainloop()


if __name__ == '__main__':
    main()
