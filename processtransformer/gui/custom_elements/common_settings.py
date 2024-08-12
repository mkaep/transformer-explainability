
import os
import tkinter as tk

import re

rel_padding = 0.01
rel_widget_height = 1.0 - 2 * rel_padding
rel_widget_width = 1.0 - 2 * rel_padding

# Change this, if you have a 2k or 4k (or whatever) screen resolution
res_x = 1920
res_y = 1080
common_padding_x = int(10 / 1920 * res_x)
common_padding_y = int(10 / 1080 * res_y)
common_width = int(150 / 1920 * res_x)
common_height = int(25 / 1080 * res_y)
left_col = int(300 / 1920 * res_x)
min_col_width = int(150 / 1920 * res_x)

scrollbar_size = int(20 / 1080 * res_y)

EMPTY = 'Empty'

# TODO: change paths to user directory
root_dir = os.path.join(os.getcwd(), '..', )
training_dir = os.path.join(root_dir, 'training', 'training_026', )
preprocessed_dir = os.path.join(root_dir, 'PLG2', 'generated', '12_long_running_dependency3', )
explaining_dir = os.path.join(root_dir, 'explaining', 'long_running_dep3', 'long_running_dep3_012', )


def enable_button(button: tk.Button):
    button['state'] = 'enabled'


def disable_button(button: tk.Button):
    button['state'] = 'disabled'


def check_float(num: str):
    return check_num(num, r'[+-]?\d*.\d*', float)


def check_int(num: str):
    return check_num(num, r'[+-]?\d+', int)


def check_num(num: str, regex: str, conversion):
    if num in ['', '+', '-']:
        return True
    try:
        if re.split(regex, num) is None:
            return False

        conversion(num)  # try conversion
        return True
    except ValueError:
        # Hint: Do not use finally here as this is executed regardless
        # of whether you return in the try-block or not.
        return False


def check_and_set_path(parent_dir, child_path, input_field):
    preprocessed = os.path.join(parent_dir, child_path)
    new_path = EMPTY
    if os.path.exists(preprocessed):
        new_path = preprocessed
    input_field.set(new_path)
