

import os
import sys
import time
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils.layer_utils import count_params
from sklearn import utils

from processtransformer.data_models.training_configuration import TrainingConfiguration
from processtransformer.models import transformer
from processtransformer.models.transformer import Transformer
from processtransformer.util.compressor import compress


class TrainNextActivityModel:
    model_directory = "model"

    def __init__(self, train_log_df: pd.DataFrame, x_word: typing.Dict, y_word: typing.Dict, max_prefix_length: int,
                 n_total_classes: int, shuffle=True):
        self.train_log_df = train_log_df
        self.x_word = x_word
        self.y_word = y_word
        self.max_prefix_length = max_prefix_length
        self.n_total_classes = n_total_classes
        self.shuffle = shuffle
        self.token_x = []
        self.token_y = []
        self.preparing_training_samples_duration = -1

        self.create_input()

    def create_input(self):
        start_time = time.time()
        x = self.train_log_df['prefix'].values
        y = self.train_log_df['next_act'].values
        if self.shuffle:
            x, y = utils.shuffle(x, y)

        for _x in x:
            self.token_x.append([self.x_word.get(s, 0) for s in str(_x).split('#')])

        for _y in y:
            self.token_y.append(self.y_word[_y])

        self.token_x = tf.keras.preprocessing.sequence.pad_sequences(self.token_x, maxlen=self.max_prefix_length)
        self.token_x = np.array(self.token_x, dtype=np.float32)
        self.token_y = np.array(self.token_y, dtype=np.float32)
        end_time = time.time()
        self.preparing_training_samples_duration = end_time - start_time

    def train_model(self, learning_rate: float, epochs: int, batch_size: int, result_dir: str, load_if_avail=False,
                    **transformer_kwargs,
                    ) -> typing.Tuple[tf.keras.models.Model, typing.List, int, int, float, float]:
        transformer_model = transformer.get_next_activity_model(
            max_case_length=self.max_prefix_length,
            vocab_size=len(self.x_word),
            output_dim=self.n_total_classes,
            **transformer_kwargs, )

        transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        tf_filepath = os.path.join(result_dir, self.model_directory)
        if load_if_avail and Path(tf_filepath).is_dir():
            print('Loading model')
            transformer_model = self.load_model(tf_filepath)
            end_time = 0
            start_time = 0
            training_history = None
        else:
            print('Fitting model')
            training_history, start_time, end_time = self._fit_model(transformer_model, tf_filepath,
                                                                     epochs, batch_size)

        # transformer_model.summary()  # moved down here, otherwise we get a build error
        trainable_weights = count_params(transformer_model.trainable_weights)

        n_training_samples = len(self.token_y)

        return transformer_model, training_history, trainable_weights, n_training_samples, end_time - start_time, \
            self.preparing_training_samples_duration

    @staticmethod
    def load_model(path_or_config: typing.Union[TrainingConfiguration, str]) -> Transformer:
        if not isinstance(path_or_config, str):
            filepath = os.path.join(path_or_config.data_source.result_dir, TrainNextActivityModel.model_directory)
        else:
            filepath = path_or_config

        tf_model = tf.keras.models.load_model(os.path.join(filepath, TrainNextActivityModel.model_directory),
                                              custom_objects={'Transformer': Transformer})

        return tf_model

    def _fit_model(self, transformer_model, filepath, epochs, batch_size):
        start_time = time.time()
        # Check if in debugging mode
        get_trace = getattr(sys, 'gettrace', None)
        if get_trace():
            print("In debugging mode - running TensorFlow eagerly (i.e. in Python)")
            # tf.config.run_functions_eagerly(True)
        training_history = transformer_model.fit(self.token_x, self.token_y, epochs=epochs, batch_size=batch_size,
                                                 shuffle=True, verbose=2)
        end_time = time.time()

        self.save_tf_model(filepath, transformer_model)

        return training_history, start_time, end_time

    @staticmethod
    def save_tf_model(filepath, transformer_model, model_dir_name=None):
        if model_dir_name is None:
            model_dir_name = TrainNextActivityModel.model_directory

        transformer_model.save(os.path.join(filepath, model_dir_name))
        compress(transformer_model.get_config(), TrainNextActivityModel._get_tf_config_path(filepath))

    @classmethod
    def _get_tf_config_path(cls, filepath):
        return os.path.join(filepath, 'tf_config')
