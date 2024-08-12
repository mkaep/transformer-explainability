import argparse
import dataclasses
import json
import os
import time as tm
import typing

import numpy as np
import pandas as pd
import pm4py
import tensorflow as tf
from pm4py.objects.conversion.log import converter as log_converter
from sklearn import metrics as sk_metrics

import feature_extraction
import processtransformer.ml.evaluate.core_metrics as metrics
import processtransformer.util.compressor as compressor
import train_model
from processtransformer.constants import Task
from processtransformer import constants
from processtransformer.data_models.metrics_model import MetricsModel
from processtransformer.data_models.training_configuration import TrainingConfiguration
from processtransformer.util.trace_util import extract_traces_and_ypred_from_xes

parser = argparse.ArgumentParser(description="Process Transformer - Model Training.")
parser.add_argument("--training_config", type=str, required=True,
                    help='Path of the training file (data, parameters).')

dict_dir_name = 'dicts'
df_dir_name = 'df'


@dataclasses.dataclass
class SampleResult:
    id: str
    pred_next_act: typing.Optional[any]
    pred_next_role: typing.Optional[any]
    pred_next_time: typing.Optional[any]
    pred_suffix: typing.Optional[typing.List[any]]
    pred_remaining_time: typing.Optional[any]


@dataclasses.dataclass
class DetailedNextActivityResult:
    id: str
    output: typing.List[typing.List[typing.Union[str, float]]]

    def to_dict(self) -> typing.Dict[str, typing.Union[str, typing.List[typing.Union[str, float]]]]:
        return {
            'id': self.id,
            'output': self.output
        }


def save_additional_measurements(additional_measurements, result_dir):
    with open(os.path.join(result_dir, 'training_statistics.json'), 'w') as f:
        json.dump(additional_measurements, f, indent=4, sort_keys=False)


def save_test_report(test_results: typing.List[SampleResult], result_dir):
    test_report_df = pd.DataFrame(test_results)
    test_report_df.to_csv(os.path.join(result_dir, 'result.csv'), sep='\t', index=False)


def get_max_prefix_length(event_log_df: pd.DataFrame):
    # Extract max prefix length
    prefixes = []
    for _x in event_log_df['prefix'].values:
        prefixes.append(len(str(_x).split('#')))

    # Determine max prefix length of training and test data; note: in the original code only the maximal prefix length
    # of training data was used.
    return max(prefixes)


def find_in_dict(a, dictionary):
    for name, index in dictionary.items():
        if index == a:
            return name
    return 'NOT_FOUND'


def next_activity_prediction(training_configuration: TrainingConfiguration,
                             original_log, train_log, x_word_dict, y_word_dict, total_classes,
                             persist=True, build_metrics=True,
                             processed_df=None, train_processed_df=None,
                             **transformer_kwargs,
                             ):
    start_time = tm.time()

    print('Extracting features from original and train log - this may take a while...')

    if processed_df is None:
        processed_df = feature_extraction.DataProcessorNextActivity.feature_extraction(original_log)
    if train_processed_df is None:
        train_processed_df = feature_extraction.DataProcessorNextActivity.feature_extraction(train_log)

    if persist:
        df_dir = os.path.join(training_configuration.data_source.result_dir, df_dir_name)
        os.makedirs(df_dir, exist_ok=True)

        processed_df_filename = os.path.join(df_dir, "processed_df")
        train_processed_df_filename = os.path.join(df_dir, "train_processed_df")
        compressor.compress(processed_df, processed_df_filename)
        compressor.compress(train_processed_df, train_processed_df_filename)

    print(f'Done extracting features. Saved to {training_configuration.data_source.result_dir}')

    # Determine max prefix length as maximum of original and training data due to possible augmentation
    max_prefix_length_original = get_max_prefix_length(processed_df)
    max_prefix_length_train = get_max_prefix_length(train_processed_df)
    # +1 so we can also feed the last prediction back to the NN
    max_prefix_length = max(max_prefix_length_original, max_prefix_length_train) + 1
    end_time = tm.time()
    model_trainer = train_model.TrainNextActivityModel(train_processed_df, x_word_dict, y_word_dict,
                                                       max_prefix_length, total_classes, True)

    transformer_model, training_history, trainable_weights, n_training_samples, \
        training_duration, creating_training_samples_duration = \
        model_trainer.train_model(
            training_configuration.transformer_params.learning_rate, training_configuration.transformer_params.epochs,
            training_configuration.transformer_params.batch_size, training_configuration.data_source.result_dir,
            **transformer_kwargs, )

    # i.e. model has been trained.
    additional_measurements = {
        'elapsed_time_training': training_duration,
        'elapsed_time_training_samples_generation': creating_training_samples_duration + (end_time - start_time),
        'training_samples': n_training_samples,
        'trainable_weights_model': trainable_weights,
        'training_history': str(training_history.history),
    }

    if build_metrics:
        eval_model(additional_measurements, max_prefix_length, persist, training_configuration,
                   transformer_model, x_word_dict, y_word_dict)

    return transformer_model, x_word_dict, y_word_dict, processed_df, train_processed_df


def eval_model(additional_measurements, max_prefix_length, persist, training_configuration,
               transformer_model, x_word_dict, y_word_dict):
    # Evaluate model
    test_data_df = pm4py.read_xes(training_configuration.data_source.test_data)
    y_word_dict_swapped = {y_word_dict[k]: k for k in y_word_dict.keys()}
    test_results = []

    trace_ytrue_tuples = extract_traces_and_ypred_from_xes(training_configuration)

    traces = list(list(zip(*trace_ytrue_tuples))[0])
    y_true = list(list(zip(*trace_ytrue_tuples))[1])

    traces = [[x_word_dict[x] for x in trace] for trace in traces]
    traces = tf.keras.preprocessing.sequence.pad_sequences(traces, maxlen=max_prefix_length)
    y_true = np.asarray([y_word_dict[y] for y in y_true])

    y_pred = _ypred_from_model(traces, transformer_model)
    original_metrics = calculate_and_save_metrics(y_pred, y_true, training_configuration if persist else None)

    loaded_model = train_model.TrainNextActivityModel.load_model(training_configuration)
    y_pred_loaded = _ypred_from_model(traces, loaded_model)
    loaded_metrics = calculate_and_save_metrics(y_pred_loaded, y_true, training_configuration if persist else None)

    assert np.allclose(y_pred, y_pred_loaded), 'Original and loaded NN-model predictions are different!'

    if persist:
        # Save results
        save_test_report(test_results, training_configuration.data_source.result_dir)

    save_additional_measurements(additional_measurements, training_configuration.data_source.result_dir)


def _ypred_from_model(traces, transformer_model):
    y_pred = transformer_model.predict(traces)
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred


def save_detailed_next_activity_results(next_activity_results_detailed, train_config):
    with open(os.path.join(train_config.data_source.result_dir, 'detailed_next_activity_results.jsonl'),
              'w', encoding='utf8') as f:
        for res in next_activity_results_detailed:
            line = json.dumps(res.to_dict())
            f.write(f'{line}\n')


def calculate_and_save_metrics(y_pred, y_true, train_config=None):
    accuracy = sk_metrics.accuracy_score(y_true, y_pred)

    precision_micro = sk_metrics.precision_score(y_true, y_pred, average='micro', zero_division=0)
    precision_macro = sk_metrics.precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = sk_metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)

    recall_micro = sk_metrics.recall_score(y_true, y_pred, average='micro', zero_division=0)
    recall_macro = sk_metrics.recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = sk_metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0)

    f1_micro = sk_metrics.f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = sk_metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = sk_metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)

    metrics_model = MetricsModel(
        accuracy=accuracy,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
        recall_micro=recall_micro,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
    )
    print('Metrics:\n', metrics_model.to_dict())
    if train_config is not None:
        with open(os.path.join(train_config.data_source.result_dir, 'metrics.json'), 'w', encoding='utf8') as f:
            json.dump(metrics_model.to_dict(), f, indent=2)

    return metrics_model


def prepare_and_train(training_configuration: TrainingConfiguration,
                      persist: bool = True,
                      build_metrics: bool = True,
                      processed_df=None, train_processed_df=None,
                      **transformer_kwargs,
                      ):
    if persist:
        os.makedirs(training_configuration.data_source.result_dir, exist_ok=True)
        with open(os.path.join(training_configuration.data_source.result_dir, training_configuration.name + ".json"),
                  'w', encoding="utf8") as file2:
            json.dump(training_configuration.to_dict(), file2, indent=2)

    print('Training config: ', training_configuration.to_dict())

    # Load training data
    original_log = pm4py.read_xes(training_configuration.data_source.original_data)
    train_log = pm4py.read_xes(training_configuration.data_source.train_data)
    dataframe_df = log_converter.apply(original_log, variant=log_converter.Variants.TO_DATA_FRAME)
    if persist:
        os.makedirs(training_configuration.data_source.result_dir, exist_ok=True)
        dataframe_df.to_csv(os.path.join(training_configuration.data_source.result_dir, 'log.csv'))

    # Extract logs metadata
    x_word_dict, y_word_dict, vocab_size, total_classes = \
        feature_extraction.DataProcessor.extract_logs_metadata(original_log)
    if persist:
        # Save dicts
        dict_dir = os.path.join(training_configuration.data_source.result_dir, dict_dir_name)
        os.makedirs(dict_dir, exist_ok=True)
        compressor.compress(x_word_dict, os.path.join(dict_dir, "x_word_dict"))
        compressor.compress(y_word_dict, os.path.join(dict_dir, "y_word_dict"))

    if training_configuration.transformer_params.task in [constants.Task.NEXT_ACTIVITY, Task.NEXT_ACTIVITY]:
        return next_activity_prediction(training_configuration, original_log, train_log,
                                        x_word_dict, y_word_dict, total_classes,
                                        persist, build_metrics, processed_df, train_processed_df,
                                        **transformer_kwargs, )
    else:
        raise ValueError(f'Task {training_configuration.transformer_params.task} not available!')


def load_training_config(training_config_path) -> TrainingConfiguration:
    if training_config_path is None:
        raise ValueError('Missing argument! Provide either explaining-config or training-config.')
    with open(training_config_path, 'r', encoding="utf8") as file:
        training_configuration = TrainingConfiguration.from_dict(json.load(file))
    return training_configuration


def main():
    args = parser.parse_args()
    training_configuration = load_training_config(args.training_config)
    prepare_and_train(training_configuration)


if __name__ == '__main__':
    main()
