

import typing

import pm4py

from processtransformer.data_models.training_configuration import TrainingConfiguration
from processtransformer.util.types import Trace, YEvent


def extract_traces_from_xes(path_or_training_config: typing.Union[str, TrainingConfiguration],
                            ) -> typing.List[Trace]:
    if isinstance(path_or_training_config, TrainingConfiguration):
        path = path_or_training_config.data_source.test_data
    else:
        path = path_or_training_config

    test_data_df = pm4py.read_xes(path)

    # Groups by case, then makes list out of traces
    return [group['concept:name'].to_list() for _, group in test_data_df.groupby('case:concept:name')]


def extract_traces_and_ypred_from_xes(path_or_training_config: typing.Union[str, TrainingConfiguration],
                                      ) -> typing.List[typing.Tuple[Trace, YEvent]]:
    traces = extract_traces_from_xes(path_or_training_config)
    # Tuples of (trace, y_pred)
    return [(trace[:-1], trace[-1]) for trace in traces]
