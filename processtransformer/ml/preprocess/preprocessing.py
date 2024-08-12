import copy

import pm4py
import typing
from abc import ABC
from pm4py.objects.log.obj import EventLog


class PreprocessingStep(ABC):
    def run(self, event_log: EventLog) -> EventLog:
        raise NotImplementedError()

# Weitere Ideen: Entfernen von Schleifen, bestimmten Tracevarianten, Duplikate, Noise, etc. Remove Duplikates


class RemoveEmptyTraces(PreprocessingStep):
    def run(self, event_log: EventLog) -> EventLog:
        print(f' # Removing empty traces from event log...')
        correct_cases = set()
        for trace in event_log:
            if len(trace) > 0:
                correct_cases.add(trace.attributes['concept:name'])
        return pm4py.filter_trace_attribute_values(event_log, 'concept:name', correct_cases)


class RemoveTracesWithMissingAttributes(PreprocessingStep):
    def __init__(self):
        super().__init__()
        self._attributes = ['concept:name', 'org:resource']

    def run(self, event_log: EventLog) -> EventLog:
        print(f' # Removing traces with missing {self._attributes} values')
        preprocessed_log = event_log.__deepcopy__()
        correct_cases = set()
        for trace in preprocessed_log:
            missing_values = False
            case_id = trace.attributes['concept:name']
            for event in trace:
                try:
                    for attribute in self._attributes:
                        _ = event[attribute]
                except KeyError:
                    print(f'\t Remove trace with id {case_id} due to missing attribute')
                    missing_values = True
                    break
            if missing_values is False:
                correct_cases.add(case_id)

        return pm4py.filter_trace_attribute_values(preprocessed_log, 'concept:name', correct_cases)


class Preprocessor:
    ALL_STEPS = {
        'remove_empty_traces': RemoveEmptyTraces,
        'remove_traces_with_missing_values': RemoveTracesWithMissingAttributes
    }

    def __init__(self, steps: typing.List[PreprocessingStep]) -> None:
        super().__init__()
        self._steps = steps

    def run_preprocessing(self, event_log: EventLog) -> EventLog:
        preprocessed_event_log = event_log.__deepcopy__()
        for step in self._steps:
            preprocessed_event_log = step.run(preprocessed_event_log)
        return preprocessed_event_log

    @staticmethod
    def build(step_short_names: typing.List[str]) -> 'Preprocessor':
        all_steps = [Preprocessor.ALL_STEPS[step_name]() for step_name in step_short_names]
        return Preprocessor(all_steps)

    @staticmethod
    def get_available_preprocessing_steps() -> typing.List:
        return list(Preprocessor.ALL_STEPS.keys())

