from abc import ABC

import pandas as pd
from pm4py.objects.log.obj import EventLog, Trace
from processtransformer.ml.core import model
from processtransformer.ml.prepare import splitter
import os
import pm4py
from processtransformer.ml.analysis import statistical_analysis
import typing


class TestSampleCreator(ABC):
    def create_samples(self, event_log: EventLog) -> typing.Tuple[EventLog, EventLog]:
        raise NotImplementedError()


class PrefixCreator(TestSampleCreator):
    def __init__(self, min_prefix_length: int):
        super().__init__()
        self.min_prefix_length = min_prefix_length

    def create_samples(self, event_log: EventLog) -> typing.Tuple[EventLog, EventLog]:
        prefix_log = EventLog()
        suffix_log = EventLog()

        for i, trace in enumerate(event_log):
            if len(trace) >= self.min_prefix_length:
                for j in range(self.min_prefix_length, len(trace)):
                    prefix_log.append(Trace(trace[:j], attributes={
                        'concept:name': f'Case_{i}_{j}',
                        'creator': PrefixCreator.get_name()
                    }))
                    suffix_log.append(Trace(trace[j:], attributes={
                        'concept:name': f'Case_{i}_{j}',
                        'creator': PrefixCreator.get_name()
                    }))
        return prefix_log, suffix_log

    @staticmethod
    def get_name() -> str:
        return "PrefixCreator"


def create_ground_truth_file(prefix_log: EventLog, suffix_log: EventLog) -> pd.DataFrame:
    pref_case_ids = statistical_analysis.Statistic.get_case_ids(prefix_log)
    suf_case_ids = statistical_analysis.Statistic.get_case_ids(suffix_log)
    assert pref_case_ids == suf_case_ids

    ground_truth_results = []
    for prefix_trace, suffix_trace in zip(prefix_log, suffix_log):
        assert prefix_trace.attributes['concept:name'] == suffix_trace.attributes['concept:name']

        case_id = prefix_trace.attributes['concept:name']
        next_activity = suffix_trace[0]['concept:name']

        # TODO das müsste irgendwie noch Rolle werden; da die jedoch nicht eindeutig sind wird dies schwierig
        try:
            next_role = suffix_trace[0]['org:resource']
        except KeyError:
            next_role = None

        suffix = []
        for event in suffix_trace:
            suffix.append(event['concept:name'])

        remaining_time = (suffix_trace[-1]['time:timestamp'] - prefix_trace[-1][
            'time:timestamp']).total_seconds() / 86400
        next_time = (suffix_trace[0]['time:timestamp'] - prefix_trace[-1]['time:timestamp']).total_seconds() / 86400

        ground_truth_results.append(model.SampleResult(case_id, next_activity, next_role, next_time, suffix,
                                                       remaining_time))
    return pd.DataFrame(ground_truth_results)


def create_test_samples(fold: splitter.SplitResult, min_pref_length: int, target_directory: str,
                        verbose: bool = False) -> typing.Tuple[str, str, str]:
    if verbose:
        print(f' Storing event log on disk')

    prefix_creator = PrefixCreator(min_pref_length)
    prefix_log, suffix_log = prefix_creator.create_samples(fold.test_log)

    test_pref_file = os.path.join(target_directory, f'test_pref.xes')
    test_suf_file = os.path.join(target_directory, f'test_suf.xes')
    pm4py.write_xes(prefix_log, test_pref_file)
    pm4py.write_xes(suffix_log, test_suf_file)

    # Build ground truth result file
    ground_truth_df = create_ground_truth_file(prefix_log, suffix_log)
    ground_truth_file = os.path.join(target_directory, 'true_result.csv')
    ground_truth_df.to_csv(ground_truth_file, sep='\t', index=False)

    return test_pref_file, test_suf_file, ground_truth_file
