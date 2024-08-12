import typing
from typing import Union, Dict, List

from pm4py.objects.log.obj import EventLog, Trace
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.filtering.log.attributes import attributes_filter

from processtransformer.ml.analysis.statistical_analysis import Statistic

T = typing.TypeVar('T', bound=typing.Any)


class AnalysisStep(typing.Generic[T]):
    def run(self, event_log: EventLog) -> T:
        raise NotImplementedError()


class ActivityDistributionAnalysis(AnalysisStep):
    def run(self, event_log: EventLog) -> typing.Dict[str, int]:
        activity_distribution = {k: 0 for k in attributes_filter.get_attribute_values(event_log, 'concept:name').keys()}
        for trace in event_log:
            for event in trace:
                key = event['concept:name']
                activity_distribution[key] = activity_distribution[key] + 1
        return activity_distribution


class TraceLengthAnalysis(AnalysisStep):
    def run(self, event_log: EventLog) -> typing.Dict[int, int]:
        length_distribution = {len(k): 0 for k in event_log}
        for trace in event_log:
            length_distribution[len(trace)] = length_distribution[len(trace)] + 1
        return length_distribution


class TraceDistributionOverTime(AnalysisStep):
    def run(self, event_log: EventLog) -> None:
        # TO DO: Was passiert bei leerer Trace?
        case_density = {k[0]['time:timestamp'].date(): 0 for k in event_log}
        for trace in event_log:
            extracted_date = trace[0]['time:timestamp'].date()
            case_density[extracted_date] = case_density[extracted_date] + 1
        print(case_density)


class TraceVariantAnalysis(AnalysisStep):
    def run(self, event_log: EventLog) -> Union[Dict[List[str], List[Trace]], Dict[str, List[Trace]]]:
        return variants_filter.get_variants(event_log)


class EventLogDescriptor:
    ALL_STEPS = {
        'number_cases': Statistic.get_number_of_traces,
        'number_events': Statistic.get_number_of_events,
        'number_activities': Statistic.get_number_of_activities,
        'avg_case_length': Statistic.get_avg_case_length,
        'max_case_length': Statistic.get_max_trace_length,
        'min_case_length': Statistic.get_min_trace_length,
        'avg_event_duration': Statistic.get_avg_event_duration,
        'max_event_duration': Statistic.get_max_event_duration,
        'min_event_duration': Statistic.get_min_event_duration,
        'avg_case_duration': Statistic.get_avg_case_duration,
        'max_case_duration': Statistic.get_max_case_duration,
        'min_case_duration': Statistic.get_min_case_duration,
        'number_variants': Statistic.get_number_of_variants,
        'number_attributes': Statistic.get_number_of_event_attributes
    }

    def __init__(self, steps) -> None:
        super().__init__()
        self._steps = steps

    def run_analysis(self, event_log: EventLog) -> typing.Dict[str, typing.Union[int, float]]:
        return {step: self.ALL_STEPS[step](event_log) for step in self._steps}

    @staticmethod
    def get_available_steps() -> typing.List[str]:
        return list(EventLogDescriptor.ALL_STEPS.keys())

    @staticmethod
    def get_column_names() -> typing.Dict[str, str]:
        return {
            'number_cases': '#Cases',
            'number_events': '#Events',
            'number_activities': '#Activities',
            'avg_case_length': 'AVG Case Length',
            'max_case_length': 'Max. Case Length',
            'min_case_length': 'Min Case Length',
            'avg_event_duration': 'AVG Event Duration',
            'max_event_duration': 'Max Event Duration',
            'min_event_duration': 'Min Event Duration',
            'avg_case_duration': 'AVG Case Duration',
            'max_case_duration': 'Max Case Duration',
            'min_case_duration': 'Min Case Duration',
            'number_variants': '#Trace Variants',
            'number_attributes': '#Attributes'
        }


class EventLogAnalysis:
    ALL_STEPS = {
        'activity_distribution': ActivityDistributionAnalysis,
        'trace_length_analysis': TraceLengthAnalysis,
        'trace_variant_analysis': TraceVariantAnalysis,
        'case_density': TraceDistributionOverTime
    }

    def __init__(self, steps: typing.List[AnalysisStep[T]]) -> None:
        super().__init__()
        self._steps = steps

    def run_analysis(self, event_log: EventLog) -> None:
        [print(step.run(event_log)) for step in self._steps]

    @staticmethod
    def build(step_short_names: typing.List[str]) -> 'EventLogAnalysis':
        all_steps = [EventLogAnalysis.ALL_STEPS[step_name]() for step_name in step_short_names]
        return EventLogAnalysis(all_steps)

    @staticmethod
    def get_available_analysis_steps() -> typing.List[str]:
        return list(EventLogAnalysis.ALL_STEPS.keys())
