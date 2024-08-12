import math
import typing
from pm4py.objects.log.obj import EventLog
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.algo.filtering.log.variants import variants_filter


class Statistic:

    @staticmethod
    def get_case_ids(event_log: EventLog) -> typing.Set:
        case_ids = set()
        for trace in event_log:
            case_ids.add(trace.attributes['concept:name'])
        return case_ids

    @staticmethod
    def get_number_of_traces(event_log: EventLog):
        return len(event_log)

    @staticmethod
    def get_number_of_events(event_log: EventLog) -> int:
        if len(event_log) == 0:
            return 0
        return sum([len(trace) for trace in event_log])

    @staticmethod
    def get_number_of_activities(event_log: EventLog) -> int:
        return len(set(attributes_filter.get_attribute_values(event_log, 'concept:name').keys()))

    @staticmethod
    def get_avg_case_length(event_log: EventLog) -> float:
        if len(event_log) == 0:
            return 0
        return sum([len(trace) for trace in event_log]) / len(event_log)

    @staticmethod
    def get_min_trace_length(event_log: EventLog) -> int:
        return min([len(trace) for trace in event_log])

    @staticmethod
    def get_min_case_duration(event_log: EventLog) -> float:
        if len(event_log) == 0:
            return 0.0
        min_case_duration = math.inf
        for trace in event_log:
            if len(trace) > 0:
                duration = (trace[len(trace)-1]['time:timestamp'] - trace[0]['time:timestamp']).total_seconds()/86400
            else:
                duration = 0.0
            if duration < min_case_duration:
                min_case_duration = duration
        return min_case_duration

    @staticmethod
    def get_max_case_duration(event_log: EventLog) -> float:
        if len(event_log) == 0:
            return 0.0
        max_case_duration = -math.inf
        for trace in event_log:
            if len(trace) > 0:
                duration = (trace[len(trace)-1]['time:timestamp'] - trace[0]['time:timestamp']).total_seconds()/86400
            else:
                duration = 0.0
            if duration > max_case_duration:
                max_case_duration = duration
        return max_case_duration

    @staticmethod
    def get_avg_case_duration(event_log: EventLog) -> float:
        if len(event_log) == 0:
            return 0.0
        total_duration = 0.0
        for trace in event_log:
            if len(trace) >= 1:
                duration = (trace[len(trace)-1]['time:timestamp'] - trace[0]['time:timestamp']).total_seconds() / 86400
                total_duration = total_duration + duration
        return total_duration / len(event_log)

    @staticmethod
    def get_min_event_duration(event_log: EventLog) -> float:
        if len(event_log) == 0:
            return 0.0
        min_duration = math.inf
        for trace in event_log:
            if len(trace) > 1:
                for i in range(1, len(trace)):
                    elapsed_time = (trace[i]['time:timestamp'] - trace[i-1]['time:timestamp']).total_seconds() / 86400
                    if elapsed_time < min_duration:
                        min_duration = elapsed_time
            else:
                min_duration = 0.0
        return min_duration

    @staticmethod
    def get_max_event_duration(event_log: EventLog) -> float:
        max_duration = 0.0
        for trace in event_log:
            if len(trace) > 1:
                for i in range(1, len(trace)):
                    elapsed_time = (trace[i]['time:timestamp'] - trace[i-1]['time:timestamp']).total_seconds() / 86400
                    if elapsed_time > max_duration:
                        max_duration = elapsed_time
        return max_duration

    @staticmethod
    def get_avg_event_duration(event_log: EventLog) -> float:
        if len(event_log) == 0:
            return 0.0
        total_duration = 0.0
        for trace in event_log:
            if len(trace) > 1:
                for i in range(1, len(trace)):
                    elapsed_time = (trace[i]['time:timestamp'] - trace[i-1]['time:timestamp']).total_seconds() / 86400
                    total_duration = total_duration + elapsed_time
        return total_duration / len(event_log)

    @staticmethod
    def get_number_of_variants(event_log: EventLog) -> int:
        variants = variants_filter.get_variants(event_log)
        return len(variants)

    @staticmethod
    def get_max_trace_length(event_log: EventLog) -> int:
        return max([len(trace) for trace in event_log])

    @staticmethod
    def get_number_of_event_attributes(event_log: EventLog) -> int:
        attributes = set()
        for trace in event_log:
            for event in trace:
                attributes.update(set(event.keys()))
        return len(attributes)

    @staticmethod
    def get_activities(event_log: EventLog) -> typing.List[str]:
        activities = []
        for trace in event_log:
            for event in trace:
                activities.append(event['concept:name'])
        activities = set(activities)
        return list(activities)

    @staticmethod
    def get_resources(event_log: EventLog) -> typing.List[str]:
        resources = []
        for trace in event_log:
            for event in trace:
                if 'org:resource' in event.keys():
                    resources.append(event['org:resource'])
        resources = set(resources)
        return list(resources)

    @staticmethod
    def get_activity_resources(event_log: EventLog) -> typing.Dict[str, typing.List[str]]:
        activities_resources = {k: set() for k in Statistic.get_activities(event_log)}
        for trace in event_log:
            for event in trace:
                if 'org:resource' in event.keys():
                    activities_resources[event['concept:name']].add(event['org:resource'])
        return {k: list(activities_resources[k]) for k in activities_resources.keys()}



