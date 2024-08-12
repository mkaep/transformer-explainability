import dataclasses
import datetime
import math
import typing
import random

from abc import ABC
from processtransformer.ml.analysis import statistical_analysis
from processtransformer.ml.util import console
from pm4py.objects.log.obj import Trace, Event, EventLog


class BaseAugmentor(ABC):
    def augment(self, trace: Trace) -> Trace:
        raise NotImplementedError()

    def is_applicable(self, task: str, trace: Trace) -> bool:
        raise NotImplementedError()

    def fit(self, event_log: EventLog):
        raise NotImplementedError()

    # noinspection PyMethodMayBeStatic
    def check_order_of_trace(self, trace: Trace) -> bool:
        previous_timestamp = trace[0]['time:timestamp']
        for event in trace[1:]:
            if event['time:timestamp'] < previous_timestamp:
                return False
            previous_timestamp = event['time:timestamp']
        return True

    @staticmethod
    def get_name() -> str:
        raise NotImplementedError()

    @staticmethod
    def preserves_control_flow():
        raise NotImplementedError()

    def to_string(self):
        raise NotImplementedError()


class RandomInsertion(BaseAugmentor):
    """
    Insert an activity at an arbitrary position (but not at begin and not at end)
    """

    def __init__(self):
        self._activities = []
        self._resources = []

    def fit(self, event_log: EventLog):
        self._activities = statistical_analysis.Statistic.get_activities(event_log)
        self._resources = statistical_analysis.Statistic.get_resources(event_log)

    def augment(self, trace: Trace) -> Trace:
        assert len(self._activities) > 0, f'You have to call fit before applying the augmentor to ensure that the' \
                                          f'list of available activities is not empty. '
        augmented_trace = trace.__deepcopy__()
        position = random.randint(1, len(trace) - 1)

        duration = (trace[position]['time:timestamp'] - trace[position - 1]['time:timestamp']).microseconds
        random_duration = random.uniform(0, duration)

        random.shuffle(self._activities)
        activity = self._activities[0]

        event = Event({
            'concept:name': activity,
            'time:timestamp': trace[position - 1]['time:timestamp'] + datetime.timedelta(microseconds=random_duration)
        })

        # If in the event log 'org:resource' is not recorded, then we could not use this information for augmentation
        if len(self._resources) > 0:
            random.shuffle(self._resources)
            event.__setitem__('org:resource', self._resources[0])

        augmented_trace.insert(position, event)

        assert self.check_order_of_trace(augmented_trace) is True, f'Augmentation violates time order ' \
                                                                   f'(case {trace.attributes["concept:name"]}); ' \
                                                                   f'Augmentation:  ' \
                                                                   f'{console.trace_to_string(augmented_trace)}!'

        return augmented_trace

    # noinspection PyMethodMayBeStatic
    def is_applicable(self, task: str, trace: Trace) -> bool:
        """
        :return: True if applicable else False
        """
        if len(trace) >= 2:
            return True
        return False

    @staticmethod
    def get_name() -> str:
        return 'RandomInsertion'

    @staticmethod
    def preserves_control_flow():
        return False

    @staticmethod
    def build(config: typing.Dict[str, any]) -> 'RandomInsertion':
        return RandomInsertion()

    def __eq__(self, other):
        if not isinstance(other, RandomInsertion):
            return NotImplementedError
        return self._activities == other._activities

    def to_string(self):
        return f'[{RandomInsertion.get_name()}, Activities: {self._activities}]'


class RandomDeletion(BaseAugmentor):
    """
    Remove randomly an event
    """

    def fit(self, event_log: EventLog):
        pass

    def augment(self, trace: Trace) -> Trace:
        assert len(trace) > 1
        position = random.randint(0, len(trace))
        augmented_trace = trace[0:position]
        if position < len(trace):
            augmented_trace.extend(trace[position + 1:])

        assert self.check_order_of_trace(augmented_trace) is True, f'Augmentation violates time order ' \
                                                                   f'(case {trace.attributes["concept:name"]}); ' \
                                                                   f'Augmentation:  ' \
                                                                   f'{console.trace_to_string(augmented_trace)}!'

        return augmented_trace

    def is_applicable(self, task: str, trace: Trace) -> bool:
        if len(trace) < 2:
            return False
        return True

    @staticmethod
    def get_name() -> str:
        return 'RandomDeletion'

    @staticmethod
    def preserves_control_flow():
        return False

    @staticmethod
    def build(config: typing.Dict[str, any]) -> 'RandomDeletion':
        return RandomDeletion()

    def to_string(self):
        return f'[{RandomDeletion.get_name()}]'


class ParallelSwap(BaseAugmentor):
    """
    Swaps activities that were executed at the same time
    """

    def fit(self, event_log: EventLog):
        pass

    def augment(self, trace: Trace) -> Trace:
        for_swap = self.extract_parallel_activities(trace)

        assert len(for_swap) > 0, f'Could not apply augmentor. There are no parallel events in the trace.'

        augmented_trace = trace.__deepcopy__()
        i, j = for_swap[0]
        augmented_trace[i], augmented_trace[j] = augmented_trace[j], augmented_trace[i]

        assert self.check_order_of_trace(augmented_trace) is True, f'Augmentation violates time order ' \
                                                                   f'(case {trace.attributes["concept:name"]}); ' \
                                                                   f'Augmentation:  ' \
                                                                   f'{console.trace_to_string(augmented_trace)}!'
        return augmented_trace

    def is_applicable(self, task: str, trace: Trace) -> bool:
        for_swap = self.extract_parallel_activities(trace)
        if len(for_swap) > 0:
            return True
        return False

    # noinspection PyMethodMayBeStatic
    def extract_parallel_activities(self, trace: Trace) -> typing.List[typing.Tuple[int, int]]:
        for_swap = []
        for i in range(0, len(trace) - 1):
            previous = (trace[i + 1]['time:timestamp'] - trace[i]['time:timestamp'])
            if previous == datetime.timedelta(0):
                for_swap.append((i + 1, i))
        return for_swap

    @staticmethod
    def get_name() -> str:
        return 'ParallelSwap'

    @staticmethod
    def preserves_control_flow():
        return True

    @staticmethod
    def build(config: typing.Dict[str, any]) -> 'ParallelSwap':
        return ParallelSwap()

    def __eq__(self, other):
        if not isinstance(other, ParallelSwap):
            return NotImplementedError
        return True

    def to_string(self):
        return f'[{ParallelSwap.get_name()}]'


class FragmentAugmentation(BaseAugmentor):
    """Extracts a fragment from a trace and uses this fragment as a new trace"""

    def fit(self, event_log: EventLog):
        pass

    def augment(self, trace: Trace) -> Trace:
        # If its only 2 long it makes no sense to extract a fragment
        assert len(trace) > 2
        start = 0
        end = 0

        # Avoid empty trace and that the fragment is identical to the given trace
        while (start == end) or (start == 0 and end == len(trace)-1) or (start == len(trace)-1 and end == 0):
            start = random.randint(0, len(trace)-1)
            end = random.randint(0, len(trace)-1)

        if end < start:
            start, end = end, start

        augmented_trace = trace[start:end + 1]

        assert len(augmented_trace) > 0, f'WARNING: Augmented trace is empty'
        assert self.check_order_of_trace(augmented_trace) is True, f'Augmentation violates time order ' \
                                                                   f'(case {trace.attributes["concept:name"]}); ' \
                                                                   f'Augmentation:  ' \
                                                                   f'{console.trace_to_string(augmented_trace)}!'
        return augmented_trace

    def is_applicable(self, task: str, trace: Trace) -> bool:
        if len(trace) > 2:
            return True
        return False

    @staticmethod
    def get_name() -> str:
        return 'FragmentAugmentation'

    @staticmethod
    def preserves_control_flow():
        return True

    @staticmethod
    def build(config: typing.Dict[str, any]) -> 'FragmentAugmentation':
        return FragmentAugmentation()

    def to_string(self):
        return f'[{FragmentAugmentation.get_name()}]'


class ReworkActivity(BaseAugmentor):
    def fit(self, event_log: EventLog):
        pass

    def augment(self, trace: Trace) -> Trace:
        assert len(trace) > 0
        augmented_trace = trace.__deepcopy__()

        pos = random.randint(0, len(trace)-1)
        if pos == len(trace) - 1:
            duration = (trace[-1]['time:timestamp'] - trace[0]['time:timestamp']).microseconds
        else:
            duration = (trace[pos+1]['time:timestamp'] - trace[pos]['time:timestamp']).microseconds
        random_duration = random.uniform(0, duration)

        rework_event = trace[pos].__deepcopy__()
        rework_event.__setitem__('time:timestamp', trace[pos]['time:timestamp'] +
                                 datetime.timedelta(microseconds=random_duration))

        augmented_trace.insert(pos + 1, rework_event)
        assert self.check_order_of_trace(augmented_trace) is True, f'Augmentation violates time order ' \
                                                                   f'(case {trace.attributes["concept:name"]}); ' \
                                                                   f'Augmentation:  ' \
                                                                   f'{console.trace_to_string(augmented_trace)}!'

        return augmented_trace

    def is_applicable(self, task: str, trace: Trace) -> bool:
        if len(trace) >= 2:
            return True
        return False

    @staticmethod
    def get_name() -> str:
        return 'ReworkActivity'

    @staticmethod
    def preserves_control_flow():
        return False

    @staticmethod
    def build(config: typing.Dict[str, any]) -> 'ReworkActivity':
        return ReworkActivity()

    def to_string(self):
        return f'[{ReworkActivity.get_name()}]'


class DeleteReworkActivity(BaseAugmentor):
    def fit(self, event_log: EventLog):
        pass

    def augment(self, trace: Trace) -> Trace:
        assert self.contains_rework_activity(trace) is True, f'Could not apply augmentor. There are no rework ' \
                                                             f'activities in the trace'
        rework_activities = []
        for i in range(0, len(trace) - 1):
            if trace[i]['concept:name'] == trace[i + 1]['concept:name']:
                rework_activities.append((i, i + 1))
        random.shuffle(rework_activities)
        (k, j) = rework_activities[0]
        augmented_trace = trace[0:j]
        augmented_trace.extend(trace[j + 1:])

        assert self.check_order_of_trace(augmented_trace) is True, f'Augmentation violates time order ' \
                                                                   f'(case {trace.attributes["concept:name"]}); ' \
                                                                   f'Augmentation:  ' \
                                                                   f'{console.trace_to_string(augmented_trace)}!'
        return augmented_trace

    def is_applicable(self, task: str, trace: Trace) -> bool:
        if self.contains_rework_activity(trace) is True:
            return True
        return False

    # noinspection PyMethodMayBeStatic
    def contains_rework_activity(self, trace) -> bool:
        for i in range(0, len(trace) - 1):
            if trace[i]['concept:name'] == trace[i + 1]['concept:name']:
                return True
        return False

    @staticmethod
    def get_name() -> str:
        return 'DeleteReworkActivity'

    @staticmethod
    def preserves_control_flow():
        return False

    @staticmethod
    def build(config: typing.Dict[str, any]) -> 'DeleteReworkActivity':
        return DeleteReworkActivity()

    def __eq__(self, other):
        if not isinstance(other, DeleteReworkActivity):
            return NotImplementedError
        return True

    def to_string(self):
        return f'[{DeleteReworkActivity.get_name()}]'


class RandomReplacement(BaseAugmentor):
    def __init__(self):
        self._activities = []
        self._resources = []

    def fit(self, event_log: EventLog):
        self._activities = statistical_analysis.Statistic.get_activities(event_log)
        self._resources = statistical_analysis.Statistic.get_resources(event_log)

    def augment(self, trace: Trace) -> Trace:
        assert len(self._activities) > 0, f'You have to call fit before applying the augmentor to ensure that the' \
                                          f'list of available activities is not empty. '
        augmented_trace = trace.__deepcopy__()
        pos = random.randint(0, len(trace) - 1)

        # Ensure that an activity is not replaced by itself
        while augmented_trace[pos]['concept:name'] == self._activities[0]:
            random.shuffle(self._activities)

        augmented_trace[pos].__setitem__('concept:name', self._activities[0])

        # If in the event log 'org:resource' is not recorded, then we could not use this information for augmentation
        if len(self._resources) > 0:
            random.shuffle(self._resources)
            augmented_trace[pos].__setitem__('org:resource', self._resources[0])

        assert self.check_order_of_trace(augmented_trace) is True, f'Augmentation violates time order ' \
                                                                   f'(case {trace.attributes["concept:name"]}); ' \
                                                                   f'Augmentation:  ' \
                                                                   f'{console.trace_to_string(augmented_trace)}!'
        return augmented_trace

    def is_applicable(self, task: str, trace: Trace) -> bool:
        if len(trace) > 0:
            return True
        return False

    @staticmethod
    def get_name() -> str:
        return 'RandomReplacement'

    @staticmethod
    def preserves_control_flow():
        return False

    @staticmethod
    def build(config: typing.Dict[str, any]) -> 'RandomReplacement':
        return RandomReplacement()

    def to_string(self):
        return f'[{RandomReplacement.get_name()}, Activities: {self._activities}]'


class RandomSwap(BaseAugmentor):
    """
    Swaps randomly two activities in a trace
    """

    def fit(self, event_log: EventLog):
        pass

    def augment(self, trace: Trace) -> Trace:
        assert len(trace) > 1
        position = random.randint(0, len(trace) - 2)
        augmented_trace = trace.__deepcopy__()

        temp = augmented_trace[position]['time:timestamp']

        augmented_trace[position].__setitem__('time:timestamp', augmented_trace[position + 1]['time:timestamp'])
        augmented_trace[position + 1].__setitem__('time:timestamp', temp)
        augmented_trace[position], augmented_trace[position + 1] = augmented_trace[position + 1], augmented_trace[
            position]

        assert self.check_order_of_trace(augmented_trace) is True, f'Augmentation violates time order ' \
                                                                   f'(case {trace.attributes["concept:name"]}); ' \
                                                                   f'Augmentation:  ' \
                                                                   f'{console.trace_to_string(augmented_trace)}!'
        return augmented_trace

    def is_applicable(self, task: str, trace: Trace) -> bool:
        if len(trace) > 1:
            return True
        return False

    @staticmethod
    def get_name() -> str:
        return 'RandomSwap'

    @staticmethod
    def preserves_control_flow():
        return False

    @staticmethod
    def build(config: typing.Dict[str, any]) -> 'RandomSwap':
        return RandomSwap()

    def to_string(self):
        return f'[{RandomSwap.get_name()}]'


@dataclasses.dataclass
class Loop:
    start: int
    end: int
    body: str

    @property
    def repetitions(self) -> int:
        return int((self.end + 1 - self.start) / len(self.body))

    @property
    def body_length(self) -> int:
        return len(self.body)


class LoopAugmentation(BaseAugmentor):
    """Identifies possible loops and adds repetitions of the loop"""

    def __init__(self, max_additional_repetitions: int, duration_tolerance: float):
        assert max_additional_repetitions > 0, f'Parameter max_additional_repetitions must be > 0, currently it ' \
                                               f'is {max_additional_repetitions}'
        self._max_additional_repetitions = max_additional_repetitions
        assert duration_tolerance > 0, f'Duration tolerance must be > 0.0'
        self._duration_tolerance = duration_tolerance
        self._activities_resources = dict()
        self._resources = []

    def fit(self, event_log: EventLog):
        self._activities_resources = statistical_analysis.Statistic.get_activity_resources(event_log)
        self._resources = statistical_analysis.Statistic.get_resources(event_log)

    def augment(self, trace: Trace) -> Trace:
        assert self.check_order_of_trace(trace) is True, f'Augmentation violates time order ' \
                                                         f'{console.trace_to_string(trace)}!'

        trace_as_character_sequence = self.trace_to_character_string(trace)
        loops = self.detect_loops(trace_as_character_sequence)
        assert len(loops) > 0, f'Could not apply augmentor. There are no loops within the trace.'

        # Select loop for augmentation
        random.shuffle(loops)
        loop_to_augment = loops[0]
        additional_repetitions = random.randint(1, self._max_additional_repetitions)

        # Get body with events
        body = [event['concept:name'] for event
                in trace[loop_to_augment.start:loop_to_augment.start + loop_to_augment.body_length]]

        # Get min and max durations of the loop repetitions
        repetition_durations = []
        for i in range(loop_to_augment.repetitions):
            duration = (trace[loop_to_augment.start - 1 + (i + 1) * loop_to_augment.body_length]['time:timestamp']
                        - trace[loop_to_augment.start + i * loop_to_augment.body_length]['time:timestamp']
                        ).microseconds
            repetition_durations.append(duration)
        min_duration = min(repetition_durations)
        max_duration = max(repetition_durations)

        # Build insertion sequence
        insertion = []
        previous_timestamp = trace[loop_to_augment.end]['time:timestamp']
        duration_of_insertion = 0.0
        for i in range(0, additional_repetitions):
            random_duration_repetition = random.uniform(min_duration * (1 - self._duration_tolerance),
                                                        max_duration * (1 + self._duration_tolerance))
            durations = self.get_durations(random_duration_repetition, len(body))

            for idx, b in enumerate(body):
                duration_of_insertion = duration_of_insertion + durations[idx]
                potential_resources = self._activities_resources[b]

                new_timestamp = previous_timestamp + datetime.timedelta(microseconds=durations[idx])
                new_event = Event({
                    'concept:name': b,
                    'time:timestamp': new_timestamp
                })

                # If in the event log 'org:resource' is not recorded, then we could not use this information for
                # augmentation
                if len(self._resources) > 0:
                    if len(potential_resources) > 0:
                        random.shuffle(potential_resources)
                        resource = potential_resources[0]
                    else:
                        random.shuffle(self._resources)
                        resource = self._resources[0]
                    new_event.__setitem__('org:resource', resource)

                insertion.append(new_event)
                previous_timestamp = new_timestamp

        previous = trace[:loop_to_augment.end + 1]
        previous.extend(insertion)
        last_pos = len(previous)
        previous.extend(trace[loop_to_augment.end + 1:])

        # Shift the remaining events into the future by adding duration_of_insertion
        for event in previous[last_pos:]:
            shifted_timestamp = event['time:timestamp'] + datetime.timedelta(microseconds=duration_of_insertion)
            event.__setitem__('time:timestamp', shifted_timestamp)

        assert self.check_order_of_trace(previous) is True, f'Augmentation violates time order ' \
                                                                   f'(case {trace.attributes["concept:name"]}); ' \
                                                                   f'Augmentation:  ' \
                                                                   f'{console.trace_to_string(previous)}!'

        return previous

    # noinspection PyMethodMayBeStatic
    def get_durations(self, duration_repetition: float, points: int) -> typing.List[float]:
        elapsed_time = []
        for i in range(points):
            value = random.uniform(0, duration_repetition)
            elapsed_time.append(value)

        elapsed_time.sort()
        durations = []
        durations.append(elapsed_time[0])
        durations.extend([elapsed_time[i + 1] - elapsed_time[i] for i in range(len(elapsed_time) - 1)])

        assert sum(durations) <= duration_repetition, f'Longer as allowed'
        return durations

    # noinspection PyMethodMayBeStatic
    def trace_to_character_string(self, trace: Trace) -> str:
        activities = [event['concept:name'] for event in trace]
        translation_dict = {act: chr(i) for i, act in enumerate(activities)}
        characterized_string = ''
        for event in trace:
            characterized_string = characterized_string + translation_dict[event['concept:name']]
        return characterized_string

    # noinspection PyMethodMayBeStatic
    def get_all_substrings_of_max_length(self, trace: str, max_substring_length: int) -> typing.Set[str]:
        substrings = []
        for i in range(len(trace)):
            for k in range(0, max_substring_length):
                substrings.append(trace[i: i + k + 1])

        return set(substrings)

    def detect_loops(self, trace: str) -> typing.List[Loop]:
        max_substring_length = math.floor(len(trace) / 2)
        substrings = self.get_all_substrings_of_max_length(trace, max_substring_length)

        loops = []
        for substr in substrings:
            for i in range(0, len(trace)):
                repetitions = 0
                if trace[i:i + len(substr)] == substr:
                    # Possible loop starts
                    start = i
                    end = i + len(substr)
                    repetitions = repetitions + 1

                    j = i + len(substr)
                    while True:
                        if trace[j: j + len(substr)] == substr:
                            repetitions = repetitions + 1
                            j = j + len(substr)
                            end = j
                        else:
                            if repetitions > 1:
                                loops.append(Loop(start, end - 1, substr))
                            break
        return loops

    def is_applicable(self, task: str, trace: Trace) -> bool:
        trace_as_character_sequence = self.trace_to_character_string(trace)
        if len(self.detect_loops(trace_as_character_sequence)) > 0:
            return True
        return False

    @staticmethod
    def get_name() -> str:
        return 'LoopAugmentation'

    @staticmethod
    def preserves_control_flow():
        return True

    @staticmethod
    def build(config: typing.Dict[str, any]) -> 'LoopAugmentation':
        assert 'max_additional_repetitions' in config.keys(), f'Requires max_additional_repetitions to build ' \
                                                              f'LoopAugmentor'
        assert 'duration_tolerance' in config.keys(), f'Requires duration_tolerance to build LoopAugmentor'
        return LoopAugmentation(max_additional_repetitions=config['max_additional_repetitions'],
                                duration_tolerance=config['duration_tolerance'])

    def to_string(self):
        return f'[{LoopAugmentation.get_name()}, max_additional_repetitions: {self._max_additional_repetitions}, ' \
               f'duration_tolerance: {self._duration_tolerance}]'
