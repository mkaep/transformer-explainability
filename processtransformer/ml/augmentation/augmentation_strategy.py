import typing
import random
import time

from random import shuffle
from processtransformer.ml.augmentation import easy_augmentors
from processtransformer.ml.core import model
from pm4py.objects.log.obj import EventLog, Trace

Augmentor = typing.TypeVar('Augmentor', bound=easy_augmentors.BaseAugmentor)


class BaseAugmentationStrategy:
    def __init__(self, idx: int, augmentors: typing.List[Augmentor], augmentation_factor: float, seed: int,
                 allow_multiple: bool = True):
        assert len(augmentors) > 0, (f'At least one augmentor must be passed to the strategy '
                                     f'(currently {len(augmentors)}).')
        assert augmentation_factor > 1, f'Augmentation factor must be greater 1 (currently {augmentation_factor}).'

        random.seed(seed)

        self.idx = idx
        self.augmentors: typing.List[Augmentor] = augmentors
        self.augmentation_factor = augmentation_factor
        self.allow_multiple = allow_multiple

    def fit(self, event_log: EventLog):
        for augmentor in self.augmentors:
            augmentor.fit(event_log)

    def augment(self, event_log: EventLog, record_augmentation: bool = True, verbose: bool = False
                ) -> typing.Tuple[EventLog, typing.Dict[str, int], typing.Dict[str, typing.List[str]], float]:
        """
        This function will augment a given event log by applying different augmentors
        :param verbose:
        :param record_augmentation:
        :param event_log:
        :return:
        """
        raise NotImplementedError()

    def calculate_traces_to_generate(self, event_log: EventLog) -> int:
        return int((self.augmentation_factor - 1) * len(event_log))

    # noinspection PyMethodMayBeStatic
    def _get_trace_by_case_id(self, case_id: str, event_log: EventLog) -> typing.Union[Trace, None]:
        for trace in event_log:
            if trace.attributes['concept:name'] == case_id:
                return trace
        return None

    # noinspection PyMethodMayBeStatic
    def _extract_case_ids(self, event_log: EventLog) -> typing.List[str]:
        return [trace.attributes['concept:name'] for trace in event_log]

    @property
    def name(self) -> str:
        return f'{self.idx}_{self.type}_{self.allow_multiple}_{self.augmentation_factor}'

    @property
    def type(self) -> str:
        raise NotImplementedError()

    def __str__(self):
        return f'[id={self.idx}, augmentors={self.augmentors}, augmentation_factor={self.augmentation_factor}, ' \
               f'allow_multiple={self.allow_multiple}]'

    def __eq__(self, other):
        if not isinstance(other, BaseAugmentationStrategy):
            return NotImplementedError
        return self.idx == other.idx and self.augmentors == other.augmentors and self.augmentation_factor == other.augmentation_factor and self.allow_multiple == other.allow_multiple


class MixedAugmentationStrategy(BaseAugmentationStrategy):

    def augment(self, event_log: EventLog, record_augmentation: bool = True, verbose: bool = False
                ) -> typing.Tuple[EventLog, typing.Dict[str, int], typing.Dict[str, typing.List[str]], float]:
        augmentation_count = {k.get_name(): 0 for k in self.augmentors}
        augmentation_record = {k.get_name(): [] for k in self.augmentors}

        case_ids = self._extract_case_ids(event_log)
        traces_to_generate = self.calculate_traces_to_generate(event_log)

        if self.allow_multiple is True:
            augmented_event_log = event_log.__deepcopy__()
        else:
            augmented_event_log = EventLog()

        if verbose:
            print('Starting augmentation')

        start_time_augmentation = time.time()
        i = 0
        while traces_to_generate > 0:
            shuffle(case_ids)
            if self.allow_multiple:
                trace = self._get_trace_by_case_id(case_ids[0], augmented_event_log)
            else:
                trace = self._get_trace_by_case_id(case_ids[0], event_log)

            # Select randomly an augmentation strategy for the selected trace
            shuffle(self.augmentors)

            if self.augmentors[0].is_applicable('', trace) is True:
                try:
                    new_id = f'{case_ids[0]}_{i}'
                    augmented_trace = self.augmentors[0].augment(trace)

                    augmented_event_log.append(Trace(augmented_trace[:], attributes={
                        'concept:name': new_id,
                        'creator': self.augmentors[0].get_name()
                    }))
                    # It is important that the following code is executed after calling augment, since in case of
                    # an exception it should not be executed!
                    if record_augmentation is True:
                        augmentation_count[self.augmentors[0].get_name()] = augmentation_count[
                                                                                self.augmentors[0].get_name()] + 1
                        augmentation_record[self.augmentors[0].get_name()].append(case_ids[0])

                    # Add trace to new event log (maybe also old)
                    if self.allow_multiple:
                        case_ids.append(new_id)

                    traces_to_generate = traces_to_generate - 1
                    i = i + 1
                except AssertionError:
                    print(f'WARNING: An assertion occurred (maybe the time order is violated due to low precision), '
                          f'we retry again with an other augmentor or an other trace.')
                    continue
            else:
                print(
                    'Augmentor could not applied to this trace. We try to augment an other '
                    'trace or with an other augmentor')

        if self.allow_multiple is False:
            for trace in event_log:
                augmented_event_log.append(trace)
        end_time_augmentation = time.time()

        if verbose:
            print('Done!')

        augmentation_duration = end_time_augmentation - start_time_augmentation
        if record_augmentation is True:
            return augmented_event_log, augmentation_count, augmentation_record, augmentation_duration
        return augmented_event_log, _, _, augmentation_duration

    @property
    def type(self) -> str:
        return 'mixed'


class SingleAugmentationStrategy(BaseAugmentationStrategy):

    def __init__(self, idx: int, augmentors: typing.List[Augmentor], augmentation_factor: float, seed: int,
                 allow_multiple: bool = True):
        super().__init__(idx, augmentors, augmentation_factor, seed, allow_multiple)
        assert len(augmentors) == 1
        self.augmentor: Augmentor = augmentors[0]

    def augment(self, event_log: EventLog, record_augmentation: bool = True, verbose: bool = False
                ) -> typing.Tuple[EventLog, typing.Dict[str, int], typing.Dict[str, typing.List[str]], float]:

        augmentation_count = {self.augmentor.get_name(): 0}
        augmentation_record = {self.augmentor.get_name(): []}

        case_ids = self._extract_case_ids(event_log)
        traces_to_generate = self.calculate_traces_to_generate(event_log)

        if self.allow_multiple is True:
            augmented_event_log = event_log.__deepcopy__()
        else:
            augmented_event_log = EventLog()

        if verbose:
            print('Starting augmentation')

        start_time_augmentation = time.time()
        i = 0
        while traces_to_generate > 0:
            shuffle(case_ids)
            if self.allow_multiple:
                trace = self._get_trace_by_case_id(case_ids[0], augmented_event_log)
            else:
                trace = self._get_trace_by_case_id(case_ids[0], event_log)

            if self.augmentor.is_applicable('', trace) is True:
                try:
                    new_id = f'{case_ids[0]}_{i}'

                    augmented_trace = self.augmentor.augment(trace)

                    augmented_event_log.append(Trace(augmented_trace[:], attributes={
                        'concept:name': str(new_id),
                        'creator': self.augmentor.get_name()
                    }))

                    # It is important that the following code is executed after calling augment, since in case of
                    # an exception it should not be executed!
                    if record_augmentation is True:
                        augmentation_count[self.augmentor.get_name()] = augmentation_count[
                                                                            self.augmentor.get_name()] + 1
                        augmentation_record[self.augmentor.get_name()].append(case_ids[0])

                    # Add trace to new event log (maybe also old)
                    if self.allow_multiple:
                        case_ids.append(new_id)

                    traces_to_generate = traces_to_generate - 1
                    i = i + 1
                except AssertionError:
                    print(f'WARNING: An assertion occurred (maybe the time order is violated due to low precision), '
                          f'we retry again with an other augmentor or an other trace.')
                    continue
            else:
                print('Augmentor could not applied to this trace. We try to augment an other trace or '
                      'with an other augmentor')

        if self.allow_multiple is False:
            for trace in event_log:
                augmented_event_log.append(trace)
        end_time_augmentation = time.time()

        if verbose:
            print('Done!')

        augmentation_duration = end_time_augmentation - start_time_augmentation

        if record_augmentation is True:
            return augmented_event_log, augmentation_count, augmentation_record, augmentation_duration
        return augmented_event_log, _, _, augmentation_duration

    @property
    def type(self) -> str:
        return 'single'


AugmentationStrategy = typing.TypeVar('AugmentationStrategy', bound=BaseAugmentationStrategy)


class AugmentorBuilder:
    SUPPORTED_AUGMENTORS = [
        'RandomInsertion',
        'RandomDeletion',
        'ParallelSwap',
        'FragmentAugmentation',
        'ReworkActivity',
        'DeleteReworkActivity',
        'RandomReplacement',
        'RandomSwap',
        'LoopAugmentation'
    ]

    def build(self, config: model.AugmentorConfig) -> Augmentor:
        assert config.name in self.SUPPORTED_AUGMENTORS, f'The given name {config.name} is not supported. ' \
                                                         f'Supported augmentors are: {self.SUPPORTED_AUGMENTORS}'
        if config.name == 'RandomInsertion':
            return easy_augmentors.RandomInsertion.build(config.parameters)
        if config.name == 'RandomDeletion':
            return easy_augmentors.RandomDeletion.build(config.parameters)
        if config.name == 'ParallelSwap':
            return easy_augmentors.ParallelSwap.build(config.parameters)
        if config.name == 'FragmentAugmentation':
            return easy_augmentors.FragmentAugmentation.build(config.parameters)
        if config.name == 'ReworkActivity':
            return easy_augmentors.ReworkActivity.build(config.parameters)
        if config.name == 'DeleteReworkActivity':
            return easy_augmentors.DeleteReworkActivity.build(config.parameters)
        if config.name == 'RandomReplacement':
            return easy_augmentors.RandomReplacement.build(config.parameters)
        if config.name == 'RandomSwap':
            return easy_augmentors.RandomSwap.build(config.parameters)
        if config.name == 'LoopAugmentation':
            return easy_augmentors.LoopAugmentation.build(config.parameters)


class AugmentationStrategyBuilder:
    SUPPORTED_STRATEGIES = ['mixed', 'single']

    def __init__(self, configuration: model.AugmentationStrategyConfig):
        assert configuration.name in self.SUPPORTED_STRATEGIES, f'The given configuration {configuration.name} is ' \
                                                                f'not supported. Supported strategies ' \
                                                                f'are {self.SUPPORTED_STRATEGIES}.'
        self.configuration = configuration
        self.augmentors = []
        self._init_augmentors(configuration.augmentors)

    def build(self) -> AugmentationStrategy:
        if self.configuration.name == 'mixed':
            return MixedAugmentationStrategy(self.configuration.id,
                                             self.augmentors,
                                             self.configuration.augmentation_factor,
                                             self.configuration.seed,
                                             self.configuration.allow_multiple)
        elif self.configuration.name == 'single':
            return SingleAugmentationStrategy(self.configuration.id,
                                              self.augmentors,
                                              self.configuration.augmentation_factor,
                                              self.configuration.seed,
                                              self.configuration.allow_multiple)
        else:
            raise ValueError(f'The given configuration {self.configuration.name} is not supported. Supported '
                             f'strategies are {self.SUPPORTED_STRATEGIES}.')

    def _init_augmentors(self, augmentors: typing.List[model.AugmentorConfig]):
        builder = AugmentorBuilder()
        for augmentor in augmentors:
            self.augmentors.append(builder.build(augmentor))
