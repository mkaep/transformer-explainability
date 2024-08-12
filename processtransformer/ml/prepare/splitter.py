import pm4py
import dataclasses
import pandas as pd
import random
import abc
import typing
import os
from pm4py.objects.log.obj import EventLog
from processtransformer.ml.core import model


@dataclasses.dataclass
class SplitResult:
    type: str
    train_log: EventLog
    test_log: EventLog

    def store(self, train_dir: str, test_dir: str, verbose: bool = False) -> typing.Tuple[str, str]:
        if verbose:
            print(f'Start storing split result')

        train_log_file = os.path.join(train_dir, f'train.xes')
        test_log_file = os.path.join(test_dir, f'test.xes')

        pm4py.write_xes(self.train_log, train_log_file)
        pm4py.write_xes(self.test_log, test_log_file)

        if verbose:
            print(f'Finished storing split result')

        return train_log_file, test_log_file


def get_partition(training_size: float, case_ids: typing.List[str]) -> typing.Tuple[typing.List[str], typing.List[str]]:
    n_training_cases = int(training_size * len(case_ids))
    training_cases = case_ids[:n_training_cases]
    test_cases = case_ids[n_training_cases:]

    return training_cases, test_cases


class AbstractSplitter(abc.ABC):
    def split(self, event_log: EventLog, **kwargs) -> typing.List[SplitResult]:
        raise NotImplementedError()

    @staticmethod
    def format_id() -> str:
        raise NotImplementedError()

    @staticmethod
    def is_repeatable() -> bool:
        raise NotImplementedError()

    @staticmethod
    def check_configuration(configuration: model.SplitterConfiguration) -> bool:
        raise NotImplementedError()


class TimeSplitter(AbstractSplitter):
    SUPPORTED_SORTING = {'first', 'last'}

    def split(self, event_log: EventLog, training_size: float = 0.8, by: str = 'first') -> typing.List[SplitResult]:

        assert 0 < training_size < 1
        assert by in self.SUPPORTED_SORTING
        result = []

        # Build a DataFrame with case id and the corresponding starting time
        df = TimeSplitter.create_sorted_event_log(event_log, by)
        training_cases, test_cases = get_partition(training_size, df['case_id'].values)

        train_log = pm4py.filter_trace_attribute_values(event_log, 'concept:name', training_cases)
        test_log = pm4py.filter_trace_attribute_values(event_log, 'concept:name', test_cases)

        result.append(SplitResult(f'{TimeSplitter.format_id()}_{by}', train_log, test_log))
        return result

    @staticmethod
    def create_sorted_event_log(event_log: EventLog, by: str) -> pd.DataFrame:
        assert by in TimeSplitter.SUPPORTED_SORTING

        data = dict()
        for trace in event_log:
            if len(trace) > 0:
                if by == 'first':
                    time = trace[0]['time:timestamp']
                else:
                    time = trace[-1]['time:timestamp']
                data[trace.attributes['concept:name']] = time
        df = pd.DataFrame(data.items(), columns=['case_id', 'time'])
        df = df.sort_values(by=['time']).reset_index(drop=True)
        return df

    @staticmethod
    def is_repeatable() -> bool:
        return False

    @staticmethod
    def format_id() -> str:
        return 'time'

    @staticmethod
    def check_configuration(configuration: model.SplitterConfiguration) -> bool:
        return configuration.by is not None and configuration.training_size is not None \
               and 0 < configuration.training_size < 1 and configuration.by in TimeSplitter.SUPPORTED_SORTING


class RandomSplitter(AbstractSplitter):
    def split(self, event_log: EventLog, training_size: float = 0.8, seed: int = 42) -> typing.List[SplitResult]:
        assert 0 < training_size < 1

        result = []
        case_ids = [trace.attributes['concept:name'] for trace in event_log]

        random.seed(seed)
        random.shuffle(case_ids)
        training_cases, test_cases = get_partition(training_size, case_ids)

        train_log = pm4py.filter_trace_attribute_values(event_log, 'concept:name', training_cases)
        test_log = pm4py.filter_trace_attribute_values(event_log, 'concept:name', test_cases)

        result.append(SplitResult(RandomSplitter.format_id(), train_log, test_log))
        return result

    @staticmethod
    def is_repeatable() -> bool:
        return True

    @staticmethod
    def format_id() -> str:
        return 'random'

    @staticmethod
    def check_configuration(configuration: model.SplitterConfiguration) -> bool:
        required_parameters = configuration.training_size is not None and \
                              configuration.seeds is not None and \
                              configuration.repetitions is not None
        seed_consistency = False
        if configuration.seeds is not None:
            seed_consistency = len(configuration.seeds) == configuration.repetitions

        return required_parameters and seed_consistency and configuration.repetitions >= 1 \
               and 0 < configuration.training_size and configuration.training_size < 1


class KFoldSplitter(AbstractSplitter):
    def split(self, event_log: EventLog, folds: int = 5, seed: int = 42) -> typing.List[SplitResult]:
        assert len(event_log) >= folds > 1

        case_ids = [trace.attributes['concept:name'] for trace in event_log]
        fold_size = int(len(case_ids) / folds)

        random.seed(seed)
        random.shuffle(case_ids)

        subsets = []
        for i in range(0, folds):
            if i < folds - 1:
                subsets.append(case_ids[i * fold_size:(i + 1) * fold_size])
            else:
                subsets.append(case_ids[i * fold_size:])

        # Build the folds
        result = []
        for i in subsets:
            temp_subset = subsets.copy()
            test_log = pm4py.filter_trace_attribute_values(event_log, 'concept:name', i)

            temp_subset.remove(i)
            training_ids = []
            for sub in temp_subset:
                training_ids.extend(sub)

            train_log = pm4py.filter_trace_attribute_values(event_log, 'concept:name', training_ids)
            result.append(SplitResult(KFoldSplitter.format_id(), train_log, test_log))

        return result

    @staticmethod
    def is_repeatable() -> bool:
        return True

    @staticmethod
    def format_id() -> str:
        return 'k_fold'

    @staticmethod
    def check_configuration(configuration: model.SplitterConfiguration) -> bool:
        required_parameters = configuration.folds is not None and \
                              configuration.seeds is not None and \
                              configuration.repetitions is not None
        seed_consistency = False
        if configuration.seeds is not None:
            seed_consistency = len(configuration.seeds) == configuration.repetitions
        return required_parameters and seed_consistency and configuration.folds > 1 and configuration.repetitions >= 1
