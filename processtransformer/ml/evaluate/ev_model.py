import dataclasses
import typing
import pandas as pd


@dataclasses.dataclass
class Fold:
    id: str
    labeled_file: str
    prediction_file: str
    test_prefixes_file: str
    additional_measures_file: str


@dataclasses.dataclass
class Repetition:
    id: str
    folds: typing.List[Fold]


@dataclasses.dataclass
class Run:
    dataset: str
    approach: str
    strategy: str
    repetitions: typing.List[Repetition]


@dataclasses.dataclass
class EvaluationModel:
    runs: typing.List[Run]

    def get_run(self, rule: typing.Callable[[Run], bool]) -> typing.Union[Run, None]:
        return next((filter(rule, self.runs)), None)

    def filter_by(self, rule: typing.Callable[[Run], bool]) -> 'EvaluationModel':
        filtered_runs = [r for r in self.runs if rule(r) is True]
        return EvaluationModel(filtered_runs)


@dataclasses.dataclass
class BaseEvaluationResult:
    pass


BaseType = typing.TypeVar("BaseType", bound=BaseEvaluationResult)


@dataclasses.dataclass
class TotalEvaluationResult(BaseEvaluationResult):
    results: typing.Dict[str, float]


@dataclasses.dataclass
class PerActivityEvaluationResult(BaseEvaluationResult):
    results: typing.Dict[str, typing.Dict[str, float]]


@dataclasses.dataclass
class PerPrefixLengthEvaluationResult(BaseEvaluationResult):
    results: typing.Dict[int, typing.Dict[str, float]]


@dataclasses.dataclass
class ArchitectureEvaluationResult(BaseEvaluationResult):
    results: typing.Dict[str, typing.Union[float, int]]

    @staticmethod
    def get_supported_items() -> typing.List[str]:
        return ['elapsed_time', 'trainable_weights']


@dataclasses.dataclass
class StatisticalPrefixEvaluationResult(BaseEvaluationResult):
    counted_prefix_lengths: typing.Dict[int, int]

    def get_value_for_prefix_length(self, value: int) -> int:
        try:
            return self.counted_prefix_lengths[value]
        except KeyError:
            raise ValueError(f'The prefix length {value} does not occur in this fold')


@dataclasses.dataclass
class StatisticalActivityEvaluationResult(BaseEvaluationResult):
    counted_activities: typing.Dict[str, int]

    def get_value_for_activity(self, value: str) -> int:
        try:
            return self.counted_activities[value]
        except KeyError:
            raise ValueError(f'The activity {value} does not occur in this fold')


@dataclasses.dataclass
class FoldResult:
    id: str
    results: BaseType


@dataclasses.dataclass
class RepetitionResult:
    id: str
    results: typing.List[FoldResult]
    aggregated_result: typing.Union[BaseType, None]


@dataclasses.dataclass
class RunResult:
    dataset: str
    approach: str
    strategy: str
    results: typing.List[RepetitionResult]
    aggregated_result: typing.Union[BaseType, None]

    def to_dataframe_row_on_run_level(self) -> typing.Tuple[typing.List[any], typing.List[str]]:
        lines = []
        if isinstance(self.aggregated_result, TotalEvaluationResult):
            columns = ['dataset', 'approach', 'strategy', 'metric', 'value']
            for metric in self.aggregated_result.results.keys():
                lines.append([self.dataset, self.approach, self.strategy, metric,
                              self.aggregated_result.results[metric]])
        elif isinstance(self.aggregated_result, PerActivityEvaluationResult):
            columns = ['dataset', 'approach', 'strategy', 'activity', 'metric', 'value']
            for activity in self.aggregated_result.results.keys():
                for metric in self.aggregated_result.results[activity].keys():
                    lines.append([self.dataset, self.approach, self.strategy, activity, metric,
                                  self.aggregated_result.results[activity][metric]])
        elif isinstance(self.aggregated_result, PerPrefixLengthEvaluationResult):
            columns = ['dataset', 'approach', 'strategy', 'prefix_length', 'metric', 'value']
            for prefix_length in self.aggregated_result.results.keys():
                for metric in self.aggregated_result.results[prefix_length].keys():
                    lines.append([self.dataset, self.approach, self.strategy, prefix_length,
                                  metric, self.aggregated_result.results[prefix_length][metric]])
        elif isinstance(self.aggregated_result, StatisticalPrefixEvaluationResult):
            columns = ['dataset', 'approach', 'strategy', 'prefix_length', 'count']
            for prefix_length in self.aggregated_result.counted_prefix_lengths.keys():
                lines.append([self.dataset, self.approach, self.strategy, prefix_length,
                              self.aggregated_result.get_value_for_prefix_length(prefix_length)])
        elif isinstance(self.aggregated_result, StatisticalActivityEvaluationResult):
            columns = ['dataset', 'approach', 'strategy', 'activity', 'count']
            for activity in self.aggregated_result.counted_activities.keys():
                lines.append([self.dataset, self.approach, self.strategy, activity,
                              self.aggregated_result.get_value_for_activity(activity)])
        elif isinstance(self.aggregated_result, ArchitectureEvaluationResult):
            columns = ['dataset', 'approach', 'strategy', 'metric', 'value']
            for metric in self.aggregated_result.results.keys():
                lines.append([self.dataset, self.approach, self.strategy, metric,
                              self.aggregated_result.results[metric]])
        else:
            raise ValueError('Should not happen. This Base Evaluation is not supported.')
        return lines, columns

    def to_dataframe_row_on_repetition_level(self) -> typing.Tuple[typing.List[any], typing.List[str]]:
        lines = []
        columns = []
        for repetition_result in self.results:
            if isinstance(repetition_result.aggregated_result, TotalEvaluationResult):
                columns = ['dataset', 'approach', 'strategy', 'repetition', 'metric', 'value']
                for metric in repetition_result.aggregated_result.results.keys():
                    lines.append(
                        [self.dataset, self.approach, self.strategy, repetition_result.id,
                         metric, repetition_result.aggregated_result.results[metric]])
            elif isinstance(repetition_result.aggregated_result, PerActivityEvaluationResult):
                columns = ['dataset', 'approach', 'strategy', 'repetition', 'activity', 'metric',
                           'value']
                for activity in repetition_result.aggregated_result.results.keys():
                    for metric in repetition_result.aggregated_result.results[activity].keys():
                        lines.append(
                            [self.dataset, self.approach, self.strategy, repetition_result.id,
                             activity, metric, repetition_result.aggregated_result.results[activity][metric]])
            elif isinstance(repetition_result.aggregated_result, PerPrefixLengthEvaluationResult):
                columns = ['dataset', 'approach', 'strategy', 'repetition', 'prefix_length', 'metric',
                           'value']
                for prefix_length in repetition_result.aggregated_result.results.keys():
                    for metric in repetition_result.aggregated_result.results[prefix_length].keys():
                        lines.append(
                            [self.dataset, self.approach, self.strategy, repetition_result.id,
                             prefix_length, metric,
                             repetition_result.aggregated_result.results[prefix_length][metric]])
            elif isinstance(repetition_result.aggregated_result, StatisticalPrefixEvaluationResult):
                columns = ['dataset', 'approach', 'strategy', 'repetition', 'prefix_length', 'count']
                for prefix_length in repetition_result.aggregated_result.counted_prefix_lengths.keys():
                    lines.append([self.dataset, self.approach, self.strategy, repetition_result.id, prefix_length,
                                  repetition_result.aggregated_result.get_value_for_prefix_length(prefix_length)])
            elif isinstance(repetition_result.aggregated_result, StatisticalActivityEvaluationResult):
                columns = ['dataset', 'approach', 'strategy', 'repetition', 'activity', 'count']
                for activity in repetition_result.aggregated_result.counted_activities.keys():
                    lines.append([self.dataset, self.approach, self.strategy, repetition_result.id, activity,
                                  repetition_result.aggregated_result.get_value_for_activity(activity)])
            elif isinstance(repetition_result.aggregated_result, ArchitectureEvaluationResult):
                columns = ['dataset', 'approach', 'strategy', 'repetition', 'metric', 'value']
                for metric in repetition_result.aggregated_result.results.keys():
                    lines.append([self.dataset, self.approach, self.strategy, repetition_result.id, metric,
                                  repetition_result.aggregated_result.results[metric]])
            else:
                raise ValueError('This should not happen. Not supported Base Evaluation')
        return lines, columns

    def to_dataframe_row_on_fold_level(self) -> typing.Tuple[typing.List[any], typing.List[str]]:
        lines = []
        columns = []
        for repetition_result in self.results:
            for fold_result in repetition_result.results:
                if isinstance(fold_result.results, TotalEvaluationResult):
                    columns = ['dataset', 'approach', 'strategy', 'repetition', 'fold', 'metric', 'value']
                    for metric in fold_result.results.results.keys():
                        lines.append(
                            [self.dataset, self.approach, self.strategy, repetition_result.id,
                             fold_result.id, metric, fold_result.results.results[metric]])
                elif isinstance(fold_result.results, PerActivityEvaluationResult):
                    columns = ['dataset', 'approach', 'strategy', 'repetition', 'fold', 'activity', 'metric',
                               'value']
                    for activity in fold_result.results.results.keys():
                        for metric in fold_result.results.results[activity].keys():
                            lines.append(
                                [self.dataset, self.approach, self.strategy, repetition_result.id,
                                 fold_result.id, activity, metric, fold_result.results.results[activity][metric]])
                elif isinstance(fold_result.results, PerPrefixLengthEvaluationResult):
                    columns = ['dataset', 'approach', 'strategy', 'repetition', 'fold', 'prefix_length', 'metric',
                               'value']
                    for prefix_length in fold_result.results.results.keys():
                        for metric in fold_result.results.results[prefix_length].keys():
                            lines.append(
                                [self.dataset, self.approach, self.strategy, repetition_result.id,
                                 fold_result.id, prefix_length, metric,
                                 fold_result.results.results[prefix_length][metric]])
                elif isinstance(fold_result.results, StatisticalPrefixEvaluationResult):
                    columns = ['dataset', 'approach', 'strategy', 'repetition', 'fold', 'prefix_length', 'count']
                    for prefix_length in fold_result.results.counted_prefix_lengths.keys():
                        lines.append([self.dataset, self.approach, self.strategy, repetition_result.id, fold_result.id,
                                      prefix_length, fold_result.results.get_value_for_prefix_length(prefix_length)])
                elif isinstance(fold_result.results, StatisticalActivityEvaluationResult):
                    columns = ['dataset', 'approach', 'strategy', 'repetition', 'fold', 'activity', 'count']
                    for activity in fold_result.results.counted_activities.keys():
                        lines.append([self.dataset, self.approach, self.strategy, repetition_result.id, fold_result.id,
                                      activity, fold_result.results.get_value_for_activity(activity)])
                elif isinstance(fold_result.results, ArchitectureEvaluationResult):
                    columns = ['dataset', 'approach', 'strategy', 'repetition', 'fold', 'metric', 'value']
                    for metric in fold_result.results.results.keys():
                        lines.append([self.dataset, self.approach, self.strategy, repetition_result.id, fold_result.id,
                                      metric, fold_result.results.results[metric]])
                else:
                    raise ValueError('This should not happen. Not supported Base Evaluation')
        return lines, columns


@dataclasses.dataclass
class EvaluationModelResult:
    SUPPORTED_AGGREGATES = {'repetition', 'fold', 'run'}
    runResults: typing.List[RunResult]

    def to_dataframe(self, aggregate_on: str) -> pd.DataFrame:
        assert aggregate_on in self.SUPPORTED_AGGREGATES
        if aggregate_on == 'repetition':
            return self.__to_dataframe_repetition_level()
        elif aggregate_on == 'fold':
            return self.__to_dataframe_fold_level()
        else:
            return self.__to_dataframe_run_level()

    def __to_dataframe_repetition_level(self) -> pd.DataFrame:
        lines = []
        columns = []
        for run_result in self.runResults:
            line, columns = run_result.to_dataframe_row_on_repetition_level()
            lines.extend(line)

        return pd.DataFrame(lines, columns=columns)

    def __to_dataframe_fold_level(self) -> pd.DataFrame:
        lines = []
        columns = []
        for run_result in self.runResults:
            line, columns = run_result.to_dataframe_row_on_fold_level()
            lines.extend(line)

        return pd.DataFrame(lines, columns=columns)

    def __to_dataframe_run_level(self) -> pd.DataFrame:
        lines = []
        columns = []
        for run_result in self.runResults:
            line, columns = run_result.to_dataframe_row_on_run_level()
            lines.extend(line)

        return pd.DataFrame(lines, columns=columns)
