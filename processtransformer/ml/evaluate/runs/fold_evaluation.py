import typing
import os
import json as js
import pm4py
import pandas as pd
from processtransformer.ml.evaluate import ev_model, base_evaluation
from processtransformer.ml.core import model
from processtransformer.ml.evaluate import core_metrics
from pm4py.objects.log.obj import EventLog


class FoldBasedEvaluation:
    ID_COLUMN = 'id'
    TRUE_LABEL_COLUMN = 'true'
    PREDICTED_COLUMN = 'pred'
    PREFIX_LENGTH_COLUMN = 'prefix_length'

    def add_prefix_length(self, fold_data_df: pd.DataFrame, test_prefixes: EventLog):
        rows = []
        for trace in test_prefixes:
            rows.append([trace.attributes['concept:name'], len(trace)])
        rows_df = pd.DataFrame(rows, columns=[self.ID_COLUMN, self.PREFIX_LENGTH_COLUMN])
        return fold_data_df.merge(rows_df, how='inner', on=self.ID_COLUMN)

    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], fold: ev_model.Fold) -> ev_model.FoldResult:
        raise NotImplementedError()

    def load_results(self, labeled_file: str, prediction_file: str) -> pd.DataFrame:
        labels_df = pd.read_csv(labeled_file, sep='\t')
        predictions_df = pd.read_csv(prediction_file, sep='\t')

        labels_df = labels_df[[self.ID_COLUMN, model.PredictionTask.NEXT_ACTIVITY.value]]
        labels_df.columns = [self.ID_COLUMN, self.TRUE_LABEL_COLUMN]

        predictions_df = predictions_df[[self.ID_COLUMN, model.PredictionTask.NEXT_ACTIVITY.value]]
        predictions_df.columns = [self.ID_COLUMN, self.PREDICTED_COLUMN]

        return predictions_df.merge(labels_df, how='inner', on=self.ID_COLUMN)

    @staticmethod
    def format_id() -> str:
        raise NotImplementedError()


class ArchitectureFoldEvaluation(FoldBasedEvaluation):
    SUPPORTED_ITEMS = {'elapsed_time', 'trainable_weights'}

    def __load_additional_measurements(self, measure_file: str) -> typing.Dict[str, any]:
        assert os.path.isfile(measure_file), f'The requested measurement file {measure_file} does not exist'
        with open(measure_file, 'r', encoding='utf8') as f:
            measurements = js.load(f)
        if self.SUPPORTED_ITEMS.issuperset(set(measurements.keys())):
            print(f'WARNING: File {measure_file} contains unsupported metrics')
        return measurements

    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], fold: ev_model.Fold) -> ev_model.FoldResult:
        assert len(metrics) == 1, f'This type of evaluation only supports the Architecture Metric'
        assert isinstance(metrics[0], core_metrics.ArchitectureWrapper), f'This type of evaluation only supports the ' \
                                                                         f'Architecture Metric'
        assert set(metrics[0].items).issubset(self.SUPPORTED_ITEMS), f'Architecture Evaluation only supports the ' \
                                                                     f'following metrics {self.SUPPORTED_ITEMS} ' \
                                                                     f'but {set(metrics[0].items)} were given'
        additional_measurements = self.__load_additional_measurements(fold.additional_measures_file)
        items = metrics[0].items
        result = base_evaluation.ArchitectureEvaluation().run_evaluation(additional_measurements, items)

        return ev_model.FoldResult(fold.id, result)

    @staticmethod
    def format_id() -> str:
        return 'architecture'


class TotalFoldEvaluation(FoldBasedEvaluation):
    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], fold: ev_model.Fold) -> ev_model.FoldResult:
        fold_data_df = self.load_results(fold.labeled_file, fold.prediction_file)
        result = base_evaluation.TotalEvaluation().run_evaluation(metrics,
                                                                  fold_data_df[self.TRUE_LABEL_COLUMN],
                                                                  fold_data_df[self.PREDICTED_COLUMN])
        return ev_model.FoldResult(fold.id, result)

    @staticmethod
    def format_id() -> str:
        return 'total'


class PerActivityFoldEvaluation(FoldBasedEvaluation):
    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], fold: ev_model) -> ev_model.FoldResult:
        fold_data_df = self.load_results(fold.labeled_file, fold.prediction_file)
        result = base_evaluation.PerActivityEvaluation().run_evaluation(metrics,
                                                                        fold_data_df[self.TRUE_LABEL_COLUMN],
                                                                        fold_data_df[self.PREDICTED_COLUMN])
        return ev_model.FoldResult(fold.id, result)

    @staticmethod
    def format_id() -> str:
        return 'per_activity'


class PerPrefixLengthFoldEvaluation(FoldBasedEvaluation):

    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], fold: ev_model,
                       min_pref_length: int = 2) -> ev_model.FoldResult:
        fold_data_df = self.load_results(fold.labeled_file, fold.prediction_file)
        test_prefixes = pm4py.read_xes(fold.test_prefixes_file)
        fold_data_df = self.add_prefix_length(fold_data_df, test_prefixes)

        result = base_evaluation.PerPrefixLengthEvaluation().run_evaluation(metrics,
                                                                            fold_data_df[self.TRUE_LABEL_COLUMN],
                                                                            fold_data_df[self.PREDICTED_COLUMN],
                                                                            fold_data_df[self.PREFIX_LENGTH_COLUMN],
                                                                            min_pref_length)
        return ev_model.FoldResult(fold.id, result)

    @staticmethod
    def format_id() -> str:
        return 'per_prefix'


class StatisticalActivityFoldEvaluation(FoldBasedEvaluation):
    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], fold: ev_model.Fold) -> ev_model.FoldResult:
        fold_data_df = self.load_results(fold.labeled_file, fold.prediction_file)

        result = base_evaluation.StatisticalActivityEvaluation().run_evaluation(fold_data_df[self.TRUE_LABEL_COLUMN])
        return ev_model.FoldResult(fold.id, result)

    @staticmethod
    def format_id() -> str:
        return 'statistical_activity'


class StatisticalPrefixFoldEvaluation(FoldBasedEvaluation):
    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], fold: ev_model.Fold) -> ev_model.FoldResult:
        test_prefixes = pm4py.read_xes(fold.test_prefixes_file)
        fold_data_df = self.load_results(fold.labeled_file, fold.prediction_file)
        fold_data_df = self.add_prefix_length(fold_data_df, test_prefixes)

        result = base_evaluation.StatisticalPrefixEvaluation().run_evaluation(fold_data_df[self.PREFIX_LENGTH_COLUMN])
        return ev_model.FoldResult(fold.id, result)

    @staticmethod
    def format_id() -> str:
        return 'statistical_prefix'


class BaseEvaluator:
    @staticmethod
    def build() -> typing.Dict[str, FoldBasedEvaluation]:
        return {
            'total': TotalFoldEvaluation(),
            'per_activity': PerActivityFoldEvaluation(),
            'per_prefix': PerPrefixLengthFoldEvaluation(),
            'statistical_prefix': StatisticalPrefixFoldEvaluation(),
            'statistical_activity': StatisticalActivityFoldEvaluation(),
            'architecture': ArchitectureFoldEvaluation()
        }
