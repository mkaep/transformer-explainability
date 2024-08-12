import math
import typing
import pandas as pd

from processtransformer.ml.evaluate import core_metrics, ev_model


class BaseEvaluation:

    @staticmethod
    def format_id() -> str:
        raise NotImplementedError()


class TotalEvaluation(BaseEvaluation):
    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], labels: typing.List,
                       predictions: typing.List) -> ev_model.TotalEvaluationResult:
        return ev_model.TotalEvaluationResult(core_metrics.MetricCalculator(metrics)
                                              .calculate_metrics(labels, predictions))

    def aggregate(self, measures: typing.List[ev_model.TotalEvaluationResult]) -> ev_model.TotalEvaluationResult:
        # TODO evtl eine check methode can_aggregated schreiben die prüft, ob immer die gleichen Keys vorliegen
        assert len(measures) > 0
        metric_names = measures[0].results.keys()
        results = dict()
        for metric in metric_names:
            results[metric] = sum([m.results[metric] for m in measures]) / len(measures)
        return ev_model.TotalEvaluationResult(results)

    @staticmethod
    def format_id() -> str:
        return 'total'


class ArchitectureEvaluation(BaseEvaluation):
    def run_evaluation(self, additional_measurements: typing.Dict[str, any], items: typing.List[str]):
        result = dict()
        for item in items:
            try:
                value = additional_measurements[item]
            except KeyError:
                value = None
            result[item] = value
        return ev_model.ArchitectureEvaluationResult(result)

    def aggregate(self, measures: typing.List[ev_model.ArchitectureEvaluationResult]
                  ) -> ev_model.ArchitectureEvaluationResult:
        if len(measures) > 0:
            items = measures[0].results.keys()
            aggregated = dict()
            for item in items:
                prototype = measures[0].results[item]
                if isinstance(prototype, int):
                    value = math.floor(sum([m.results[item] for m in measures]) / len(measures))
                elif isinstance(prototype, float):
                    value = sum([m.results[item] for m in measures]) / len(measures)
                elif prototype is None:
                    value = None
                else:
                    raise ValueError('This should not happen. Not supported datatype')
                aggregated[item] = value
        else:
            aggregated = dict()
        return ev_model.ArchitectureEvaluationResult(aggregated)

    @staticmethod
    def format_id() -> str:
        return 'architecture'


class StatisticalActivityEvaluation(BaseEvaluation):
    def count_activities(self, labels: typing.List) -> typing.Dict[str, int]:
        counted_activities = dict()
        for label in labels:
            try:
                counted_activities[label] = counted_activities[label] + 1
            except KeyError:
                counted_activities[label] = 1
        return counted_activities

    def run_evaluation(self, labels: typing.List) -> ev_model.StatisticalActivityEvaluationResult:
        counted_activities = self.count_activities(labels)
        return ev_model.StatisticalActivityEvaluationResult(counted_activities)

    def aggregate(self, measures: typing.List[ev_model.StatisticalActivityEvaluationResult]
                  ) -> ev_model.StatisticalActivityEvaluationResult:
        assert len(measures) > 0
        # todo was passiert wenn die measures unterschiedliche prefixe und aktivitäten haben...
        activities = measures[0].counted_activities.keys()
        aggregated_counted_activities = dict()
        for activity in activities:
            aggregated_counted_activities[activity] = math.floor(sum([m.counted_activities[activity]
                                                                      for m in measures]) / len(measures))

        return ev_model.StatisticalActivityEvaluationResult(aggregated_counted_activities)

    @staticmethod
    def format_id() -> str:
        return 'statistical_activity'


class StatisticalPrefixEvaluation(BaseEvaluation):
    def count_prefix_lengths(self, prefix_lengths: typing.List[int]) -> typing.Dict[int, int]:
        counted_prefix_lengths = dict()
        for length in prefix_lengths:
            try:
                counted_prefix_lengths[length] = counted_prefix_lengths[length] + 1
            except KeyError:
                counted_prefix_lengths[length] = 1
        return counted_prefix_lengths

    def run_evaluation(self, prefix_lengths: typing.List[int]) -> ev_model.StatisticalPrefixEvaluationResult:
        counted_prefix_lengths = self.count_prefix_lengths(prefix_lengths)
        return ev_model.StatisticalPrefixEvaluationResult(counted_prefix_lengths)

    def aggregate(self, measures: typing.List[ev_model.StatisticalPrefixEvaluationResult]
                  ) -> ev_model.StatisticalPrefixEvaluationResult:
        assert len(measures) > 0
        # todo was passiert wenn die measures unterschiedliche prefixe und aktivitäten haben...
        prefix_lengths = measures[0].counted_prefix_lengths.keys()
        aggregated_counted_prefix_lengths = dict()
        for prefix_length in prefix_lengths:
            aggregated_counted_prefix_lengths[prefix_length] = math.floor(sum([m.counted_prefix_lengths[prefix_length]
                                                                               for m in measures]) / len(measures))
        return ev_model.StatisticalPrefixEvaluationResult(aggregated_counted_prefix_lengths)

    @staticmethod
    def format_id() -> str:
        return 'statistical_prefix'


class PerActivityEvaluation(BaseEvaluation):
    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], labels: typing.List, predictions: typing.List) \
            -> ev_model.PerActivityEvaluationResult:
        activities = set(labels)
        combined = list(zip(labels, predictions))
        combined_df = pd.DataFrame(combined, columns=['label', 'predicted'])

        results = dict()
        for activity in activities:
            merged_filtered_df = combined_df[combined_df['label'] == activity]
            results[activity] = core_metrics.MetricCalculator(metrics)\
                .calculate_metrics(merged_filtered_df['label'], merged_filtered_df['predicted'])

        return ev_model.PerActivityEvaluationResult(results)

    def aggregate(self, measures: typing.List[ev_model.PerActivityEvaluationResult]
                  ) -> ev_model.PerActivityEvaluationResult:
        assert len(measures) > 0

        activities = []
        for item in measures:
            activities.extend(item.results.keys())
        activities = set(activities)

        random_activity = list(measures[0].results.keys())[0]
        metrics = list(measures[0].results[random_activity].keys())

        for item in measures:
            for idx in item.results.keys():
                assert set(metrics) == set(item.results[idx].keys())
                metrics.extend(item.results[idx].keys())

        metrics = set(metrics)

        aggregated = dict()
        for activity in activities:
            aggregated_metrics = dict()
            for metric in metrics:
                collected_values = []
                for item in measures:
                    try:
                        collected_values.append(item.results[activity][metric])
                    except KeyError:
                        pass
                aggregated_metrics[metric] = sum(collected_values) / len(collected_values)
            aggregated[activity] = aggregated_metrics
        return ev_model.PerActivityEvaluationResult(aggregated)

    @staticmethod
    def format_id() -> str:
        return 'per_activity'


class PerPrefixLengthEvaluation(BaseEvaluation):
    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], labels: typing.List, predictions: typing.List,
                       prefix_lengths: typing.List[int],
                       min_pref_length: int) -> ev_model.PerPrefixLengthEvaluationResult:

        combined = list(zip(labels, predictions, prefix_lengths))
        combined_df = pd.DataFrame(combined, columns=['label', 'predicted', 'prefix_length'])

        max_pref_length = max(combined_df['prefix_length'])
        real_min_pref_length = min(combined_df['prefix_length'])

        min_pref_length = min(real_min_pref_length, min_pref_length)

        results = dict()
        for k in range(min_pref_length, max_pref_length):
            merged_filtered_df = combined_df[combined_df['prefix_length'] == k]
            results[k] = core_metrics.MetricCalculator(metrics).calculate_metrics(merged_filtered_df['label'],
                                                                                       merged_filtered_df['predicted'])

        return ev_model.PerPrefixLengthEvaluationResult(results)

    def aggregate(self, measures: typing.List[ev_model.PerPrefixLengthEvaluationResult]
                  ) -> ev_model.PerPrefixLengthEvaluationResult:
        assert len(measures) > 0
        prefix_lengths = []
        for item in measures:
            prefix_lengths.extend(item.results.keys())
        prefix_lengths = set(prefix_lengths)

        random_prefix_length = list(measures[0].results.keys())[0]
        metrics = list(measures[0].results[random_prefix_length].keys())

        for item in measures:
            for idx in item.results.keys():
                assert set(metrics) == set(item.results[idx].keys())
                metrics.extend(item.results[idx].keys())

        metrics = set(metrics)

        aggregated = dict()
        for prefix_length in prefix_lengths:
            aggregated_metrics = dict()
            for metric in metrics:
                collected_values = []
                for item in measures:
                    try:
                        collected_values.append(item.results[prefix_length][metric])
                    except KeyError:
                        pass
                aggregated_metrics[metric] = sum(collected_values) / len(collected_values)
            aggregated[prefix_length] = aggregated_metrics
        return ev_model.PerPrefixLengthEvaluationResult(aggregated)

    @staticmethod
    def format_id() -> str:
        return 'per_prefix'


class BaseEvaluator:
    @staticmethod
    def build() -> typing.Dict[str, BaseEvaluation]:
        return {
            'total': TotalEvaluation(),
            'per_activity': PerActivityEvaluation(),
            'per_prefix': PerPrefixLengthEvaluation(),
            'statistical_prefix': StatisticalPrefixEvaluation(),
            'statistical_activity': StatisticalActivityEvaluation(),
            'architecture': ArchitectureEvaluation()
        }
