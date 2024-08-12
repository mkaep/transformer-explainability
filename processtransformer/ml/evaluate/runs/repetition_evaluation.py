import typing
from processtransformer.ml.evaluate import ev_model
from processtransformer.ml.evaluate.runs import fold_evaluation
from processtransformer.ml.evaluate import base_evaluation, core_metrics


class RepetitionBasedEvaluation:
    def run_evaluation(self, metric_names: typing.List[str],
                       repetition: ev_model.Repetition) -> ev_model.RepetitionResult:
        raise NotImplementedError()

    @staticmethod
    def format_id() -> str:
        raise NotImplementedError()


class TotalRepetitionEvaluation(RepetitionBasedEvaluation):
    def run_evaluation(self, metrics: typing.List[core_metrics.Metric],
                       repetition: ev_model.Repetition) -> ev_model.RepetitionResult:
        fold_results = []
        aggregated_results = []
        for fold in repetition.folds:
            result = fold_evaluation.TotalFoldEvaluation().run_evaluation(metrics, fold)
            fold_results.append(result)
            aggregated_results.append(result.results)
        aggregated = base_evaluation.TotalEvaluation().aggregate(aggregated_results)
        return ev_model.RepetitionResult(repetition.id, fold_results, aggregated)

    @staticmethod
    def format_id() -> str:
        return 'total'


class ArchitectureRepetitionEvaluation(RepetitionBasedEvaluation):
    def run_evaluation(self, metrics: typing.List[core_metrics.Metric],
                       repetition: ev_model.Repetition) -> ev_model.RepetitionResult:
        fold_results = []
        aggregated_results = []
        for fold in repetition.folds:
            result = fold_evaluation.ArchitectureFoldEvaluation().run_evaluation(metrics, fold)
            fold_results.append(result)
            aggregated_results.append(result.results)
        aggregated = base_evaluation.ArchitectureEvaluation().aggregate(aggregated_results)
        return ev_model.RepetitionResult(repetition.id, fold_results, aggregated)

    @staticmethod
    def format_id() -> str:
        return 'architecture'


class StatisticalActivityRepetitionEvaluation(RepetitionBasedEvaluation):
    def run_evaluation(self, metrics: typing.List[core_metrics.Metric],
                       repetition: ev_model.Repetition) -> ev_model.RepetitionResult:
        fold_results = []
        aggregated_results = []
        for fold in repetition.folds:
            result = fold_evaluation.StatisticalActivityFoldEvaluation().run_evaluation(metrics, fold)
            fold_results.append(result)
            aggregated_results.append(result.results)
        aggregated = base_evaluation.StatisticalActivityEvaluation().aggregate(aggregated_results)
        return ev_model.RepetitionResult(repetition.id, fold_results, aggregated)

    @staticmethod
    def format_id() -> str:
        return 'statistical_activity'


class StatisticalPrefixRepetitionEvaluation(RepetitionBasedEvaluation):
    def run_evaluation(self, metrics: typing.List[core_metrics.Metric],
                       repetition: ev_model.Repetition) -> ev_model.RepetitionResult:
        fold_results = []
        aggregated_results = []
        for fold in repetition.folds:
            result = fold_evaluation.StatisticalPrefixFoldEvaluation().run_evaluation(metrics, fold)
            fold_results.append(result)
            aggregated_results.append(result.results)
        aggregated = base_evaluation.StatisticalPrefixEvaluation().aggregate(aggregated_results)
        return ev_model.RepetitionResult(repetition.id, fold_results, aggregated)

    @staticmethod
    def format_id() -> str:
        return 'statistical_prefix'


class PerActivityRepetitionEvaluation(RepetitionBasedEvaluation):
    def run_evaluation(self, metrics: typing.List[core_metrics.Metric],
                       repetition: ev_model.Repetition) -> ev_model.RepetitionResult:
        fold_results = []
        aggregated_results = []
        for fold in repetition.folds:
            result = fold_evaluation.PerActivityFoldEvaluation().run_evaluation(metrics, fold)
            fold_results.append(result)
            aggregated_results.append(result.results)

        aggregated = base_evaluation.PerActivityEvaluation().aggregate(aggregated_results)
        return ev_model.RepetitionResult(repetition.id, fold_results, aggregated)

    @staticmethod
    def format_id() -> str:
        return 'per_activity'


class PerPrefixLengthRepetitionEvaluation(RepetitionBasedEvaluation):
    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], repetition: ev_model.Repetition,
                       min_pref_length: int = 2) -> ev_model.RepetitionResult:
        fold_results = []
        aggregated_results = []
        for fold in repetition.folds:
            result = fold_evaluation.PerPrefixLengthFoldEvaluation().run_evaluation(metrics, fold, min_pref_length)
            fold_results.append(result)
            aggregated_results.append(result.results)

        aggregated = base_evaluation.PerPrefixLengthEvaluation().aggregate(aggregated_results)

        return ev_model.RepetitionResult(repetition.id, fold_results, aggregated)

    @staticmethod
    def format_id() -> str:
        return 'per_prefix'
