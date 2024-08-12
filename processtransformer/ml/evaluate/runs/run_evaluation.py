import typing
from processtransformer.ml.evaluate import ev_model, base_evaluation, core_metrics
from processtransformer.ml.evaluate.runs import repetition_evaluation


class RunBasedEvaluation:

    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], run: ev_model.Run) -> ev_model.RunResult:
        raise NotImplementedError()


class TotalRunEvaluation(RunBasedEvaluation):

    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], run: ev_model.Run) -> ev_model.RunResult:
        repetition_results = []
        aggregated_results = []
        for repetition in run.repetitions:
            result = repetition_evaluation.TotalRepetitionEvaluation().run_evaluation(metrics, repetition)
            repetition_results.append(result)
            aggregated_results.append(result.aggregated_result)
        aggregated = base_evaluation.TotalEvaluation().aggregate(aggregated_results)
        return ev_model.RunResult(run.dataset, run.approach, run.strategy, repetition_results, aggregated)


class ArchitectureRunEvaluation(RunBasedEvaluation):
    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], run: ev_model.Run) -> ev_model.RunResult:
        repetition_results = []
        aggregated_results = []
        for repetition in run.repetitions:
            result = repetition_evaluation.ArchitectureRepetitionEvaluation().run_evaluation(metrics, repetition)
            repetition_results.append(result)
            aggregated_results.append(result.aggregated_result)
        aggregated = base_evaluation.ArchitectureEvaluation().aggregate(aggregated_results)
        return ev_model.RunResult(run.dataset, run.approach, run.strategy, repetition_results, aggregated)


class StatisticalActivityRunEvaluation(RunBasedEvaluation):

    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], run: ev_model.Run) -> ev_model.RunResult:
        repetition_results = []
        aggregated_results = []
        for repetition in run.repetitions:
            result = repetition_evaluation.StatisticalActivityRepetitionEvaluation().run_evaluation(metrics,
                                                                                                    repetition)
            repetition_results.append(result)
            aggregated_results.append(result.aggregated_result)
        aggregated = base_evaluation.StatisticalActivityEvaluation().aggregate(aggregated_results)
        return ev_model.RunResult(run.dataset, run.approach, run.strategy, repetition_results, aggregated)


class StatisticalPrefixRunEvaluation(RunBasedEvaluation):

    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], run: ev_model.Run) -> ev_model.RunResult:
        repetition_results = []
        aggregated_results = []
        for repetition in run.repetitions:
            result = repetition_evaluation.StatisticalPrefixRepetitionEvaluation().run_evaluation(metrics,
                                                                                                  repetition)
            repetition_results.append(result)
            aggregated_results.append(result.aggregated_result)
        aggregated = base_evaluation.StatisticalPrefixEvaluation().aggregate(aggregated_results)
        return ev_model.RunResult(run.dataset, run.approach, run.strategy, repetition_results, aggregated)


class PerActivityRunEvaluation(RunBasedEvaluation):

    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], run: ev_model.Run) -> ev_model.RunResult:
        repetition_results = []
        aggregated_results = []
        for repetition in run.repetitions:
            result = repetition_evaluation.PerActivityRepetitionEvaluation().run_evaluation(metrics, repetition)
            repetition_results.append(result)
            aggregated_results.append(result.aggregated_result)

        aggregated = base_evaluation.PerActivityEvaluation().aggregate(aggregated_results)
        return ev_model.RunResult(run.dataset, run.approach, run.strategy, repetition_results, aggregated)


class PerPrefixLengthRunEvaluation(RunBasedEvaluation):

    def run_evaluation(self, metrics: typing.List[core_metrics.Metric], run: ev_model.Run,
                       min_pref_length: int = 2) -> ev_model.RunResult:
        repetition_results = []
        aggregated_results = []
        for repetition in run.repetitions:
            result = repetition_evaluation.PerPrefixLengthRepetitionEvaluation().run_evaluation(metrics,
                                                                                                repetition,
                                                                                                min_pref_length)
            repetition_results.append(result)
            aggregated_results.append(result.aggregated_result)

        aggregated = base_evaluation.PerPrefixLengthEvaluation().aggregate(aggregated_results)
        return ev_model.RunResult(run.dataset, run.approach, run.strategy, repetition_results, aggregated)
