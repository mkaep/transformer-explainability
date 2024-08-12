import typing

import click

from processtransformer.ml.evaluate import ev_model, core_metrics
from processtransformer.ml.evaluate.runs import run_evaluation


class ExperimentBasedEvaluation:

    def run_evaluation(self, evaluation_model: ev_model.EvaluationModel,
                       metrics: typing.List[core_metrics.Metric]) -> ev_model.EvaluationModelResult:
        raise NotImplementedError


class TotalExperimentEvaluation(ExperimentBasedEvaluation):
    def run_evaluation(self, evaluation_model: ev_model.EvaluationModel,
                       metrics: typing.List[core_metrics.Metric]) -> ev_model.EvaluationModelResult:
        results = []
        for run in evaluation_model.runs:
            results.append(run_evaluation.TotalRunEvaluation().run_evaluation(metrics, run))
        return ev_model.EvaluationModelResult(results)


class ArchitectureExperimentEvaluation(ExperimentBasedEvaluation):
    def run_evaluation(self, evaluation_model: ev_model.EvaluationModel,
                       metrics: typing.List[core_metrics.Metric]) -> ev_model.EvaluationModelResult:
        results = []
        for run in evaluation_model.runs:
            results.append(run_evaluation.ArchitectureRunEvaluation().run_evaluation(metrics, run))
        return ev_model.EvaluationModelResult(results)


class StatisticalActivityExperimentEvaluation(ExperimentBasedEvaluation):
    def run_evaluation(self, evaluation_model: ev_model.EvaluationModel,
                       metrics: typing.List[core_metrics.Metric]) -> ev_model.EvaluationModelResult:
        if len(metrics) > 0:
            click.secho('WARNING: Metrics does have any effects to this analysis and will be ignored', fg='yellow')
        results = []
        for run in evaluation_model.runs:
            results.append(run_evaluation.StatisticalActivityRunEvaluation().run_evaluation([], run))
        return ev_model.EvaluationModelResult(results)


class StatisticalPrefixExperimentEvaluation(ExperimentBasedEvaluation):
    def run_evaluation(self, evaluation_model: ev_model.EvaluationModel,
                       metrics: typing.List[core_metrics.Metric]) -> ev_model.EvaluationModelResult:
        if len(metrics) > 0:
            click.secho('WARNING: Metrics does have any effects to this analysis and will be ignored', fg='yellow')
        results = []
        for run in evaluation_model.runs:
            results.append(run_evaluation.StatisticalPrefixRunEvaluation().run_evaluation([], run))
        return ev_model.EvaluationModelResult(results)


class PerActivityExperimentEvaluation(ExperimentBasedEvaluation):
    def run_evaluation(self, evaluation_model: ev_model.EvaluationModel,
                       metrics: typing.List[core_metrics.Metric]) -> ev_model.EvaluationModelResult:
        results = []
        for run in evaluation_model.runs:
            results.append(run_evaluation.PerActivityRunEvaluation().run_evaluation(metrics, run))
        return ev_model.EvaluationModelResult(results)


class PerPrefixLengthExperimentEvaluation(ExperimentBasedEvaluation):
    def run_evaluation(self, evaluation_model: ev_model.EvaluationModel,
                       metrics: typing.List[core_metrics.Metric]) -> ev_model.EvaluationModelResult:
        results = []
        for run in evaluation_model.runs:
            results.append(run_evaluation.PerPrefixLengthRunEvaluation().run_evaluation(metrics, run, 2))
        return ev_model.EvaluationModelResult(results)
