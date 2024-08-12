import math
import typing

import numpy as np
import pandas as pd
import os

from processtransformer.ml.evaluate.significance import stuart_maxwell
from processtransformer.ml.evaluate import ev_model, core_metrics
from processtransformer.ml.evaluate.runs import experiment_evaluation
from processtransformer.ml import evaluate
from processtransformer.ml.visualize import visualizer

JOBS_DIR = '.jobs'
COMMON_DATA_DIR = '_common_data'


def build_evaluation_model(experiment_dir: str, strategies: typing.List[str]) -> ev_model.EvaluationModel:
    assert evaluate.is_finished(os.path.join(experiment_dir, JOBS_DIR), ignore_failed=True)
    failed_jobs = evaluate.load_failed_jobs(experiment_dir)

    found_datasets = os.listdir(experiment_dir)
    found_datasets.remove(JOBS_DIR)
    runs = []
    for dataset in found_datasets:
        dataset_dir = os.path.join(experiment_dir, dataset)
        found_approaches = os.listdir(dataset_dir)
        found_approaches.remove(COMMON_DATA_DIR)
        for approach in found_approaches:
            if evaluate.approach_failed_on_dataset(approach, dataset, failed_jobs) is True:
                continue
            approach_dir = os.path.join(dataset_dir, approach)
            found_repetitions = os.listdir(approach_dir)
            for strategy in strategies:
                repetitions = []
                for rep in found_repetitions:
                    repetitions_dir = os.path.join(approach_dir, rep)
                    found_folds = os.listdir(repetitions_dir)
                    folds = []
                    for fold in found_folds:
                        folds_dir = os.path.join(repetitions_dir, fold)
                        strategy_dir = os.path.join(folds_dir, strategy)

                        labeled_file = os.path.join(dataset_dir, COMMON_DATA_DIR, rep, fold, 'test', 'true_result.csv')
                        test_prefixes_file = os.path.join(dataset_dir, COMMON_DATA_DIR, rep, fold, 'test',
                                                          'test_pref.xes')
                        prediction_file = os.path.join(strategy_dir, 'result.csv')
                        additional_measures_file = os.path.join(strategy_dir, 'additional_measurements.json')

                        folds.append(ev_model.Fold(id=fold,
                                                   labeled_file=labeled_file,
                                                   prediction_file=prediction_file,
                                                   test_prefixes_file=test_prefixes_file,
                                                   additional_measures_file=additional_measures_file))
                    repetitions.append(ev_model.Repetition(id=rep, folds=folds))
                runs.append(ev_model.Run(dataset, approach, strategy, repetitions))
    return ev_model.EvaluationModel(runs)


def get_run_result(results: typing.List[ev_model.RunResult],
                   rule: typing.Callable[[ev_model.RunResult], bool]) -> typing.Union[ev_model.RunResult, None]:
    return next((filter(rule, results)), None)


def get_fold_result_from_run_result(result: ev_model.RunResult, rep_id: str,
                                    fold_id: str) -> typing.Union[ev_model.BaseType, None]:
    repetition = next((r for r in result.results if r.id == rep_id), None)
    if repetition is not None:
        fold = next((f for f in repetition.results if f.id == fold_id), None)
        if fold is not None:
            return fold.results
    return None


def evaluate_architecture(experiment_dir: str, strategies: typing.List[str], aggregate_on='run', n_precision: int = 2,
                          target_file: str = None) -> pd.DataFrame:
    assert aggregate_on in {'fold', 'repetition', 'run'}

    evaluation_model = build_evaluation_model(experiment_dir, strategies)
    architecture_result = experiment_evaluation.ArchitectureExperimentEvaluation()\
        .run_evaluation(evaluation_model, [core_metrics.ArchitectureWrapper(['elapsed_time', 'trainable_weights'])])
    data_df = architecture_result.to_dataframe(aggregate_on=aggregate_on)

    data_df['value'] = data_df['value'].apply(
        lambda x: str(round(x, n_precision)) if math.floor(x) - x != 0.0 else str(int(x)))

    if aggregate_on == 'fold':
        data_df = data_df.pivot(columns=['approach', 'repetition', 'fold'], index=['dataset', 'strategy', 'metric'],
                                values='value')
    elif aggregate_on == 'repetition':
        data_df = data_df.pivot(columns=['approach', 'repetition'], index=['dataset', 'strategy', 'metric'],
                                values='value')
    elif aggregate_on == 'run':
        data_df = data_df.pivot(columns=['approach'], index=['dataset', 'strategy', 'metric'],
                                values='value')
    else:
        raise ValueError(f'The given aggregate value {aggregate_on} is not supported.')

    if target_file is not None:
        with open(target_file, 'w', encoding='utf8') as f:
            f.write(data_df.to_latex())
    return data_df


def build_metric(metric_name: str) -> core_metrics.Metric:
    if metric_name == 'Accuracy':
        return core_metrics.Accuracy()
    if metric_name == 'Recall':
        return core_metrics.Recall()
    if metric_name == 'Precision':
        return core_metrics.Precision()
    if metric_name == 'F1-Score (macro)':
        return core_metrics.MacroF1Score()
    if metric_name == 'F1-Score (micro)':
        return core_metrics.MacroF1Score()


def calculcate_significance_of_strategies_on_datasets_and_approaches(experiment_dir: str, strategies: typing.List[str],
                                                           datasets: typing.List[str],
                                                           approaches: typing.List[str], n_precision: int = 3,
                                                           target_file: str = None):
    if 'base' in strategies:
        strategies.remove('base')
    data_df = pd.DataFrame()
    for i, strategy in enumerate(strategies):
        result = calculate_significance_of_strategy_on_datasets_and_approaches(experiment_dir, strategy, datasets,
                                                                      approaches)
        if i == 0:
            data_df = result
        else:
            data_df = pd.concat([data_df, result], ignore_index=True)

    data_df = data_df.pivot(columns=['dataset'], index=['approach', 'strategy'], values='p_value')
    return data_df


def calculate_significance_of_strategy_on_datasets_and_approaches(experiment_dir: str, strategy: str,
                                                                  datasets: typing.List[str],
                                                                  approaches: typing.List[str]) -> pd.DataFrame:
    assert strategy != 'base', f'Cannot compare base with itself'
    base_evaluation_model = build_evaluation_model(experiment_dir, ['base'])
    strategy_evaluation_model = build_evaluation_model(experiment_dir, [strategy])

    filtered_base_evaluation_model = base_evaluation_model.filter_by(
        lambda x: x.dataset in datasets and x.approach in approaches and x.strategy == 'base')

    filtered_strategy_evaluation_model = strategy_evaluation_model.filter_by(
        lambda x: x.dataset in datasets and x.approach in approaches and x.strategy == strategy)

    lines = []
    for dataset in datasets:
        for approach in approaches:
            base_run = filtered_base_evaluation_model.filter_by(lambda x: x.dataset == dataset and x.approach == approach and x.strategy == 'base')
            strat_run = filtered_strategy_evaluation_model.filter_by(lambda x: x.dataset == dataset and x.approach == approach and x.strategy == strategy)

            if len(base_run.runs) == 1 and len(strat_run.runs) == 1:
                assert len(base_run.runs[0].repetitions) == 1 and len(strat_run.runs[0].repetitions) == 1
                base_folds = base_run.runs[0].repetitions[0].folds
                strat_folds = strat_run.runs[0].repetitions[0].folds
                assert len(base_folds) == 1 and len(strat_folds) == 1
                base_prediction = base_folds[0].prediction_file
                strat_prediction = strat_folds[0].prediction_file

                _, _, p_value = stuart_maxwell.perform_significance_test(base_prediction, strat_prediction)

                lines.append([dataset, approach, strategy, p_value])
            elif len(base_run.runs) == 0 or len(strat_run.runs) == 0:
                lines.append([dataset, approach, strategy, np.Inf])
            else:
                raise ValueError(f'Should not happen. There are more runs than expected for {dataset, approach, strategy}')

    return pd.DataFrame(lines, columns=['dataset', 'approach', 'strategy', 'p_value'])


def evaluate_correlations(experiment_dir: str, strategies: typing.List[str], datasets: typing.List[str],
                  approaches: typing.List[str], x_property: str = '', y_property: str='', aggregate_on: str = 'run',
                  target_file: str = None) -> pd.DataFrame:

    supported_properties = ['']
    assert x_property in supported_properties, f'Currently only {supported_properties} are supported x_properties'
    assert y_property in supported_properties, f'Currently only {supported_properties} are supported y_properties'

    evaluation_model = build_evaluation_model(experiment_dir, strategies)
    architecture_result = experiment_evaluation.ArchitectureExperimentEvaluation() \
        .run_evaluation(evaluation_model, [core_metrics.ArchitectureWrapper(['trainable_weights_model'])])
    data_x_df = architecture_result.to_dataframe(aggregate_on=aggregate_on)

    data_y_df = pd.DataFrame()
    if 'base' in strategies:
        strategies.remove('base')
    for i, strategy in enumerate(strategies):
        result = evaluate_gain_of_strategy_on_datasets_and_approaches(experiment_dir, strategy, datasets,
                                                                      approaches, 'Accuracy',
                                                                      aggregate_on=aggregate_on)
        if i == 0:
            data_y_df = result
        else:
            data_y_df = pd.concat([data_y_df, result], ignore_index=True)

    data_x_df.drop('metric', axis=1, inplace=True)
    data_y_df.drop('metric', axis=1, inplace=True)
    data_x_df.rename(columns={'value': 'x_property'}, inplace=True)
    data_y_df.rename(columns={'gain': 'y_property'}, inplace=True)

    if aggregate_on == 'fold':
        columns = ['dataset', 'approach', 'strategy', 'repetition', 'fold']
    elif aggregate_on == 'repetition':
        columns = ['dataset', 'approach', 'strategy', 'repetition']
    elif aggregate_on == 'run':
        columns = ['dataset', 'approach', 'strategy']
    else:
        raise ValueError(f'The given aggregate value {aggregate_on} is not supported.')
    print(data_y_df)
    merged_df = data_x_df.merge(data_y_df, how='inner', on=columns)

    viz = visualizer.Visualizer(export_dir=r'D:\PBPM_Approaches\experiment\evaluation\img',
                                silent_mode=False)
    viz.plot_correlations(merged_df)
    print(merged_df)

    return pd.DataFrame()


def evaluate_gain_of_strategies_on_datasets_and_approaches(experiment_dir: str, strategies: typing.List[str],
                                                           datasets: typing.List[str],
                                                           approaches: typing.List[str], metric_name: str,
                                                           aggregate_on: str = 'fold', n_precision: int = 3,
                                                           target_file: str = None) -> pd.DataFrame:

    if 'base' in strategies:
        strategies.remove('base')
    data_df = pd.DataFrame()
    for i, strategy in enumerate(strategies):
        result = evaluate_gain_of_strategy_on_datasets_and_approaches(experiment_dir, strategy, datasets,
                                                                      approaches, metric_name,
                                                                      aggregate_on=aggregate_on)
        if i == 0:
            data_df = result
        else:
            data_df = pd.concat([data_df, result], ignore_index=True)

    data_df['gain'] = data_df['gain'].apply(
        lambda x: str(round(x, n_precision)) if math.floor(x) - x != 0.0 else str(int(x)))

    assert len(data_df['metric'].unique()) == 1, f'Result contains values for more than one metric'
    data_df.drop('metric', axis=1, inplace=True)

    if aggregate_on == 'fold':
        data_df = data_df.pivot(columns=['dataset', 'repetition', 'fold'], index=['approach', 'strategy'],
                                values='gain')
    elif aggregate_on == 'repetition':
        data_df = data_df.pivot(columns=['dataset', 'repetition'], index=['approach', 'strategy'], values='gain')
    elif aggregate_on == 'run':
        data_df = data_df.pivot(columns=['dataset'], index=['approach', 'strategy'], values='gain')
    else:
        raise ValueError(f'The given aggregate value {aggregate_on} is not supported.')

    if target_file is not None:
        with open(target_file, 'w', encoding='utf8') as f:
            f.write(data_df.to_latex())

    return data_df


def evaluate_gain_of_strategy_on_datasets_and_approaches(experiment_dir: str, strategy: str,
                                                         datasets: typing.List[str],
                                                         approaches: typing.List[str], metric_name: str,
                                                         aggregate_on: str = 'fold') -> pd.DataFrame:
    assert strategy != 'base', f'Cannot compare base with itself'
    base_evaluation_model = build_evaluation_model(experiment_dir, ['base'])
    strategy_evaluation_model = build_evaluation_model(experiment_dir, [strategy])
    filtered_base_evaluation_model = base_evaluation_model.filter_by(
        lambda x: x.dataset in datasets and x.approach in approaches and x.strategy == 'base')

    filtered_strategy_evaluation_model = strategy_evaluation_model.filter_by(
        lambda x: x.dataset in datasets and x.approach in approaches and x.strategy == strategy)

    total_result_base = experiment_evaluation.TotalExperimentEvaluation().run_evaluation(
        filtered_base_evaluation_model, [build_metric(metric_name)])
    total_result_base_df = total_result_base.to_dataframe(aggregate_on=aggregate_on)
    total_result_base_df.rename(columns={'value': 'base_value'}, inplace=True)
    total_result_base_df.drop('strategy', axis=1, inplace=True)

    total_result_strategy = experiment_evaluation.TotalExperimentEvaluation().run_evaluation(
        filtered_strategy_evaluation_model, [build_metric(metric_name)])
    total_result_strategy_df = total_result_strategy.to_dataframe(aggregate_on=aggregate_on)
    total_result_strategy_df.rename(columns={'value': 'strategy_value'}, inplace=True)
    total_result_strategy_df.drop('strategy', axis=1, inplace=True)

    if aggregate_on == 'fold':
        columns = ['dataset', 'approach', 'repetition', 'fold', 'metric']
    elif aggregate_on == 'repetition':
        columns = ['dataset', 'approach', 'repetition', 'metric']
    elif aggregate_on == 'run':
        columns = ['dataset', 'approach', 'metric']
    else:
        raise ValueError(f'The given aggregate value {aggregate_on} is not supported.')

    merged_df = total_result_base_df.merge(total_result_strategy_df, how='inner', on=columns)
    merged_df['gain'] = merged_df.apply(lambda x: x['strategy_value'] - x['base_value'], axis=1)
    merged_df['strategy'] = strategy
    merged_df.drop(['base_value', 'strategy_value'], axis=1, inplace=True)

    return merged_df


def evaluate_strategies_on_datasets_and_approaches(experiment_dir: str, strategies: typing.List[str],
                                                   datasets: typing.List[str],
                                                   approaches: typing.List[str], metric: str,
                                                   aggregate_on: str = 'fold', n_precision: int = 3,
                                                   target_file: str = None) -> pd.DataFrame:
    evaluation_model = build_evaluation_model(experiment_dir, strategies)
    filtered_evaluation_model = evaluation_model.filter_by(
        lambda x: x.dataset in datasets and x.approach in approaches and x.strategy in strategies)

    total_result = experiment_evaluation.TotalExperimentEvaluation().run_evaluation(
        filtered_evaluation_model, [build_metric(metric)])
    total_result_df = total_result.to_dataframe(aggregate_on=aggregate_on)

    total_result_df['value'] = total_result_df['value'].apply(
        lambda x: str(round(x, n_precision)) if math.floor(x) - x != 0.0 else str(int(x)))

    assert len(total_result_df['metric'].unique()) == 1, f'Result contains values for more than one metric'
    total_result_df.drop('metric', axis=1, inplace=True)

    if aggregate_on == 'fold':
        total_result_df = total_result_df.pivot(columns=['dataset', 'repetition', 'fold'],
                                                index=['approach', 'strategy'],
                                                values='value')
    elif aggregate_on == 'repetition':
        total_result_df = total_result_df.pivot(columns=['dataset', 'repetition'],
                                                index=['approach', 'strategy'],
                                                values='value')
    elif aggregate_on == 'run':
        total_result_df = total_result_df.pivot(columns=['dataset'],
                                                index=['approach', 'strategy'],
                                                values='value')
    else:
        raise ValueError(f'The given aggregate value {aggregate_on} is not supported.')

    if target_file is not None:
        with open(target_file, 'w', encoding='utf8') as f:
            f.write(total_result_df.to_latex())

    return total_result_df


def evaluate_strategies_on_dataset_and_approach(experiment_dir: str, dataset: str, approach: str,
                                                strategies: typing.List[str], metric: str):

    evaluation_model = build_evaluation_model(experiment_dir, strategies)
    filtered_evaluation_model = evaluation_model.filter_by(
        lambda x: x.dataset == dataset and x.approach == approach and x.strategy in strategies)
    total_result = experiment_evaluation.TotalExperimentEvaluation().run_evaluation(
        filtered_evaluation_model, [build_metric(metric)])
    per_activity_result = experiment_evaluation.PerActivityExperimentEvaluation().run_evaluation(
        filtered_evaluation_model, [build_metric(metric)])
    per_prefix_result = experiment_evaluation.PerPrefixLengthExperimentEvaluation().run_evaluation(
        filtered_evaluation_model, [build_metric(metric)])
    statistical_prefix_result = experiment_evaluation.StatisticalPrefixExperimentEvaluation().run_evaluation(
        filtered_evaluation_model, [build_metric(metric)])
    statistical_activity_result = experiment_evaluation.StatisticalActivityExperimentEvaluation().run_evaluation(
        filtered_evaluation_model, [build_metric(metric)])

    viz = visualizer.Visualizer(export_dir=r'D:\PBPM_Approaches\experiment\evaluation\img',
                                silent_mode=True)
    viz.plot_prefix_analysis_strategies_on_dataset_on_approach(per_prefix_result.to_dataframe(aggregate_on='run'),
                                                               statistical_prefix_result.to_dataframe(
                                                                   aggregate_on='run'),
                                                               dataset,
                                                               approach)
    viz.plot_activity_analysis_strategies_on_dataset_on_approach(per_activity_result.to_dataframe(aggregate_on='run'),
                                                                 statistical_activity_result.to_dataframe(
                                                                     aggregate_on='run'),
                                                                 dataset,
                                                                 approach)
    viz.plot_total_analysis_strategies_on_dataset_on_approach(total_result.to_dataframe(aggregate_on='run'),
                                                              dataset,
                                                              approach)
