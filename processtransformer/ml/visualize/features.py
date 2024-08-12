import dataclasses
import os
import typing
import pandas as pd
import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog
from processtransformer.ml.core import model
from processtransformer.ml.evaluate import evaluation
from processtransformer.ml.evaluate import core_metrics
from processtransformer.ml.evaluate import significance
from scipy import stats


@dataclasses.dataclass
class DataFrameRow:
    id: str
    prefix_length: int


def extract_dataset_names_from_directory(experiment_dir: str):
    return os.listdir(experiment_dir)


@dataclasses.dataclass
class JobResultDataFrameRow:
    dataset: str
    approach: str
    repetition: int
    fold: int
    ground_truth_file: str
    result_file: str
    pref_file: str


def build_pairs(n_repetitions: int, n_folds: int, experiment: model.Experiment):
    experiment_dir = os.path.join(experiment.run_dir, experiment.name)

    incomplete = []
    complete = []
    for dataset in experiment.event_logs:
        for approach in experiment.approaches:
            for i in range(n_repetitions):
                for j in range(n_folds):
                    ground_truth_file = os.path.join(experiment_dir, dataset.name, '_common_data', f'rep_{i}',
                                                     f'fold_{j}', 'true_result.csv')
                    result_file = os.path.join(experiment_dir, dataset.name, approach.name, f'rep_{i}', f'fold_{j}',
                                               'result.csv')
                    pref_file = os.path.join(experiment_dir, dataset.name, '_common_data', f'rep_{i}', f'fold_{j}',
                                             'test_pref.xes')
                    if os.path.exists(ground_truth_file) and os.path.exists(result_file) and os.path.exists(pref_file):
                        complete.append(JobResultDataFrameRow(dataset.name, approach.name, i, j, ground_truth_file,
                                                              result_file, pref_file))
                    else:
                        incomplete.append((dataset.name, approach.name))
                        print(f'WARNING: At least one of the requested files {ground_truth_file} and {result_file} '
                              f'does not exist. We will proceed without approach {approach.name} on '
                              f'dataset {dataset.name}')

    # Remove datasets that are incomplete with regard to repetitions and folds due to comparison
    complete_df = pd.DataFrame(complete, columns=['dataset', 'approach', 'repetition', 'fold', 'ground_truth_file',
                                                  'result_file', 'pref_file'])

    for value in set(incomplete):
        complete_df = complete_df[complete_df.apply(lambda x: not (x['dataset'] == value[0]
                                                                   and x['approach'] == value[1]), axis=1)]

    return complete_df


def add_prefix_length(res_file: pd.DataFrame, test_log: EventLog):
    rows = []
    for trace in test_log:
        rows.append(DataFrameRow(trace.attributes['concept:name'], len(trace)))
    return res_file.merge(pd.DataFrame(rows), how='inner', on='id')


def summarize_fold_results(true_file: str, predictions_file: str, test_samples_file: str) -> pd.DataFrame:
    test_log = pm4py.read_xes(test_samples_file)

    true_df = pd.read_csv(true_file, sep='\t')
    true_df = add_prefix_length(true_df, test_log)
    true_df = true_df[['id', model.PredictionTask.NEXT_ACTIVITY.value, 'prefix_length']]
    true_df.columns = ['id', f'true', 'prefix_length']

    pred_df = pd.read_csv(predictions_file, sep='\t')
    pred_df = pred_df[['id', model.PredictionTask.NEXT_ACTIVITY.value]]
    pred_df.columns = ['id', f'pred']

    return pred_df.merge(true_df, how='inner', on='id')


def evaluate_single_fold(true_file: str, predictions_file: str, test_samples_file: str,
                         metric_names: typing.List[str]) -> typing.Dict[str, float]:
    merged_df = summarize_fold_results(true_file, predictions_file, test_samples_file)
    print(merged_df)
    # Calculate metrics
    assert 'true' in merged_df.columns and 'pred' in merged_df.columns
    assert set(metric_names).issubset(set(metrics.MetricCalculator.get_available_metrics()))

    return metrics.MetricCalculator(metric_names).calculate_metrics(merged_df['true'], merged_df['pred'])


def evaluate_single_repetition(dataset: str, approach: str, repetition: int, experiment: model.Experiment,
                               metric_names: typing.List[str]):
    pairs_df = build_pairs(2, 2, experiment)

    pairs_df = pairs_df[pairs_df.apply(lambda x: x['dataset'] == dataset and x['approach'] == approach and
                                                 x['repetition'] == repetition, axis=1)]
    print(pairs_df)
    results = []
    for _, row in pairs_df.iterrows():
        results.append(
            evaluate_single_fold(row['ground_truth_file'], row['result_file'], row['pref_file'], metric_names))

    print(results)
    n_folds = len(pairs_df)
    results_repetition = {}
    for metric in metric_names:
        results_repetition[metric] = sum([result[metric] for result in results]) / n_folds

    print(results_repetition)
    return results_repetition


def compare_approaches_on(approach_1, approach_2, dataset, experiment, metric_name: str):
    n_repetitions = 2
    n_folds = 2

    pairs_df = build_pairs(n_repetitions, n_folds, experiment)
    pairs_df = pairs_df[pairs_df.apply(lambda x: x['dataset'] == dataset and (x['approach'] == approach_1
                                                                              or x['approach'] == approach_2), axis=1)]

    appr_1 = pairs_df[pairs_df.apply(lambda x: x['approach'] == approach_1, axis=1)]
    appr_2 = pairs_df[pairs_df.apply(lambda x: x['approach'] == approach_2, axis=1)]

    assert len(appr_1) == n_repetitions * n_folds
    assert len(appr_2) == n_repetitions * n_folds
    assert len(appr_1) == len(appr_2)

    # Apply significance tests
    if n_repetitions == 1 and n_folds == 1:
        print('Perform McNemar Significance Test')
    elif n_repetitions > 1 and n_folds == 1:
        print('perform corrected t test Significance Test')
    elif n_repetitions > 1 and n_folds > 1:
        observations = []
        for i in range(n_repetitions):
            for j in range(n_folds):
                observations.append([approach_1, i, j, evaluate_single_repetition(dataset, approach_1, i, experiment, [metric_name])[metric_name]])
                observations.append([approach_2, i, j, evaluate_single_repetition(dataset, approach_2, i, experiment, [metric_name])[metric_name]])

        t = significance.CorrectedRepeatedKFoldCVTest().calculate_t_value(n_repetitions, n_folds, pd.DataFrame(observations, columns=['approach', 'repetition', 'fold', 'metric']))
        print(t)
        degrees_of_freedom = n_repetitions * n_folds - 1
        # https://www.reneshbedre.com/blog/ttest-from-scratch.html
        p = 2 * (1 - stats.t.cdf(x=t, df=degrees_of_freedom))
        print(f'p_value: {p}')
        alpha = 0.05
        # Compare with p value from t statistic
        if p > alpha:
            print('Accept null hypothesis. There is no significant difference')
        else:
            print('Reject the null hypothesis. There is a significant difference')

        print('perform cv corrected t-test')
    else:
        raise ValueError('Not appropriate significance test available')

    pass