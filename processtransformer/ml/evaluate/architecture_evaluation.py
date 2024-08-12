import pandas as pd
import os
import typing
import json as js
from processtransformer.ml import evaluate


def access_measurement(measure_file: str, name: str) -> typing.Any:
    assert os.path.isfile(measure_file), f'The requested measurement file {measure_file} does not exist'
    with open(measure_file, 'r', encoding='utf8') as f:
        measurements = js.load(f)

    try:
        value = measurements[name]
    except KeyError:
        value = None
    return value


def evaluate_training_time(experiment_dir: str, target_file: str = None, aggregated: bool = False):
    assert evaluate.is_finished(os.path.join(experiment_dir, '.jobs'), ignore_failed=True)
    failed_jobs = evaluate.load_failed_jobs(experiment_dir)

    lines = []

    datasets = os.listdir(experiment_dir)
    datasets.remove('.jobs')
    for dataset in datasets:
        dataset_dir = os.path.join(experiment_dir, dataset)
        approaches = os.listdir(dataset_dir)
        approaches.remove('_common_data')
        for approach in approaches:
            if evaluate.approach_failed_on_dataset(approach, dataset, failed_jobs) is True:
                continue
            approach_dir = os.path.join(dataset_dir, approach)
            repetitions = os.listdir(approach_dir)
            for rep in repetitions:
                repetitions_dir = os.path.join(approach_dir, rep)
                folds = os.listdir(repetitions_dir)
                for fold in folds:
                    folds_dir = os.path.join(repetitions_dir, fold)
                    strategies = os.listdir(folds_dir)
                    for strategy in strategies:
                        strategy_dir = os.path.join(folds_dir, strategy)
                        file = os.path.join(strategy_dir, 'additional_measurements.json')
                        value = access_measurement(file, 'elapsed_time')
                        lines.append([dataset, approach, rep, fold, strategy, value])

    dataframe = pd.DataFrame(lines, columns=['dataset', 'approach', 'rep', 'fold', 'strategy', 'elapsed_time'])
    dataframe = dataframe.astype(str)
    dataframe = dataframe.pivot(columns=['approach', 'rep', 'fold'], index=['dataset', 'strategy'],
                                values='elapsed_time')

    if target_file is not None:
        with open(target_file, 'w', encoding='utf8') as f:
            f.write(dataframe.to_latex())
    print(dataframe)


def evaluate_trainable_weights(experiment_dir: str, target_file: str = None, detailed: bool = False):
    assert evaluate.is_finished(os.path.join(experiment_dir, '.jobs'), ignore_failed=True)

    failed_jobs = evaluate.load_failed_jobs(experiment_dir)

    lines = []
    datasets = os.listdir(experiment_dir)
    datasets.remove('.jobs')
    for dataset in datasets:
        dataset_dir = os.path.join(experiment_dir, dataset)
        approaches = os.listdir(dataset_dir)
        approaches.remove('_common_data')
        for approach in approaches:
            if evaluate.approach_failed_on_dataset(approach, dataset, failed_jobs) is True:
                continue

            approach_dir = os.path.join(dataset_dir, approach)
            if detailed is False:
                file = os.path.join(approach_dir, 'rep_0', 'fold_0', 'base', 'additional_measurements.json')
                value = access_measurement(file, 'trainable_weights')
                lines.append([dataset, approach, value['trainable_weights']])
            else:
                repetitions = os.listdir(approach_dir)
                for rep in repetitions:
                    repetitions_dir = os.path.join(approach_dir, rep)
                    folds = os.listdir(repetitions_dir)
                    for fold in folds:
                        folds_dir = os.path.join(repetitions_dir, fold)
                        strategies = os.listdir(folds_dir)
                        for strategy in strategies:
                            strategy_dir = os.path.join(folds_dir, strategy)
                            file = os.path.join(strategy_dir, 'additional_measurements.json')
                            value = access_measurement(file, 'trainable_weights')
                            lines.append([dataset, approach, rep, fold, strategy, value])

    if detailed is True:
        dataframe = pd.DataFrame(lines, columns=['dataset', 'approach', 'rep', 'fold', 'strategy', 'trainable_weights'])
        dataframe = dataframe.astype(str)
        dataframe = dataframe.pivot(columns=['approach', 'rep', 'fold'], index=['dataset', 'strategy'],
                                    values='trainable_weights')
    else:
        dataframe = pd.DataFrame(lines, columns=['dataset', 'approach', 'trainable_weights'])
        dataframe = dataframe.astype(str)
        dataframe = dataframe.pivot(index='approach', columns='dataset', values='trainable_weights')

    if target_file is not None:
        with open(target_file, 'w', encoding='utf8') as f:
            f.write(dataframe.to_latex())
    print(dataframe)
