import os
import typing
import json as js
import pm4py

from processtransformer.ml.core import model, loader, sample_creator
from processtransformer.ml.prepare import splitter
from processtransformer.ml.persistence import json
from processtransformer.ml.pipeline import job_executor
from processtransformer.ml.augmentation import augmentation_strategy
from processtransformer.ml import pipeline

from pm4py.objects.log.obj import EventLog


def split(event_log: EventLog, splitter_configuration: model.SplitterConfiguration, repetition,
          verbose=False) -> typing.List[splitter.SplitResult]:
    if verbose:
        print(f' Start splitting the event log into training and test data')
    if splitter_configuration.name == splitter.TimeSplitter.format_id():
        event_log_splitter = splitter.TimeSplitter()

        assert event_log_splitter.check_configuration(splitter_configuration)

        split_result = event_log_splitter.split(event_log,
                                                training_size=splitter_configuration.training_size,
                                                by=splitter_configuration.by)
    elif splitter_configuration.name == splitter.RandomSplitter.format_id():
        event_log_splitter = splitter.RandomSplitter()

        assert event_log_splitter.check_configuration(splitter_configuration)
        assert repetition <= len(splitter_configuration.seeds)-1

        split_result = event_log_splitter.split(event_log,
                                                training_size=splitter_configuration.training_size,
                                                seed=splitter_configuration.seeds[repetition])
    elif splitter_configuration.name == splitter.KFoldSplitter.format_id():
        event_log_splitter = splitter.KFoldSplitter()

        assert event_log_splitter.check_configuration(splitter_configuration)
        assert repetition <= len(splitter_configuration.seeds)-1

        split_result = event_log_splitter.split(event_log,
                                                folds=splitter_configuration.folds,
                                                seed=splitter_configuration.seeds[repetition])
    else:
        raise ValueError(f'Splitter with name {splitter_configuration.name} does not exist.')

    if verbose:
        print(f' Splitting done!')
    return split_result


def create_fold_dir(common_data_dir: str, repetition: int, fold: int) -> typing.Tuple[str, str, str]:
    fold_dir = os.path.join(os.path.join(common_data_dir, f'rep_{repetition}'), f'fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)

    train_dir = os.path.join(fold_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)

    test_dir = os.path.join(fold_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)

    return fold_dir, train_dir, test_dir


def build_augmentation_strategies_from_config(
        augmentation_strategies_configs: typing.List[model.AugmentationStrategyConfig]):
    augmentation_strategies = []
    for augmentation_strategy_config in augmentation_strategies_configs:
        augmentation_strategies.append(augmentation_strategy.AugmentationStrategyBuilder(
            configuration=augmentation_strategy_config).build())

    return augmentation_strategies


def run_pipeline(experiment: model.AugmentationExperiment, verbose=False) -> None:
    experiment_dir = os.path.join(experiment.run_dir, experiment.name)
    pipeline.create_basic_folder_structure(experiment, verbose)

    jobs: typing.List[model.Job] = []
    for dataset in experiment.event_logs:
        event_log = loader.Loader.load_event_log(dataset.file_path, verbose)

        augmentation_strategies = build_augmentation_strategies_from_config(experiment.augmentation_strategies)

        assert event_log is not None

        # TODO Call preprocessing if necessary

        common_data_dir = os.path.join(experiment_dir, dataset.name, '_common_data')
        pm4py.write_xes(event_log, os.path.join(common_data_dir, 'preprocessed.xes'))

        n_repetitions = experiment.splitter_configuration.repetitions
        if experiment.splitter_configuration.name == splitter.TimeSplitter.format_id():
            n_repetitions = 1

        for repetition in range(0, n_repetitions):
            split_result = split(event_log, experiment.splitter_configuration, repetition, verbose)

            for i, fold in enumerate(split_result):

                # Create fold directory
                fold_dir, train_dir, test_dir = create_fold_dir(common_data_dir, repetition, i)

                # Store split data
                train_log_file, test_log_file = fold.store(train_dir, test_dir, verbose)

                # Create test samples (i.e prefixes)
                test_pref_file, _, _ = sample_creator.create_test_samples(fold, experiment.min_pref_length, test_dir,
                                                                          verbose)

                # Augment training data
                if verbose:
                    print('Start augmentation...')
                augmented_files = {}
                for strategy in augmentation_strategies:
                    aug_dir = os.path.join(fold_dir, strategy.name)
                    os.makedirs(aug_dir, exist_ok=True)

                    print("BEFORE FIT:")
                    for aug in strategy.augmentors:
                        print(aug.to_string())
                    strategy.fit(event_log)
                    print("AFTER FIT:")
                    for aug in strategy.augmentors:
                        print(aug.to_string())
                    aug_event_log, aug_count, aug_record, aug_duration = strategy.augment(fold.train_log,
                                                                                          record_augmentation=True,
                                                                                          verbose=verbose)
                    # Store the aug_count and aug_record
                    with open(os.path.join(aug_dir, 'aug_count.json'), 'w', encoding='utf8') as f:
                        f.write(js.dumps(aug_count))
                    with open(os.path.join(aug_dir, 'aug_record.json'), 'w', encoding='utf8') as f:
                        f.write(js.dumps(aug_record))
                    with open(os.path.join(aug_dir, 'aug_time.json'), 'w', encoding='utf8') as f:
                        f.write(js.dumps({
                            'augmentation_duration': aug_duration
                        }))

                    augmented_train_file = os.path.join(aug_dir, 'train.xes')
                    pm4py.write_xes(aug_event_log, augmented_train_file)
                    augmented_files[strategy.name] = augmented_train_file

                # Create Jobs for training and testing
                for approach in experiment.approaches:

                    base_job_directory = os.path.join(experiment_dir, dataset.name, approach.name, f'rep_{repetition}',
                                                      f'fold_{i}', 'base')
                    jobs.append(model.Job(approach, dataset.name, repetition, i, train_log_file, test_pref_file,
                                          os.path.join(common_data_dir, 'preprocessed.xes'), base_job_directory, {}))
                    for aug_strat in augmented_files.keys():
                        job_directory = os.path.join(experiment_dir, dataset.name, approach.name, f'rep_{repetition}',
                                                     f'fold_{i}', f'{aug_strat}')
                        jobs.append(model.Job(approach, dataset.name, repetition, i, augmented_files[aug_strat],
                                              test_pref_file, os.path.join(common_data_dir, 'preprocessed.xes'),
                                              job_directory, {}))

                # Store the jobs as jsonl
                job_targets = os.path.join(experiment_dir, '.jobs', 'jobs.jsonl')
                json.JsonJobExporter(job_targets).save(jobs, verbose)

    executor = job_executor.JobExecutor(os.path.join(experiment_dir, '.jobs'), jobs, verbose)
    executor.run()
