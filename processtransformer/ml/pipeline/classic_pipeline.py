import os
import typing
import pm4py

from processtransformer.ml.prepare import splitter
from processtransformer.ml import pipeline
from processtransformer.ml.core.loader import Loader
from processtransformer.ml.core import sample_creator, model
from processtransformer.ml.persistence import json
from processtransformer.ml.pipeline import job_executor


def run_pipeline(experiment: model.Experiment, verbose=False):
    experiment_dir = os.path.join(experiment.run_dir, experiment.name)
    pipeline.create_basic_folder_structure(experiment, verbose)

    # Split dataset into train and test
    jobs: typing.List[model.Job] = []
    for dataset in experiment.event_logs:
        event_log = Loader.load_event_log(dataset.file_path, verbose)

        assert event_log is not None

        common_data_dir = os.path.join(experiment_dir, dataset.name, '_common_data')
        pm4py.write_xes(event_log, os.path.join(common_data_dir, 'preprocessed.xes'))

        valid_configuration = False
        if experiment.splitter_configuration.name == splitter.TimeSplitter.format_id():
            valid_configuration = splitter.TimeSplitter.check_configuration(experiment.splitter_configuration)
        elif experiment.splitter_configuration.name == splitter.RandomSplitter.format_id():
            valid_configuration = splitter.RandomSplitter.check_configuration(experiment.splitter_configuration)
        elif experiment.splitter_configuration.name == splitter.KFoldSplitter.format_id():
            valid_configuration = splitter.KFoldSplitter.check_configuration(experiment.splitter_configuration)
        else:
            raise ValueError(f'Splitter with name {experiment.splitter_configuration.name} does not exist.')

        assert valid_configuration

        n_repetitions = experiment.splitter_configuration.repetitions
        if experiment.splitter_configuration.name == splitter.TimeSplitter.format_id():
            n_repetitions = 1

        for repetition in range(0, n_repetitions):
            if verbose:
                # TODO Iteration mit aufnehmen, Progressbafr
                print(f' Start splitting the event log into training and test data')
            if experiment.splitter_configuration.name == splitter.TimeSplitter.format_id():
                split_result = splitter.TimeSplitter().split(event_log,
                                                             training_size=experiment.splitter_configuration.training_size,
                                                             by=experiment.splitter_configuration.by)
            elif experiment.splitter_configuration.name == splitter.KFoldSplitter.format_id():
                split_result = splitter.KFoldSplitter().split(event_log,
                                                              folds=experiment.splitter_configuration.folds,
                                                              seed=experiment.splitter_configuration.seeds[repetition])
            else:
                split_result = splitter.RandomSplitter().split(event_log,
                                                               training_size=experiment.splitter_configuration.training_size,
                                                               seed=experiment.splitter_configuration.seeds[repetition])
            if verbose:
                print(f' Splitting done!')

            for i, fold in enumerate(split_result):
                if verbose:
                    print(f' Storing event log on disk')
                fold_dir = os.path.join(common_data_dir, f'rep_{repetition}', f'fold_{i}')
                os.makedirs(fold_dir, exist_ok=True)

                train_dir = os.path.join(fold_dir, 'train')
                os.makedirs(train_dir, exist_ok=True)

                test_dir = os.path.join(fold_dir, 'test')
                os.makedirs(test_dir, exist_ok=True)

                # Store split data
                train_log_file, test_log_file = fold.store(train_dir, test_dir, verbose)

                # Create test samples (i.e prefixes)
                test_pref_file, test_suf_file, ground_truth_file = sample_creator.create_test_samples(fold, experiment.min_pref_length,
                                                                                       test_dir, verbose)

                for approach in experiment.approaches:
                    job_directory = os.path.join(experiment_dir, dataset.name, approach.name, f'rep_{repetition}',
                                                 f'fold_{i}')
                    jobs.append(model.Job(approach, dataset.name, repetition, i, train_log_file, test_pref_file,
                                          os.path.join(common_data_dir, 'preprocessed.xes'), job_directory, {}))

                # Store the jobs as JSON
                job_targets = os.path.join(experiment_dir, '.jobs', 'jobs.jsonl')
                json.JsonJobExporter(job_targets).save(jobs, verbose)

    executor = job_executor.JobExecutor(os.path.join(experiment_dir, '.jobs'), jobs, verbose)
    executor.run()
