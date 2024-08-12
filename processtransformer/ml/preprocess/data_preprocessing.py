

import argparse
import os

import pm4py

from processtransformer.ml.core import model, loader, sample_creator
from processtransformer.ml.persistence import json
from processtransformer.ml.pipeline import augmentation_pipeline
from processtransformer.ml.prepare import splitter


def preprocess(experiment: model.AugmentationExperiment, verbose=False):
    for dataset in experiment.event_logs:
        file_path = os.path.join(os.getcwd(), dataset.file_path)

        dataset_dir = os.path.join(os.path.dirname(file_path), experiment.name)
        print(dataset_dir)
        os.makedirs(dataset_dir, exist_ok=True)

        json.JsonExperimentExporter(os.path.join(dataset_dir, experiment.name + '.json')).save(experiment)

        event_log = loader.Loader.load_event_log(file_path, verbose)
        assert event_log is not None

        pm4py.write_xes(event_log, os.path.join(dataset_dir, 'preprocessed.xes'))

        n_repetitions = experiment.splitter_configuration.repetitions
        if experiment.splitter_configuration.name == splitter.TimeSplitter.format_id():
            n_repetitions = 1

        for repetition in range(0, n_repetitions):
            split_result = augmentation_pipeline.split(event_log, experiment.splitter_configuration,
                                                       repetition, verbose)

            for i, fold in enumerate(split_result):
                # Store split data
                fold.store(dataset_dir, dataset_dir, verbose)

                # Create test samples (i.e prefixes)
                test_pref_file, _, _ = sample_creator.create_test_samples(fold, experiment.min_pref_length,
                                                                          dataset_dir, verbose)


def run_experiment(experiment_file: str, verbose: bool = False):
    experiment = json.JsonExperimentImporter(experiment_file).load(verbose)

    preprocess(experiment, verbose)


def main():
    parser = argparse.ArgumentParser(description="Process Transformer - Data Preparation.")
    parser.add_argument("--experiment_file_path", type=str, help='Path of the experiment file')
    args = parser.parse_args()
    run_experiment(args.experiment_file_path)


if __name__ == "__main__":
    main()
