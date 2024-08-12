from processtransformer.ml.core import model
import typing
import os


def create_basic_folder_structure(experiment: typing.Union[model.Experiment, model.AugmentationExperiment],
                                  verbose=False) -> None:
    """
    Creates the experiment directory and within this directory for each dataset a directory. In each dataset directory
    for each approach an own approach directory is created as well as a shared directory called _common_data which
    contains data that is used by all approaches (e.g. train and test data)

    Note: If one of the directories that should be created already exists they are not overwritten.

    :param experiment:
    :param verbose:
    :return: nothing
    """
    if verbose:
        print(f' Start creating folder for {len(experiment.event_logs)} event logs and '
              f'{len(experiment.approaches)} approaches')

    experiment_dir = os.path.join(experiment.run_dir, experiment.name)
    os.makedirs(experiment_dir, exist_ok=True)

    for dataset in experiment.event_logs:
        if verbose:
            print(f' Create directory for event log {dataset.name}')
        for approach in experiment.approaches:
            if verbose:
                print(f'\t Create directory for approach {approach.name}')
            os.makedirs(os.path.join(experiment_dir, dataset.name, approach.name), exist_ok=True)
