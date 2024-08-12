import os
import json
import typing
from processtransformer.ml.core import model

ExperimentType = typing.TypeVar('ExperimentType', bound=model.BaseExperiment)


class JsonExperimentImporter:
    def __init__(self, file_path: str):
        self._file_path = file_path

    def load(self, verbose=False) -> ExperimentType:
        if verbose:
            print(f' Start loading experiment from {self._file_path}.')

        assert os.path.getsize(self._file_path) > 0
        with open(self._file_path, 'r', encoding='utf8') as f:
            experiment_loaded = json.load(f)

        if 'augmentation_strategies' in experiment_loaded.keys():
            return model.AugmentationExperiment.from_dict(experiment_loaded)
        return model.Experiment.from_dict(experiment_loaded)


class JsonExperimentExporter:
    def __init__(self, target_file: str):
        self._target_file = target_file

    def save(self, experiment: ExperimentType, verbose=False):
        if verbose:
            print(f'Writing an experiment with {len(experiment.event_logs)} event logs and {len(experiment.approaches)}'
                  f'approaches to {self._target_file}.')
        os.makedirs(os.path.dirname(self._target_file), exist_ok=True)
        with open(self._target_file, 'w', encoding='utf8') as f:
            json.dump(experiment.to_dict(), f, indent=2)
        if verbose:
            print(f'Done!')


class JsonJobExporter:
    def __init__(self, target_file: str):
        self._target_file = target_file

    def save(self, jobs: typing.List[model.Job], verbose=False):
        if verbose:
            print(f'Storing {len(jobs)} jobs in {self._target_file}.')
        os.makedirs(os.path.dirname(self._target_file), exist_ok=True)
        with open(self._target_file, 'w', encoding='utf8') as f:
            for job in jobs:
                line = json.dumps(job.to_dict())
                f.write(f'{line}\n')
        if verbose:
            print(f'Done!')


class JsonJobImporter:
    def __init__(self, file_path: str):
        self._file_path = file_path

    def load(self, verbose=False) -> typing.List[model.Job]:
        if verbose:
            print(f'Loading jobs from {self._file_path}')

        jobs: typing.List[model.Job] = []
        with open(self._file_path, 'r', encoding='utf8') as f:
            for line in f:
                data = json.loads(line)
                jobs.append(model.Job.from_dict(data))

        if verbose:
            print('Done!')
        return jobs
