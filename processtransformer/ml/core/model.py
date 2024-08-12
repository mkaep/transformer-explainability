import typing
import enum
import dataclasses
import abc


@dataclasses.dataclass
class Approach:
    name: str
    env_name: str
    dir: str
    hyperparameter: typing.Dict[str, typing.Union[str, int, float]]

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return {
            'name': self.name,
            'env_name': self.env_name,
            'dir': self.dir,
            'hyperparameter': self.hyperparameter
        }

    @staticmethod
    def from_dict(data) -> 'Approach':
        return Approach(
            name=data['name'],
            env_name=data['env_name'],
            dir=data['dir'],
            hyperparameter=data['hyperparameter']
        )


@dataclasses.dataclass
class Dataset:
    name: str
    file_path: str

    def to_dict(self) -> typing.Dict[str, str]:
        return {
            'name': self.name,
            'file_path': self.file_path
        }

    @staticmethod
    def from_dict(data) -> 'Dataset':
        return Dataset(
            name=data['name'],
            file_path=data['file_path']
        )


@dataclasses.dataclass
class SplitterConfiguration:
    name: str
    training_size: typing.Optional[float] = None
    by: typing.Optional[str] = None
    seeds: typing.Optional[typing.List[int]] = None
    repetitions: typing.Optional[int] = None
    folds: typing.Optional[int] = None

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return {
            'name': self.name,
            'training_size': self.training_size,
            'by': self.by,
            'seeds': self.seeds,
            'repetitions': self.repetitions,
            'folds': self.folds
        }

    @staticmethod
    def from_dict(data) -> 'SplitterConfiguration':
        return SplitterConfiguration(
            name=data['name'],
            training_size=data['training_size'],
            by=data['by'],
            seeds=data['seeds'],
            repetitions=data['repetitions'],
            folds=data['folds']
        )


@dataclasses.dataclass
class BaseExperiment(abc.ABC):
    name: str
    data_dir: str
    run_dir: str
    evaluation_dir: str
    event_logs: typing.List[Dataset]
    approaches: typing.List[Approach]
    splitter_configuration: SplitterConfiguration
    min_pref_length: int

    def get_dataset_names(self) -> typing.List[str]:
        dataset_names = [dataset.name for dataset in self.event_logs]
        return dataset_names

    def get_approach_names(self) -> typing.List[str]:
        approach_names = [approach.name for approach in self.approaches]
        return approach_names

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return {
            'name': self.name,
            'data_dir': self.data_dir,
            'run_dir': self.run_dir,
            'evaluation_dir': self.evaluation_dir,
            'event_logs': [event_log.to_dict() for event_log in self.event_logs],
            'approaches': [approach.to_dict() for approach in self.approaches],
            'splitter_configuration': self.splitter_configuration.to_dict(),
            'min_pref_length': self.min_pref_length
        }

    @staticmethod
    def from_dict(data):
        raise NotImplementedError()


@dataclasses.dataclass
class Experiment(BaseExperiment):
    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return super().to_dict()

    @staticmethod
    def from_dict(data) -> 'Experiment':
        return Experiment(
            name=data['name'],
            data_dir=data['data_dir'],
            run_dir=data['run_dir'],
            evaluation_dir=data['evaluation_dir'],
            event_logs=[Dataset.from_dict(t) for t in data['event_logs']],
            approaches=[Approach.from_dict(t) for t in data['approaches']],
            splitter_configuration=SplitterConfiguration.from_dict(data['splitter_configuration']),
            min_pref_length=data['min_pref_length']
        )


@dataclasses.dataclass
class AugmentorConfig:
    name: str
    parameters: typing.Dict[str, any]

    def to_dict(self):
        return {
            'name': self.name,
            'parameters': self.parameters
        }

    @staticmethod
    def from_dict(data) -> 'AugmentorConfig':
        return AugmentorConfig(
            name=data['name'],
            parameters=data['parameters']
        )


@dataclasses.dataclass
class AugmentationStrategyConfig:
    id: int
    name: str
    seed: int
    augmentors: typing.List[AugmentorConfig]
    augmentation_factor: float
    allow_multiple: bool

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'seed': self.seed,
            'augmentors': [augmentor.to_dict() for augmentor in self.augmentors],
            'augmentation_factor': self.augmentation_factor,
            'allow_multiple': self.allow_multiple,
        }

    @staticmethod
    def from_dict(data) -> 'AugmentationStrategyConfig':
        return AugmentationStrategyConfig(
            id=data['id'],
            name=data['name'],
            seed=data['seed'],
            augmentors=[AugmentorConfig.from_dict(augmentor) for augmentor in data['augmentors']],
            augmentation_factor=data['augmentation_factor'],
            allow_multiple=data['allow_multiple'],
        )

    def get_representation(self):
        return f'{self.id}_{self.name}_{self.allow_multiple}_{self.augmentation_factor}'


@dataclasses.dataclass
class AugmentationExperiment(BaseExperiment):
    augmentation_strategies: typing.List[AugmentationStrategyConfig]

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        dict_repr = super().to_dict()
        dict_repr['augmentation_strategies'] = [s.to_dict() for s in self.augmentation_strategies]
        return dict_repr

    @staticmethod
    def from_dict(data) -> 'AugmentationExperiment':
        return AugmentationExperiment(
            name=data['name'],
            data_dir=data['data_dir'],
            run_dir=data['run_dir'],
            evaluation_dir=data['evaluation_dir'],
            event_logs=[Dataset.from_dict(t) for t in data['event_logs']],
            approaches=[Approach.from_dict(t) for t in data['approaches']],
            splitter_configuration=SplitterConfiguration.from_dict(data['splitter_configuration']),
            min_pref_length=data['min_pref_length'],
            augmentation_strategies=[AugmentationStrategyConfig.from_dict(s)
                                     for s in data['augmentation_strategies']]
        )


@dataclasses.dataclass
class Job:
    approach: Approach
    dataset: str
    iteration: int
    fold: int
    path_training_file: str
    path_test_file: str
    path_complete_log: str
    job_directory: str
    metadata: typing.Dict[str, typing.Any]

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return {
            'approach': self.approach.to_dict(),
            'dataset': self.dataset,
            'iteration': self.iteration,
            'fold': self.fold,
            'path_training_file': self.path_training_file,
            'path_test_file': self.path_test_file,
            'path_complete_log': self.path_complete_log,
            'job_directory': self.job_directory,
            'metadata': self.metadata
        }

    @staticmethod
    def from_dict(data) -> 'Job':
        return Job(
            approach=Approach.from_dict(data['approach']),
            dataset=data['dataset'],
            iteration=data['iteration'],
            fold=data['fold'],
            path_training_file=data['path_training_file'],
            path_test_file=data['path_test_file'],
            path_complete_log=data['path_complete_log'],
            job_directory=data['job_directory'],
            metadata=data['metadata']
        )

    def get_conda_command(self):
        params = f'--original_data {self.path_complete_log} ' \
                 f'--train_data {self.path_training_file} ' \
                 f'--test_data {self.path_test_file} ' \
                 f'--result_dir {self.job_directory} '
        for param in self.approach.hyperparameter.keys():
            params = params + f'--{param} {str(self.approach.hyperparameter[param])} '

        return f'conda run --no-capture-output -n {self.approach.env_name} python {self.approach.dir} {params}'


@dataclasses.dataclass
class SampleResult:
    id: str
    pred_next_act: typing.Optional[any]
    pred_next_role: typing.Optional[any]
    pred_next_time: typing.Optional[any]
    pred_suffix: typing.Optional[typing.List[any]]
    pred_remaining_time: typing.Optional[any]


@enum.unique
class PredictionTask(enum.Enum):
    NEXT_ACTIVITY = 'pred_next_act'
    NEXT_ROLE = 'pred_next_role'
    NEXT_TIME = 'pred_next_time'
    SUFFIX = 'pred_suffix'
    REMAINING_TIME = 'pred_remaining_time'
