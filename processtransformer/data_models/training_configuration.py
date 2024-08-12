
import dataclasses
from processtransformer import constants
from processtransformer.data_models.json_serializable import JsonSerializable


@dataclasses.dataclass
class TransformerParameters(JsonSerializable):
    task: constants.Task
    epochs: int
    batch_size: int
    learning_rate: float
    gpu: int

    def to_dict(self):
        return {
            'task': self.task.value,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'gpu': self.gpu,
        }

    @staticmethod
    def from_dict(data) -> 'TransformerParameters':
        value = data['task']
        return TransformerParameters(
            task=(constants.Task(value)),
            epochs=data['epochs'],
            batch_size=data['batch_size'],
            learning_rate=data['learning_rate'],
            gpu=data['gpu'],
        )


@dataclasses.dataclass
class DataSource(JsonSerializable):
    original_data: str
    train_data: str
    test_data: str
    result_dir: str

    def to_dict(self):
        return {
            'original_data': self.original_data,
            'train_data': self.train_data,
            'test_data': self.test_data,
            'result_dir': self.result_dir,
        }

    @staticmethod
    def from_dict(data) -> 'DataSource':
        return DataSource(
            original_data=data['original_data'],
            train_data=data['train_data'],
            test_data=data['test_data'],
            result_dir=data['result_dir'],
        )


@dataclasses.dataclass
class TrainingConfiguration(JsonSerializable):
    name: str
    prefix_and_y_true_log: str  # may be None
    data_source: DataSource
    transformer_params: TransformerParameters

    def to_dict(self):
        return {
            'name': self.name,
            'prefix_and_y_true_log: str': self.prefix_and_y_true_log,
            'data_source': self.data_source.to_dict(),
            'transformer_params': self.transformer_params.to_dict(),
        }

    @staticmethod
    def from_dict(data) -> 'TrainingConfiguration':
        try:
            prefix_and_y_true_log = data['prefix_and_y_true_log']
        except KeyError:
            prefix_and_y_true_log = None

        return TrainingConfiguration(
            name=data['name'],
            prefix_and_y_true_log=prefix_and_y_true_log,
            data_source=DataSource.from_dict(data['data_source']),
            transformer_params=TransformerParameters.from_dict(data['transformer_params'])
        )
