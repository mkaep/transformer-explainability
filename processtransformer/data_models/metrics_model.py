
import dataclasses

from processtransformer.data_models.json_serializable import JsonSerializable


@dataclasses.dataclass
class MetricsModel(JsonSerializable):
    accuracy: float
    precision_micro: float
    precision_macro: float
    precision_weighted: float
    recall_micro: float
    recall_macro: float
    recall_weighted: float
    f1_micro: float
    f1_macro: float
    f1_weighted: float

    def to_dict(self):
        return {
            'accuracy': self.accuracy,
            'precision_micro': self.precision_micro,
            'precision_macro': self.precision_macro,
            'precision_weighted': self.precision_weighted,
            'recall_micro': self.recall_micro,
            'recall_macro': self.recall_macro,
            'recall_weighted': self.recall_weighted,
            'f1_micro': self.f1_micro,
            'f1_macro': self.f1_macro,
            'f1_weighted': self.f1_weighted,
        }

    @staticmethod
    def from_dict(data):
        return MetricsModel(
            accuracy=data['accuracy'],
            precision_micro=data['precision_micro'],
            precision_macro=data['precision_macro'],
            precision_weighted=data['precision_weighted'],
            recall_micro=data['recall_micro'],
            recall_macro=data['recall_macro'],
            recall_weighted=data['recall_weighted'],
            f1_micro=data['f1_micro'],
            f1_macro=data['f1_macro'],
            f1_weighted=data['f1_weighted'],
        )
