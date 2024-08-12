import typing
from sklearn import metrics as metr


class BaseMetric:
    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def format(self) -> str:
        raise NotImplementedError()

    def calculate(self, y_true: typing.List, y_pred: typing.List) -> float:
        raise NotImplementedError()

    @staticmethod
    def aggregate(measures):
        raise NotImplementedError


Metric = typing.TypeVar('Metric', bound=BaseMetric)


class ArchitectureWrapper(BaseMetric):
    def __init__(self, items: typing.List[str]):
        self.items = items

    @property
    def format(self) -> str:
        return '.2%'

    @property
    def name(self) -> str:
        return 'Architecture'

    def calculate(self, y_true: typing.List, y_pred: typing.List) -> float:
        raise NotImplementedError()

    @staticmethod
    def aggregate(measures: typing.List[float]) -> float:
        raise NotImplementedError()


class Accuracy(BaseMetric):
    @property
    def format(self) -> str:
        return '.2%'

    @property
    def name(self) -> str:
        return 'Accuracy'

    def calculate(self, y_true: typing.List, y_pred: typing.List) -> float:
        assert len(y_true) == len(y_pred)
        return Accuracy.get_correct_predictions(y_true, y_pred) / len(y_true)

    @staticmethod
    def get_correct_predictions(y_true: typing.List, y_pred: typing.List) -> int:
        n_correct_predictions = 0
        for label, prediction in zip(y_true, y_pred):
            if label == prediction:
                n_correct_predictions += 1
        return n_correct_predictions

    @staticmethod
    def aggregate(measures: typing.List[float]) -> float:
        return sum([x for x in measures]) / len(measures)


class Recall(BaseMetric):
    @property
    def format(self) -> str:
        return '.%2'

    @property
    def name(self) -> str:
        return 'Recall'

    def calculate(self, y_true: typing.List, y_pred: typing.List) -> float:
        assert len(y_true) == len(y_pred)
        return metr.recall_score(y_true, y_pred, average='macro', zero_division=0)

    @staticmethod
    def aggregate(measures: typing.List[float]) -> float:
        return sum([x for x in measures]) / len(measures)


class Precision(BaseMetric):
    @property
    def format(self) -> str:
        return '.%2'

    @property
    def name(self) -> str:
        return 'Precision'

    def calculate(self, y_true: typing.List, y_pred: typing.List) -> float:
        assert len(y_true) == len(y_pred)
        return metr.precision_score(y_true, y_pred, average='macro', zero_division=0)

    @staticmethod
    def aggregate(measures: typing.List[float]) -> float:
        return sum([x for x in measures]) / len(measures)


class MacroF1Score(BaseMetric):
    @property
    def format(self) -> str:
        return '.%2'

    @property
    def name(self) -> str:
        return 'F1-Score (macro)'

    def calculate(self, y_true: typing.List, y_pred: typing.List) -> float:
        assert len(y_true) == len(y_pred)
        return metr.f1_score(y_true, y_pred, average='macro', zero_division=0)

    @staticmethod
    def aggregate(measures: typing.List[float]) -> float:
        return sum([x for x in measures]) / len(measures)


class MicroF1Score(BaseMetric):
    @property
    def format(self) -> str:
        return '.%2'

    @property
    def name(self) -> str:
        return 'F1-Score (micro)'

    def calculate(self, y_true: typing.List, y_pred: typing.List) -> float:
        assert len(y_true) == len(y_pred)
        return metr.f1_score(y_true, y_pred, average='micro', zero_division=0)

    @staticmethod
    def aggregate(measures: typing.List[float]) -> float:
        return sum([x for x in measures]) / len(measures)


class MetricCalculator:
    def __init__(self, metrics: typing.List[Metric]):
        self._metrics = metrics

    def calculate_metrics(self, y_true: typing.List, y_pred: typing.List) -> typing.Dict[str, typing.Any]:
        return {step.name: step.calculate(y_true, y_pred) for step in self._metrics}
