
import abc
import dataclasses
import typing

from processtransformer.xai.visualization.output_models.output_data import OutputData


class VizOutput:
    """
    Output from visualization-funcs, e.g. where resulting files are stored.
    This can then be picked up by the GUI and shown to the user.
    """
    pass


@dataclasses.dataclass
class FigureOutput(VizOutput):
    path_to_figure: str


@dataclasses.dataclass
class TextOutput(VizOutput):
    text: str


class BaseViz(abc.ABC):
    accepts: typing.List[OutputData] = []

    @classmethod
    def get_accepted_formats(cls):
        return cls.accepts

    @abc.abstractmethod
    def visualize(self) -> typing.List[VizOutput]:
        pass
