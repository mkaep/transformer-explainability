
import dataclasses

from pm4py.objects.petri_net.obj import Marking, PetriNet

from processtransformer.xai.visualization.common.figure_data import FigureData
from processtransformer.xai.visualization.output_models.output_data import OutputData


@dataclasses.dataclass
class PetriNetOutput(OutputData):
    net: PetriNet
    initial_marking: Marking
    final_marking: Marking
    figure_data: FigureData
