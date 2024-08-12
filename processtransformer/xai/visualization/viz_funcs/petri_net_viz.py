
import pm4py
import typing

from processtransformer.xai.visualization.viz_funcs.base_viz import BaseViz, VizOutput
from processtransformer.xai.visualization.output_models.petri_net_output import PetriNetOutput


class PetriNetViz(BaseViz):
    accepts = [PetriNetOutput]

    def __init__(self, petri_net_output: PetriNetOutput) -> None:
        super().__init__()
        self.petri_net_output = petri_net_output

    def visualize(self) -> typing.List[VizOutput]:
        # ToDo save figure and return path
        pm4py.view_petri_net(self.petri_net_output.net,
                             self.petri_net_output.initial_marking,
                             self.petri_net_output.final_marking)

        return []
