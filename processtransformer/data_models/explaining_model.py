
import dataclasses
import typing

from processtransformer.data_models.json_serializable import JsonSerializable
from processtransformer.xai.explainer import Explainer
from processtransformer.xai.visual_attn_mtrx_xai import VisualAttentionMatrixExplainer
from processtransformer.util.subclassing import EvaluatorSubclasses, ExplainerSubclasses

evaluators_keyword = 'evaluators'


@dataclasses.dataclass
class ExplainingModel(JsonSerializable):
    name: str
    prefix_and_y_true_log: str
    trace_to_explain: typing.Union[str, typing.List[str]]
    neural_network_model_dir: str
    dict_dir: str
    result_dir: str
    explainer: typing.Type[Explainer]  # will be any of its subclasses
    explainer_kwargs: typing.Dict

    def to_dict(self):
        explainer_kwargs = {key: value for key, value in self.explainer_kwargs.items()}
        if evaluators_keyword in self.explainer_kwargs.keys():
            explainer_kwargs[evaluators_keyword] = [
                (eval_kwargs, evaluator.get_name())
                for eval_kwargs, evaluator in self.explainer_kwargs[evaluators_keyword]
            ]

        return {
            'name': self.name,
            'prefix_and_y_true_log': self.prefix_and_y_true_log,
            'trace_to_explain': self.trace_to_explain,
            'neural_network_model_dir': self.neural_network_model_dir,
            'dict_dir': self.dict_dir,
            'result_dir': self.result_dir,
            'explainer': self.explainer.get_name(),
            'explainer_kwargs': explainer_kwargs,
        }

    @staticmethod
    def from_dict(data):
        try:
            explainer_type = ExplainerSubclasses().get_type(data['explainer'])
        except (ValueError, KeyError):
            # Default to this if not available - used before this field has been added
            explainer_type = VisualAttentionMatrixExplainer
            print(f"Defaulting to {explainer_type.get_name()}. Is this intended?")

        try:
            explainer_kwargs: dict = data['explainer_kwargs']
            if evaluators_keyword in explainer_kwargs.keys():
                explainer_subclasses = EvaluatorSubclasses()
                explainer_kwargs[evaluators_keyword] = [(eval_kwargs, explainer_subclasses.get_type(evaluator))
                                                        for eval_kwargs, evaluator
                                                        in explainer_kwargs[evaluators_keyword]]
        except KeyError:
            # Empty dictionary
            explainer_kwargs = {}
            print(f"Did not load any explainer-kwargs. Is this intended?")

        try:
            trace_to_explain = data['trace_to_explain']
        except KeyError:
            trace_to_explain = None

        return ExplainingModel(
            name=data['name'],
            trace_to_explain=trace_to_explain,
            prefix_and_y_true_log=data['prefix_and_y_true_log'],
            neural_network_model_dir=data['neural_network_model_dir'],
            dict_dir=data['dict_dir'],
            result_dir=data['result_dir'],
            explainer=explainer_type,
            explainer_kwargs=explainer_kwargs,
        )
