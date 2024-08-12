

import abc
import dataclasses

from pm4py.util import typing

from processtransformer.models.helper import Predictor
from processtransformer.util.attn_funcs import attn_transform
from processtransformer.util.types import XEvent, YEvent, LocalScoreDict


@dataclasses.dataclass
class BasePredictionInfo:
    trace: typing.List[XEvent]
    attn: typing.List[typing.Tuple[XEvent, float]]
    softmax: typing.List[typing.Tuple[YEvent, float]]
    # float is softmax value, int is the index
    relevant_predictions: typing.List[typing.Tuple[YEvent, float, int]]
    # index in trace
    relevant_attn_indices: typing.List[int]


class Evaluator(abc.ABC):
    """
    Important: Import your subclass in the __init__.py of this module!
    """

    def __init__(self, predictor: Predictor,
                 allowed_pred_delta=0.25,
                 ) -> None:
        self.predictor = predictor
        self.allowed_pred_delta = allowed_pred_delta

    @staticmethod
    def build_masked_indices(bool_list, prefix, relevant_attn_indices, mask_true_values=True):
        assert len(relevant_attn_indices) == len(bool_list)
        masked_local_indices = [index for index, boolean in zip(relevant_attn_indices, bool_list)
                                if boolean]
        non_masked_local_indices = [index for index, boolean in zip(relevant_attn_indices, bool_list)
                                    if not boolean]
        masked_events = [(prefix[index], index) for index in masked_local_indices]
        non_masked_events = [(prefix[index], index) for index in non_masked_local_indices]
        if mask_true_values:
            return masked_events, non_masked_events, masked_local_indices, non_masked_local_indices
        return non_masked_events, masked_events, non_masked_local_indices, masked_local_indices

    def evaluate_event_influence(self, prefix, base_attn, local_score_dict, masked_events, masked_trace,
                                 non_masked_events, relevant_predictions):
        _, _, m_attn, _, m_softmax, _ = self.predictor.make_prediction(masked_trace,
                                                                       return_softmax=True)
        m_attn = attn_transform(prefix, m_attn)
        for relevant_pred_event, pred_val, index in relevant_predictions:
            # Per predicted event:
            new_pred_val = m_softmax[index][1]
            # Plus -> masked event was important
            prediction_is_similar = False

            if pred_val * (1.0 - self.allowed_pred_delta) <= new_pred_val <= pred_val * (1.0 + self.allowed_pred_delta):
                # Minus -> masked event was not (that) important
                prediction_is_similar = True

            for masked_event, m_e_index in masked_events:
                score = pred_val * base_attn[m_e_index][1]
                if prediction_is_similar:
                    score = -score
                local_score_dict[relevant_pred_event][masked_event].append(score)

            # Also consider non-masked-events
            for non_masked_event, n_m_e_index in non_masked_events:
                base_attn_for_event = base_attn[n_m_e_index][1]
                new_attn_for_event = m_attn[n_m_e_index][1]

                # attn_diff = new_attn_for_event - base_attn_for_event

                if prediction_is_similar:
                    # Award if attention is still big
                    score = new_attn_for_event * pred_val
                else:
                    # Score is big if both attention and prediction have changed a lot
                    score = abs(base_attn_for_event - new_attn_for_event) * abs(pred_val - new_pred_val)

                local_score_dict[relevant_pred_event][non_masked_event].append(score)

    @staticmethod
    @abc.abstractmethod
    def get_name() -> str:
        pass

    @abc.abstractmethod
    def eval(self, info: BasePredictionInfo, local_score_dict: LocalScoreDict):
        pass
