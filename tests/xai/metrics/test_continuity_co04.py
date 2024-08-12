
import typing
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from processtransformer.util.types import Trace, RelationsDict, YEvent, XEvent
from processtransformer.xai.metrics.continuity_co04 import eval_continuity
from processtransformer.xai.visualization.output_models.output_data import TransformerInfoForExplanation


class TestEvalContinuity(unittest.TestCase):
    @patch('processtransformer.xai.metrics.continuity_co04.get_blackbox_feature_importance', autospec=True)
    @patch('processtransformer.xai.metrics.continuity_co04.maybe_gen_real_local_env', autospec=True)
    def test_eval_continuity1(self, mock_maybe_gen_real_local_env, mock_get_bb_fi):
        mock_pred = MagicMock()

        def make_prediction(event_trace: typing.List[str],
                            return_softmax: bool = False):
            # Softmax: A, B, C, D, E, R, S, X
            if event_trace == ['A', 'B', 'X']:
                # Predict R
                sm = [0.01, 0.01, 0.01, 0.01, 0.01, 0.77, 0.17, 0.01]
            elif event_trace == ['A', 'C', 'X']:
                # Predict R
                sm = [0.01, 0.01, 0.01, 0.01, 0.01, 0.77, 0.17, 0.01]
            elif event_trace == ['A', 'D', 'X']:
                # Predict R
                sm = [0.01, 0.01, 0.01, 0.01, 0.01, 0.77, 0.17, 0.01]
            elif event_trace == ['A', 'E', 'X']:
                # Predict S
                sm = [0.01, 0.01, 0.01, 0.01, 0.01, 0.17, 0.77, 0.01]
            else:
                assert False
            return None, None, None, None, None, sm

        def make_multi_predictions(traces, return_softmax=False):
            return [make_prediction(trace) for trace in traces]

        mock_pred.make_prediction.side_effect = make_prediction
        mock_pred.make_multi_predictions.side_effect = make_multi_predictions

        mock_relations_out = MagicMock()

        def get_explanation_for_trace(trace: Trace,
                                      transformer_info: TransformerInfoForExplanation,
                                      ) -> RelationsDict:
            if trace == ['A', 'B', 'X']:
                # Predict R
                return {('A',): ('C', 'D', 'E'), ('B',): ('R',), ('X',): ('R', 'S')}
            elif trace == ['A', 'C', 'X']:
                # Predict R
                return {('A',): ('B', 'D', 'E'), ('C',): ('R',), ('X',): ('R', 'S')}
            elif trace == ['A', 'D', 'X']:
                # Predict R
                return {('A',): ('B', 'C', 'E'), ('D',): ('R',), ('X',): ('R', 'S')}
            elif trace == ['A', 'E', 'X']:
                # Predict S
                return {('A',): ('B', 'C', 'D'), ('E',): ('S',), ('X',): ('R', 'S')}
            else:
                assert False

        mock_relations_out.get_explanation_for_trace.side_effect = get_explanation_for_trace

        def get_blackbox_feature_importance(trace: Trace, y_word_dict, event_log, predictor, dist_func=None,
                                            **kwargs,
                                            ) -> typing.Tuple[typing.Dict[XEvent, float], typing.List[YEvent]]:
            if trace == ['A', 'B', 'X']:
                dct = {'A': 0.05, 'B': 0.70, 'C': 0.00, 'D': 0.00, 'E': 0.00, 'R': 0.00, 'S': 0.00, 'X': 0.25}
            elif trace == ['A', 'C', 'X']:
                # This does not make sense, but for testing it is ok.
                dct = {'A': 0.05, 'B': 0.70, 'C': 0.00, 'D': 0.00, 'E': 0.00, 'R': 0.00, 'S': 0.00, 'X': 0.25}
            elif trace == ['A', 'D', 'X']:
                dct = {'A': 0.05, 'B': 0.00, 'C': 0.00, 'D': 0.70, 'E': 0.00, 'R': 0.00, 'S': 0.00, 'X': 0.25}
            elif trace == ['A', 'E', 'X']:
                dct = {'A': 0.05, 'B': 0.00, 'C': 0.00, 'D': 0.00, 'E': 0.70, 'R': 0.00, 'S': 0.00, 'X': 0.25}
            else:
                assert False

            return dct, None

        mock_get_bb_fi.side_effect = get_blackbox_feature_importance

        base_trace = ['A', 'B', 'X']
        mock_maybe_gen_real_local_env.return_value = [['A', 'C', 'X'], ['A', 'D', 'X'], ['A', 'E', 'X']]

        continuity_metrics = eval_continuity(base_trace, MagicMock(), mock_pred, mock_relations_out, MagicMock())
        assert np.isclose(continuity_metrics.similarity, 0.0)
        assert np.isclose(continuity_metrics.relaxed_similarity, 0.0)
        assert np.isclose(continuity_metrics.feat_imp_similarity, 0.0)
        assert continuity_metrics.similarity_count == 2
        assert continuity_metrics.relaxed_similarity_count == 2
        assert continuity_metrics.feat_imp_similarity_count == 1

        # For this test we have to lower the similarity-threshold significantly
        continuity_metrics = eval_continuity(base_trace, MagicMock(), mock_pred, mock_relations_out, MagicMock(),
                                             similarity_threshold=0.45)
        assert np.isclose(continuity_metrics.similarity, 0.0)
        assert np.isclose(continuity_metrics.relaxed_similarity, 1.0)
        assert np.isclose(continuity_metrics.feat_imp_similarity, 0.0)
        assert continuity_metrics.similarity_count == 2
        assert continuity_metrics.relaxed_similarity_count == 2
        assert continuity_metrics.feat_imp_similarity_count == 1

        # For this test we have to lower the similarity-threshold heavily
        continuity_metrics = eval_continuity(base_trace, MagicMock(), mock_pred, mock_relations_out, MagicMock(),
                                             similarity_threshold=0.80)
        assert np.isclose(continuity_metrics.similarity, 1.0)
        assert np.isclose(continuity_metrics.relaxed_similarity, 1.0)
        assert np.isclose(continuity_metrics.feat_imp_similarity, 1.0)
        assert continuity_metrics.similarity_count == 2
        assert continuity_metrics.relaxed_similarity_count == 2
        assert continuity_metrics.feat_imp_similarity_count == 1
        pass
