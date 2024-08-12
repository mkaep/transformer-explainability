
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from processtransformer.xai.metrics.contrastivity_co05 import eval_contrastivity
from processtransformer.util.types import RelationsDict


class TestEvalContrastivity(unittest.TestCase):
    @patch('processtransformer.xai.metrics.contrastivity_co05.maybe_gen_real_local_env', autospec=True)
    def test_eval_contrastivity(self, mock_maybe_gen_real_local_env):
        mock_pred = MagicMock()

        mock_maybe_gen_real_local_env.return_value = [['A', 'C', 'X'], ['A', 'D', 'X'], ['A', 'E', 'X']]

        def make_prediction(event_trace, return_softmax: bool = False):
            # Softmax: A, B, C, D, E, R, S, X
            if event_trace == ['A', 'B', 'X']:
                # Predict R
                sm = [0.01, 0.01, 0.01, 0.01, 0.01, 0.77, 0.07, 0.01]
            elif event_trace == ['A', 'C', 'X']:
                # Predict S
                sm = [0.01, 0.01, 0.01, 0.01, 0.01, 0.07, 0.77, 0.01]
            elif event_trace == ['A', 'D', 'X']:
                # Predict S
                sm = [0.01, 0.01, 0.01, 0.01, 0.01, 0.07, 0.77, 0.01]
            elif event_trace == ['A', 'E', 'X']:
                # Predict S
                sm = [0.01, 0.01, 0.01, 0.01, 0.01, 0.07, 0.77, 0.01]
            else:
                assert False
            return None, None, None, None, None, sm

        mock_pred.make_prediction.side_effect = make_prediction

        def get_explanation_for_trace(trace, transformer_info) -> RelationsDict:
            if trace == ['A', 'B', 'X']:
                # Predict R
                return {('A',): ('C', 'D', 'E'), ('B',): ('R',), ('X',): ('R', 'S')}
            elif trace == ['A', 'C', 'X']:
                # Predict R
                return {('A',): ('B', 'D', 'E'), ('C',): ('R',), ('X',): ('R', 'S')}
            elif trace == ['A', 'D', 'X']:
                # Predict S
                return {('A',): ('B', 'C', 'E'), ('D',): ('S',), ('X',): ('R', 'S')}
            elif trace == ['A', 'E', 'X']:
                # Predict S
                return {('A',): ('B', 'C', 'D'), ('E',): ('S',), ('X',): ('R', 'S')}
            else:
                assert False

        mock_relations_out = MagicMock()
        mock_relations_out.get_explanation_for_trace.side_effect = get_explanation_for_trace

        base_trace = ['A', 'B', 'X']
        contrastivity_metrics = eval_contrastivity(base_trace, MagicMock(), mock_pred, mock_relations_out)
        assert np.isclose(contrastivity_metrics.diff_sim, 0.806, atol=0.001)
        assert np.isclose(contrastivity_metrics.diff_sim_relaxed, 0.598, atol=0.001)
        assert contrastivity_metrics.count == 3
        pass
