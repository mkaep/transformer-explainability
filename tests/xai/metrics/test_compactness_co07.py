

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from processtransformer.xai.metrics.compactness_co07 import eval_compactness


class TestEvalCompactness(unittest.TestCase):
    @patch('processtransformer.xai.metrics.compactness_co07.get_blackbox_feature_importance')
    def test_eval_compactness(self, mock_get_bb_fi):
        mock_pred = MagicMock()
        mock_pred.make_prediction.return_value = None, None, None, None, \
            [('D', 0.40), ('E', 0.20), ('F', 0.20), ('G', 0.20)], None

        mock_relations_output = MagicMock()
        mock_relations_output.get_explanation_for_trace.return_value = {('C',): ('D', 'E'), ('B',): ('F', )}

        mock_get_bb_fi.return_value = {'A': 0.15, 'B': 0.15, 'C': 0.70}, None

        metrics = eval_compactness(['A', 'B', 'C', 'D'], mock_pred, MagicMock(), MagicMock(), mock_relations_output)
        assert metrics.num_rules == 2
        assert np.isclose(metrics.avg_left_side_length, 1.0)
        assert np.isclose(metrics.avg_right_side_length, 1.5)

        assert np.isclose(metrics.ratio_left_to_trace, 0.50, atol=0.001)
        assert np.isclose(metrics.ratio_left_to_feat_imp, 0.667, atol=0.001)
        assert np.isclose(metrics.ratio_xai_right_to_pred, 0.75, atol=0.001)
        pass
