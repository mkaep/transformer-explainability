

import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import euclidean

from processtransformer.util.types import YWordDict, RelationsDict
from processtransformer.xai.metrics.feature_importance import get_blackbox_feature_importance, \
    get_xai_feature_importance
from processtransformer.xai.visualization.output_models.relations_output import RelationsOutput
from tests.xai.metrics.common import _generate_df_from_events


class TestBlackboxFeatureImportance(unittest.TestCase):
    @patch('processtransformer.models.helper.Predictor', autospec=True)
    def test_get_blackbox_feature_importance1(self, predictor_mock):
        trace = ['A', 'B', 'D']
        y_word_dict: YWordDict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
        # A (B|C) D (E|F)
        log = _generate_df_from_events([
            # Assumption: B leads more often to E than to F. Does not have an effect here.
            ['A', 'B', 'D', 'E'],
            ['A', 'B', 'D', 'E'],
            ['A', 'B', 'D', 'E'],
            ['A', 'B', 'D', 'F'],
            # Assumption: C leads more often to F than to E. Does not have an effect here.
            ['A', 'C', 'D', 'E'],
            ['A', 'C', 'D', 'F'],
            ['A', 'C', 'D', 'F'],
            ['A', 'C', 'D', 'F'],
        ])

        mock_pred = predictor_mock.return_value

        def make_prediction(trace_in, return_softmax):
            # Dicts are softmax-vectors that are to be returned
            if trace_in == ['A', 'B', 'D']:
                softmax_dict = {'A': 0.01, 'B': 0.01, 'C': 0.01, 'D': 0.01, 'E': 0.72, 'F': 0.24}
            elif trace_in == ['M-A', 'B', 'D']:
                softmax_dict = {'A': 0.08, 'B': 0.01, 'C': 0.01, 'D': 0.01, 'E': 0.70, 'F': 0.19}
            elif trace_in == ['A', 'M-B', 'D']:
                softmax_dict = {'A': 0.01, 'B': 0.01, 'C': 0.01, 'D': 0.01, 'E': 0.47, 'F': 0.49}
            elif trace_in == ['A', 'C', 'D']:
                softmax_dict = {'A': 0.01, 'B': 0.01, 'C': 0.01, 'D': 0.01, 'E': 0.23, 'F': 0.73}
            elif trace_in == ['A', 'B', 'M-D']:
                softmax_dict = {'A': 0.01, 'B': 0.01, 'C': 0.01, 'D': 0.78, 'E': 0.09, 'F': 0.10}
            else:
                # Default case - should not occur!
                assert False

            return None, None, None, None, softmax_dict, list(softmax_dict.values())

        def make_multi_predictions(traces, return_softmax=False):
            return [make_prediction(t, return_softmax) for t in traces]

        mock_pred.make_prediction.side_effect = make_prediction
        mock_pred.make_multi_predictions.side_effect = make_multi_predictions

        mock_pred.x_word_dict = y_word_dict

        dct, lst = get_blackbox_feature_importance(trace, y_word_dict, log, mock_pred)
        assert np.isclose(dct['A'], 0.006, atol=0.01)
        assert np.isclose(dct['B'], 0.238, atol=0.01)
        assert np.isclose(dct['C'], 0.0, atol=0.01)
        assert np.isclose(dct['D'], 0.756, atol=0.01)
        assert np.isclose(dct['E'], 0.0, atol=0.01)
        assert np.isclose(dct['F'], 0.0, atol=0.01)
        assert lst[0] == 'D'
        assert lst[1] == 'B'

        dct, lst = get_blackbox_feature_importance(trace, y_word_dict, log, mock_pred, dist_func=euclidean)
        assert np.isclose(dct['A'], 0.055, atol=0.01)
        assert np.isclose(dct['B'], 0.324, atol=0.01)
        assert np.isclose(dct['C'], 0.0, atol=0.01)
        assert np.isclose(dct['D'], 0.622, atol=0.01)
        assert np.isclose(dct['E'], 0.0, atol=0.01)
        assert np.isclose(dct['F'], 0.0, atol=0.01)
        assert lst[0] == 'D'
        assert lst[1] == 'B'
        pass

    @patch('processtransformer.models.helper.Predictor', autospec=True)
    def test_get_blackbox_feature_importance2(self, predictor_mock):
        trace = ['A']
        y_word_dict: YWordDict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
        # (AB|CD|EF)
        log = _generate_df_from_events([
            ['A', 'B'],
            ['C', 'D'],
            ['E', 'F'],
        ])

        mock_pred = predictor_mock.return_value
        mock_pred.x_word_dict = y_word_dict

        def make_prediction(trace_in, return_softmax):
            # Dicts are softmax-vectors that are to be returned
            if trace_in == ['A']:
                softmax_dict = {'A': 0.01, 'B': 0.89, 'C': 0.01, 'D': 0.04, 'E': 0.01, 'F': 0.04}
            elif trace_in == ['M-A']:
                softmax_dict = {'A': 0.01, 'B': 0.33, 'C': 0.01, 'D': 0.33, 'E': 0.01, 'F': 0.32}
            elif trace_in == ['C']:
                softmax_dict = {'A': 0.01, 'B': 0.01, 'C': 0.01, 'D': 0.95, 'E': 0.01, 'F': 0.01}
            elif trace_in == ['E']:
                softmax_dict = {'A': 0.01, 'B': 0.01, 'C': 0.01, 'D': 0.01, 'E': 0.01, 'F': 0.95}
            else:
                # Default case - should not occur!
                assert False

            return None, None, None, None, softmax_dict, list(softmax_dict.values())

        def make_multi_predictions(traces, return_softmax=False):
            return [make_prediction(t, return_softmax) for t in traces]

        mock_pred.make_prediction.side_effect = make_prediction
        mock_pred.make_multi_predictions.side_effect = make_multi_predictions

        dct, lst = get_blackbox_feature_importance(trace, y_word_dict, log, mock_pred)
        assert np.isclose(dct['A'], 1.0, atol=0.01)
        assert np.isclose(dct['B'], 0.0, atol=0.01)
        assert np.isclose(dct['C'], 0.0, atol=0.01)
        assert np.isclose(dct['D'], 0.0, atol=0.01)
        assert np.isclose(dct['E'], 0.0, atol=0.01)
        assert np.isclose(dct['F'], 0.0, atol=0.01)
        assert lst[0] == 'A'

        dct, lst = get_blackbox_feature_importance(trace, y_word_dict, log, mock_pred, dist_func=euclidean)
        assert np.isclose(dct['A'], 1.0, atol=0.01)
        assert np.isclose(dct['B'], 0.0, atol=0.01)
        assert np.isclose(dct['C'], 0.0, atol=0.01)
        assert np.isclose(dct['D'], 0.0, atol=0.01)
        assert np.isclose(dct['E'], 0.0, atol=0.01)
        assert np.isclose(dct['F'], 0.0, atol=0.01)
        assert lst[0] == 'A'
        pass


class TestXaiFeatureImportance(unittest.TestCase):
    def test_get_xai_feature_importance1(self):
        trace = ['A', 'B', 'D']
        y_word_dict: YWordDict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
        # A (B|C) D (E|F)
        log = _generate_df_from_events([
            # Assumption: B leads more often to E than to F. Does not have an effect here.
            ['A', 'B', 'D', 'E'],
            ['A', 'B', 'D', 'E'],
            ['A', 'B', 'D', 'E'],
            ['A', 'B', 'D', 'F'],
            # Assumption: C leads more often to F than to E. Does not have an effect here.
            ['A', 'C', 'D', 'E'],
            ['A', 'C', 'D', 'F'],
            ['A', 'C', 'D', 'F'],
            ['A', 'C', 'D', 'F'],
        ])

        relations_dict: RelationsDict = dict()
        relations_dict[('A',)] = ('B', 'C',)
        relations_dict[('B',)] = ('D',)
        relations_dict[('C',)] = ('D',)
        relations_dict[('D',)] = ('E', 'F',)
        relations_dict[('E',)] = ()
        relations_dict[('F',)] = ()
        relations_output = RelationsOutput(relations_dict, None, None, ['A'])

        dct, lst = get_xai_feature_importance(trace, y_word_dict, log, relations_output)
        assert np.isclose(dct['A'], 0.182, atol=0.01)
        assert np.isclose(dct['B'], 0.232, atol=0.01)
        assert np.isclose(dct['C'], 0.01, atol=0.01)
        assert np.isclose(dct['D'], 0.587, atol=0.01)
        assert np.isclose(dct['E'], 0.01, atol=0.01)
        assert np.isclose(dct['F'], 0.01, atol=0.01)
        assert lst[0:3] == ['D', 'B', 'A']

        dct, lst = get_xai_feature_importance(trace, y_word_dict, log, relations_output, dist_func=euclidean)
        assert np.isclose(dct['A'], 0.254, atol=0.01)
        assert np.isclose(dct['B'], 0.306, atol=0.01)
        assert np.isclose(dct['C'], 0.0, atol=0.01)
        assert np.isclose(dct['D'], 0.440, atol=0.01)
        assert np.isclose(dct['E'], 0.0, atol=0.01)
        assert np.isclose(dct['F'], 0.0, atol=0.01)
        assert lst[0:3] == ['D', 'B', 'A']
        pass

    def test_get_xai_feature_importance2(self):
        trace = ['A']
        y_word_dict: YWordDict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
        # (AB|CD|EF)
        log = _generate_df_from_events([
            ['A', 'B'],
            ['C', 'D'],
            ['E', 'F'],
        ])

        relations_dict: RelationsDict = dict()
        relations_dict[('A',)] = ('B',)
        relations_dict[('C',)] = ('D',)
        relations_dict[('E',)] = ('F',)
        relations_output = RelationsOutput(relations_dict, None, None, ['A', 'B', 'C'])

        dct, lst = get_xai_feature_importance(trace, y_word_dict, log, relations_output)
        assert np.isclose(dct['A'], 1.0, atol=0.01)
        assert np.isclose(dct['B'], 0.0, atol=0.01)
        assert np.isclose(dct['C'], 0.0, atol=0.01)
        assert np.isclose(dct['D'], 0.0, atol=0.01)
        assert np.isclose(dct['E'], 0.0, atol=0.01)
        assert np.isclose(dct['F'], 0.0, atol=0.01)
        assert lst[0] == 'A'

        dct, lst = get_xai_feature_importance(trace, y_word_dict, log, relations_output, dist_func=euclidean)
        assert np.isclose(dct['A'], 1.0, atol=0.01)
        assert np.isclose(dct['B'], 0.0, atol=0.01)
        assert np.isclose(dct['C'], 0.0, atol=0.01)
        assert np.isclose(dct['D'], 0.0, atol=0.01)
        assert np.isclose(dct['E'], 0.0, atol=0.01)
        assert np.isclose(dct['F'], 0.0, atol=0.01)
        assert lst[0] == 'A'
        pass


if __name__ == '__main__':
    unittest.main()
