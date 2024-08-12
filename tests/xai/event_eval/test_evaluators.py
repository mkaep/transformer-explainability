

import unittest
from unittest.mock import patch

import numpy as np

from processtransformer.xai.event_eval import evaluator, position_evaluator
from util.attn_funcs import attn_transform
from util.types import LocalScoreDict


class TestEvaluator(unittest.TestCase):
    # Single, with mask-True-values

    def test_masking_single_true(self):
        masked_events, non_masked_events, masked_local_indices, non_masked_local_indices = \
            evaluator.Evaluator.build_masked_indices([True], ['A', 'B', 'C'], [0])
        self.assertEqual(masked_events, [('A', 0)])
        self.assertEqual(non_masked_events, [])
        self.assertEqual(masked_local_indices, [0])
        self.assertEqual(non_masked_local_indices, [])

    def test_masking_single_false(self):
        masked_events, non_masked_events, masked_local_indices, non_masked_local_indices = \
            evaluator.Evaluator.build_masked_indices([False], ['A', 'B', 'C'], [0], mask_true_values=False)
        self.assertEqual(masked_events, [('A', 0)])
        self.assertEqual(non_masked_events, [])
        self.assertEqual(masked_local_indices, [0])
        self.assertEqual(non_masked_local_indices, [])

    # Single, with mask-False-values

    def test_masking_single_true_inverted(self):
        masked_events, non_masked_events, masked_local_indices, non_masked_local_indices = \
            evaluator.Evaluator.build_masked_indices([False], ['A', 'B', 'C'], [0])
        self.assertEqual(masked_events, [])
        self.assertEqual(non_masked_events, [('A', 0)])
        self.assertEqual(masked_local_indices, [])
        self.assertEqual(non_masked_local_indices, [0])

    def test_masking_single_false_inverted(self):
        masked_events, non_masked_events, masked_local_indices, non_masked_local_indices = \
            evaluator.Evaluator.build_masked_indices([True], ['A', 'B', 'C'], [0], mask_true_values=False)
        self.assertEqual(masked_events, [])
        self.assertEqual(non_masked_events, [('A', 0)])
        self.assertEqual(masked_local_indices, [])
        self.assertEqual(non_masked_local_indices, [0])

    # Multiple, but with mask-True-values

    def test_masking_multiple_true_v1(self):
        masked_events, non_masked_events, masked_local_indices, non_masked_local_indices = \
            evaluator.Evaluator.build_masked_indices([True, False, False], ['A', 'B', 'C', 'D', 'E'], [0, 2, 3],
                                                     mask_true_values=True)
        self.assertEqual(masked_events, [('A', 0)])
        self.assertEqual(non_masked_events, [('C', 2), ('D', 3)])
        self.assertEqual(masked_local_indices, [0])
        self.assertEqual(non_masked_local_indices, [2, 3])

    def test_masking_multiple_true_v2(self):
        masked_events, non_masked_events, masked_local_indices, non_masked_local_indices = \
            evaluator.Evaluator.build_masked_indices([True, False, True], ['A', 'B', 'C', 'D', 'E'], [0, 2, 3],
                                                     mask_true_values=True)
        self.assertEqual(masked_events, [('A', 0), ('D', 3)])
        self.assertEqual(non_masked_events, [('C', 2)])
        self.assertEqual(masked_local_indices, [0, 3])
        self.assertEqual(non_masked_local_indices, [2])

    def test_masking_multiple_true_v1_inverted(self):
        masked_events, non_masked_events, masked_local_indices, non_masked_local_indices = \
            evaluator.Evaluator.build_masked_indices([False, True, True], ['A', 'B', 'C', 'D', 'E'], [0, 2, 3],
                                                     mask_true_values=True)
        self.assertEqual(non_masked_events, [('A', 0)])
        self.assertEqual(masked_events, [('C', 2), ('D', 3)])
        self.assertEqual(non_masked_local_indices, [0])
        self.assertEqual(masked_local_indices, [2, 3])

    def test_masking_multiple_true_v2_inverted(self):
        masked_events, non_masked_events, masked_local_indices, non_masked_local_indices = \
            evaluator.Evaluator.build_masked_indices([False, True, False], ['A', 'B', 'C', 'D', 'E'], [0, 2, 3],
                                                     mask_true_values=True)
        self.assertEqual(non_masked_events, [('A', 0), ('D', 3)])
        self.assertEqual(masked_events, [('C', 2)])
        self.assertEqual(non_masked_local_indices, [0, 3])
        self.assertEqual(masked_local_indices, [2])

    # Multiple, but with mask-False-values

    def test_masking_multiple_false_v1(self):
        masked_events, non_masked_events, masked_local_indices, non_masked_local_indices = \
            evaluator.Evaluator.build_masked_indices([True, False, False], ['A', 'B', 'C', 'D', 'E'], [0, 2, 3],
                                                     mask_true_values=False)
        self.assertEqual(non_masked_events, [('A', 0)])
        self.assertEqual(masked_events, [('C', 2), ('D', 3)])
        self.assertEqual(non_masked_local_indices, [0])
        self.assertEqual(masked_local_indices, [2, 3])

    def test_masking_multiple_false_v2(self):
        masked_events, non_masked_events, masked_local_indices, non_masked_local_indices = \
            evaluator.Evaluator.build_masked_indices([True, False, True], ['A', 'B', 'C', 'D', 'E'], [0, 2, 3],
                                                     mask_true_values=False)
        self.assertEqual(non_masked_events, [('A', 0), ('D', 3)])
        self.assertEqual(masked_events, [('C', 2)])
        self.assertEqual(non_masked_local_indices, [0, 3])
        self.assertEqual(masked_local_indices, [2])

    def test_masking_multiple_false_v1_inverted(self):
        masked_events, non_masked_events, masked_local_indices, non_masked_local_indices = \
            evaluator.Evaluator.build_masked_indices([False, True, True], ['A', 'B', 'C', 'D', 'E'], [0, 2, 3],
                                                     mask_true_values=False)
        self.assertEqual(masked_events, [('A', 0)])
        self.assertEqual(non_masked_events, [('C', 2), ('D', 3)])
        self.assertEqual(masked_local_indices, [0])
        self.assertEqual(non_masked_local_indices, [2, 3])

    def test_masking_multiple_false_v2_inverted(self):
        masked_events, non_masked_events, masked_local_indices, non_masked_local_indices = \
            evaluator.Evaluator.build_masked_indices([False, True, False], ['A', 'B', 'C', 'D', 'E'], [0, 2, 3],
                                                     mask_true_values=False)
        self.assertEqual(masked_events, [('A', 0), ('D', 3)])
        self.assertEqual(non_masked_events, [('C', 2)])
        self.assertEqual(masked_local_indices, [0, 3])
        self.assertEqual(non_masked_local_indices, [2])

    #
    # Test evaluate_event_influence
    #

    @patch('processtransformer.models.helper.Predictor', autospec=True)
    def test_eval_influence(self, predictor_mock):
        base_attn_matrix = np.asarray([[
            [[.3, .0, .7],
             [.9, .1, .0],
             [.6, .0, .4]],
            [[.3, .0, .7],
             [.9, .1, .0],
             [.6, .0, .4]],
            [[.3, .0, .7],
             [.9, .1, .0],
             [.6, .0, .4]],
            [[.3, .0, .7],
             [.9, .1, .0],
             [.6, .0, .4]],
        ]])
        masked_attn_matrix = np.asarray([[
            [[.0, .4, .6],
             [.0, .8, .2],
             [.2, .4, .4]],
            [[.0, .4, .6],
             [.0, .8, .2],
             [.2, .4, .4]],
            [[.0, .4, .6],
             [.0, .8, .2],
             [.2, .4, .4]],
            [[.0, .4, .6],
             [.0, .8, .2],
             [.2, .4, .4]],
        ]])
        events = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

        pred = predictor_mock.return_value
        predictions = {'D': 0.3, 'E': 0.4}
        softmax = [(event, 0.0) if event not in predictions.keys() else
                   (event, predictions[event]) for event in events]
        relevant_predictions = [(pred_event, val, index)
                                for index, (pred_event, val) in enumerate(softmax) if val > 0.0]

        masked_predictions = {'E': 0.5, 'F': 0.3}
        masked_softmax = [(event, 0.0) if event not in masked_predictions.keys() else
                          (event, masked_predictions[event]) for event in events]

        test_eval = position_evaluator.PositionEvaluator(pred)

        local_score_dict: LocalScoreDict = {y_event: {x_event: [] for x_event in events} for y_event in events}

        masked_events = [('A', 0)]
        non_masked_events = [('C', 2)]
        trace = ['A', 'B', 'C']
        masked_trace = ['M-A', 'B', 'C']
        base_attn = attn_transform(trace, base_attn_matrix)

        pred.make_prediction.return_value = None, None, masked_attn_matrix, None, masked_softmax, None
        test_eval.evaluate_event_influence(trace, base_attn, local_score_dict, masked_events,
                                           masked_trace, non_masked_events, relevant_predictions)

        score_DA = sum(local_score_dict['D']['A'])
        score_DC = sum(local_score_dict['D']['C'])
        score_EA = sum(local_score_dict['E']['A'])
        score_EC = sum(local_score_dict['E']['C'])
        self.assertEqual(np.isclose(score_DA, 0.18), True)
        self.assertEqual(np.isclose(score_DC, 0.01), True)
        self.assertEqual(np.isclose(score_EA, -0.24), True)
        self.assertEqual(np.isclose(score_EC, 0.16), True)


if __name__ == '__main__':
    unittest.main()
