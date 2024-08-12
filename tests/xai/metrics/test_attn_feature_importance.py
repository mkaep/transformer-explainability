
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from processtransformer.xai.metrics.attn_feature_importance import eval_attn_vs_bb_feature_importance


@patch('pm4py.discover_dfg', autospec=True)
@patch('processtransformer.xai.metrics.trace_generation.generate_artificial_local_env', autospec=True)
@patch('processtransformer.xai.metrics.feature_importance.get_blackbox_feature_importance', autospec=True)
class TestAttnFeatureImportance(unittest.TestCase):
    def test_eval_attn_vs_bb_feature_importance1(self, mock_fi, mock_env_gen, mock_discover_dfg):
        mock_discover_dfg.return_value = None

        traces = [['A', 'B', 'D']]
        mock_fi.return_value = {'PAD': 0.0, 'A': 0.05, 'B': 0.9, 'C': 0.0, 'D': 0.05, 'E': 0.0, 'F': 0.0}, None
        mock_env_gen.return_value = [
            (None, ['M-A', 'B', 'D']),
            (None, ['A', 'C', 'D']),
            (None, ['A', 'M-B', 'D']),
            (None, ['A', 'B', 'M-D']),
        ]

        mock_pred = MagicMock()

        def make_prediction(trace, return_softmax=False):
            # softmax = [A, B, C, D, E, F]
            if trace == ['A', 'B', 'D']:
                attn = np.asarray([[[[0.1, 0.8, 0.1],
                                     [0.1, 0.8, 0.1],
                                     [0.1, 0.8, 0.1],
                                     ]]])
                sm = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            elif trace == ['M-A', 'B', 'D']:
                attn = np.asarray([[[[0.0, 0.85, 0.15],
                                     [0.0, 0.85, 0.15],
                                     [0.0, 0.85, 0.15],
                                     ]]])
                sm = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            elif trace == ['A', 'C', 'D']:
                attn = np.asarray([[[[0.1, 0.8, 0.1],
                                     [0.1, 0.8, 0.1],
                                     [0.1, 0.8, 0.1],
                                     ]]])
                sm = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            elif trace == ['A', 'M-B', 'D']:
                attn = np.asarray([[[[0.4, 0.0, 0.6],
                                     [0.4, 0.0, 0.6],
                                     [0.4, 0.0, 0.6],
                                     ]]])
                sm = [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
            elif trace == ['A', 'B', 'M-D']:
                attn = np.asarray([[[[0.15, 0.85, 0.0],
                                     [0.15, 0.85, 0.0],
                                     [0.15, 0.85, 0.0],
                                     ]]])
                sm = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            else:
                assert False

            return None, None, attn, None, None, sm

        def make_multi_predictions(traces, return_softmax=False):
            return [make_prediction(trace) for trace in traces]

        mock_pred.make_prediction.side_effect = make_prediction
        mock_pred.make_multi_predictions.side_effect = make_multi_predictions

        mock_pred.x_word_dict = {'PAD': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}

        # Should be fairly equal
        metrics = eval_attn_vs_bb_feature_importance(traces, mock_pred, MagicMock())
        assert np.isclose(metrics.spearman_mean_corr, 1.0)
        assert np.isclose(metrics.kendalltau_mean_corr, 1.0)
        assert np.isclose(metrics.pearson_mean_corr, 0.9923, atol=0.001)

    def test_eval_attn_vs_bb_feature_importance2(self, mock_fi, mock_env_gen, mock_discover_dfg):
        mock_discover_dfg.return_value = None

        traces = [['A', 'B', 'D']]
        mock_fi.return_value = {'PAD': 0.0, 'A': 0.0, 'B': 0.2, 'C': 0.8, 'D': 0.0, 'E': 0.0, 'F': 0.0}, None
        mock_env_gen.return_value = [
            (None, ['M-A', 'B', 'D']),
            (None, ['A', 'C', 'D']),
            (None, ['A', 'M-B', 'D']),
            (None, ['A', 'B', 'M-D']),
        ]

        mock_pred = MagicMock()

        def make_prediction(trace, return_softmax=False):
            # softmax = [A, B, C, D, E, F]
            if trace == ['A', 'B', 'D']:
                sm = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            elif trace == ['M-A', 'B', 'D']:
                sm = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            elif trace == ['A', 'C', 'D']:
                sm = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            elif trace == ['A', 'M-B', 'D']:
                sm = [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
            elif trace == ['A', 'B', 'M-D']:
                sm = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            else:
                assert False

            attn = np.asarray([[[[0.33, 0.33, 0.33],
                                 [0.33, 0.33, 0.33],
                                 [0.33, 0.33, 0.33],
                                 ]]])
            return None, None, attn, None, None, sm

        def make_multi_predictions(traces, return_softmax=False):
            return [make_prediction(trace) for trace in traces]

        mock_pred.make_prediction.side_effect = make_prediction
        mock_pred.make_multi_predictions.side_effect = make_multi_predictions

        mock_pred.x_word_dict = {'PAD': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}

        # Should be different
        metrics = eval_attn_vs_bb_feature_importance(traces, mock_pred, MagicMock())
        assert np.isclose(metrics.spearman_mean_corr, -0.1491, atol=0.001)
        assert np.isclose(metrics.kendalltau_mean_corr, -0.1612, atol=0.001)
        assert np.isclose(metrics.pearson_mean_corr, -0.2382, atol=0.001)

    def test_eval_attn_vs_bb_feature_importance3(self, mock_fi, mock_env_gen, mock_discover_dfg):
        mock_discover_dfg.return_value = None

        traces = [
            ['A', 'B', 'D'],
            ['A', 'C', 'D'],
        ]
        mock_fi.side_effect = [
            ({'PAD': 0.0, 'A': 0.05, 'B': 0.9, 'C': 0.0, 'D': 0.05, 'E': 0.0, 'F': 0.0}, None),
            ({'PAD': 0.0, 'A': 0.30, 'B': 0.0, 'C': 0.3, 'D': 0.40, 'E': 0.0, 'F': 0.0}, None),
        ]
        mock_env_gen.side_effect = [
            [
                (None, ['M-A', 'B', 'D']),
                (None, ['A', 'C', 'D']),
                (None, ['A', 'M-B', 'D']),
                (None, ['A', 'B', 'M-D']),
            ],
            [
                (None, ['M-A', 'C', 'D']),
                (None, ['A', 'B', 'D']),
                (None, ['A', 'M-C', 'D']),
                (None, ['A', 'C', 'M-D']),
            ],
        ]

        mock_pred = MagicMock()

        def make_prediction(trace, return_softmax=False):
            # softmax = [A, B, C, D, E, F]
            if trace == ['A', 'B', 'D']:
                attn = np.asarray([[[[0.1, 0.8, 0.1],
                                     [0.1, 0.8, 0.1],
                                     [0.1, 0.8, 0.1],
                                     ]]])
                sm = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            elif trace == ['M-A', 'B', 'D']:
                attn = np.asarray([[[[0.0, 0.85, 0.15],
                                     [0.0, 0.85, 0.15],
                                     [0.0, 0.85, 0.15],
                                     ]]])
                sm = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            elif trace == ['M-A', 'C', 'D']:
                attn = np.asarray([[[[0.0, 0.85, 0.15],
                                     [0.0, 0.85, 0.15],
                                     [0.0, 0.85, 0.15],
                                     ]]])
                sm = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            elif trace == ['A', 'C', 'D']:
                attn = np.asarray([[[[0.1, 0.8, 0.1],
                                     [0.1, 0.8, 0.1],
                                     [0.1, 0.8, 0.1],
                                     ]]])
                sm = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            elif trace == ['A', 'M-B', 'D'] or trace == ['A', 'M-C', 'D']:
                attn = np.asarray([[[[0.4, 0.0, 0.6],
                                     [0.4, 0.0, 0.6],
                                     [0.4, 0.0, 0.6],
                                     ]]])
                sm = [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
            elif trace == ['A', 'B', 'M-D']:
                attn = np.asarray([[[[0.15, 0.85, 0.0],
                                     [0.15, 0.85, 0.0],
                                     [0.15, 0.85, 0.0],
                                     ]]])
                sm = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            elif trace == ['A', 'C', 'M-D']:
                attn = np.asarray([[[[0.15, 0.85, 0.0],
                                     [0.15, 0.85, 0.0],
                                     [0.15, 0.85, 0.0],
                                     ]]])
                sm = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            else:
                assert False

            return None, None, attn, None, None, sm

        def make_multi_predictions(traces, return_softmax=False):
            return [make_prediction(trace) for trace in traces]

        mock_pred.make_prediction.side_effect = make_prediction
        mock_pred.make_multi_predictions.side_effect = make_multi_predictions

        mock_pred.x_word_dict = {'PAD': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}

        # Should be fairly equal
        metrics = eval_attn_vs_bb_feature_importance(traces, mock_pred, MagicMock())
        assert np.isclose(metrics.spearman_mean_corr, 0.95, atol=0.001)
        assert np.isclose(metrics.spearman_std_dev_corr, 0.07071, atol=0.001)
        assert np.isclose(metrics.df['spearman-correlation'][0], 1.0)
        assert np.isclose(metrics.df['spearman-correlation'][1], 0.9)
        assert np.isclose(metrics.kendalltau_mean_corr, 0.8929, atol=0.001)
        assert np.isclose(metrics.kendalltau_std_dev_corr, 0.1515, atol=0.001)
        assert np.isclose(metrics.df['kendalltau-correlation'][0], 1.0)
        assert np.isclose(metrics.df['kendalltau-correlation'][1], 0.78571, atol=0.001)
        assert np.isclose(metrics.pearson_mean_corr, 0.7716, atol=0.001)
        assert np.isclose(metrics.pearson_std_dev_corr, 0.3121, atol=0.001)
        assert np.isclose(metrics.df['pearson-correlation'][0], 0.9923, atol=0.001)
        assert np.isclose(metrics.df['pearson-correlation'][1], 0.5509, atol=0.001)
