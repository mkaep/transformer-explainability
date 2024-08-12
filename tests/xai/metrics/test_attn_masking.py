

import typing
import unittest
from unittest.mock import MagicMock

import numpy as np

from processtransformer.xai.metrics.attn_masking import eval_attn_vs_trace_masking


class TestEvalAttnVsTraceMasking(unittest.TestCase):
    def test_eval_attn_vs_trace_masking1(self):
        mock_pred = MagicMock()

        def make_prediction(event_trace: typing.List[str],
                            return_softmax: bool = False,
                            attn_indices_to_mask=None,
                            ):
            # sm = [A, B, C, E, R, S]
            if attn_indices_to_mask is None:
                if event_trace == ['M-A', 'B', 'E']:
                    ev, sm = 'R', [0.0, 0.0, 0.0, 0.0, 0.9, 0.1]
                elif event_trace == ['A', 'M-B', 'E']:
                    ev, sm = 'S', [0.0, 0.0, 0.0, 0.0, 0.49, 0.51]
                elif event_trace == ['A', 'B', 'M-E']:
                    ev, sm = 'R', [0.0, 0.0, 0.0, 0.0, 0.9, 0.1]
                elif event_trace == ['M-A', 'C', 'E']:
                    ev, sm = 'S', [0.0, 0.0, 0.0, 0.0, 0.1, 0.9]
                elif event_trace == ['A', 'M-C', 'E']:
                    ev, sm = 'S', [0.0, 0.0, 0.0, 0.0, 0.49, 0.51]
                elif event_trace == ['A', 'C', 'M-E']:
                    ev, sm = 'S', [0.0, 0.0, 0.0, 0.0, 0.1, 0.9]
                else:
                    assert False
            else:
                if event_trace == ['A', 'B', 'E'] and attn_indices_to_mask == [0]:
                    ev, sm = 'R', [0.0, 0.0, 0.0, 0.0, 0.87, 0.13]
                elif event_trace == ['A', 'B', 'E'] and attn_indices_to_mask == [1]:
                    # Slightly heavier S-bias
                    ev, sm = 'S', [0.0, 0.0, 0.0, 0.0, 0.45, 0.55]
                elif event_trace == ['A', 'B', 'E'] and attn_indices_to_mask == [2]:
                    ev, sm = 'R', [0.0, 0.0, 0.0, 0.0, 0.87, 0.13]
                elif event_trace == ['A', 'C', 'E'] and attn_indices_to_mask == [0]:
                    ev, sm = 'S', [0.0, 0.0, 0.0, 0.0, 0.13, 0.87]
                elif event_trace == ['A', 'C', 'E'] and attn_indices_to_mask == [1]:
                    # Slightly heavier R-bias
                    ev, sm = 'R', [0.0, 0.0, 0.0, 0.0, 0.55, 0.45]
                elif event_trace == ['A', 'C', 'E'] and attn_indices_to_mask == [2]:
                    ev, sm = 'S', [0.0, 0.0, 0.0, 0.0, 0.13, 0.87]
                else:
                    assert False

            return ev, None, None, None, None, sm

        def make_multi_predictions(_traces, return_softmax=False, attn_indices_to_mask_list=None):
            if attn_indices_to_mask_list is None:
                attn_indices_to_mask_list = [None] * len(_traces)
            return [make_prediction(trace, return_softmax, attn_indices)
                    for trace, attn_indices in zip(_traces, attn_indices_to_mask_list)]

        mock_pred.make_prediction.side_effect = make_prediction
        mock_pred.make_multi_predictions.side_effect = make_multi_predictions

        # A, then (B | C), then E, then (R | S)
        traces = [
            ['A', 'B', 'E'],
            ['A', 'C', 'E'],
        ]
        metrics = eval_attn_vs_trace_masking(mock_pred,
                                             traces)
        assert np.isclose(metrics.first_quantile, 0.03)
        assert np.isclose(metrics.median, 0.03)
        assert np.isclose(metrics.third_quantile, 0.0375)
        pass

    def test_eval_attn_vs_trace_masking2(self):
        mock_pred = MagicMock()

        def make_prediction(event_trace: typing.List[str],
                            return_softmax: bool = False,
                            attn_indices_to_mask=None,
                            ):
            # sm = [A, B, C, E, R, S]
            if attn_indices_to_mask is None:
                if event_trace == ['M-A', 'B', 'E']:
                    ev, sm = 'R', [0.0, 0.0, 0.0, 0.0, 0.9, 0.1]
                elif event_trace == ['A', 'M-B', 'E']:
                    ev, sm = 'S', [0.0, 0.0, 0.0, 0.0, 0.49, 0.51]
                elif event_trace == ['A', 'B', 'M-E']:
                    ev, sm = 'R', [0.0, 0.0, 0.0, 0.0, 0.9, 0.1]
                elif event_trace == ['M-A', 'C', 'E']:
                    ev, sm = 'S', [0.0, 0.0, 0.0, 0.0, 0.1, 0.9]
                elif event_trace == ['A', 'M-C', 'E']:
                    ev, sm = 'S', [0.0, 0.0, 0.0, 0.0, 0.49, 0.51]
                elif event_trace == ['A', 'C', 'M-E']:
                    ev, sm = 'S', [0.0, 0.0, 0.0, 0.0, 0.1, 0.9]
                else:
                    assert False
            else:
                if event_trace == ['A', 'B', 'E'] and attn_indices_to_mask == [0]:
                    ev, sm = 'R', [0.0, 0.0, 0.0, 0.0, 0.65, 0.35]
                elif event_trace == ['A', 'B', 'E'] and attn_indices_to_mask == [1]:
                    # Slightly heavier S-bias
                    ev, sm = 'S', [0.0, 0.0, 0.0, 0.0, 0.25, 0.75]
                elif event_trace == ['A', 'B', 'E'] and attn_indices_to_mask == [2]:
                    ev, sm = 'R', [0.0, 0.0, 0.0, 0.0, 0.7, 0.3]
                elif event_trace == ['A', 'C', 'E'] and attn_indices_to_mask == [0]:
                    ev, sm = 'S', [0.0, 0.0, 0.0, 0.0, 0.15, 0.85]
                elif event_trace == ['A', 'C', 'E'] and attn_indices_to_mask == [1]:
                    # Slightly heavier R-bias
                    ev, sm = 'R', [0.0, 0.0, 0.0, 0.0, 0.75, 0.25]
                elif event_trace == ['A', 'C', 'E'] and attn_indices_to_mask == [2]:
                    ev, sm = 'S', [0.0, 0.0, 0.0, 0.0, 0.25, 0.75]
                else:
                    assert False

            return ev, None, None, None, None, sm

        def make_multi_predictions(_traces, return_softmax=False, attn_indices_to_mask_list=None):
            if attn_indices_to_mask_list is None:
                attn_indices_to_mask_list = [None] * len(_traces)

            return [make_prediction(trace, return_softmax, attn_indices)
                    for trace, attn_indices in zip(_traces, attn_indices_to_mask_list)]

        mock_pred.make_prediction.side_effect = make_prediction
        mock_pred.make_multi_predictions.side_effect = make_multi_predictions

        # A, then (B | C), then E, then (R | S)
        traces = [
            ['A', 'B', 'E'],
            ['A', 'C', 'E'],
        ]
        metrics = eval_attn_vs_trace_masking(mock_pred,
                                             traces)
        assert np.isclose(metrics.first_quantile, 0.1625)
        assert np.isclose(metrics.median, 0.22)
        assert np.isclose(metrics.third_quantile, 0.2475)
        pass
