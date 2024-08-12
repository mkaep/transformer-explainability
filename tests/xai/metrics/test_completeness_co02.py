

import typing
import unittest
from unittest.mock import patch

import numpy as np
from pm4py.objects.log.obj import EventLog

from processtransformer.util.types import RelationsDict, YWordDict, Trace
from processtransformer.xai.metrics.completeness_co02 import MultiLabelConfusionMatrix, TrueFalsePosNeg, \
    eval_completeness
from processtransformer.xai.visualization.output_models.relations_output import RelationsOutput
from tests.xai.metrics.common import _generate_df_from_events


class TestMultiLabelConfusionMatrix(unittest.TestCase):
    def test_add1(self):
        # Test case according to paper "MLCM: Multi-Label Confusion Matrix"
        classes = ['C0', 'C1', 'C2']
        i1 = (['C0', 'C1'], ['C0', 'C1'])
        i2 = (['C0', 'C1', 'C2'], ['C0', 'C2'])
        i3 = ([], [])

        i4 = (['C0'], ['C0', 'C1', 'C2'])
        i5 = (['C0', 'C1'], ['C0', 'C1', 'C2'])
        i6 = ([], ['C1', 'C2'])

        i7 = (['C0'], ['C1', 'C2'])
        i8 = (['C0', 'C1'], ['C0', 'C2'])
        i9 = (['C0', 'C1'], ['C2'])

        mlcm = MultiLabelConfusionMatrix(classes)
        mlcm.add(set(i1[0]), set(i1[1]))
        assert (mlcm.matrix.values == [[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       ]).all()
        assert mlcm.last_cat == 1

        mlcm = MultiLabelConfusionMatrix(classes)
        mlcm.add(set(i2[0]), set(i2[1]))
        assert (mlcm.matrix.values == [[1, 0, 0, 0],
                                       [0, 0, 0, 1],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 0],
                                       ]).all()
        assert mlcm.last_cat == 1

        mlcm = MultiLabelConfusionMatrix(classes)
        mlcm.add(set(i3[0]), set(i3[1]))
        assert (mlcm.matrix.values == [[0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 1],
                                       ]).all()
        assert mlcm.last_cat == 0  # paper states that it actually terminates before the 2nd step

        mlcm = MultiLabelConfusionMatrix(classes)
        mlcm.add(set(i4[0]), set(i4[1]))
        assert (mlcm.matrix.values == [[1, 1, 1, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       ]).all()
        assert mlcm.last_cat == 2

        mlcm = MultiLabelConfusionMatrix(classes)
        mlcm.add(set(i5[0]), set(i5[1]))
        assert (mlcm.matrix.values == [[1, 0, 1, 0],
                                       [0, 1, 1, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       ]).all()
        assert mlcm.last_cat == 2

        mlcm = MultiLabelConfusionMatrix(classes)
        mlcm.add(set(i6[0]), set(i6[1]))
        assert (mlcm.matrix.values == [[0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       [0, 1, 1, 0],
                                       ]).all()
        assert mlcm.last_cat == 2

        mlcm = MultiLabelConfusionMatrix(classes)
        mlcm.add(set(i7[0]), set(i7[1]))
        assert (mlcm.matrix.values == [[0, 1, 1, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       ]).all()
        assert mlcm.last_cat == 3

        mlcm = MultiLabelConfusionMatrix(classes)
        mlcm.add(set(i8[0]), set(i8[1]))
        assert (mlcm.matrix.values == [[1, 0, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       ]).all()
        assert mlcm.last_cat == 3

        mlcm = MultiLabelConfusionMatrix(classes)
        mlcm.add(set(i9[0]), set(i9[1]))
        assert (mlcm.matrix.values == [[0, 0, 1, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       ]).all()
        assert mlcm.last_cat == 3

        mlcm = MultiLabelConfusionMatrix(classes)
        mlcm.add(set(i1[0]), set(i1[1]))
        mlcm.add(set(i2[0]), set(i2[1]))
        mlcm.add(set(i3[0]), set(i3[1]))
        mlcm.add(set(i4[0]), set(i4[1]))
        mlcm.add(set(i5[0]), set(i5[1]))
        mlcm.add(set(i6[0]), set(i6[1]))
        mlcm.add(set(i7[0]), set(i7[1]))
        mlcm.add(set(i8[0]), set(i8[1]))
        mlcm.add(set(i9[0]), set(i9[1]))
        assert (mlcm.matrix.values == [[5, 2, 4, 0],
                                       [0, 2, 3, 1],
                                       [0, 0, 1, 0],
                                       [0, 1, 1, 1],
                                       ]).all()

        stats = mlcm.get_all_stats()
        assert stats['C0'].TN == 4
        assert stats['C0'].FP == 0
        assert stats['C0'].FN == 6
        assert stats['C0'].TP == 5
        assert stats['C1'].TN == 7
        assert stats['C1'].FP == 3
        assert stats['C1'].FN == 4
        assert stats['C1'].TP == 2
        assert stats['C2'].TN == 8
        assert stats['C2'].FP == 8
        assert stats['C2'].FN == 0
        assert stats['C2'].TP == 1
        assert stats[MultiLabelConfusionMatrix.NTL].TN == 8
        assert stats[MultiLabelConfusionMatrix.NTL].FP == 1
        assert stats[MultiLabelConfusionMatrix.NTL].FN == 2
        assert stats[MultiLabelConfusionMatrix.NTL].TP == 1
        assert stats['sum'].TN == 27
        assert stats['sum'].FP == 12
        assert stats['sum'].FN == 12
        assert stats['sum'].TP == 9

    def test_add2(self):
        # Another test case according to paper "MLCM: Multi-Label Confusion Matrix"
        classes = ['C0', 'C1', 'C2', 'C3', 'C4']
        i1 = (['C0', 'C1', 'C2'], ['C0', 'C3', 'C4'])

        mlcm = MultiLabelConfusionMatrix(classes)
        mlcm.add(set(i1[0]), set(i1[1]))
        assert (mlcm.matrix.values == [[1, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 1, 1, 0],
                                       [0, 0, 0, 1, 1, 0],
                                       [0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0],
                                       ]).all()

    @patch('processtransformer.xai.metrics.completeness_co02.MultiLabelConfusionMatrix.get_stats_for_class',
           autospec=True)
    def test_metrics(self, mock_get_stats_for_class):
        # See table 9
        C0 = TrueFalsePosNeg(58, 453, 17, 23)
        C1 = TrueFalsePosNeg(105, 406, 18, 20)
        C2 = TrueFalsePosNeg(24, 487, 5, 5)
        C3 = TrueFalsePosNeg(9, 502, 7, 12)
        C4 = TrueFalsePosNeg(54, 457, 11, 20)
        C5 = TrueFalsePosNeg(10, 501, 38, 41)
        C6 = TrueFalsePosNeg(48, 463, 13, 51)
        C7 = TrueFalsePosNeg(42, 469, 10, 37)
        C8 = TrueFalsePosNeg(161, 350, 17, 34)
        NTL = TrueFalsePosNeg(0, 511, 107, 0)

        # noinspection PyUnusedLocal,PyShadowingNames
        def get_stats(self, clazz, clazz_col=None) -> TrueFalsePosNeg:
            if clazz == 'C0':
                return C0
            elif clazz == 'C1':
                return C1
            elif clazz == 'C2':
                return C2
            elif clazz == 'C3':
                return C3
            elif clazz == 'C4':
                return C4
            elif clazz == 'C5':
                return C5
            elif clazz == 'C6':
                return C6
            elif clazz == 'C7':
                return C7
            elif clazz == 'C8':
                return C8
            elif clazz == 'NTL':
                return NTL

        mock_get_stats_for_class.side_effect = get_stats

        mcml = MultiLabelConfusionMatrix(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'])
        # See table 8 (a)
        mcml.matrix.loc['C0'] = [58, 1, 0, 1, 0, 5, 4, 2, 3, 7]
        mcml.matrix.loc['C1'] = [1, 105, 0, 0, 1, 1, 0, 0, 4, 13]
        mcml.matrix.loc['C2'] = [0, 2, 24, 0, 0, 0, 0, 0, 0, 3]
        mcml.matrix.loc['C3'] = [1, 1, 1, 9, 0, 4, 1, 0, 0, 4]
        mcml.matrix.loc['C4'] = [2, 5, 2, 1, 54, 2, 1, 0, 0, 7]
        mcml.matrix.loc['C5'] = [5, 3, 1, 0, 1, 10, 4, 2, 5, 20]
        mcml.matrix.loc['C6'] = [1, 0, 0, 5, 4, 9, 48, 6, 2, 24]
        mcml.matrix.loc['C7'] = [3, 1, 1, 0, 1, 9, 1, 42, 3, 18]
        mcml.matrix.loc['C8'] = [4, 5, 0, 0, 4, 8, 2, 0, 161, 11]
        mcml.matrix.loc['NTL'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Compare with table 12
        class_metrics = mcml.get_metrics_for_all_classes()
        cm = class_metrics['C0']
        assert np.isclose(cm.precision, 0.77, atol=0.005)
        assert np.isclose(cm.recall, 0.72, atol=0.005)
        assert np.isclose(cm.f_score, 0.74, atol=0.005)
        cm = class_metrics['C1']
        assert np.isclose(cm.precision, 0.85, atol=0.005)
        assert np.isclose(cm.recall, 0.84, atol=0.005)
        assert np.isclose(cm.f_score, 0.85, atol=0.005)
        cm = class_metrics['C2']
        assert np.isclose(cm.precision, 0.83, atol=0.005)
        assert np.isclose(cm.recall, 0.83, atol=0.005)
        assert np.isclose(cm.f_score, 0.83, atol=0.005)
        cm = class_metrics['C3']
        assert np.isclose(cm.precision, 0.56, atol=0.005)
        assert np.isclose(cm.recall, 0.43, atol=0.005)
        assert np.isclose(cm.f_score, 0.49, atol=0.005)
        cm = class_metrics['C4']
        assert np.isclose(cm.precision, 0.83, atol=0.005)
        assert np.isclose(cm.recall, 0.73, atol=0.005)
        assert np.isclose(cm.f_score, 0.78, atol=0.005)
        cm = class_metrics['C5']
        assert np.isclose(cm.precision, 0.21, atol=0.005)
        assert np.isclose(cm.recall, 0.20, atol=0.005)
        assert np.isclose(cm.f_score, 0.20, atol=0.005)
        cm = class_metrics['C6']
        assert np.isclose(cm.precision, 0.79, atol=0.005)
        assert np.isclose(cm.recall, 0.48, atol=0.005)
        assert np.isclose(cm.f_score, 0.60, atol=0.005)
        cm = class_metrics['C7']
        assert np.isclose(cm.precision, 0.81, atol=0.005)
        assert np.isclose(cm.recall, 0.53, atol=0.005)
        assert np.isclose(cm.f_score, 0.64, atol=0.005)
        cm = class_metrics['C8']
        assert np.isclose(cm.precision, 0.90, atol=0.005)
        assert np.isclose(cm.recall, 0.83, atol=0.005)
        assert np.isclose(cm.f_score, 0.86, atol=0.005)

        overall_metrics = mcml.get_overall_metrics()
        # Commented out on purpose - results in table 12 are not in accordance to eq. (5).
        # Confirmed this via manual calculation of the micro-scores. If we add NLT, it is ok though.
        # assert np.isclose(overall_metrics.precision_micro, 0.68, atol=0.005)
        # assert np.isclose(overall_metrics.recall_micro, 0.68, atol=0.005)
        # assert np.isclose(overall_metrics.f_micro, 0.68, atol=0.005)
        assert np.isclose(overall_metrics.precision_macro, 0.73, atol=0.005)
        assert np.isclose(overall_metrics.recall_macro, 0.62, atol=0.005)
        assert np.isclose(overall_metrics.f_macro, 0.67, atol=0.005)
        assert np.isclose(overall_metrics.precision_weighted, 0.79, atol=0.005)
        assert np.isclose(overall_metrics.recall_weighted, 0.68, atol=0.005)
        assert np.isclose(overall_metrics.f_weighted, 0.72, atol=0.005)
        pass


class TestEvalCompleteness(unittest.TestCase):
    @patch('processtransformer.models.helper.Predictor', autospec=True)
    @patch('processtransformer.xai.metrics.completeness_co02.generate_real_local_env', autospec=True)
    def test_eval_completeness1(self, mock_gen_local_env, mock_predictor):
        base_trace = ['A', 'B', 'D']
        y_word_dict: YWordDict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}

        def gen_local_env(trace: Trace, event_log: EventLog, return_base_trace=False) -> typing.List[Trace]:
            if len(trace) == 0:
                return [[]]
            elif trace == ['A']:
                return [['A']]
            elif trace == ['A', 'B']:
                return [['A', 'B'], ['A', 'C']]
            elif trace == ['A', 'B', 'D']:
                return [['A', 'B', 'D'], ['A', 'C', 'D']]

        mock_gen_local_env.side_effect = gen_local_env

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

        pred = mock_predictor.return_value

        def make_pred(trace_in, return_softmax):
            # Dicts are softmax-vectors that are to be returned
            if len(trace_in) == 0:
                softmax_dict = {'A': 0.95, 'B': 0.01, 'C': 0.01, 'D': 0.01, 'E': 0.01, 'F': 0.01}
            elif trace_in == ['A']:
                softmax_dict = {'A': 0.01, 'B': 0.55, 'C': 0.41, 'D': 0.01, 'E': 0.01, 'F': 0.01}
            elif trace_in == ['A', 'B']:
                softmax_dict = {'A': 0.01, 'B': 0.01, 'C': 0.01, 'D': 0.89, 'E': 0.04, 'F': 0.04}
            elif trace_in == ['A', 'C']:
                softmax_dict = {'A': 0.01, 'B': 0.01, 'C': 0.01, 'D': 0.89, 'E': 0.04, 'F': 0.04}
            elif trace_in == ['A', 'B', 'D']:
                softmax_dict = {'A': 0.01, 'B': 0.01, 'C': 0.01, 'D': 0.01, 'E': 0.72, 'F': 0.24}
            elif trace_in == ['A', 'C', 'D']:
                softmax_dict = {'A': 0.01, 'B': 0.01, 'C': 0.01, 'D': 0.01, 'E': 0.23, 'F': 0.73}
            else:
                # Default case - should not occur!
                assert False

            return None, None, None, None, [(k, v) for k, v in softmax_dict.items()], list(softmax_dict.values())

        pred.make_prediction.side_effect = make_pred

        def make_multi_predictions(_traces, return_softmax):
            return [make_pred(t, return_softmax) for t in _traces]

        pred.make_multi_predictions.side_effect = make_multi_predictions

        relations_dict: RelationsDict = dict()
        relations_dict[('A',)] = ('B', 'C',)
        relations_dict[('B',)] = ('D',)
        relations_dict[('C',)] = ('D',)
        relations_dict[('D',)] = ('E', 'F',)
        relations_dict[('E',)] = ()
        relations_dict[('F',)] = ()
        relations_output = RelationsOutput(relations_dict, None, None, ['A'])

        dct_single, overall, _, _ = eval_completeness(base_trace, log, y_word_dict, pred, relations_output)
        assert np.isclose(dct_single['A'].precision, 1.0, atol=0.001)
        assert np.isclose(dct_single['A'].recall, 1.0, atol=0.001)
        assert np.isclose(dct_single['A'].f_score, 1.0, atol=0.001)
        assert np.isclose(dct_single['B'].precision, 0.25, atol=0.001)
        assert np.isclose(dct_single['B'].recall, 1.0, atol=0.001)
        assert np.isclose(dct_single['B'].f_score, 0.4, atol=0.001)
        assert np.isclose(dct_single['C'].precision, 0.25, atol=0.001)
        assert np.isclose(dct_single['C'].recall, 1.0, atol=0.001)
        assert np.isclose(dct_single['C'].f_score, 0.4, atol=0.001)
        assert np.isclose(dct_single['D'].precision, 1.0, atol=0.001)
        assert np.isclose(dct_single['D'].recall, 0.5, atol=0.001)
        assert np.isclose(dct_single['D'].f_score, 0.667, atol=0.001)
        assert np.isclose(dct_single['E'].precision, 1.0, atol=0.001)
        assert np.isclose(dct_single['E'].recall, 0.5, atol=0.001)
        assert np.isclose(dct_single['E'].f_score, 0.667, atol=0.001)
        assert np.isclose(dct_single['F'].precision, 1.0, atol=0.001)
        assert np.isclose(dct_single['F'].recall, 0.5, atol=0.001)
        assert np.isclose(dct_single['F'].f_score, 0.667, atol=0.001)

        assert np.isclose(overall.precision_micro, 0.6, atol=0.001)
        assert np.isclose(overall.recall_micro, 0.6, atol=0.001)
        assert np.isclose(overall.f_micro, 0.6, atol=0.001)
        assert np.isclose(overall.precision_macro, 0.75, atol=0.001)
        assert np.isclose(overall.recall_macro, 0.75, atol=0.001)
        assert np.isclose(overall.f_macro, 0.6333, atol=0.001)
        assert np.isclose(overall.precision_weighted, 0.9, atol=0.001)
        assert np.isclose(overall.recall_weighted, 0.6, atol=0.001)
        assert np.isclose(overall.f_weighted, 0.6533, atol=0.001)
        pass


if __name__ == '__main__':
    unittest.main()
