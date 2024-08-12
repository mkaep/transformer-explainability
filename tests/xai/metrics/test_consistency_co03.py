

import dataclasses
import typing
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from pm4py.objects.log.obj import EventLog
from scipy.stats import cosine
from tensorflow import Variable

from processtransformer.models.helper import Predictor
from processtransformer.models.transformer import Transformer
from processtransformer.util.types import Trace, YWordDict, YEvent
from processtransformer.xai.metrics.common import get_jaccard_value
from processtransformer.xai.metrics.consistency_co03 import eval_consistency, PackedOutput
from processtransformer.xai.visualization.output_models.relations_output import RelationsOutput
from util.types import RelationsDict


@dataclasses.dataclass
class Model:
    trainable_weights: typing.List[Variable]


class TestEvalConsistency(unittest.TestCase):
    @patch('processtransformer.xai.metrics.feature_importance.get_blackbox_feature_importance', autospec=True)
    @patch('processtransformer.xai.metrics.consistency_co03._find_adversarial_model', autospec=True)
    @patch('processtransformer.xai.metrics.consistency_co03.prepare_and_train', autospec=True)
    def test_eval_consistency1(self, mock_prepare_and_train, mock_find_adv_model, mock_get_bb_fi):
        mock_pred1 = MagicMock()
        mock_pred2 = MagicMock()
        model1 = Model([Variable(np.asarray([[0, 1, 0], [0, 1, 0], [0, 1, 0]]))])
        model2 = Model([Variable(np.asarray([[0, 1, 0], [0, 1, 0], [0, 1, 0]]))])
        mock_prepare_and_train.return_value = model1, None, None, None, None
        mock_find_adv_model.return_value = model2, mock_pred1, mock_pred2, None, None

        def get_relations_output(model: Transformer,
                                 x_dict: typing.Dict[str, int],
                                 y_dict: typing.Dict[str, int],
                                 ) -> PackedOutput:
            class A(PackedOutput):
                def explain_trace(self, trace) -> typing.List[RelationsDict]:
                    return [RelationsOutput({('A',): ('B', 'C'), ('B',): 'D', ('C',): 'D', ('D',): 'E',
                                             ('R',): ('S',), ('S',): ('T', 'U'), ('T',): 'U', ('U',): 'T'},
                                            None, None, ['A', 'R']).get_explanation_for_trace(trace, None)]

            return A()

        eval_traces = [
            (['A', 'B', 'D'], '-'),
            (['A', 'C', 'D'], '-'),
            (['R', 'S', 'T'], '-'),
            (['R', 'S', 'U'], '-'),
        ]

        def make_prediction1(event_trace: typing.List[str],
                             return_softmax: bool = False):
            # Softmax: A, B, C, D, E, R, S, T, U
            if event_trace == ['A', 'B', 'D']:
                softmax = [0.01, 0.01, 0.01, 0.01, 0.95, 0.00, 0.00, 0.00, 0.00]
            elif event_trace == ['A', 'C', 'D']:
                softmax = [0.01, 0.01, 0.01, 0.01, 0.95, 0.00, 0.00, 0.00, 0.00]
            elif event_trace == ['R', 'S', 'T']:
                softmax = [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.97]
            elif event_trace == ['R', 'S', 'U']:
                softmax = [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.97, 0.01]
            else:
                assert False
            return None, None, None, None, None, softmax

        mock_pred1.make_prediction.side_effect = make_prediction1

        def make_multi_predictions(_traces, return_softmax=False):
            return [make_prediction1(t, return_softmax) for t in _traces]

        mock_pred1.make_multi_predictions.side_effect = make_multi_predictions

        mock_pred2.make_prediction.side_effect = make_prediction1
        mock_pred2.make_multi_predictions.side_effect = make_multi_predictions

        def get_bb_fi(trace: Trace,
                      y_word_dict: YWordDict,
                      event_log: EventLog,
                      predictor: Predictor,
                      dist_func: typing.Callable[[typing.Any, typing.Any], float] = cosine,
                      art_env_wrapper=None,
                      ) -> typing.Tuple[typing.Dict[YEvent, float], typing.List[YEvent]]:
            if trace == ['A', 'B', 'D']:
                dct = {'A': 0.0, 'B': 0.8, 'C': 0.0, 'D': 0.2, 'E': 0.0,
                       'R': 0.0, 'S': 0.0, 'T': 0.0, 'U': 0.0}
            elif trace == ['A', 'C', 'D']:
                dct = {'A': 0.0, 'B': 0.0, 'C': 0.8, 'D': 0.2, 'E': 0.0,
                       'R': 0.0, 'S': 0.0, 'T': 0.0, 'U': 0.0}
            elif trace == ['R', 'S', 'T']:
                dct = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0,
                       'R': 0.0, 'S': 0.4, 'T': 0.6, 'U': 0.0}
            elif trace == ['R', 'S', 'U']:
                dct = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0,
                       'R': 0.0, 'S': 0.4, 'T': 0.0, 'U': 0.6}
            else:
                assert False
            return dct, None

        mock_get_bb_fi.side_effect = get_bb_fi

        metrics = eval_consistency(None, get_relations_output, None, eval_traces,
                                   art_env_wrapper=MagicMock())
        assert np.isclose(metrics.consistency, 1.0)
        assert metrics.count == 4
        pass

    @patch('processtransformer.xai.metrics.feature_importance.get_blackbox_feature_importance', autospec=True)
    @patch('processtransformer.xai.metrics.consistency_co03._find_adversarial_model', autospec=True)
    @patch('processtransformer.xai.metrics.consistency_co03.prepare_and_train', autospec=True)
    def test_eval_consistency2(self, mock_prepare_and_train, mock_find_adv_model, mock_get_bb_fi):
        mock_pred1 = MagicMock()
        mock_pred2 = MagicMock()
        model1 = Model([Variable(np.asarray([[0, 1, 0], [0, 1, 0], [0, 1, 0]]))])
        model2 = Model([Variable(np.asarray([[0, 1, 0], [0, 1, 0], [0, 1, 0]]))])
        mock_prepare_and_train.return_value = model1, None, None, None, None
        mock_find_adv_model.return_value = model2, mock_pred1, mock_pred2, None, None

        def get_relations_output(model: Transformer,
                                 x_dict: typing.Dict[str, int],
                                 y_dict: typing.Dict[str, int],
                                 ) -> PackedOutput:
            class A(PackedOutput):
                def __init__(self, _model):
                    super().__init__()
                    self._model = _model

                def explain_trace(self, trace) -> typing.List[RelationsDict]:
                    if id(self._model) == id(model1):
                        return [RelationsOutput({('A',): ('B', 'C'), ('B',): 'D', ('C',): 'D', ('D',): 'E',
                                                 ('R',): ('S',), ('S',): ('T', 'U'), ('T',): 'U', ('U',): 'T'},
                                                None, None, ['A', 'R']).get_explanation_for_trace(trace, None)]
                    else:
                        return [RelationsOutput({('A',): ('B', 'C'), ('B',): 'D', ('C',): 'D', ('D',): 'E',
                                                 ('R',): ('S',), ('S',): ('T', 'U'), ('T',): 'U', ('U',): 'T'},
                                                None, None, ['A', 'R']).get_explanation_for_trace(trace, None)]

            return A(model)

        eval_traces = [
            (['A', 'B', 'D'], '-'),
            (['A', 'C', 'D'], '-'),
            (['R', 'S', 'T'], '-'),
            (['R', 'S', 'U'], '-'),
        ]

        def make_prediction1(event_trace: typing.List[str],
                             return_softmax: bool = False):
            # Softmax: A, B, C, D, E, R, S, T, U
            if event_trace == ['A', 'B', 'D']:
                # Changed softmax
                softmax = [0.01, 0.01, 0.01, 0.01, 0.95, 0.00, 0.00, 0.00, 0.00]
            elif event_trace == ['A', 'C', 'D']:
                softmax = [0.01, 0.01, 0.01, 0.01, 0.95, 0.00, 0.00, 0.00, 0.00]
            elif event_trace == ['R', 'S', 'T']:
                softmax = [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.97]
            elif event_trace == ['R', 'S', 'U']:
                softmax = [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.97, 0.01]
            else:
                assert False
            return None, None, None, None, None, softmax

        mock_pred1.make_prediction.side_effect = make_prediction1

        def make_multi_predictions1(_traces, return_softmax=False):
            return [make_prediction1(t, return_softmax) for t in _traces]

        mock_pred1.make_multi_predictions.side_effect = make_multi_predictions1

        def make_prediction2(event_trace: typing.List[str],
                             return_softmax: bool = False):
            # Softmax: A, B, C, D, E, R, S, T, U
            if event_trace == ['A', 'B', 'D']:
                # Changed softmax
                softmax = [0.95, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00]
            elif event_trace == ['A', 'C', 'D']:
                softmax = [0.01, 0.01, 0.01, 0.01, 0.95, 0.00, 0.00, 0.00, 0.00]
            elif event_trace == ['R', 'S', 'T']:
                softmax = [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.97]
            elif event_trace == ['R', 'S', 'U']:
                softmax = [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.97, 0.01]
            else:
                assert False
            return None, None, None, None, None, softmax

        mock_pred2.make_prediction.side_effect = make_prediction2

        def make_multi_predictions2(_traces, return_softmax=False):
            return [make_prediction2(t, return_softmax) for t in _traces]

        mock_pred2.make_multi_predictions.side_effect = make_multi_predictions2

        def get_bb_fi(trace: Trace,
                      y_word_dict: YWordDict,
                      event_log: EventLog,
                      predictor: Predictor,
                      dist_func: typing.Callable[[typing.Any, typing.Any], float] = cosine,
                      art_env_wrapper=None,
                      ) -> typing.Tuple[typing.Dict[YEvent, float], typing.List[YEvent]]:
            if trace == ['A', 'B', 'D']:
                dct = {'A': 0.0, 'B': 0.8, 'C': 0.0, 'D': 0.2, 'E': 0.0,
                       'R': 0.0, 'S': 0.0, 'T': 0.0, 'U': 0.0}
            elif trace == ['A', 'C', 'D']:
                dct = {'A': 0.0, 'B': 0.0, 'C': 0.8, 'D': 0.2, 'E': 0.0,
                       'R': 0.0, 'S': 0.0, 'T': 0.0, 'U': 0.0}
            elif trace == ['R', 'S', 'T']:
                dct = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0,
                       'R': 0.0, 'S': 0.4, 'T': 0.6, 'U': 0.0}
            elif trace == ['R', 'S', 'U']:
                dct = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0,
                       'R': 0.0, 'S': 0.4, 'T': 0.0, 'U': 0.6}
            else:
                assert False
            return dct, None

        mock_get_bb_fi.side_effect = get_bb_fi

        metrics = eval_consistency(None, get_relations_output, None, eval_traces,
                                   art_env_wrapper=MagicMock())
        assert np.isclose(metrics.consistency, 1.0)
        assert metrics.count == 3
        pass

    @patch('processtransformer.xai.metrics.feature_importance.get_blackbox_feature_importance', autospec=True)
    @patch('processtransformer.xai.metrics.consistency_co03._find_adversarial_model', autospec=True)
    @patch('processtransformer.xai.metrics.consistency_co03.prepare_and_train', autospec=True)
    def test_eval_consistency3(self, mock_prepare_and_train, mock_find_adv_model, mock_get_bb_fi):
        mock_pred1 = MagicMock()
        mock_pred2 = MagicMock()
        model1 = Model([Variable(np.asarray([[0, 1, 0], [0, 1, 0], [0, 1, 0]]))])
        model2 = Model([Variable(np.asarray([[0, 1, 0], [0, 1, 0], [0, 1, 0]]))])
        mock_prepare_and_train.return_value = model1, None, None, None, None
        mock_find_adv_model.return_value = model2, mock_pred1, mock_pred2, None, None

        def get_relations_output(model: Transformer,
                                 x_dict: typing.Dict[str, int],
                                 y_dict: typing.Dict[str, int],
                                 ) -> PackedOutput:
            class A(PackedOutput):
                def __init__(self, _model):
                    super().__init__()
                    self._model = _model

                def explain_trace(self, trace) -> typing.List[RelationsDict]:
                    if id(self._model) == id(model1):
                        return [RelationsOutput({('A',): ('B', 'C'), ('B',): 'D', ('C',): 'D', ('D',): 'E',
                                                 ('R',): ('S',), ('S',): ('T', 'U'), ('T',): 'U', ('U',): 'T'},
                                                None, None, ['A', 'R']).get_explanation_for_trace(trace, None)]
                    else:
                        return [RelationsOutput({('A',): ('B', 'C'), ('B',): 'D', ('C',): 'D', ('D',): 'E',
                                                 ('R',): ('S',), ('S',): ('T', 'U'), ('T',): 'U', ('U',): 'T'},
                                                None, None, ['A', 'R']).get_explanation_for_trace(trace, None)]

            return A(model)

        eval_traces = [
            (['A', 'B', 'D'], '-'),
            (['A', 'C', 'D'], '-'),
            (['R', 'S', 'T'], '-'),
            (['R', 'S', 'U'], '-'),
        ]

        def make_prediction1(event_trace: typing.List[str],
                             return_softmax: bool = False):
            # Softmax: A, B, C, D, E, R, S, T, U
            if event_trace == ['A', 'B', 'D']:
                # Changed softmax
                softmax = [0.01, 0.01, 0.01, 0.01, 0.95, 0.00, 0.00, 0.00, 0.00]
            elif event_trace == ['A', 'C', 'D']:
                softmax = [0.01, 0.01, 0.01, 0.01, 0.95, 0.00, 0.00, 0.00, 0.00]
            elif event_trace == ['R', 'S', 'T']:
                softmax = [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.97]
            elif event_trace == ['R', 'S', 'U']:
                softmax = [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.97, 0.01]
            else:
                assert False
            return None, None, None, None, None, softmax

        mock_pred1.make_prediction.side_effect = make_prediction1

        def make_multi_predictions1(_traces, return_softmax=False):
            return [make_prediction1(t, return_softmax) for t in _traces]

        mock_pred1.make_multi_predictions.side_effect = make_multi_predictions1

        def make_prediction2(event_trace: typing.List[str],
                             return_softmax: bool = False):
            # Softmax: A, B, C, D, E, R, S, T, U
            if event_trace == ['A', 'B', 'D']:
                # Changed softmax
                softmax = [0.95, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00]
            elif event_trace == ['A', 'C', 'D']:
                softmax = [0.01, 0.01, 0.01, 0.01, 0.95, 0.00, 0.00, 0.00, 0.00]
            elif event_trace == ['R', 'S', 'T']:
                softmax = [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.97]
            elif event_trace == ['R', 'S', 'U']:
                softmax = [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.97, 0.01]
            else:
                assert False
            return None, None, None, None, None, softmax

        mock_pred2.make_prediction.side_effect = make_prediction2

        def make_multi_predictions2(_traces, return_softmax=False):
            return [make_prediction2(t, return_softmax) for t in _traces]

        mock_pred2.make_multi_predictions.side_effect = make_multi_predictions2

        def get_bb_fi(trace: Trace,
                      y_word_dict: YWordDict,
                      event_log: EventLog,
                      predictor: Predictor,
                      dist_func: typing.Callable[[typing.Any, typing.Any], float] = cosine,
                      art_env_wrapper=None,
                      ) -> typing.Tuple[typing.Dict[YEvent, float], typing.List[YEvent]]:
            if predictor == mock_pred1:
                if trace == ['A', 'B', 'D']:
                    dct = {'A': 0.0, 'B': 0.8, 'C': 0.0, 'D': 0.2, 'E': 0.0,
                           'R': 0.0, 'S': 0.0, 'T': 0.0, 'U': 0.0}
                elif trace == ['A', 'C', 'D']:
                    dct = {'A': 0.0, 'B': 0.0, 'C': 0.8, 'D': 0.2, 'E': 0.0,
                           'R': 0.0, 'S': 0.0, 'T': 0.0, 'U': 0.0}
                elif trace == ['R', 'S', 'T']:
                    dct = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0,
                           'R': 0.0, 'S': 0.4, 'T': 0.6, 'U': 0.0}
                elif trace == ['R', 'S', 'U']:
                    # Changed
                    dct = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0,
                           'R': 0.0, 'S': 0.8, 'T': 0.0, 'U': 0.2}
                else:
                    assert False
                return dct, None
            else:
                if trace == ['A', 'B', 'D']:
                    dct = {'A': 0.0, 'B': 0.8, 'C': 0.0, 'D': 0.2, 'E': 0.0,
                           'R': 0.0, 'S': 0.0, 'T': 0.0, 'U': 0.0}
                elif trace == ['A', 'C', 'D']:
                    dct = {'A': 0.0, 'B': 0.0, 'C': 0.8, 'D': 0.2, 'E': 0.0,
                           'R': 0.0, 'S': 0.0, 'T': 0.0, 'U': 0.0}
                elif trace == ['R', 'S', 'T']:
                    dct = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0,
                           'R': 0.0, 'S': 0.4, 'T': 0.6, 'U': 0.0}
                elif trace == ['R', 'S', 'U']:
                    # Changed
                    dct = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0,
                           'R': 0.0, 'S': 0.4, 'T': 0.0, 'U': 0.6}
                else:
                    assert False
                return dct, None

        mock_get_bb_fi.side_effect = get_bb_fi

        metrics = eval_consistency(None, get_relations_output, None, eval_traces,
                                   art_env_wrapper=MagicMock())
        assert np.isclose(metrics.consistency, 1.0)
        assert metrics.count == 2
        pass

    @patch('processtransformer.xai.metrics.feature_importance.get_blackbox_feature_importance', autospec=True)
    @patch('processtransformer.xai.metrics.consistency_co03._find_adversarial_model', autospec=True)
    @patch('processtransformer.xai.metrics.consistency_co03.prepare_and_train', autospec=True)
    def test_eval_consistency4(self, mock_prepare_and_train, mock_find_adv_model, mock_get_bb_fi):
        mock_pred1 = MagicMock()
        mock_pred2 = MagicMock()
        model1 = Model([Variable(np.asarray([[0, 1, 0], [0, 1, 0], [0, 1, 0]]))])
        model2 = Model([Variable(np.asarray([[0, 1, 0], [0, 1, 0], [0, 1, 0]]))])
        mock_prepare_and_train.return_value = model1, None, None, None, None
        mock_find_adv_model.return_value = model2, mock_pred1, mock_pred2, None, None

        def get_relations_output(model: Transformer,
                                 x_dict: typing.Dict[str, int],
                                 y_dict: typing.Dict[str, int],
                                 ) -> PackedOutput:
            # Half-way different
            class A(PackedOutput):
                def __init__(self, _model):
                    super().__init__()
                    self._model = _model

                def explain_trace(self, trace) -> typing.List[RelationsDict]:
                    if id(self._model) == id(model1):
                        return [RelationsOutput({('A',): ('B', 'C'), ('B',): 'D', ('C',): 'D', ('D',): 'E',
                                                 ('R',): ('S',), ('S',): ('T', 'U'), ('T',): 'U', ('U',): 'T'},
                                                None, None, ['A', 'R']).get_explanation_for_trace(trace, None)]
                    else:
                        return [RelationsOutput({('A',): ('R',), ('B',): 'R', ('C',): 'R', ('D',): 'R',
                                                 ('R',): ('S',), ('S',): ('T', 'U'), ('T',): 'U', ('U',): 'T'},
                                                None, None, ['A', 'R']).get_explanation_for_trace(trace, None)]

            return A(model)

        eval_traces = [
            (['A', 'B', 'D'], '-'),
            (['A', 'C', 'D'], '-'),
            (['R', 'S', 'T'], '-'),
            (['R', 'S', 'U'], '-'),
        ]

        def make_prediction1(event_trace: typing.List[str],
                             return_softmax: bool = False):
            # Softmax: A, B, C, D, E, R, S, T, U
            if event_trace == ['A', 'B', 'D']:
                # Changed softmax
                softmax = [0.01, 0.01, 0.01, 0.01, 0.95, 0.00, 0.00, 0.00, 0.00]
            elif event_trace == ['A', 'C', 'D']:
                softmax = [0.01, 0.01, 0.01, 0.01, 0.95, 0.00, 0.00, 0.00, 0.00]
            elif event_trace == ['R', 'S', 'T']:
                softmax = [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.97]
            elif event_trace == ['R', 'S', 'U']:
                softmax = [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.97, 0.01]
            else:
                assert False
            return None, None, None, None, None, softmax

        mock_pred1.make_prediction.side_effect = make_prediction1

        def make_multi_predictions(_traces, return_softmax=False):
            return [make_prediction1(t, return_softmax) for t in _traces]

        mock_pred1.make_multi_predictions.side_effect = make_multi_predictions

        mock_pred2.make_prediction.side_effect = make_prediction1
        mock_pred2.make_multi_predictions.side_effect = make_multi_predictions

        def get_bb_fi(trace: Trace,
                      y_word_dict: YWordDict,
                      event_log: EventLog,
                      predictor: Predictor,
                      dist_func: typing.Callable[[typing.Any, typing.Any], float] = cosine,
                      art_env_wrapper=None,
                      ) -> typing.Tuple[typing.Dict[YEvent, float], typing.List[YEvent]]:
            if trace == ['A', 'B', 'D']:
                dct = {'A': 0.0, 'B': 0.8, 'C': 0.0, 'D': 0.2, 'E': 0.0,
                       'R': 0.0, 'S': 0.0, 'T': 0.0, 'U': 0.0}
            elif trace == ['A', 'C', 'D']:
                dct = {'A': 0.0, 'B': 0.0, 'C': 0.8, 'D': 0.2, 'E': 0.0,
                       'R': 0.0, 'S': 0.0, 'T': 0.0, 'U': 0.0}
            elif trace == ['R', 'S', 'T']:
                dct = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0,
                       'R': 0.0, 'S': 0.4, 'T': 0.6, 'U': 0.0}
            elif trace == ['R', 'S', 'U']:
                dct = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0,
                       'R': 0.0, 'S': 0.4, 'T': 0.0, 'U': 0.6}
            else:
                assert False
            return dct, None

        mock_get_bb_fi.side_effect = get_bb_fi

        metrics = eval_consistency(None, get_relations_output, None, eval_traces,
                                   art_env_wrapper=MagicMock())
        assert np.isclose(metrics.consistency, 0.5)
        assert metrics.count == 4
        pass

    def test_get_jaccard_value(self):
        # Perfect match
        jaccard = get_jaccard_value([], {('A',): ('B',), ('B',): ('C',)},
                                    {('A',): ('B',), ('B',): ('C',)})
        assert np.isclose(jaccard, 1.0)

        # No match at all
        jaccard = get_jaccard_value([], {('A',): ('B',), ('B',): ('C',)},
                                    {('X',): ('Y',), ('Y',): 'Z'})
        assert np.isclose(jaccard, 0.0)

        # Keys match, one right side does not
        jaccard = get_jaccard_value([], {'A': ('B',), ('B',): ('C',)},
                                    {'A': ('B',), ('B',): ('Y',)})
        assert np.isclose(jaccard, 0.5)

        # First right side matches only partially
        jaccard = get_jaccard_value([], {('A',): ('B', 'C'), ('B',): ('C',)},
                                    {('A',): ('B', 'D'), ('B',): ('C',)})
        assert np.isclose(jaccard, 0.667, atol=0.001)

        # Only one (perfect) key match, one unmatched key and one false key
        jaccard = get_jaccard_value([], {('A',): ('B', 'D'), ('B',): ('C',)},
                                    {('B',): ('C',), ('C',): ('B', 'D')})
        assert np.isclose(jaccard, 0.333, atol=0.001)
        pass
