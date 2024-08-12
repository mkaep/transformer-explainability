
import dataclasses
import threading

import pandas as pd
from pm4py.objects.log.obj import EventLog

from processtransformer.models.helper import Predictor
from processtransformer.util.types import Trace, YWordDict, XWordDict
from processtransformer.xai.metrics.feature_importance import get_blackbox_feature_importance, BlackboxFIWrapper
from processtransformer.xai.visualization.output_models.output_data import ExplainableOutputData


@dataclasses.dataclass
class CompactnessMetrics:
    num_rules: int
    avg_left_side_length: float
    avg_right_side_length: float

    ratio_left_to_trace: float
    ratio_xai_right_to_pred: float
    ratio_left_to_feat_imp: float

    base_trace: Trace

    def get_as_df(self):
        return pd.DataFrame(data=[[
            self.num_rules, self.avg_left_side_length, self.avg_right_side_length,
            self.ratio_left_to_trace, self.ratio_xai_right_to_pred, self.ratio_left_to_feat_imp,
            self.base_trace,
        ]],
            columns=['num_rules', 'avg_left_side_length', 'avg_right_side_length',
                     'ratio_left_to_trace', 'ratio_xai_right_to_pred', 'ratio_left_to_feat_imp',
                     'base_trace',
                     ]
        )


class CompactnessCo07Wrapper:
    _bb_wrapper: BlackboxFIWrapper = None

    def __init__(self,
                 predictor: Predictor,
                 x_word_dict: XWordDict,
                 y_word_dict: YWordDict,
                 event_log: EventLog,
                 relations_output: ExplainableOutputData,
                 prediction_threshold=0.1,
                 feature_importance_threshold=0.1,
                 bb_wrapper: BlackboxFIWrapper = None,
                 ) -> None:
        super().__init__()
        self.predictor = predictor
        self.y_word_dict = y_word_dict
        self.event_log = event_log
        self.relations_output = relations_output
        self.prediction_threshold = prediction_threshold
        self.feature_importance_threshold = feature_importance_threshold

        if bb_wrapper is not None:
            self.bb_wrapper = bb_wrapper
        elif CompactnessCo07Wrapper._bb_wrapper is not None:
            self.bb_wrapper = CompactnessCo07Wrapper._bb_wrapper
        else:
            self.bb_wrapper = BlackboxFIWrapper(x_word_dict, event_log, predictor)

        self.lock = threading.Lock()
        self.cache = dict()

    @classmethod
    def set_global_bb_wrapper(cls, bb_wrapper: BlackboxFIWrapper):
        cls._bb_wrapper = bb_wrapper

    def eval(self,
             base_trace: Trace,
             ) -> CompactnessMetrics:
        key = tuple(base_trace)
        if key in self.cache.keys():
            return self.cache[key]

        result = eval_compactness(base_trace,
                                  self.predictor,
                                  self.y_word_dict,
                                  self.event_log,
                                  self.relations_output,
                                  self.prediction_threshold,
                                  self.feature_importance_threshold,
                                  self.bb_wrapper,
                                  )
        with self.lock:
            self.cache[key] = result
        return result


def eval_compactness(base_trace: Trace,
                     predictor: Predictor,
                     y_word_dict: YWordDict,
                     event_log: EventLog,
                     relations_output: ExplainableOutputData,
                     prediction_threshold=0.1,
                     feature_importance_threshold=0.1,
                     bb_wrapper: BlackboxFIWrapper = None,
                     ) -> CompactnessMetrics:
    xai = relations_output.get_explanation_for_trace(base_trace, None)
    # Rule lengths
    num_rules = len(xai.keys())
    len_left_side = sum(len(left_side) for left_side in xai.keys())
    len_right_side = sum(len(right_side) for right_side in xai.values())
    if num_rules > 0:
        avg_left_side_length = len_left_side / num_rules
        avg_right_side_length = len_right_side / num_rules
    else:
        avg_left_side_length = 0.0
        avg_right_side_length = 0.0

    # Ratio
    num_events_in_trace = len(set(base_trace))
    if num_events_in_trace > 0:
        ratio_left_to_trace = len_left_side / num_events_in_trace
    else:
        ratio_left_to_trace = 0.0
    sm = predictor.make_prediction(base_trace, return_softmax=True)[4]
    predictions = set([s[0] for s in sm if s[1] > prediction_threshold])
    xai_right_side = set(event for tpl in xai.values() for event in tpl)
    len_union = len(predictions.union(xai_right_side))
    if len_union > 0:
        ratio_xai_right_to_pred = len(predictions.intersection(xai_right_side)) / len_union
    else:
        ratio_xai_right_to_pred = 0.0

    if bb_wrapper is None:
        dct_fi, _ = get_blackbox_feature_importance(base_trace, y_word_dict, event_log, predictor)
    else:
        dct_fi, _ = bb_wrapper.eval_bb_fi(base_trace)
    dct_fi = {key: value for key, value in dct_fi.items() if value > feature_importance_threshold}
    dct_fi = set(dct_fi.keys())
    left_side = set(event for tpl in xai.keys() for event in tpl)
    len_union = len(dct_fi.union(left_side))
    if len_union > 0:
        ratio_left_to_feat_imp = len(dct_fi.intersection(left_side)) / len_union
    else:
        ratio_left_to_feat_imp = 0.0

    return CompactnessMetrics(num_rules, avg_left_side_length, avg_right_side_length,
                              ratio_left_to_trace, ratio_xai_right_to_pred, ratio_left_to_feat_imp,
                              base_trace)
