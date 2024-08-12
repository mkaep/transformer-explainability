
import dataclasses
import functools
import threading
import typing

import numpy as np
import pandas as pd
from pm4py.objects.log.obj import EventLog

from processtransformer.models.helper import Predictor
from processtransformer.util.types import Trace, YWordDict
from processtransformer.xai.metrics.trace_generation import generate_real_local_env, RealEnvWrapper
from processtransformer.xai.trace_series_generators import prefix_trace_series
from processtransformer.xai.visualization.output_models.output_data import ExplainableOutputData


@dataclasses.dataclass
class TrueFalsePosNeg:
    TP: int
    TN: int
    FP: int
    FN: int


@dataclasses.dataclass
class SingleClassMetrics:
    precision: float
    recall: float
    f_score: float


@dataclasses.dataclass
class OverallMetrics:
    precision_micro: float
    precision_macro: float
    precision_weighted: float
    recall_micro: float
    recall_macro: float
    recall_weighted: float
    f_micro: float
    f_macro: float
    f_weighted: float


class CompletenessCo02Wrapper:
    _real_env_wrapper: RealEnvWrapper = None

    def __init__(self,
                 event_log: EventLog,
                 y_word_dict: YWordDict,
                 predictor: Predictor,
                 relations_output: ExplainableOutputData,
                 prediction_threshold=0.1,
                 real_env_wrapper: RealEnvWrapper = None,
                 ) -> None:
        super().__init__()
        self.event_log = event_log
        self.y_word_dict = y_word_dict
        self.predictor = predictor
        self.relations_output = relations_output
        self.prediction_threshold = prediction_threshold

        if real_env_wrapper is not None:
            self.real_env_wrapper = real_env_wrapper
        elif CompletenessCo02Wrapper._real_env_wrapper is not None:
            self.real_env_wrapper = CompletenessCo02Wrapper._real_env_wrapper
        else:
            self.real_env_wrapper = RealEnvWrapper(event_log)

        self.lock = threading.Lock()
        self.cache = dict()

    @classmethod
    def set_global_env_wrapper(cls, real_env_wrapper: RealEnvWrapper):
        cls._real_env_wrapper = real_env_wrapper

    def eval(self,
             base_trace: Trace,
             ) -> typing.Tuple[typing.Dict[typing.Any, SingleClassMetrics], OverallMetrics, pd.DataFrame, pd.DataFrame]:
        key = tuple(base_trace)
        if key in self.cache.keys():
            return self.cache[key]

        result = eval_completeness(base_trace, self.event_log, self.y_word_dict, self.predictor,
                                   self.relations_output, self.prediction_threshold, self.real_env_wrapper, )
        with self.lock:
            self.cache[key] = result
        return result


def eval_completeness(base_trace: Trace,
                      event_log: EventLog,
                      y_word_dict: YWordDict,
                      predictor: Predictor,
                      relations_output: ExplainableOutputData,
                      prediction_threshold=0.1,
                      real_env_wrapper: RealEnvWrapper = None
                      ) -> typing.Tuple[typing.Dict[typing.Any, SingleClassMetrics], OverallMetrics,
                                        pd.DataFrame, pd.DataFrame]:
    mlcm = MultiLabelConfusionMatrix(list(y_word_dict.keys()))

    for prefix in [[]] + list(prefix_trace_series(base_trace)) + [base_trace]:
        if real_env_wrapper is None:
            local_traces = generate_real_local_env(prefix, event_log, return_base_trace=True)
        else:
            local_traces = real_env_wrapper.get_env(prefix, True)
        # should be sufficient
        local_traces = local_traces[:20]

        nn_results = predictor.make_multi_predictions(local_traces, return_softmax=True)
        nn_results = [r[4] for r in nn_results]
        for nn_res, trace in zip(nn_results, local_traces):
            nn_res = set([pair[0] for pair in nn_res if pair[1] > prediction_threshold])
            xai_res = relations_output.get_explanation_for_trace(trace, None)
            xai_res = [event for right_side in xai_res.values() for event in right_side]
            xai_res = set(xai_res)

            # Notation from paper
            T = nn_res  # true labels
            P = xai_res  # predicted labels
            mlcm.add(T, P)

    single_class_dct = mlcm.get_metrics_for_all_classes()
    single_class_dfs = []
    for event, metric in single_class_dct.items():
        df = pd.DataFrame(data=[[event, metric.recall, metric.precision, metric.f_score, base_trace]],
                          columns=['event', 'recall', 'precision', 'f-score', 'base_trace'])
        single_class_dfs.append(df)
    single_class_dfs = pd.concat(single_class_dfs, ignore_index=True)

    overall_metrics = mlcm.get_overall_metrics()
    overall_df = pd.DataFrame(data=[[
        overall_metrics.precision_micro, overall_metrics.precision_macro, overall_metrics.precision_weighted,
        overall_metrics.recall_micro, overall_metrics.recall_macro, overall_metrics.recall_weighted,
        overall_metrics.f_micro, overall_metrics.f_macro, overall_metrics.f_weighted,
        base_trace,
    ]],
        columns=['precision_micro', 'precision_macro', 'precision_weighted',
                 'recall_micro', 'recall_macro', 'recall_weighted',
                 'f_micro', 'f_macro', 'f_weighted',
                 'base_trace',
                 ])

    return single_class_dct, overall_metrics, single_class_dfs, overall_df


class MultiLabelConfusionMatrix:
    # See "MLCM: Multi-Label Confusion Matrix" paper
    NPL = 'NPL'  # i.e., one or more true labels are not predicted ("no_predicted_label")
    NTL = 'NTL'  # i.e., no true label for this instance ("no_true_label")

    def __init__(self, classes: typing.List) -> None:
        super().__init__()
        self.classes = classes
        # Multi-label confusion matrix. Rows = true values (i.e. BB), columns = predicted values (i.e. XAI)
        self.matrix = pd.DataFrame(0,
                                   index=classes + [self.NTL],
                                   columns=classes + [self.NPL])
        self.last_cat = 0

    def add(self, T: typing.Set, P: typing.Set):
        T1 = T.intersection(P)  # predicted, true labels
        T2 = T.difference(P)  # not-predicted, true labels
        P1 = T1  # correctly predicted
        P2 = P.difference(T)  # incorrectly predicted

        # First step of MLCM (paper, algo 2), i.e., update all TP
        self.last_cat = 0
        for t1 in T1:
            self.matrix.loc[t1, t1] += 1

        if len(T) == 0 and len(P) == 0:
            self.matrix.loc[self.NTL, self.NPL] += 1
            return

        # Second step
        if P.issubset(T):
            # Category 1; algo 3. Add FN to all non-predicted labels
            self.last_cat = 1
            self.matrix.loc[T2, self.NPL] += 1
        elif T.issubset(P) and len(T) < len(P):
            # Category 2; algo 4.
            self.last_cat = 2
            self.matrix.loc[T, P2] += 1
            if len(T) == 0:
                self.matrix.loc[self.NTL, P2] += 1
        else:
            # Category 3 (T2 not empty, P2 not empty)
            self.last_cat = 3
            self.matrix.loc[T2, P2] += 1

    def get_stats_for_class(self, clazz, clazz_col=None) -> TrueFalsePosNeg:
        if clazz_col is None:
            clazz_col = clazz
        # See paper as well, equations (15) - (18)
        TP = self.matrix.loc[clazz, clazz_col]
        TN = sum(np.diag(self.matrix)) - TP
        FP = sum(self.matrix.loc[:, clazz_col]) - TP
        FN = sum(self.matrix.loc[clazz, :]) - TP

        return TrueFalsePosNeg(TP, TN, FP, FN)

    def get_all_stats(self) -> typing.Dict[typing.Any, TrueFalsePosNeg]:
        classes = [(clazz, clazz) for clazz in self.classes]
        classes.append((self.NTL, self.NPL))
        dct = {clazz[0]: self.get_stats_for_class(clazz[0], clazz[1]) for clazz in classes}
        sum_metrics = functools.reduce(lambda x, y: TrueFalsePosNeg(x.TP + y.TP, x.TN + y.TN,
                                                                    x.FP + y.FP, x.FN + y.FN), dct.values())
        dct['sum'] = sum_metrics
        return dct

    def get_metrics_for_class(self, clazz, beta=1.0) -> SingleClassMetrics:
        stats = self.get_stats_for_class(clazz)
        return self.calc_precision_recall_fscore(beta, stats)

    def get_metrics_for_all_classes(self, beta=1.0) -> typing.Dict[typing.Any, SingleClassMetrics]:
        return {clazz: self.get_metrics_for_class(clazz, beta) for clazz in self.classes}

    @classmethod
    def calc_precision_recall_fscore(cls, beta, stats) -> SingleClassMetrics:
        precision = stats.TP / np.nansum([stats.TP, stats.FP])
        recall = stats.TP / np.nansum([stats.TP, stats.FN])
        beta = beta ** 2
        f_score = ((beta + 1) * stats.TP) / \
                  ((beta + 1) * stats.TP + beta * stats.FN + stats.FP)
        return SingleClassMetrics(precision, recall, f_score)

    def get_overall_metrics(self) -> OverallMetrics:
        all_stats = self.get_all_stats()
        # Attention: Only want 'real' classes, according to equation (5)
        # (You actually do not get the values from the paper with the two pop's - however it does fit eq. (5))
        all_stats.pop(self.NTL, None)
        all_stats.pop('sum', None)
        summed = functools.reduce(lambda x, y: TrueFalsePosNeg(x.TP + y.TP, x.TN + y.TN,
                                                               x.FP + y.FP, x.FN + y.FN), all_stats.values())
        micro_stats = self.calc_precision_recall_fscore(1.0, summed)

        all_metrics = [self.get_metrics_for_class(clazz) for clazz in self.classes]
        precision_macro = 1.0 / len(self.classes) * np.nansum([metric.precision for metric in all_metrics])
        recall_macro = 1.0 / len(self.classes) * np.nansum([metric.recall for metric in all_metrics])
        f_macro_score = 1.0 / len(self.classes) * np.nansum([metric.f_score for metric in all_metrics])

        # Sum up each row
        weights = self.matrix.sum(axis=1).values[:-1]  # exclude NTL row
        precisions = [metric.precision for metric in all_metrics]
        precision_weighted = np.nansum(precisions * weights) / np.nansum(weights)
        recalls = [metric.recall for metric in all_metrics]
        recall_weighted = np.nansum(recalls * weights) / np.nansum(weights)
        f_scores = [metric.f_score for metric in all_metrics]
        f_weighted_score = np.nansum(f_scores * weights) / np.nansum(weights)

        return OverallMetrics(micro_stats.precision, precision_macro, precision_weighted,
                              micro_stats.recall, recall_macro, recall_weighted,
                              micro_stats.f_score, f_macro_score, f_weighted_score)
