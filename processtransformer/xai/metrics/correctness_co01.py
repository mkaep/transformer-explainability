
import dataclasses
import threading
import typing

import pandas as pd
from pm4py.objects.log.obj import EventLog
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, kendalltau, pearsonr

from processtransformer.models.helper import Predictor
from processtransformer.util.types import Trace, XWordDict
from processtransformer.xai.metrics.common import dct_to_ordered_list
from processtransformer.xai.metrics.feature_importance import get_blackbox_feature_importance, \
    get_xai_feature_importance, BlackboxFIWrapper
from processtransformer.xai.metrics.trace_generation import ArtEnvWrapper
from processtransformer.xai.visualization.output_models.output_data import ExplainableOutputData


@dataclasses.dataclass
class CorrectnessMetrics:
    df: pd.DataFrame
    spearman_corr: float
    kendalltau_corr: float
    pearson_corr: float


class CorrectnessCo01Wrapper:
    _bb_wrapper: BlackboxFIWrapper = None
    _art_env_wrapper: ArtEnvWrapper = None

    def __init__(self,
                 x_word_dict: XWordDict,
                 event_log: EventLog,
                 predictor: Predictor,
                 relations_output: ExplainableOutputData,
                 dist_func: typing.Callable[[typing.Any, typing.Any], float] = cosine,
                 bb_wrapper: BlackboxFIWrapper = None,
                 art_env_wrapper: ArtEnvWrapper = None,
                 ) -> None:
        super().__init__()
        self.x_word_dict = x_word_dict
        self.event_log = event_log
        self.predictor = predictor
        self.relations_output = relations_output
        self.dist_func = dist_func

        if art_env_wrapper is not None:
            self.art_env_wrapper = art_env_wrapper
        elif CorrectnessCo01Wrapper._art_env_wrapper is not None:
            self.art_env_wrapper = CorrectnessCo01Wrapper._art_env_wrapper
        else:
            self.art_env_wrapper = ArtEnvWrapper(event_log)

        if bb_wrapper is not None:
            self.bb_wrapper = bb_wrapper
        elif CorrectnessCo01Wrapper._bb_wrapper is not None:
            self.bb_wrapper = CorrectnessCo01Wrapper._bb_wrapper
        else:
            self.bb_wrapper = BlackboxFIWrapper(x_word_dict, event_log, predictor,
                                                art_env_wrapper=self.art_env_wrapper)

        self.lock = threading.Lock()
        self.cache = dict()

    @classmethod
    def set_global_bb_wrapper(cls, bb_wrapper: BlackboxFIWrapper):
        cls._bb_wrapper = bb_wrapper

    @classmethod
    def set_global_art_env_wrapper(cls, art_env_wrapper: ArtEnvWrapper):
        cls._art_env_wrapper = art_env_wrapper

    def eval(self,
             trace: Trace,
             ) -> CorrectnessMetrics:
        key = tuple(trace)
        if key in self.cache.keys():
            return self.cache[key]

        result = eval_correctness(trace,
                                  self.x_word_dict,
                                  self.event_log,
                                  self.predictor,
                                  self.relations_output,
                                  self.dist_func,
                                  self.bb_wrapper,
                                  self.art_env_wrapper,
                                  )

        with self.lock:
            self.cache[key] = result
        return result


def eval_correctness(trace: Trace,
                     x_word_dict: XWordDict,
                     event_log: EventLog,
                     predictor: Predictor,
                     relations_output: ExplainableOutputData,
                     dist_func: typing.Callable[[typing.Any, typing.Any], float] = cosine,
                     bb_wrapper: BlackboxFIWrapper = None,
                     art_env_wrapper: ArtEnvWrapper = None,
                     ) -> CorrectnessMetrics:
    """
    Returns the correlation measured via Spearman correlation coefficient.
    Blackbox is measured against XAI.
    """
    if bb_wrapper is None:
        bb_dct, _ = get_blackbox_feature_importance(trace, x_word_dict, event_log, predictor, dist_func)
    else:
        bb_dct, _ = bb_wrapper.eval_bb_fi(trace)

    xai_dct, _ = get_xai_feature_importance(trace, x_word_dict, event_log, relations_output, dist_func,
                                            art_env_wrapper=art_env_wrapper)

    bb_lst = dct_to_ordered_list(bb_dct)
    xai_lst = dct_to_ordered_list(xai_dct)
    s = spearmanr(bb_lst, xai_lst)[0]
    k = kendalltau(bb_lst, xai_lst)[0]
    p = pearsonr(bb_lst, xai_lst)[0]
    df = pd.DataFrame(data=[[s, k, p, trace]], columns=['spearman-corr', 'kendalltau-corr', 'pearson-corr', 'trace'])
    return CorrectnessMetrics(df, s, k, p)
