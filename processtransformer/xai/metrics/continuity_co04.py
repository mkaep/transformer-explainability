
import dataclasses
import threading

import pandas as pd
from pm4py.objects.log.obj import EventLog
from scipy.spatial.distance import cosine

from processtransformer.models.helper import Predictor
from processtransformer.util.types import Trace, YWordDict
from processtransformer.xai.metrics.common import get_jaccard_value, dct_to_vector, get_relaxed_jaccard_value
from processtransformer.xai.metrics.feature_importance import get_blackbox_feature_importance, BlackboxFIWrapper
from processtransformer.xai.metrics.trace_generation import maybe_gen_real_local_env, MaybeGenWrapper, ArtEnvWrapper
from processtransformer.xai.visualization.output_models.output_data import ExplainableOutputData


@dataclasses.dataclass
class ContinuityMetrics:
    similarity: float
    similarity_count: int
    relaxed_similarity: float
    relaxed_similarity_count: int
    feat_imp_similarity: float
    feat_imp_similarity_count: int
    threshold_less_sim: float
    threshold_less_relaxed_sim: float
    base_trace: Trace

    def get_as_df(self):
        return pd.DataFrame(data=[[
            self.similarity, self.similarity_count,
            self.relaxed_similarity, self.relaxed_similarity_count,
            self.feat_imp_similarity, self.feat_imp_similarity_count,
            self.threshold_less_sim, self.threshold_less_relaxed_sim,
            self.base_trace,
        ]],
            columns=[
                'similarity', 'similarity_count',
                'relaxed_similarity', 'relaxed_similarity_count',
                'feat_imp_similarity', 'feat_imp_similarity_count',
                'threshold_less_sim', 'threshold_less_relaxed_sim',
                'base_trace',
            ])


class ContinuityCo04Wrapper:
    _bb_fi_wrapper: BlackboxFIWrapper = None
    _maybe_gen_wrapper: MaybeGenWrapper = None
    _art_env_wrapper: ArtEnvWrapper = None

    def __init__(self,
                 event_log: EventLog,
                 predictor: Predictor,
                 relations_output: ExplainableOutputData,
                 y_word_dict: YWordDict,
                 dist_func=cosine,
                 dist_threshold=0.2,
                 similarity_threshold=0.15,
                 bb_fi_wrapper: BlackboxFIWrapper = None,
                 maybe_gen_wrapper: MaybeGenWrapper = None,
                 art_env_wrapper: ArtEnvWrapper = None,
                 ) -> None:
        super().__init__()
        self.event_log = event_log
        self.predictor = predictor
        self.relations_output = relations_output
        self.y_word_dict = y_word_dict
        self.dist_func = dist_func
        self.dist_threshold = dist_threshold
        self.similarity_threshold = similarity_threshold

        if art_env_wrapper is not None:
            self.art_env_wrapper = art_env_wrapper
        elif ContinuityCo04Wrapper._art_env_wrapper is not None:
            self.art_env_wrapper = ContinuityCo04Wrapper._art_env_wrapper
        else:
            self.art_env_wrapper = ArtEnvWrapper(event_log)

        if bb_fi_wrapper is not None:
            self.bb_fi_wrapper = bb_fi_wrapper
        elif ContinuityCo04Wrapper._bb_fi_wrapper is not None:
            self.bb_fi_wrapper = ContinuityCo04Wrapper._bb_fi_wrapper
        else:
            self.bb_fi_wrapper = BlackboxFIWrapper(y_word_dict, event_log, predictor,
                                                   art_env_wrapper=self.art_env_wrapper)

        if maybe_gen_wrapper is not None:
            self.maybe_gen_wrapper = maybe_gen_wrapper
        elif ContinuityCo04Wrapper._maybe_gen_wrapper is not None:
            self.maybe_gen_wrapper = ContinuityCo04Wrapper._maybe_gen_wrapper
        else:
            self.maybe_gen_wrapper = MaybeGenWrapper(event_log)

        self.lock = threading.Lock()
        self.cache = dict()

    @classmethod
    def set_global_bb_fi_wrapper(cls, bb_fi_wrapper: BlackboxFIWrapper):
        cls._bb_fi_wrapper = bb_fi_wrapper

    @classmethod
    def set_global_maybe_gen_wrapper(cls, maybe_gen_wrapper: MaybeGenWrapper):
        cls._maybe_gen_wrapper = maybe_gen_wrapper

    @classmethod
    def set_global_art_env_wrapper(cls, art_env_wrapper: ArtEnvWrapper):
        cls._art_env_wrapper = art_env_wrapper

    def eval(self, base_trace: Trace,
             ) -> ContinuityMetrics:
        key = tuple(base_trace)
        if key in self.cache.keys():
            return self.cache[key]

        result = eval_continuity(base_trace, self.event_log, self.predictor, self.relations_output,
                                 self.y_word_dict, self.dist_func, self.dist_threshold, self.similarity_threshold,
                                 self.bb_fi_wrapper, self.maybe_gen_wrapper, self.art_env_wrapper)

        with self.lock:
            self.cache[key] = result
        return result


def eval_continuity(base_trace: Trace,
                    event_log: EventLog,
                    predictor: Predictor,
                    relations_output: ExplainableOutputData,
                    y_word_dict: YWordDict,
                    dist_func=cosine,
                    dist_threshold=0.2,
                    similarity_threshold=0.15,
                    bb_fi_wrapper: BlackboxFIWrapper = None,
                    maybe_gen_wrapper: MaybeGenWrapper = None,
                    art_env_wrapper: ArtEnvWrapper = None,
                    ) -> ContinuityMetrics:
    # Restrict to 10 - this should be sufficient
    if maybe_gen_wrapper is None:
        local_traces = maybe_gen_real_local_env(base_trace, event_log, return_base_trace=False)
    else:
        local_traces = maybe_gen_wrapper.eval(base_trace, False)
    # should be sufficient
    local_traces = local_traces[:10]

    base_sm = predictor.make_prediction(base_trace, return_softmax=True)[5]
    if bb_fi_wrapper is None:
        base_fi_dct, _ = get_blackbox_feature_importance(base_trace, y_word_dict, event_log, predictor,
                                                         art_env_wrapper=art_env_wrapper)
    else:
        base_fi_dct, _ = bb_fi_wrapper.eval_bb_fi(base_trace)
    base_fi_dct = dct_to_vector(base_fi_dct)

    base_xai = relations_output.get_explanation_for_trace(base_trace, None)

    similarity = 0
    dissimilarity = 0
    relaxed_similarity = 0
    relaxed_dissimilarity = 0
    similarity_feat_imp = 0
    dissimilarity_feat_imp = 0
    threshold_less_sim = 0.0
    threshold_less_relaxed_sim = 0.0
    threshold_less_counter = 0

    nn_res = predictor.make_multi_predictions(local_traces, return_softmax=True)
    softmaxes = [res[5] for res in nn_res]
    for local_trace, local_sm in zip(local_traces, softmaxes):
        # NN predictions similar?
        # local_sm = predictor.make_prediction(local_trace, return_softmax=True)[5]
        if bb_fi_wrapper is None:
            local_fi_dct, _ = get_blackbox_feature_importance(local_trace, y_word_dict, event_log, predictor,
                                                              art_env_wrapper=art_env_wrapper)
        else:
            local_fi_dct, _ = bb_fi_wrapper.eval_bb_fi(local_trace)
        dist = dist_func(base_sm, local_sm)
        if dist > dist_threshold:
            continue
        blackbox_similarity = 1.0 - dist
        threshold_less_counter += 1

        # XAI similarity
        local_xai = relations_output.get_explanation_for_trace(local_trace, None)
        jaccard = get_jaccard_value(local_trace, base_xai, local_xai)

        # NN and XAI approximately equally similar?
        closeness = abs(blackbox_similarity - jaccard)
        threshold_less_sim += 1.0 - closeness
        are_NN_XAI_similar = closeness < similarity_threshold
        if are_NN_XAI_similar:
            # Similar
            similarity += 1
        else:
            dissimilarity += 1

        local_fi_dct = dct_to_vector(local_fi_dct)
        if dist_func(base_fi_dct, local_fi_dct) < dist_threshold:
            if are_NN_XAI_similar:
                similarity_feat_imp += 1
            else:
                dissimilarity_feat_imp += 1

        relaxed_jaccard = get_relaxed_jaccard_value(base_xai, local_xai)
        relaxed_closeness = abs(blackbox_similarity - relaxed_jaccard)
        threshold_less_relaxed_sim += 1.0 - relaxed_closeness
        if relaxed_closeness < similarity_threshold:
            relaxed_similarity += 1
        else:
            relaxed_dissimilarity += 1

    sim_count = similarity + dissimilarity
    if sim_count > 0:
        similarity /= sim_count
    else:
        similarity = 0.0

    relaxed_sim_count = relaxed_similarity + relaxed_dissimilarity
    if relaxed_sim_count > 0:
        relaxed_similarity /= relaxed_sim_count
    else:
        relaxed_similarity = 0.0

    similarity_feat_imp_count = similarity_feat_imp + dissimilarity_feat_imp
    if similarity_feat_imp_count > 0:
        similarity_feat_imp /= similarity_feat_imp_count
    else:
        similarity_feat_imp = 0.0

    if threshold_less_counter > 0:
        threshold_less_sim /= threshold_less_counter
        threshold_less_relaxed_sim /= threshold_less_counter

    return ContinuityMetrics(similarity, sim_count,
                             relaxed_similarity, relaxed_sim_count,
                             similarity_feat_imp, similarity_feat_imp_count,
                             threshold_less_sim, threshold_less_relaxed_sim,
                             base_trace)
