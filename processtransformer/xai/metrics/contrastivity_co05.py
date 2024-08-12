
import dataclasses
import threading

import pandas as pd
from pm4py.objects.log.obj import EventLog
from scipy.spatial.distance import cosine

from processtransformer.models.helper import Predictor
from processtransformer.util.types import Trace
from processtransformer.xai.metrics.common import get_jaccard_value, get_relaxed_jaccard_value
from processtransformer.xai.metrics.trace_generation import maybe_gen_real_local_env, MaybeGenWrapper
from processtransformer.xai.visualization.output_models.output_data import ExplainableOutputData


@dataclasses.dataclass
class ContrastivityMetrics:
    diff_sim: float
    diff_sim_relaxed: float
    count: int
    base_trace: Trace

    def get_as_df(self):
        return pd.DataFrame(
            data=[[self.diff_sim, self.diff_sim_relaxed, self.count, self.base_trace]],
            columns=['diff_sim', 'diff_sim_relaxed', 'count', 'base_trace']
        )


class ContrastivityCo05Wrapper:
    def __init__(self,
                 event_log,
                 predictor,
                 relations_output,
                 dist_func=cosine,
                 dist_threshold=0.25,
                 ) -> None:
        super().__init__()
        self.event_log = event_log
        self.predictor = predictor
        self.relations_output = relations_output
        self.dist_func = dist_func
        self.dist_threshold = dist_threshold
        self.maybe_gen_wrapper = MaybeGenWrapper(event_log)

        self.lock = threading.Lock()
        self.cache = dict()

    def eval(self,
             base_trace: Trace,
             ) -> ContrastivityMetrics:
        key = tuple(base_trace)
        if key in self.cache.keys():
            return self.cache[key]

        result = eval_contrastivity(base_trace,
                                    self.event_log,
                                    self.predictor,
                                    self.relations_output,
                                    self.dist_func,
                                    self.dist_threshold,
                                    self.maybe_gen_wrapper,
                                    )
        with self.lock:
            self.cache[key] = result
        return result


def eval_contrastivity(base_trace: Trace,
                       event_log: EventLog,
                       predictor: Predictor,
                       relations_output: ExplainableOutputData,
                       dist_func=cosine,
                       dist_threshold=0.25,
                       maybe_gen_wrapper: MaybeGenWrapper = None,
                       ) -> ContrastivityMetrics:
    if maybe_gen_wrapper is None:
        local_traces = maybe_gen_real_local_env(base_trace, event_log, return_base_trace=False)
    else:
        local_traces = maybe_gen_wrapper.eval(base_trace, False)
    # should be sufficient
    local_traces = local_traces[:20]

    base_sm = predictor.make_prediction(base_trace, return_softmax=True)[5]
    base_xai = relations_output.get_explanation_for_trace(base_trace, None)

    diff_sim = 0.0
    diff_sim_relaxed = 0.0
    diff_sim_count = 0

    for local_trace in local_traces:
        local_sm = predictor.make_prediction(local_trace, return_softmax=True)[5]

        dist_sm = dist_func(base_sm, local_sm)
        if dist_sm < dist_threshold:
            continue
        similarity_sm = 1.0 - dist_sm

        local_xai = relations_output.get_explanation_for_trace(local_trace, None)
        jaccard = get_jaccard_value(local_trace, base_xai, local_xai)
        relaxed_jaccard = get_relaxed_jaccard_value(base_xai, local_xai)

        diff_sim += abs(similarity_sm - jaccard)
        diff_sim_relaxed += abs(similarity_sm - relaxed_jaccard)
        diff_sim_count += 1

    if diff_sim_count == 0:
        # Set to worst values
        diff_sim = 0.0
        diff_sim_relaxed = 0.0
    else:
        diff_sim /= diff_sim_count
        diff_sim = 1.0 - diff_sim  # "invert", i.e., 1.0 is good, 0.0 is bad.
        diff_sim_relaxed /= diff_sim_count
        diff_sim_relaxed = 1.0 - diff_sim_relaxed
    return ContrastivityMetrics(diff_sim, diff_sim_relaxed, diff_sim_count, base_trace)
