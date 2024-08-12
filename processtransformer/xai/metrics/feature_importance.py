
import threading
import typing as tp
from collections import Counter

from pm4py.objects.log.obj import EventLog
from scipy.spatial.distance import cosine

from processtransformer.models.helper import Predictor
from processtransformer.util.types import Trace, XEvent, XWordDict
from processtransformer.xai.metrics.trace_generation import generate_artificial_local_env, ArtEnvWrapper
from processtransformer.xai.visualization.output_models.output_data import ExplainableOutputData


class BlackboxFIWrapper:
    def __init__(self,
                 x_word_dict: XWordDict,
                 event_log: EventLog,
                 predictor: Predictor,
                 dist_func: tp.Callable[[tp.Any, tp.Any], float] = cosine,
                 art_env_wrapper: ArtEnvWrapper = None,
                 ) -> None:
        super().__init__()
        self.x_word_dict = x_word_dict
        self.event_log = event_log
        self.predictor = predictor
        self.dist_func = dist_func
        self.art_env_wrapper = art_env_wrapper

        self.lock = threading.Lock()
        self.cache = dict()

    def eval_bb_fi(self, trace: Trace,
                   ) -> tp.Tuple[tp.Dict[XEvent, float], tp.List[XEvent]]:
        key = tuple(trace)
        if key in self.cache.keys():
            return self.cache[key]

        result = get_blackbox_feature_importance(trace,
                                                 self.x_word_dict,
                                                 self.event_log,
                                                 self.predictor,
                                                 self.dist_func,
                                                 self.art_env_wrapper,
                                                 )
        with self.lock:
            self.cache[key] = result
        return result


def get_blackbox_feature_importance(trace: Trace,
                                    x_word_dict: XWordDict,
                                    event_log: EventLog,
                                    predictor: Predictor,
                                    dist_func: tp.Callable[[tp.Any, tp.Any], float] = cosine,
                                    art_env_wrapper: ArtEnvWrapper = None,
                                    ) -> tp.Tuple[tp.Dict[XEvent, float], tp.List[XEvent]]:
    def get_vector_for_trace(traces_in) -> tp.List[tp.List[float]]:
        result = predictor.make_multi_predictions(traces_in, return_softmax=True)
        return [res[5] for res in result]

    return _get_feature_importance(trace, predictor.x_word_dict, event_log, get_vector_for_trace, dist_func,
                                   art_env_wrapper=art_env_wrapper)


def get_xai_feature_importance(trace: Trace,
                               x_word_dict: XWordDict,
                               event_log: EventLog,
                               relations_output: ExplainableOutputData,
                               dist_func: tp.Callable[[tp.Any, tp.Any], float] = cosine,
                               art_env_wrapper: ArtEnvWrapper = None,
                               ) -> tp.Tuple[tp.Dict[XEvent, float], tp.List[XEvent]]:
    def get_vector_for_trace(traces_in) -> tp.List[tp.List[float]]:
        res = []
        for trace_in in traces_in:
            rel_dict = relations_output.get_explanation_for_trace(trace_in, None)
            predictions = list(rel_dict.values())
            # flat map, e.g. [('A', 'B'), ('B', 'C')] --> ['A', 'B', 'B', 'C']
            predictions = [e for t in predictions for e in t]
            # Gives {'A': 1, 'B': 2, 'C': 1}. Unknown key returns 0.
            predictions = Counter(predictions)
            # To vector
            res.append([predictions[key] for key in x_word_dict.keys()])
        return res

    return _get_feature_importance(trace, x_word_dict, event_log, get_vector_for_trace, dist_func,
                                   art_env_wrapper=art_env_wrapper)


def _get_feature_importance(trace: Trace,
                            x_word_dict: XWordDict,
                            event_log: EventLog,
                            get_vector_for_traces: tp.Callable[[tp.List[Trace]], tp.List[tp.List[float]]],
                            dist_func: tp.Callable[[tp.Any, tp.Any], float] = cosine,
                            art_env_wrapper: ArtEnvWrapper = None,
                            ) -> tp.Tuple[tp.Dict[XEvent, float], tp.List[XEvent]]:
    # Dict of event: (total-score, count)
    score_dict = {key: (0.0, 0) for key in x_word_dict.keys()}
    # Just the softmax vector
    base_pred = get_vector_for_traces([trace])[0]

    if art_env_wrapper is None:
        local_env = list(generate_artificial_local_env(trace, event_log))
    else:
        local_env = art_env_wrapper.get_env(trace)
    # Should be sufficient
    local_env = local_env[:20]

    local_traces = []
    events = []
    for event, local_trace in local_env:
        if event not in score_dict.keys():
            continue

        local_traces.append(local_trace)
        events.append(event)

    local_preds = get_vector_for_traces(local_traces)

    for event, local_pred in zip(events, local_preds):
        dist = dist_func(base_pred, local_pred)

        new_score = score_dict[event][0] + dist
        new_count = score_dict[event][1] + 1
        score_dict[event] = (new_score, new_count)

    # Divide score by count (do not divide by zero)
    score_dict = {key: 0.0 if pair[1] == 0 else pair[0] / pair[1] for key, pair in score_dict.items()}
    score_list = [(key, value) for key, value in score_dict.items()]
    score_list.sort(key=lambda pair: pair[1], reverse=True)
    score_list = [pair[0] for pair in score_list]

    sum_score_dct = sum(score_dict.values())
    if sum_score_dct == 0.0:
        sum_score_dct = len(score_dict.keys())
    score_dict = {key: value / sum_score_dct for key, value in score_dict.items()}
    return score_dict, score_list
