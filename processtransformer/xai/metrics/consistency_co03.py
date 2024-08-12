
import dataclasses
import logging
import typing

import numpy as np
import pandas as pd
import pm4py
import tensorflow as tf
from pm4py.objects.log.obj import EventLog
from scipy.spatial.distance import cosine
from tqdm import tqdm

from main import prepare_and_train
from processtransformer.data_models.training_configuration import TrainingConfiguration
from processtransformer.models.helper import Predictor
from processtransformer.models.transformer import Transformer
from processtransformer.util.types import XEvent, YEvent, RelationsDict
from processtransformer.xai.metrics.common import dct_to_vector
from processtransformer.xai.metrics.common import get_jaccard_value
from processtransformer.xai.metrics.feature_importance import BlackboxFIWrapper
from processtransformer.xai.metrics.trace_generation import ArtEnvWrapper


class PackedOutput:
    def explain_trace(self, trace) -> typing.List[RelationsDict]:
        pass


@dataclasses.dataclass
class ConsistencyMetrics:
    consistency: float
    count: int
    diff_mean: float
    diff_std: float
    df: pd.DataFrame
    model2: Transformer
    complete_df: pd.DataFrame = None


def eval_consistency(training_config: TrainingConfiguration,
                     get_relations_output: typing.Callable[[Transformer,
                                                            typing.Dict[str, int],
                                                            typing.Dict[str, int], ], PackedOutput],
                     event_log: EventLog,
                     eval_traces: typing.List[typing.Tuple[typing.List[XEvent], YEvent]],
                     dist_func=cosine,
                     dist_threshold=0.1,
                     similarity_threshold=0.1,
                     logger: logging.Logger = None,
                     art_env_wrapper: ArtEnvWrapper = None,
                     predictor: Predictor = None,
                     processed_df=None,
                     train_processed_df=None,
                     previous_output: ConsistencyMetrics = None,
                     ) -> ConsistencyMetrics:
    if predictor is None or processed_df is None or train_processed_df is None:
        model1, x_word_dict, y_word_dict, processed_df, train_processed_df = \
            prepare_and_train(training_config, persist=False, build_metrics=False)
    else:
        model1 = predictor.model
        x_word_dict = predictor.x_word_dict
        y_word_dict = predictor.y_word_dict

    weights1 = [np.abs(w.numpy().reshape(-1)) for w in model1.trainable_weights]
    weights1 = np.concatenate(weights1)

    if previous_output is None:
        model2, predictor1, predictor2, diff_mean, diff_std = \
            _find_adversarial_model(model1, processed_df, train_processed_df,
                                    training_config, weights1,
                                    x_word_dict, y_word_dict,
                                    logger=logger)
    else:
        predictor1 = predictor
        model2 = previous_output.model2
        predictor2 = Predictor(model2, x_word_dict, y_word_dict)
        diff_mean = previous_output.diff_mean
        diff_std = previous_output.diff_std

    # Fit XAI to model1 and model2 independently
    packed_xai1 = get_relations_output(model1, x_word_dict, y_word_dict)
    packed_xai2 = get_relations_output(model2, x_word_dict, y_word_dict)

    # Caching
    if art_env_wrapper is None:
        art_env_wrapper = ArtEnvWrapper(event_log, dfg=pm4py.discover_directly_follows_graph(event_log))
    bb_fi_1 = BlackboxFIWrapper(y_word_dict, event_log, predictor1, art_env_wrapper=art_env_wrapper)
    bb_fi_2 = BlackboxFIWrapper(y_word_dict, event_log, predictor2, art_env_wrapper=art_env_wrapper)

    consistent = 0
    inconsistent = 0

    relaxed_consistency = 0.0
    relaxed_consistency_count = 0

    eval_traces = [t[0] for t in eval_traces]
    res1 = predictor1.make_multi_predictions(eval_traces, return_softmax=True)
    res1 = [r[5] for r in res1]
    res2 = predictor2.make_multi_predictions(eval_traces, return_softmax=True)
    res2 = [r[5] for r in res2]

    complete_df = []
    it = list(zip(eval_traces, res1, res2))
    # Evaluate on local/global traces, if model1 and model2 are close
    for i, (trace, sm1, sm2) in enumerate(tqdm(it, 'Running over traces [Co03]')):
        # Similarity of NNs w.r.t. predictions
        dist = dist_func(sm1, sm2)
        if dist > dist_threshold:
            continue
        similarity = 1.0 - dist

        # Similarity of NNs w.r.t. feature importance
        dct1, _ = bb_fi_1.eval_bb_fi(trace)
        dct2, _ = bb_fi_2.eval_bb_fi(trace)
        dct1 = dct_to_vector(dct1)
        dct2 = dct_to_vector(dct2)
        if dist_func(dct1, dct2) > dist_threshold:
            continue

        # Similarity of XAI, w.r.t. predictions
        dcts1 = packed_xai1.explain_trace(trace)
        dcts2 = packed_xai2.explain_trace(trace)
        local_cons, local_incons = 0, 0
        local_rel_cons, local_rel_count = 0.0, 0
        for dct1, dct2 in zip(dcts1, dcts2):
            jaccard = get_jaccard_value(trace, dct1, dct2)

            diff_bb_xai = abs(similarity - jaccard)
            local_rel_cons += 1.0 - diff_bb_xai  # if BB and XAI are similar, we get values close to 1.0
            local_rel_count += 1
            if diff_bb_xai < similarity_threshold:
                # Similar
                local_cons += 1
            else:
                local_incons += 1
        consistent += local_cons
        inconsistent += local_incons
        relaxed_consistency += local_rel_cons
        relaxed_consistency_count += local_rel_count

        local_cons, local_count, local_rel_cons = _calc_consistency(local_cons, local_incons,
                                                                    local_rel_cons, local_rel_count)

        local_df = pd.DataFrame(data=[[local_cons, local_count,
                                       local_rel_cons, local_rel_count,
                                       trace]],
                                columns=['consistency', 'count',
                                         'relaxed_consistency', 'relaxed_consistency_count',
                                         'trace'])
        complete_df.append(local_df)

        if logger is not None and (i + 1) % 20 == 0:
            logger.info(f'Co03: Iterated over {i + 1} traces')

    consistency, count, relaxed_consistency = _calc_consistency(consistent, inconsistent,
                                                                relaxed_consistency, relaxed_consistency_count)

    complete_df = pd.concat(complete_df)
    df = pd.DataFrame(data=[[consistency, count, diff_mean, diff_std,
                             relaxed_consistency, relaxed_consistency_count]],
                      columns=['consistency', 'count', 'diff_mean', 'diff_std',
                               'relaxed_consistency', 'relaxed_consistency_count'])
    return ConsistencyMetrics(consistency, count, diff_mean, diff_std, df, model2, complete_df)


def _calc_consistency(consistent, inconsistent, relaxed_consistency, relaxed_consistency_count):
    count = (consistent + inconsistent)
    if count > 0:
        consistency = consistent / count
    else:
        consistency = 0.0

    if relaxed_consistency_count > 0:
        relaxed_consistency /= relaxed_consistency_count
    return consistency, count, relaxed_consistency


def _find_adversarial_model(model1, processed_df, train_processed_df, training_config, weights1, x_word_dict,
                            y_word_dict, logger: logging.Logger = None):
    stat_list = []
    num_models = 5
    for i in range(num_models):
        tf.random.set_seed(np.random.randint(1e9))
        model2, _, _, _, _ = prepare_and_train(training_config, persist=False, build_metrics=False,
                                               processed_df=processed_df, train_processed_df=train_processed_df)

        weights2 = [np.abs(w.numpy().reshape(-1)) for w in model2.trainable_weights]
        weights2 = np.concatenate(weights2)

        w2_mean = np.mean(weights2)
        w2_std = np.std(weights2)

        weight_diff_linear = np.abs(weights1 - weights2)
        diff_mean = float(np.mean(weight_diff_linear))
        diff_std = float(np.std(weight_diff_linear))

        stat_list.append((model2, w2_mean, w2_std, diff_mean, diff_std))

        if logger is not None:
            logger.info(f'Co03: Trained {i + 1}/{num_models} models')

    # Sort by diff-mean
    stat_list.sort(key=lambda x: x[3], reverse=True)
    model2 = stat_list[0][0]
    predictor1 = Predictor(model1, x_word_dict, y_word_dict)
    predictor2 = Predictor(model2, x_word_dict, y_word_dict)
    return model2, predictor1, predictor2, stat_list[0][3], stat_list[0][4]
