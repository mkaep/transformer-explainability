
import dataclasses
import typing

import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, kendalltau, pearsonr
from tqdm import tqdm

from processtransformer.models.helper import Predictor
from processtransformer.util.attn_funcs import reduce_attention_to_row
from processtransformer.util.types import Trace
from processtransformer.xai.metrics.common import dct_to_ordered_list
from processtransformer.xai.metrics.common import summarize_df
from processtransformer.xai.metrics.feature_importance import BlackboxFIWrapper
from processtransformer.xai.metrics.trace_generation import ArtEnvWrapper


@dataclasses.dataclass
class AttentionVsBlackBoxFiMetrics:
    df: pd.DataFrame
    df_summary: pd.DataFrame
    spearman_mean_corr: float
    spearman_std_dev_corr: float
    kendalltau_mean_corr: float
    kendalltau_std_dev_corr: float
    pearson_mean_corr: float
    pearson_std_dev_corr: float


def eval_attn_vs_bb_feature_importance(traces: typing.List[Trace],
                                       predictor: Predictor,
                                       event_log: EventLog,
                                       dist_func: typing.Callable[[typing.Any, typing.Any], float] = cosine,
                                       dist_threshold=0.1,
                                       ) -> AttentionVsBlackBoxFiMetrics:
    df_list = []
    dfg = pm4py.discover_dfg(event_log)

    art_env_wrapper = ArtEnvWrapper(event_log, dfg)
    bb_fi_wrapper = BlackboxFIWrapper(predictor.x_word_dict, event_log, predictor,
                                      art_env_wrapper=art_env_wrapper)

    for trace in tqdm(traces, 'Running over traces [Attn-vs-BB FI]'):
        overall_attn_scores = {key: (0.0, 0) for key in predictor.x_word_dict.keys()}
        _, _, attn_scores, _, _, base_sm = predictor.make_prediction(trace, return_softmax=True)
        _update_attn_dct(attn_scores, trace, overall_attn_scores)

        bb_dct, _ = bb_fi_wrapper.eval_bb_fi(trace)
        # Restrict to 100 traces
        local_traces = [local_trace for _, local_trace in art_env_wrapper.get_env(trace)][:100]
        res = predictor.make_multi_predictions(local_traces, return_softmax=True)

        for (_, _, attn_scores, _, _, sm), local_trace in zip(res, local_traces):

            if dist_func(base_sm, sm) < dist_threshold:
                _update_attn_dct(attn_scores, local_trace, overall_attn_scores)

        # Normalize - not necessary for spearman, but may be used differently in the future
        overall_attn_scores = {key: 0.0 if pair[1] == 0 else pair[0] / pair[1]
                               for key, pair in overall_attn_scores.items()}

        bb_lst = dct_to_ordered_list(bb_dct)
        attn_lst = dct_to_ordered_list(overall_attn_scores)

        s = spearmanr(bb_lst, attn_lst)
        k = kendalltau(bb_lst, attn_lst)
        p = pearsonr(bb_lst, attn_lst)
        df = pd.DataFrame(data=[[s[0], s[1], k[0], k[1], p[0], p[1]]],
                          columns=['spearman-correlation', 'spearman-p-value',
                                   'kendalltau-correlation', 'kendalltau-p-value',
                                   'pearson-correlation', 'pearson-p-value',
                                   ])
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)
    s_mean = df['spearman-correlation'].mean()
    s_std = df['spearman-correlation'].std()
    k_mean = df['kendalltau-correlation'].mean()
    k_std = df['kendalltau-correlation'].std()
    p_mean = df['pearson-correlation'].mean()
    p_std = df['pearson-correlation'].std()

    df_summary = summarize_df(df)
    return AttentionVsBlackBoxFiMetrics(df, df_summary, s_mean, s_std, k_mean, k_std, p_mean, p_std)


def _update_attn_dct(attn_scores, local_trace, overall_attn_scores):
    attn_scores = reduce_attention_to_row(attn_scores)
    for i, ev in enumerate(local_trace):
        # ev may be masked --> skip
        if ev not in overall_attn_scores.keys():
            continue
        old_val = overall_attn_scores[ev]
        overall_attn_scores[ev] = (old_val[0] + attn_scores[i], old_val[1] + 1)
