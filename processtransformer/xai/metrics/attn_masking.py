
import dataclasses
import os.path
import typing

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from processtransformer.models.helper import Predictor
from processtransformer.util.types import Trace
from processtransformer.xai.metrics.common import file_endings
from processtransformer.xai.metrics.common import total_variation_distance, summarize_df


@dataclasses.dataclass
class AttentionMaskingMetrics:
    df: pd.DataFrame
    df_summary: pd.DataFrame
    first_quantile: float
    median: float
    third_quantile: float


def eval_attn_vs_trace_masking(predictor: Predictor,
                               traces: typing.List[Trace],
                               result_dir=None,
                               ) -> AttentionMaskingMetrics:
    df_list = []
    for trace in tqdm(traces, 'Iterating over traces [Attn- vs. trace-masking]'):
        masked_traces = [
            [event if j != i else f'M-{event}' for j, event in enumerate(trace)]
            for i, event in enumerate(trace)
        ]

        # Call with masked trace
        results = predictor.make_multi_predictions(masked_traces, return_softmax=True)
        results = [(p[0], p[5]) for p in results]
        # Call with normal trace, but masked attention
        attn_results = predictor.make_multi_predictions([trace for _ in range(len(trace))],
                                                        return_softmax=True,
                                                        attn_indices_to_mask_list=[[i] for i in range(len(trace))])
        attn_results = [p[5] for p in attn_results]

        data = [[total_variation_distance(normal_masked_sm, attn_masked_sm), pred_event]
                for (pred_event, normal_masked_sm), attn_masked_sm in zip(results, attn_results)]
        df = pd.DataFrame(data=data, columns=['TVD', 'class'])
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    plot_hist_combined(df, result_dir)
    plot_hist_by_class(df, result_dir)

    first, second, third = df['TVD'].quantile([0.25, 0.50, 0.75])
    df_summary = summarize_df(df)

    return AttentionMaskingMetrics(df, df_summary, first, second, third)


def plot_hist_by_class(df, result_dir):
    sns.histplot(data=df, x='TVD', hue='class', stat='probability', bins=20)
    file_name = 'tvd_hist_by_class'
    _plot_hist(file_name, df['TVD'].max() * 1.05, result_dir)


def plot_hist_combined(df, result_dir, ax=None):
    sns.histplot(data=df, x='TVD', stat='probability', bins=10, ax=ax)
    if ax is not None:
        return
    file_name = 'tvd_hist_combined'
    _plot_hist(file_name, df['TVD'].max() * 1.05, result_dir)


def _plot_hist(file_name, max_x, result_dir):
    plt.xlim(0.0, max_x)
    plt.xlabel('TVD')
    plt.tight_layout()
    if result_dir is not None:
        for file_ending in file_endings:
            plt.savefig(os.path.join(result_dir, file_name + file_ending))
    plt.close()
