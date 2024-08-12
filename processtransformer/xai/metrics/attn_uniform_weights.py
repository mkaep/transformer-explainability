
import dataclasses
import os.path
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.spatial.distance import cosine
from tqdm import tqdm

from main import prepare_and_train
from processtransformer.data_models.training_configuration import TrainingConfiguration
from processtransformer.models.helper import Predictor
from processtransformer.util.trace_util import extract_traces_from_xes
from processtransformer.util.types import Trace
from processtransformer.xai.metrics.common import total_variation_distance, jensen_shannon_divergence, \
    summarize_df, file_endings

attn1 = '[0.00,\n0.25)'
attn2 = '[0.25,\n0.50)'
attn3 = '[0.50,\n0.75)'
attn4 = '[0.75,\n1.00]'
attnX = [attn1, attn2, attn3, attn4]


@dataclasses.dataclass
class AttnUniformMetrics:
    jsd_tvd_df: pd.DataFrame
    jsd_tvd_df_summary: pd.DataFrame
    attn_df: pd.DataFrame
    attn_df_summary: pd.DataFrame


def eval_attn_uniform_weights(training_config: TrainingConfiguration,
                              tvd_threshold=0.1,
                              result_dir=None,
                              max_trace_num=None,
                              traces: typing.List[Trace] = None,
                              predictor_base: Predictor = None,
                              processed_df=None,
                              train_processed_df=None,
                              ) -> AttnUniformMetrics:
    if predictor_base is None or processed_df is None or train_processed_df is None:
        model_base, x_word_dict, y_word_dict, processed_df, train_processed_df = \
            prepare_and_train(training_config, persist=False, build_metrics=False)
        predictor_base = Predictor(model_base, x_word_dict, y_word_dict)
    else:
        x_word_dict = predictor_base.x_word_dict
        y_word_dict = predictor_base.y_word_dict

    predictors_changed = []

    num_seeds = 5
    for i in tqdm(range(num_seeds), 'Training models with different seeds [Attn-Uni-Seed]'):
        tf.random.set_seed(np.random.randint(1e9))
        model_uniform, _, _, _, _ = prepare_and_train(training_config, persist=False, build_metrics=False,
                                                      processed_df=processed_df, train_processed_df=train_processed_df)
        predictor_uniform = Predictor(model_uniform, x_word_dict, y_word_dict)
        predictors_changed.append((predictor_uniform, f'seed-{i}'))

    num_uniform = 5
    for i in tqdm(range(num_uniform), 'Training uniform models [Attn-Uni-Seed]'):
        tf.random.set_seed(np.random.randint(1e9))
        model_uniform, _, _, _, _ = prepare_and_train(training_config, persist=False, build_metrics=False,
                                                      processed_df=processed_df, train_processed_df=train_processed_df,
                                                      **dict(train_attention=False))
        predictor_uniform = Predictor(model_uniform, x_word_dict, y_word_dict)
        predictors_changed.append((predictor_uniform, f'uniform-{i}'))

    if traces is None:
        traces = extract_traces_from_xes(training_config)

    jsd_tvd_df = pd.DataFrame(0.0,
                              index=[i for i in range(len(predictors_changed))],
                              columns=['JSD', 'TVD', 'cosine'])

    attn_list = []

    if max_trace_num is not None:
        traces = traces[:max_trace_num]

    steps = 50
    for step in tqdm(range(0, len(traces), steps), 'Iterating over traces [Attn-Uni-Seed]'):
        sub_traces = traces[step:step + steps]
        base_res = predictor_base.make_multi_predictions(sub_traces, return_softmax=True)
        changed_res = list(zip(*[pred.make_multi_predictions(sub_traces, return_softmax=True)
                                 for pred, _ in predictors_changed]))

        for (ev, _, base_attn, _, _, base_sm), c_res in zip(base_res, changed_res):

            base_attn = base_attn.reshape(-1)
            base_attn /= np.sum(base_attn)

            for i, (_, _, uni_attn, _, _, uni_sm) in enumerate(c_res):

                uni_attn = uni_attn.reshape(-1)
                if len(uni_attn) == 0:
                    continue
                max_attn = uni_attn.max()
                uni_attn /= np.sum(uni_attn)
                jsd = jensen_shannon_divergence(base_attn, uni_attn)
                tvd = total_variation_distance(base_sm, uni_sm)
                cos = cosine(base_sm, uni_sm)
                jsd_tvd_df.loc[i, 'JSD'] += jsd
                jsd_tvd_df.loc[i, 'TVD'] += tvd
                jsd_tvd_df.loc[i, 'cosine'] += cos

                if tvd > tvd_threshold:
                    continue

                # ONLY for rather similar predictions
                # Examples: 0.20 --> index 0; 0.6 --> index 2, 1.0 --> index 3
                max_attn_index = np.min([len(attnX) - 1, int(np.floor(max_attn * len(attnX)))])
                attn_df = pd.DataFrame(data=[[attnX[max_attn_index], jsd, ev, max_attn]],
                                       columns=['attnX', 'JSD', 'event', 'max-attn'])
                attn_list.append(attn_df)

    attn_df = pd.concat(attn_list, ignore_index=True)

    plot_jsd_vs_max_attn(attn_df, result_dir)
    plot_jsd_vs_max_attn_by_class(attn_df, result_dir)

    attn_df_summary = []
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    for name, group in attn_df.groupby('attnX'):
        lst = group['JSD'].quantile(quantiles)
        attn_df_summary.append(pd.DataFrame(data=[lst.values],
                                            columns=[f'p-{q:.2f}' for q in quantiles],
                                            index=[f'JSD for attn: {name}']))

    attn_df_summary = pd.concat(attn_df_summary)

    jsd_tvd_df /= len(traces)
    jsd_tvd_df['model-name'] = list(list(zip(*predictors_changed))[1])

    seed = summarize_df(jsd_tvd_df[jsd_tvd_df['model-name'].str.contains('seed-')])
    seed['type'] = 'seed'
    uni = summarize_df(jsd_tvd_df[jsd_tvd_df['model-name'].str.contains('uniform-')])
    uni['type'] = 'uniform'
    jsd_tvd_df_summary = pd.concat([seed, uni])

    plot_jsd_vs_tvd(jsd_tvd_df, result_dir)

    return AttnUniformMetrics(jsd_tvd_df, jsd_tvd_df_summary, attn_df, attn_df_summary)


def plot_jsd_vs_max_attn(attn_df, result_dir, ax=None):
    sns.violinplot(data=attn_df, x='JSD', y='attnX', order=attnX,
                   scale='count', inner='quartile', color='#7c78ab',
                   linewidth=1, cut=0, dodge=True, ax=ax)
    if ax is not None:
        return
    if result_dir is not None:
        result_dir = os.path.join(result_dir, 'jsd_vs_max_attn_combined_classes')
    _common_jsd_vs_max_attn(result_dir)


def plot_jsd_vs_max_attn_by_class(attn_df, result_dir):
    sns.violinplot(data=attn_df, x='JSD', y='attnX', order=attnX, hue='event',
                   scale='width', inner='quartile',
                   linewidth=1, cut=0, dodge=True)
    if result_dir is not None:
        result_dir = os.path.join(result_dir, 'jsd_vs_max_attn_by_class')
    _common_jsd_vs_max_attn(result_dir)


def _common_jsd_vs_max_attn(result_dir):
    plt.ylabel('Max Attention')
    plt.xlabel('Max JSD within epsilon')
    plt.tight_layout()
    if result_dir is not None:
        for file_ending in file_endings:
            plt.savefig(result_dir + file_ending)
    plt.close()


def plot_jsd_vs_tvd(jsd_tvd_df, result_dir):
    seeds = jsd_tvd_df[jsd_tvd_df['model-name'].str.contains('seed-')]
    plt.scatter(seeds['JSD'], seeds['TVD'], marker='^', color='green', label='Seeds')

    uniform = jsd_tvd_df[jsd_tvd_df['model-name'].str.contains('uniform-')]
    plt.scatter(uniform['JSD'], uniform['TVD'], marker='s', color='cyan', label='Uniform')

    max_tvd = max(jsd_tvd_df['TVD'].max(), jsd_tvd_df['cosine'].max())
    plt.xlim(0.0, 0.70)
    plt.ylim(0.0, max(0.10, max_tvd * 1.05))

    plt.xlabel('Attentions JSD')
    plt.ylabel('Predictions TVD')
    plt.legend()
    plt.tight_layout()
    if result_dir is not None:
        for file_ending in file_endings:
            plt.savefig(os.path.join(result_dir, 'jsd_vs_tvd' + file_ending))
    plt.close()


def plot_jsd_vs_tvd_axis(jsd_tvd_df, axis):
    seeds = jsd_tvd_df[jsd_tvd_df['model-name'].str.contains('seed-')]
    axis.scatter(seeds['JSD'], seeds['TVD'], marker='^', color='green', label='random seeds')

    uniform = jsd_tvd_df[jsd_tvd_df['model-name'].str.contains('uniform-')]
    axis.scatter(uniform['JSD'], uniform['TVD'], marker='s', color='cyan', label='uniform weights')

    max_tvd = jsd_tvd_df['TVD'].max()
    return max_tvd
