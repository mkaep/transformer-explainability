
import dataclasses
import os.path
import typing
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

plt.rcParams["figure.figsize"] = (5.4, 2.5)
plt.rcParams["svg.fonttype"] = 'none'
plt.rcParams["font.size"] = 10.0
plt.rcParams["axes.unicode_minus"] = False  # otherwise LaTeX has issues rendering the minus sign


@dataclasses.dataclass
class Metric:
    mean: float
    std: float
    df: pd.DataFrame


@dataclasses.dataclass
class DatasetResults:
    xai_name: str  # name of XAI-approach
    co01: Metric  # kendall-tau
    co02_precision: Metric  # precision
    co02_recall: Metric  # recall
    co02_f_score: Metric  # f-score
    co03: Metric  # relaxed_consistency
    co04: Metric  # threshold_less_sim
    co05: Metric  # diff_sim
    co07_num_rules: Metric  # num_rules
    co07_avg_right_side_length: Metric  # avg_right_side_length

    def to_df(self):
        return pd.DataFrame(data=[[self.xai_name,
                                   f'{self.co01.mean:.2f}',
                                   f'{self.co02_precision.mean:.2f}',
                                   f'{self.co02_recall.mean:.2f}',
                                   f'{self.co02_f_score.mean:.2f}',
                                   f'{self.co03.mean:.2f}',
                                   f'{self.co04.mean:.2f}',
                                   f'{self.co05.mean:.2f}',
                                   f'{self.co07_num_rules.mean:.2f}',
                                   f'{self.co07_avg_right_side_length.mean:.2f}',
                                   ],
                                  ['',
                                   f'$\pm${self.co01.std:.2f}',
                                   f'$\pm${self.co02_precision.std:.2f}',
                                   f'$\pm${self.co02_recall.std:.2f}',
                                   f'$\pm${self.co02_f_score.std:.2f}',
                                   f'$\pm${self.co03.std:.2f}',
                                   f'$\pm${self.co04.std:.2f}',
                                   f'$\pm${self.co05.std:.2f}',
                                   f'$\pm${self.co07_num_rules.std:.2f}',
                                   f'$\pm${self.co07_avg_right_side_length.std:.2f}',
                                   ]],
                            columns=['XAI approach', 'Co01',
                                     'Co02 p.', 'Co02 r.', 'Co02 f.',
                                     'Co03', 'Co04', 'Co05',
                                     'Co07 $|\mathcal{R}|$', 'Co07 $S_\mu^R$'])


bpic12 = {
    'Attn-Expl': r'explaining/bpic12/bpic12-attn_expl_xai',
    'Attn-restr DFG': r'explaining/bpic12/bpic12-attn_restr_dfg_xai',
    'Backward': r'explaining/bpic12/bpic12-trace_backward_xai',
    'Mask-Variety': r'explaining/bpic12/bpic12-trace_mask_variety_xai',
    'Mask-Walk': r'explaining/bpic12/bpic12-trace_mask_walk_xai',
    'Mask': r'explaining/bpic12/bpic12-trace_mask_xai',
    'Variety': r'explaining/bpic12/bpic12-trace_variety_xai',
}

bpic12_o = {
    'Attn-Expl': r'explaining/bpic12_o/bpic12_o-attn_expl_xai',
    'Attn-restr DFG': r'explaining/bpic12_o/bpic12_o-attn_restr_dfg_xai',
    'Backward': r'explaining/bpic12_o/bpic12_o-trace_backward_xai',
    'Mask-Variety': r'explaining/bpic12_o/bpic12_o-trace_mask_variety_xai',
    'Mask-Walk': r'explaining/bpic12_o/bpic12_o-trace_mask_walk_xai',
    'Mask': r'explaining/bpic12_o/bpic12_o-trace_mask_xai',
    'Variety': r'explaining/bpic12_o/bpic12_o-trace_variety_xai',
}

bpic12_w = {
    'Attn-Expl': r'explaining/bpic12_w/bpic12_w-attn_expl_xai',
    'Attn-restr DFG': r'explaining/bpic12_w/bpic12_w-attn_restr_dfg_xai',
    'Backward': r'explaining/bpic12_w/bpic12_w-trace_backward_xai',
    'Mask-Variety': r'explaining/bpic12_w/bpic12_w-trace_mask_variety_xai',
    'Mask-Walk': r'explaining/bpic12_w/bpic12_w-trace_mask_walk_xai',
    'Mask': r'explaining/bpic12_w/bpic12_w-trace_mask_xai',
    'Variety': r'explaining/bpic12_w/bpic12_w-trace_variety_xai',
}

bpic12_wc = {
    'Attn-Expl': r'explaining/bpic12_wc/bpic12_wc-attn_expl_xai',
    'Attn-restr DFG': r'explaining/bpic12_wc/bpic12_wc-attn_restr_dfg_xai',
    'Backward': r'explaining/bpic12_wc/bpic12_wc-trace_backward_xai',
    'Mask-Variety': r'explaining/bpic12_wc/bpic12_wc-trace_mask_variety_xai',
    'Mask-Walk': r'explaining/bpic12_wc/bpic12_wc-trace_mask_walk_xai',
    'Mask': r'explaining/bpic12_wc/bpic12_wc-trace_mask_xai',
    'Variety': r'explaining/bpic12_wc/bpic12_wc-trace_variety_xai',
}

bpic13_cp = {
    'Attn-Expl': r'explaining/bpic13_closed_problems/bpic13_closed_problems-attn_expl_xai',
    'Attn-restr DFG': r'explaining/bpic13_closed_problems/bpic13_closed_problems-attn_restr_dfg_xai',
    'Backward': r'explaining/bpic13_closed_problems/bpic13_closed_problems-trace_backward_xai',
    'Mask-Variety': r'explaining/bpic13_closed_problems/bpic13_closed_problems-trace_mask_variety_xai',
    'Mask-Walk': r'explaining/bpic13_closed_problems/bpic13_closed_problems-trace_mask_walk_xai',
    'Mask': r'explaining/bpic13_closed_problems/bpic13_closed_problems-trace_mask_xai',
    'Variety': r'explaining/bpic13_closed_problems/bpic13_closed_problems-trace_variety_xai',
}

bpic13_i = {
    'Attn-Expl': r'explaining/bpic13_incidents/bpic13_incidents-attn_expl_xai',
    'Attn-restr DFG': r'explaining/bpic13_incidents/bpic13_incidents-attn_restr_dfg_xai',
    'Backward': r'explaining/bpic13_incidents/bpic13_incidents-trace_backward_xai',
    'Mask-Variety': r'explaining/bpic13_incidents/bpic13_incidents-trace_mask_variety_xai',
    'Mask-Walk': r'explaining/bpic13_incidents/bpic13_incidents-trace_mask_walk_xai',
    'Mask': r'explaining/bpic13_incidents/bpic13_incidents-trace_mask_xai',
    'Variety': r'explaining/bpic13_incidents/bpic13_incidents-trace_variety_xai',
}

sepsis = {
    'Attn-Expl': r'explaining/sepsis/sepsis-attn_expl_xai',
    'Attn-restr DFG': r'explaining/sepsis/sepsis-attn_restr_dfg_xai',
    'Backward': r'explaining/sepsis/sepsis-trace_backward_xai',
    'Mask-Variety': r'explaining/sepsis/sepsis-trace_mask_variety_xai',
    'Mask-Walk': r'explaining/sepsis/sepsis-trace_mask_walk_xai',
    'Mask': r'explaining/sepsis/sepsis-trace_mask_xai',
    'Variety': r'explaining/sepsis/sepsis-trace_variety_xai',
}

helpdesk = {
    'Attn-Expl': r'explaining/helpdesk/helpdesk-attn_expl_xai',
    'Attn-restr DFG': r'explaining/helpdesk/helpdesk-attn_restr_dfg_xai',
    'Backward': r'explaining/helpdesk/helpdesk-trace_backward_xai',
    'Mask-Variety': r'explaining/helpdesk/helpdesk-trace_mask_variety_xai',
    'Mask-Walk': r'explaining/helpdesk/helpdesk-trace_mask_walk_xai',
    'Mask': r'explaining/helpdesk/helpdesk-trace_mask_xai',
    'Variety': r'explaining/helpdesk/helpdesk-trace_variety_xai',
}

complex_model = {
    'Attn-Expl': r'explaining/14_complex_models/complex_model_001/complex_model_001-attn_expl_xai',
    'Attn-restr DFG': r'explaining/14_complex_models/complex_model_001/complex_model_001-attn_restr_dfg_xai',
    'Backward': r'explaining/14_complex_models/complex_model_001/complex_model_001-trace_backward_xai',
    'Mask-Variety': r'explaining/14_complex_models/complex_model_001/complex_model_001-trace_mask_variety_xai',
    'Mask-Walk': r'explaining/14_complex_models/complex_model_001/complex_model_001-trace_mask_walk_xai',
    'Mask': r'explaining/14_complex_models/complex_model_001/complex_model_001-trace_mask_xai',
    'Variety': r'explaining/14_complex_models/complex_model_001/complex_model_001-trace_variety_xai',
}

long_distance_dep = {
    'Attn-Expl': r'explaining/long_dist_dep2/long_dist_dep2-attn_expl_xai',
    'Attn-restr DFG': r'explaining/long_dist_dep2/long_dist_dep2-attn_restr_dfg_xai',
    'Backward': r'explaining/long_dist_dep2/long_dist_dep2-trace_backward_xai',
    'Mask-Variety': r'explaining/long_dist_dep2/long_dist_dep2-trace_mask_variety_xai',
    'Mask-Walk': r'explaining/long_dist_dep2/long_dist_dep2-trace_mask_walk_xai',
    'Mask': r'explaining/long_dist_dep2/long_dist_dep2-trace_mask_xai',
    'Variety': r'explaining/long_dist_dep2/long_dist_dep2-trace_variety_xai',
}

looped_and = {
    'Attn-Expl': r'explaining/15_looped_AND/looped_and-attn_expl_xai',
    'Attn-restr DFG': r'explaining/15_looped_AND/looped_and-attn_restr_dfg_xai',
    'Backward': r'explaining/15_looped_AND/looped_and-trace_backward_xai',
    'Mask-Variety': r'explaining/15_looped_AND/looped_and-trace_mask_variety_xai',
    'Mask-Walk': r'explaining/15_looped_AND/looped_and-trace_mask_walk_xai',
    'Mask': r'explaining/15_looped_AND/looped_and-trace_mask_xai',
    'Variety': r'explaining/15_looped_AND/looped_and-trace_variety_xai',
}
sequence = {
    'Attn-Expl': r'explaining/sequence/sequence-attn_expl_xai',
    'Attn-restr DFG': r'explaining/sequence/sequence-attn_restr_dfg_xai',
    'Backward': r'explaining/sequence/sequence-trace_backward_xai',
    'Mask-Variety': r'explaining/sequence/sequence-trace_mask_variety_xai',
    'Mask-Walk': r'explaining/sequence/sequence-trace_mask_walk_xai',
    'Mask': r'explaining/sequence/sequence-trace_mask_xai',
    'Variety': r'explaining/sequence/sequence-trace_variety_xai',
}


def get_result_for_xai(xai_name: str, df_dir: str):
    co01_metric = get_co01(df_dir)

    co02_f_score, co02_precision, co02_recall = get_co02(df_dir)

    co03_metric = get_co03(df_dir)

    co04_metric = get_co04(df_dir)

    co05_metric = get_co05(df_dir)

    co07_avg_right_side, co07_num_rules = get_co07(df_dir)

    return DatasetResults(xai_name, co01_metric, co02_precision, co02_recall, co02_f_score,
                          co03_metric, co04_metric, co05_metric, co07_num_rules, co07_avg_right_side)


def get_co07(df_dir):
    df = pd.read_csv(os.path.join(df_dir, 'compactness_co07.csv'))
    df_num_rules = df['num_rules']
    co07_num_rules = Metric(df_num_rules.mean(), df_num_rules.std(), df_num_rules)
    df_avg_right_side_length = df['avg_right_side_length']
    co07_avg_right_side = Metric(df_avg_right_side_length.mean(), df_avg_right_side_length.std(),
                                 df_avg_right_side_length)
    print('Co07-----------------------------------------')
    print(df.mean())
    return co07_avg_right_side, co07_num_rules


def get_co05(df_dir):
    df = pd.read_csv(os.path.join(df_dir, 'contrastivity_co05.csv'))
    df_diff_sim = df['diff_sim']
    co05_metric = Metric(df_diff_sim.mean(), df_diff_sim.std(), df_diff_sim)
    df = df[df['count'] > 0]
    print('Co05-----------------------------------------')
    print(df.mean())
    return co05_metric


def get_co04(df_dir):
    df = pd.read_csv(os.path.join(df_dir, 'continuity_co04.csv'))
    df_threshold_less_sim = df['threshold_less_sim']
    co04_metric = Metric(df_threshold_less_sim.mean(), df_threshold_less_sim.std(), df_threshold_less_sim)
    df = df[df['similarity_count'] > 0]
    print('Co04-----------------------------------------')
    print(df.mean())
    return co04_metric


def get_co03(df_dir):
    df = pd.read_csv(os.path.join(df_dir, 'consistency_co03_complete.csv'))
    df_relaxed_consistency = df['relaxed_consistency']
    co03_metric = Metric(df_relaxed_consistency.mean(), df_relaxed_consistency.std(), df_relaxed_consistency)
    print('Co03-----------------------------------------')
    print(df.mean())
    return co03_metric


def get_co02(df_dir):
    df = pd.read_csv(os.path.join(df_dir, 'completeness_co02_combined.csv'))
    df_precision = df['precision_weighted']
    co02_precision = Metric(df_precision.mean(), df_precision.std(), df_precision)
    df_recall = df['recall_weighted']
    co02_recall = Metric(df_recall.mean(), df_recall.std(), df_recall)
    df_f_score = df['f_weighted']
    co02_f_score = Metric(df_f_score.mean(), df_f_score.std(), df_f_score)
    print('Co02-----------------------------------------')
    print(df.mean())
    return co02_f_score, co02_precision, co02_recall


def get_co01(df_dir):
    df = pd.read_csv(os.path.join(df_dir, 'correctness_co01.csv'))
    df_kendalltau = df['kendalltau-corr']
    co01_metric = Metric(df_kendalltau.mean(), df_kendalltau.std(), df_kendalltau)
    print('Co01-----------------------------------------')
    print(df.mean())
    return co01_metric


def process_dataset(xai_paths):
    results = []
    for xai_name, xai_path in xai_paths.items():
        results.append(get_result_for_xai(xai_name, os.path.join(xai_path, 'metrics')))

    dfs = pd.concat([metric.to_df() for metric in results])
    dfs = dfs.set_index('XAI approach')

    # Choose max value for Co01 through Co05
    make_columns_bold(dfs.columns[:-2], dfs, np.nanmax)
    # Choose min value for Co07
    make_columns_bold(dfs.columns[-2:], dfs, np.nanmin)

    res_dir = os.path.dirname(list(xai_paths.values())[0])
    dfs.to_latex(os.path.join(res_dir, 'metric_df_mean.tex'), escape=False)
    pass


def make_columns_bold(columns, dfs, eval_func):
    for col_name in columns:
        entries = [float(entry) for entry in dfs[col_name].to_list() if not entry.startswith('$')]
        eval_val = f'{eval_func(entries):.2f}'
        for i in range(len(dfs[col_name])):
            if dfs[col_name][i] == eval_val:
                dfs[col_name][i] = '\\textbf{' + eval_val + '}'


def main():
    process_dataset(bpic12)
    process_dataset(bpic12_o)
    process_dataset(bpic12_w)
    process_dataset(bpic12_wc)

    process_dataset(bpic13_cp)
    process_dataset(bpic13_i)
    process_dataset(sepsis)
    process_dataset(helpdesk)

    process_dataset(complex_model)
    process_dataset(long_distance_dep)
    process_dataset(looped_and)
    process_dataset(sequence)


@dataclasses.dataclass
class MetricInfo:
    best_metric_func: Callable
    metric_func: Callable
    metric_name: str
    bounds: typing.Tuple[int, int] = None


def process_per_metric():
    datasets = {'BPIC12': bpic12, 'BPIC12-O': bpic12_o, 'BPIC12-W': bpic12_w, 'BPIC12-WC': bpic12_wc,
                'BPIC13-CP': bpic13_cp, 'BPIC13-i': bpic13_i, 'Sepsis': sepsis, 'Helpdesk': helpdesk,
                'Complex': complex_model, 'LDD': long_distance_dep, 'Loop AND': looped_and, 'Sequence': sequence}
    # Some yaxis major ticks and y-limits were set manually.
    metric_infos = [
        (MetricInfo(np.nanmax, get_co01, 'Co01', bounds=(-1, 1))),
        (MetricInfo(np.nanmax, lambda x: get_co02(x)[0], 'Co02f', bounds=(0, 1))),  # f score
        (MetricInfo(np.nanmax, lambda x: get_co02(x)[1], 'Co02p', bounds=(0, 1))),  # precision
        (MetricInfo(np.nanmax, lambda x: get_co02(x)[2], 'Co02r', bounds=(0, 1))),  # recall
        (MetricInfo(np.nanmax, get_co03, 'Co03', bounds=(0, 1))),
        (MetricInfo(np.nanmax, get_co04, 'Co04', bounds=(0, 1))),
        (MetricInfo(np.nanmax, get_co05, 'Co05', bounds=(0, 1))),
        (MetricInfo(np.nanmin, lambda x: get_co07(x)[0], 'Co07right')),  # average right side length
        (MetricInfo(np.nanmin, lambda x: get_co07(x)[1], 'Co07rules')),  # num rules
    ]

    section_refs = {
        'Attn-Expl': r'\cref{sec:xai:approaches:attn-explr-xai}',
        'Attn-restr DFG': r'\cref{sec:xai:approaches:attn-restr-dfg}',
        'Backward': r'\cref{sec:xai:approaches:trace-mod:trace-backward}',
        'Mask-Variety': r'\cref{sec:xai:approaches:trace-mod:mask-variety}',
        'Mask-Walk': r'\cref{sec:xai:approaches:trace-mod:mask-walk}',
        'Mask': r'\cref{sec:xai:approaches:trace-mod:mask}',
        'Variety': r'\cref{sec:xai:approaches:trace-mod:variety}',
    }
    xai_name_mapping = {
        'Mask': ('Mask XAI [l]', 1),
        'Mask-Walk': ('MaskWalk XAI [l]', 2),
        'Variety': ('Variety XAI [l]', 3),
        'Mask-Variety': ('MaskVariety XAI [l]', 4),
        'Backward': ('Backward XAI [g]', 5),
        'Attn-restr DFG': ('AttnRestr. DFG [g]', 6),
        'Attn-Expl': ('AttnExpl. XAI [g]', 7),
    }

    explainer_names = list(datasets['BPIC12'].keys())
    explainer_names.sort(key=lambda x: xai_name_mapping[x][1])

    for metric_info in metric_infos:
        index = []
        new_index = []  # for later (renaming)
        for xai_name in explainer_names:
            index.append(xai_name)
            index.append(xai_name + 'std_dev')
            new_index.append(xai_name_mapping[xai_name][0])
            new_index.append(section_refs[xai_name])

        dfs_for_boxplots = {xai: [] for xai in explainer_names}
        overall_df = pd.DataFrame(index=index, columns=(datasets.keys()))
        for dataset_name, dataset in datasets.items():
            for explainer, df_dir in dataset.items():
                df_dir = os.path.join(df_dir, 'metrics')
                metric = metric_info.metric_func(df_dir)
                overall_df[dataset_name][explainer] = f'{metric.mean:.2f}'
                overall_df[dataset_name][explainer + 'std_dev'] = f'$\pm${metric.std:.2f}'
                dfs_for_boxplots[explainer].append(metric.df)
        dfs_for_boxplots = {xai: pd.concat(lst) for xai, lst in dfs_for_boxplots.items()}

        dfs_for_boxplots = pd.concat([pd.DataFrame(data=[[v, xai_name_mapping[name][0]] for v in lst.tolist()],
                                                   columns=['values', 'approach'])
                                      for name, lst in dfs_for_boxplots.items()])
        sns.boxplot(dfs_for_boxplots, x='approach', y='values', orient='v',
                    flierprops={"marker": "x"}, saturation=1.0, linewidth=1.0, color='tab:blue')
        plt.tick_params('x', labelrotation=30)
        # if metric_info.bounds is not None:
        #     plt.ylim(np.asarray(metric_info.bounds) * 1.05)
        # plt.ylim(-0.55, 1.05)  # for co01 - comment out previous calls to ylim!
        # plt.gca().yaxis.set_major_locator(MultipleLocator(2))  # set y-axis ticks manually
        plt.xlabel('')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join('misc', 'co-metrics-results', f'metric-{metric_info.metric_name}.svg'))
        plt.show()

        # Highlight best metric values
        make_columns_bold(overall_df.columns, overall_df, metric_info.best_metric_func)
        # Change index to XAI approaches
        overall_df['XAI approach'] = new_index
        overall_df = overall_df.set_index('XAI approach')
        # Rotate column names (dataset names) by X degrees (LaTeX command: \newcommand*\rot{\rotatebox{45}})
        overall_df.columns = [r'\rot{' + col + r'}' for col in overall_df.columns]
        overall_df.to_latex(os.path.join('misc', 'co-metrics-results',
                                         f'metric-{metric_info.metric_name}.tex'),
                            escape=False, column_format='lrrrrrrrrrrrr')


if __name__ == '__main__':
    process_per_metric()
