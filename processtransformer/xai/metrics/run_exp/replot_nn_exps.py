

import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

from processtransformer.xai.metrics.attn_masking import plot_hist_combined, plot_hist_by_class
from processtransformer.xai.metrics.attn_uniform_weights import plot_jsd_vs_tvd, plot_jsd_vs_max_attn, \
    plot_jsd_vs_tvd_axis, plot_jsd_vs_max_attn_by_class

plt.rcParams["figure.figsize"] = (5.4 * 0.9, 4)
plt.rcParams["svg.fonttype"] = 'none'
plt.rcParams["font.size"] = 11.0
plt.rcParams["axes.unicode_minus"] = False  # otherwise LaTeX has issues rendering the minus sign

source_arr = [
    ['training_bpic12_GPU', 'training_bpic13_closed_problems_GPU', 'training_complex_model'],
    ['training_bpic12_o_GPU', 'training_bpic13_incidents_GPU', 'training_looped_and'],
    ['training_bpic12_w_GPU', 'training_helpdesk_GPU', 'training_long_dist_dep'],
    ['training_bpic12_wc_GPU', 'training_sepsis_GPU', 'training_sequence'],
]
dataset_names = [
    ['BPIC12', 'BPIC13-CP', 'Complex Model'],
    ['BPIC12-O', 'BPIC13-I', 'Looped And'],
    ['BPIC12-W', 'Helpdesk', 'LDD'],
    ['BPIC12-WC', 'Sepsis', 'Sequence'],
]
width = 5.4 * 0.9
rows = 4
cols = 3

for i, row in enumerate(source_arr):
    source_arr[i] = [os.path.join('training', dataset, 'metrics') for dataset in row]


def plot_attn_feature_importance_table():
    fig, axes = plt.subplots(rows, cols, figsize=(width, width / cols * rows / 1.5), sharex=True, sharey=True)

    # fig.suptitle('Horizontally stacked subplots')
    dfs = []
    for i, row in enumerate(source_arr):
        for j, dataset in enumerate(row):
            attn_fi_df = pd.read_csv(os.path.join(dataset, 'attn_feature_importance.csv'))
            dataset_name = dataset_names[i][j]
            corr_column = attn_fi_df['kendalltau-correlation']
            quantiles = [0.00, 0.25, 0.50, 0.75, 1.00]
            quantiles_df = corr_column.quantile(quantiles)
            quantiles_df = pd.DataFrame(data=[[dataset_name] + quantiles_df.values.tolist()],
                                        columns=['Dataset'] + [f'p-{q:.2f}' for q in quantiles])
            dfs.append(quantiles_df)

            ax = axes[i][j]
            sns.boxplot(x=corr_column, ax=ax, flierprops={"marker": "x"},
                        saturation=0.5, linewidth=1.0)
            ax.set_xlabel('')
            ax.set_ylabel('')

            ax.set_title(dataset_name)
            ax.xaxis.set_major_locator(MultipleLocator(0.50))
            ax.xaxis.set_minor_locator(MultipleLocator(0.25))
            ax.tick_params('x', labelrotation=45)

            # Labels only on left side/bottom
            if i == rows - 1:
                ax.set_xlabel(r'\$\tau^K\$ correlation')

    plt.xlim(-1.0, 1.0)
    plt.tight_layout()

    fig.savefig('attn_fi.svg')
    fig.show()

    dfs = reorder_dfs(dfs)
    to_latex(dfs, 'attn_fi_table.tex')


def jsd_vs_tvd_subplot():
    fig, axes = plt.subplots(rows, cols, figsize=(width, width / cols * rows), sharex=True, sharey=True)

    # fig.suptitle('Horizontally stacked subplots')
    dfs = []
    for i, row in enumerate(source_arr):
        for j, dataset in enumerate(row):
            jsd_tvd_df = pd.read_csv(os.path.join(dataset, 'attn_uniform_weights_jsd_tvd.csv'))
            dataset_name = dataset_names[i][j]
            mean_seed = jsd_tvd_df[jsd_tvd_df['model-name'].str.contains('seed')].mean()
            mean_uni = jsd_tvd_df[jsd_tvd_df['model-name'].str.contains('uniform')].mean()
            dfs.append(pd.DataFrame(data=[[dataset_name, mean_seed['TVD'], mean_seed['JSD'],
                                           mean_uni['TVD'], mean_uni['JSD']]],
                                    columns=['Dataset', r'TVD (seed, $\downarrow$)', r'JSD (seed, $\uparrow$)',
                                             r'TVD (uni, $\downarrow$)', r'JSD (uni, $\uparrow$)']))

            ax = axes[i][j]
            ax.set_title(dataset_name)
            ax.xaxis.set_major_locator(MultipleLocator(0.20))
            ax.xaxis.set_minor_locator(MultipleLocator(0.05))
            ax.yaxis.set_major_locator(MultipleLocator(0.20))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))

            # Labels only on left side/bottom
            if i == rows - 1:
                ax.set_xlabel('Attentions JSD')
            if j == 0:
                ax.set_ylabel('Predictions TVD')
            ax.plot([0.0, 0.7], [0.1, 0.1], color='black')
            plot_jsd_vs_tvd_axis(jsd_tvd_df, ax)

    plt.xlim(0.0, 0.70)
    plt.ylim(0.0, 0.40)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    axes[rows - 1][1].legend(loc='upper center',
                             bbox_to_anchor=(0.5, -0.5), fancybox=False, shadow=False, ncol=2)

    fig.savefig('jsd_vs_tvd.svg')
    fig.show()

    dfs = reorder_dfs(dfs)
    to_latex(dfs, 'tvd_jsd_table.tex')


def to_latex(dfs, file_name):
    dfs.to_latex(file_name, escape=False, float_format="%.3f", index=False)


def reorder_dfs(dfs):
    new_dfs = [None] * rows * cols
    step = 0
    for i in range(rows):
        for j in range(cols):
            new_dfs[i * cols + j] = dfs[step]
            step += cols
            if step % (rows * cols) < step:
                step = (step % (rows * cols)) + 1
    dfs = pd.concat(new_dfs)
    return dfs


def violins():
    fig, axes = plt.subplots(rows, cols, figsize=(width, width / cols * rows * 1.2),
                             sharex=True, sharey=True)

    # fig.suptitle('Horizontally stacked subplots')
    dfs = []
    for i, row in enumerate(source_arr):
        for j, dataset in enumerate(row):
            dataset_name = dataset_names[i][j]
            ax = axes[i][j]
            attn_df = pd.read_csv(os.path.join(dataset, 'attn_uniform_weights_attn.csv'))

            plot_jsd_vs_max_attn(attn_df, None, ax)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim([0.0, 0.7])

            ax.set_title(dataset_name)
            ax.xaxis.set_major_locator(MultipleLocator(0.35))
            ax.xaxis.set_minor_locator(MultipleLocator(0.05))

            ax.tick_params('x', labelrotation=45)

            # Labels only on left side/bottom
            if i == rows - 1:
                ax.set_xlabel('JSD (seeds+uni\nvs. original)')
            if j == 0:
                ax.set_ylabel('Max Attention')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.2)
    fig.show()
    fig.savefig('maxattn-vs-jsd.svg')


def combine_hists():
    fig, axes = plt.subplots(rows, cols, figsize=(width, width / cols * rows * 1.2),
                             sharex=True, sharey=True)

    # fig.suptitle('Horizontally stacked subplots')
    dfs = []
    for i, row in enumerate(source_arr):
        for j, dataset in enumerate(row):
            dataset_name = dataset_names[i][j]
            ax = axes[i][j]
            df = pd.read_csv(os.path.join(dataset, 'attn_masking.csv'))

            quantiles = [0.00, 0.25, 0.50, 0.75, 1.00]
            quantiles_df = df['TVD'].quantile(quantiles)
            quantiles_df = pd.DataFrame(data=[[dataset_name] + quantiles_df.values.tolist()],
                                        columns=['Dataset'] + [f'p-{q:.2f}' for q in quantiles])
            dfs.append(quantiles_df)

            plot_hist_combined(df, None, ax)

            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim([0.0, 1.0])
            ax.set_xlim([0.0, 1.0])
            ax.set_title(dataset_name)

            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.tick_params('x', labelrotation=45)

            # Labels only on left side/bottom
            if i == rows - 1:
                ax.set_xlabel('Predictions TVD')
            if j == 0:
                ax.set_ylabel('Probability')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)
    fig.show()
    fig.savefig('tvd-hists.svg')

    dfs = reorder_dfs(dfs)
    to_latex(dfs, 'tvd_hists.tex')


def fix_all_attn_uni_weights():
    for row in source_arr:
        for data in row:
            source_dir = data
            result_dir = os.path.join(source_dir, 'attn_uniform_weights')
            attn_df = pd.read_csv(os.path.join(source_dir, 'attn_uniform_weights_attn.csv'))
            jsd_tvd_df = pd.read_csv(os.path.join(source_dir, 'attn_uniform_weights_jsd_tvd.csv'))

            plot_jsd_vs_max_attn(attn_df, result_dir)
            plot_jsd_vs_max_attn_by_class(attn_df, result_dir)

            plot_jsd_vs_tvd(jsd_tvd_df, result_dir)


def fix_all_attn_masking():
    for row in source_arr:
        for data in row:
            source_dir = data
            result_dir = os.path.join(source_dir, 'attn_masking')
            attn_mask_df = pd.read_csv(os.path.join(source_dir, 'attn_masking.csv'))

            plot_hist_by_class(attn_mask_df, result_dir)
            plot_hist_combined(attn_mask_df, result_dir)


if __name__ == '__main__':
    violins()
