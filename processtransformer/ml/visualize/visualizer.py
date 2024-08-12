import os
import typing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self, theme=None, export_dir: typing.Optional[str] = None, silent_mode=False):
        if theme is None:
            sns.set_theme()
        else:
            sns.set_theme(theme)
        self.export_dir = export_dir
        self.silent_mode = silent_mode

    def _save_plot(self, title: str):
        if self.export_dir is not None:
            os.makedirs(self.export_dir, exist_ok=True)
            plt.savefig(os.path.join(self.export_dir, title.replace('/', '-')))

    def plot_prefix_analysis_strategies_on_dataset_on_approach(self, data_df: pd.DataFrame,
                                                               prefix_stats_df: pd.DataFrame,
                                                               dataset: str, approach: str):
        assert len(data_df) > 0 and len(prefix_stats_df) > 0
        assert set(data_df.columns).issubset({'dataset', 'approach', 'strategy', 'repetition', 'fold', 'prefix_length',
                                              'metric', 'value', 'count'})
        assert set(prefix_stats_df.columns).issubset({'dataset', 'approach', 'strategy', 'repetition', 'fold',
                                                      'prefix_length', 'value', 'count'})

        n_strategies = len(data_df['strategy'].unique())
        df = data_df.merge(prefix_stats_df, how='inner', on=['dataset', 'approach', 'strategy', 'prefix_length'])

        plt.figure()
        ax = sns.lineplot(data=df, x='prefix_length', y='value', hue='strategy')
        ax = sns.scatterplot(data=df, x='prefix_length', y='value', hue='strategy', size='count', legend='brief')

        h, l = ax.get_legend_handles_labels()
        plt.legend(h[n_strategies:], l[n_strategies:])
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        self._save_plot(f'{dataset}_{approach}_prefix_analysis')
        if self.silent_mode is False:
            plt.show()

    def plot_activity_analysis_strategies_on_dataset_on_approach(self, data_df: pd.DataFrame,
                                                                 activity_stats_df: pd.DataFrame,
                                                                 dataset: str, approach: str):
        assert len(data_df) > 0 and len(activity_stats_df) > 0
        assert set(data_df.columns).issubset({'dataset', 'approach', 'strategy', 'repetition', 'fold', 'activity',
                                              'value', 'metric', 'count'})
        assert set(activity_stats_df.columns).issubset({'dataset', 'approach', 'strategy', 'repetition', 'fold',
                                                        'activity', 'count'})

        # todo haengt vom aggregate level ab hier run
        df = data_df.merge(activity_stats_df, how='inner', on=['dataset', 'approach', 'strategy', 'activity'])
        f, axes = plt.subplots(2, 1, sharex='True')

        # Make density plot along x-axis without legend
        sns.barplot(data=df, x='activity', y='count', ax=axes[0], color='b')
        sns.barplot(data=df, x='activity', y='value', hue='strategy', ax=axes[1])

        handles, labels = axes[1].get_legend_handles_labels()

        _ = plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xticks(rotation=90)
        plt.tight_layout()
        self._save_plot(f'{dataset}_{approach}_activity_analysis')
        if self.silent_mode is False:
            plt.show()

    def plot_total_analysis_strategies_on_dataset_on_approach(self, data_df: pd.DataFrame, dataset: str, approach: str):
        assert len(data_df) > 0, f'Empty DataFrame. There is nothing to plot.'
        assert set(data_df.columns).issubset({'dataset', 'approach', 'strategy', 'repetition', 'fold', 'value',
                                              'metric'})
        plt.figure()
        sns.catplot(data=data_df, x='strategy', y='value', kind='box')
        sns.stripplot(data=data_df, x='strategy', y='value', dodge=True, color='.25')
        plt.xticks(rotation=90)
        plt.tight_layout()
        self._save_plot(f'{dataset}_{approach}_total_analysis')
        if self.silent_mode is False:
            plt.show()

    def plot_correlations(self, data_df: pd.DataFrame):
        assert len(data_df) > 0, f'Empty DataFrame. There is nothing to plot.'
        assert set(data_df.columns).issubset({'dataset', 'approach', 'strategy', 'repetition', 'fold', 'x_property',
                                              'y_property'})
        assert 'x_property' in data_df.columns
        assert 'y_property' in data_df.columns

        plt.figure()
        ax = sns.scatterplot(data=data_df, x='x_property', y='y_property', hue='approach', style='dataset')
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()

        self._save_plot(f'Correlation_analysis')
        if self.silent_mode is False:
            plt.show()
