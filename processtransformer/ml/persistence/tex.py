import typing

import pandas as pd
from processtransformer.ml.analysis import event_log_analysis


def round_to_precision(stats: typing.Dict[str, typing.Dict], num_precision: int) -> typing.Dict[str, typing.Dict]:
    assert num_precision > 0
    for event_log in stats.keys():
        for step in stats[event_log].keys():
            value = stats[event_log][step]
            if isinstance(value, float):
                stats[event_log][step] = round(value, num_precision)
    return stats


def export_event_log_statistics_to_tex_file(stats: typing.Dict[str, typing.Dict], target_file: str,
                                            orientation: str = 'columns', clear_names: bool = True,
                                            num_precision: int = 2) -> None:
    assert orientation in {'row', 'columns'}

    stats = round_to_precision(stats, num_precision)
    table_df = pd.DataFrame.from_dict(stats, orient='index')

    if clear_names:
        table_df = table_df.rename(columns=event_log_analysis.EventLogDescriptor.get_column_names())

    # Convert to string for export
    table_df = table_df.astype(str)

    if orientation == 'row':
        table_df = table_df.transpose()

    print(table_df)

    if target_file is not None:
        with open(target_file, 'w', encoding='utf8') as f:
            f.write(table_df.to_latex())
