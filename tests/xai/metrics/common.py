

import time
import typing

import pandas as pd


def _generate_df_from_events(traces: typing.List[typing.List[str]]):
    base_time = time.time()
    data = []
    for case_id, trace in enumerate(traces):
        for i, event in enumerate(trace):
            entry = (event, 'p123', pd.to_datetime((base_time + 1e6) * 1e9, unit='ns'), str(case_id + 1))
            data.append(entry)

    return pd.DataFrame(
        columns=['concept:name', 'org:group', 'time:timestamp', 'case:concept:name'],
        data=data,
    )
