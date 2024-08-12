
import argparse
import os

import pandas as pd
import pm4py

parser = argparse.ArgumentParser(description="Misc - Traces to Histogram.")
parser.add_argument("--xes_input", type=str, required=True, help='Path of the input XES file.')
parser.add_argument("--xes_output", type=str, required=True,
                    help='Path of the output XES file. Can be identical to the input.')
args = parser.parse_args()


class _InternalState:
    def __init__(self, original: pd.DataFrame):
        self.indices = []
        self.original = original
        self.resulting_log = pd.DataFrame()
        self.case_id = 1

    def add_new_case_trace(self):
        if len(self.indices) < 1:
            return

        unique_case_trace = self.original.iloc[self.indices, :].copy()
        unique_case_trace['case:concept:name'] = str(self.case_id)
        self.case_id += 1
        self.indices = []
        self.resulting_log = pd.concat([self.resulting_log, unique_case_trace])


def make_case_ids_unique(input_xes_file: str, output_xes_file: str):
    """
    Reads the XES-file, splits into unique case IDs and
    overwrites the XES-file with the new one
    """
    original: pd.DataFrame = pm4py.read_xes(input_xes_file)

    internal_state = _InternalState(original)
    for name, group in original.groupby(['case:concept:name']):
        previous_timestamp = 0

        for index, row in group.iterrows():
            timestamp = pd.Timestamp(row['time:timestamp']).timestamp()

            if timestamp < previous_timestamp:
                internal_state.add_new_case_trace()

            internal_state.indices.append(index)
            previous_timestamp = timestamp

        internal_state.add_new_case_trace()

    assert len(original) == len(internal_state.resulting_log)
    os.makedirs(os.path.dirname(output_xes_file), exist_ok=True)
    pm4py.write_xes(internal_state.resulting_log, output_xes_file)


if __name__ == "__main__":
    make_case_ids_unique(args.xes_input, args.xes_output)
