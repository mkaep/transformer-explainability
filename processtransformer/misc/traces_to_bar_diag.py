
import pm4py
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Misc - Traces to Histogram.")
parser.add_argument("--xes_log", type=str, required=True, help='Path of the XES file.')
args = parser.parse_args()


def read_file(file_path: str):
    return pm4py.read_xes(file_path)


def log_to_bins(log: pd.DataFrame, title: str):
    by_trace_df = pd.DataFrame()
    for name, group in log.groupby(['case:concept:name']):
        trace_as_str = ','.join(group['concept:name'].values)
        by_trace_df = pd.concat([by_trace_df, pd.DataFrame([trace_as_str], columns=['trace'])])

    bins = pd.DataFrame()
    for name, group in by_trace_df.groupby(['trace']):
        trace = group.iloc[0][0]
        num_traces = len(group)
        bins = pd.concat([bins, pd.DataFrame([[trace, num_traces]], columns=['trace', 'count'])])

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(len(bins) * 3 * cm, 12 * cm))
    ax.bar(bins['trace'], bins['count'])
    ax.set_ylabel('Num occurrences')
    plt.xticks(rotation=10)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    event_log = read_file(args.xes_log)
    log_to_bins(event_log, 'File name: ' + args.xes_log)
