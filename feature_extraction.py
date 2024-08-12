

import datetime
import typing
from abc import ABC

import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog
from tqdm import tqdm


class DataProcessor(ABC):
    @staticmethod
    def extract_logs_metadata(event_log: EventLog) -> typing.Tuple[typing.Dict, typing.Dict, int, int]:
        keys = ['[PAD]', '[UNK]']
        event_log_df = pm4py.convert_to_dataframe(event_log)
        activities = list(event_log_df['concept:name'].unique())
        keys.extend(activities)
        val = range(len(keys))
        coded_activity = dict({'x_word_dict': dict(zip(keys, val))})
        code_activity_normal = dict({'y_word_dict': dict(zip(activities, range(len(activities))))})
        coded_activity.update(code_activity_normal)

        x_word_dict = coded_activity['x_word_dict']
        y_word_dict = coded_activity['y_word_dict']
        vocab_size = len(x_word_dict)
        total_classes = len(y_word_dict)

        return x_word_dict, y_word_dict, vocab_size, total_classes

    @staticmethod
    def filter_attributes(event_log: EventLog) -> pd.DataFrame:
        # Removed all preprocessing of the activities to ensure that the approach deals with the same number of
        # activities
        event_log_df = pm4py.convert_to_dataframe(event_log)
        # Remove the string parsing of the timestamp and set UTC true
        event_log_df['time:timestamp'] = pd.to_datetime(event_log_df['time:timestamp'], dayfirst=True, utc=True) \
            .map(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        event_log_df = event_log_df[['case:concept:name', 'concept:name', 'time:timestamp']]
        return event_log_df

    @staticmethod
    def feature_extraction(event_log: EventLog) -> pd.DataFrame:
        raise NotImplementedError()


class DataProcessorNextTime(DataProcessor):
    @staticmethod
    def feature_extraction(event_log: EventLog) -> pd.DataFrame:
        event_log_df = DataProcessorNextTime.filter_attributes(event_log)
        processed_df = pd.DataFrame(columns=['case_id', 'prefix', 'k', 'time_passed', 'recent_time', 'latest_time',
                                             'next_time', 'remaining_time_days'])

        idx = 0
        unique_cases = event_log_df['case:concept:name'].unique()
        for _, case in enumerate(unique_cases):
            # ['Assign seriousness', 'Take in charge ticket', 'Take in charge ticket', 'Resolve ticket', 'Closed']
            act = event_log_df[event_log_df['case:concept:name'] == case]['concept:name'].to_list()
            # ['2012-10-09 11:50:17', '2012-10-09 11:51:01', '2012-10-12 12:02:56',
            # '2012-10-25 08:54:26', '2012-11-09 10:54:39']
            time = event_log_df[event_log_df['case:concept:name'] == case]['time:timestamp'].str[:19].to_list()
            time_passed = 0
            # 0:00:00
            latest_diff = datetime.timedelta()
            # 0:00:00
            recent_diff = datetime.timedelta()
            next_time = datetime.timedelta()
            for i in range(0, len(act)):
                prefix = np.where(i == 0, act[0], '#'.join(act[:i + 1]))
                if i > 0:
                    latest_diff = datetime.datetime.strptime(time[i], '%Y-%m-%d %H:%M:%S') - \
                                  datetime.datetime.strptime(time[i - 1], '%Y-%m-%d %H:%M:%S')
                if i > 1:
                    recent_diff = datetime.datetime.strptime(time[i], '%Y-%m-%d %H:%M:%S') - \
                                  datetime.datetime.strptime(time[i - 2], '%Y-%m-%d %H:%M:%S')
                latest_time = np.where(i == 0, 0, latest_diff.days)
                recent_time = np.where(i <= 1, 0, recent_diff.days)
                time_passed = time_passed + latest_time

                if i + 1 < len(time):
                    next_time = datetime.datetime.strptime(time[i + 1], '%Y-%m-%d %H:%M:%S') - \
                                datetime.datetime.strptime(time[i], '%Y-%m-%d %H:%M:%S')
                    next_time_days = str(int(next_time.days))
                else:
                    next_time_days = str(1)

                processed_df.at[idx, 'case_id'] = case
                processed_df.at[idx, 'prefix'] = prefix
                processed_df.at[idx, 'k'] = i
                processed_df.at[idx, 'time_passed'] = time_passed
                processed_df.at[idx, 'recent_time'] = recent_time
                processed_df.at[idx, 'latest_time'] = latest_time
                processed_df.at[idx, 'next_time'] = next_time_days
                idx = idx + 1

        return processed_df[['case_id', 'prefix', 'k', 'time_passed', 'recent_time', 'latest_time', 'next_time']]


class DataProcessorRemainingTime(DataProcessor):
    @staticmethod
    def feature_extraction(event_log: EventLog) -> pd.DataFrame:
        event_log_df = DataProcessorRemainingTime.filter_attributes(event_log)
        processed_df = pd.DataFrame(columns=['case_id', 'prefix', 'k', 'time_passed', 'recent_time', 'latest_time',
                                             'next_act', 'remaining_time_days'])

        idx = 0
        unique_cases = event_log_df['case:concept:name'].unique()
        for _, case in enumerate(unique_cases):
            act = event_log_df[event_log_df['case:concept:name'] == case]['concept:name'].to_list()
            time = event_log_df[event_log_df['case:concept:name'] == case]['time:timestamp'].str[:19].to_list()
            time_passed = 0
            latest_diff = datetime.timedelta()
            recent_diff = datetime.timedelta()
            for i in range(0, len(act)):
                prefix = np.where(i == 0, act[0], '#'.join(act[:i + 1]))
                if i > 0:
                    latest_diff = datetime.datetime.strptime(time[i], '%Y-%m-%d %H:%M:%S') - \
                                  datetime.datetime.strptime(time[i - 1], '%Y-%m-%d %H:%M:%S')
                if i > 1:
                    recent_diff = datetime.datetime.strptime(time[i], '%Y-%m-%d %H:%M:%S') - \
                                  datetime.datetime.strptime(time[i - 2], '%Y-%m-%d %H:%M:%S')

                latest_time = np.where(i == 0, 0, latest_diff.days)
                recent_time = np.where(i <= 1, 0, recent_diff.days)
                time_passed = time_passed + latest_time

                time_stamp = str(np.where(i == 0, time[0], time[i]))
                ttc = datetime.datetime.strptime(time[-1], '%Y-%m-%d %H:%M:%S') - \
                      datetime.datetime.strptime(time_stamp, '%Y-%m-%d %H:%M:%S')
                ttc = str(ttc.days)

                processed_df.at[idx, 'case_id'] = case
                processed_df.at[idx, 'prefix'] = prefix
                processed_df.at[idx, 'k'] = i
                processed_df.at[idx, 'time_passed'] = time_passed
                processed_df.at[idx, 'recent_time'] = recent_time
                processed_df.at[idx, 'latest_time'] = latest_time
                processed_df.at[idx, 'remaining_time_days'] = ttc
                idx = idx + 1

        return processed_df[['case_id', 'prefix', 'k', 'time_passed', 'recent_time', 'latest_time',
                             'remaining_time_days']]


class DataProcessorNextActivity(DataProcessor):
    @staticmethod
    def feature_extraction(event_log: EventLog) -> pd.DataFrame:
        event_log_df = DataProcessorNextActivity.filter_attributes(event_log)

        df_list = []
        for case, group in tqdm(event_log_df.groupby('case:concept:name'), 'Transforming into DataFrame'):
            act = group['concept:name'].to_list()
            for i in range(len(act)):
                prefix = '#'.join(act[:i])
                df = pd.DataFrame(
                    data=[[case, prefix, i, act[i]]],
                    columns=['case_id', 'prefix', 'k', 'next_act'])
                df_list.append(df)
        dfs = pd.concat(df_list, ignore_index=True)

        return dfs
