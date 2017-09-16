import pytest

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

import fishact


def test_sleep_latency():
    df = pd.DataFrame({'zeit': np.linspace(0.0, 19.0, 20),
                       'sleep': np.ones(20, dtype=int)})

    assert np.isnan(fishact.summarize._sleep_latency(df))

    df.loc[6, 'sleep'] = 0
    assert np.isclose(fishact.summarize._sleep_latency(df), 1.0)

    df.loc[7, 'sleep'] = 0
    assert np.isclose(fishact.summarize._sleep_latency(df), 2.0)

    df.loc[5, 'sleep'] = 0
    assert np.isclose(fishact.summarize._sleep_latency(df), 3.0)

    df.loc[15, 'sleep'] = 0
    assert np.isclose(fishact.summarize._sleep_latency(df), 3.0)

    df.loc[0:15, 'sleep'] = 0
    assert np.isclose(fishact.summarize._sleep_latency(df), 16.0)

    df['sleep'] = np.zeros(len(df))
    assert np.isnan(fishact.summarize._sleep_latency(df))


def test_compute_bouts():
    df = pd.DataFrame({'zeit': np.linspace(0.0, 19.0, 20),
                       'sleep': np.ones(20, dtype=int),
                       'time': pd.to_datetime(['2017-03-30 14:00:00',
                                               '2017-03-30 14:01:00',
                                               '2017-03-30 14:02:00',
                                               '2017-03-30 14:03:00',
                                               '2017-03-30 14:04:00',
                                               '2017-03-30 14:05:00',
                                               '2017-03-30 14:06:00',
                                               '2017-03-30 14:07:00',
                                               '2017-03-30 14:08:00',
                                               '2017-03-30 14:09:00',
                                               '2017-03-30 14:10:00',
                                               '2017-03-30 14:11:00',
                                               '2017-03-30 14:12:00',
                                               '2017-03-30 14:13:00',
                                               '2017-03-30 14:14:00',
                                               '2017-03-30 14:15:00',
                                               '2017-03-30 14:16:00',
                                               '2017-03-30 14:17:00',
                                               '2017-03-30 14:18:00',
                                               '2017-03-30 14:19:00']),
                       'light': [True]*20,
                       'day': [5]*20})

    correct_df = pd.DataFrame(
        columns=['day_start', 'day_end', 'light_start', 'light_end',
                 'bout_start_zeit', 'bout_end_zeit',
                 'bout_start_clock', 'bout_end_clock', 'bout_length'])
    assert_frame_equal(fishact.summarize._compute_bouts(df), correct_df)

    df['sleep'] = np.array([0]*5 + [1]*15,dtype=int)
    assert_frame_equal(fishact.summarize._compute_bouts(df), correct_df)

    df['sleep'] = np.array([1]*5 + [0]*15,dtype=int)
    assert_frame_equal(fishact.summarize._compute_bouts(df), correct_df)

    df['sleep'] = np.array([0]*3 + [1]*4 + [0]*13,dtype=int)
    correct_df = correct_df.append(
            {'day_start': 5,
               'day_end': 5,
               'light_start': True,
               'light_end': True,
               'bout_start_zeit': 3.0,
               'bout_end_zeit': 7.0,
               'bout_start_clock': pd.to_datetime('2017-03-30 14:03:00'),
               'bout_end_clock': pd.to_datetime('2017-03-30 14:07:00'),
               'bout_length': 4.0}, ignore_index=True)
    assert_frame_equal(fishact.summarize._compute_bouts(df), correct_df,
                       check_dtype=False)

    df['sleep'] = np.array([0]*3 + [1]*4 + [0] + [1]*12, dtype=int)
    assert_frame_equal(fishact.summarize._compute_bouts(df), correct_df,
                       check_dtype=False)

    df['sleep'] = np.array([0]*3 + [1]*4 + [0]*2 + [1]*10 + [0],dtype=int)
    correct_df = pd.DataFrame(
            {'day_start': [5, 5],
               'day_end': [5, 5],
               'light_start': [True, True],
               'light_end': [True, True],
               'bout_start_zeit': [3.0, 9.0],
               'bout_end_zeit': [7.0, 19.0],
               'bout_start_clock': pd.to_datetime(['2017-03-30 14:03:00',
                                                   '2017-03-30 14:09:00']),
               'bout_end_clock': pd.to_datetime(['2017-03-30 14:07:00',
                                                 '2017-03-30 14:19:00']),
               'bout_length': [4.0, 10.0]})
    correct_df = correct_df.sort_index(axis=1)
    assert_frame_equal(fishact.summarize._compute_bouts(df).sort_index(axis=1),
                       correct_df, check_dtype=False)
