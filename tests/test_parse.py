import pytest

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

import fishact

def test_sniffer():
    n_header, delimiter, line = fishact.parse._sniff_file_info(
                                                'tests/single_gtype.txt')
    assert n_header == 2
    assert delimiter is None
    assert line == '1\n'

    n_header, delimiter, line = fishact.parse._sniff_file_info(
                                                'tests/multiple_gtype.txt')
    assert n_header == 2
    assert delimiter == '\t'
    assert line == '1\t5\t2\n'


def test_gtype_loader():
    df = fishact.parse.load_gtype('tests/single_gtype.txt', quiet=True,
                                  rstrip=True)
    assert all(df.columns == ['genotype', 'fish'])
    assert all(df['genotype'] == 'all fish')
    assert all(df['fish'] == [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96])


def test_resample_array():
    x  = np.arange(10, dtype=float)
    assert np.isclose(fishact.parse._resample_array(x, 10),
                      np.array([45.])).all()
    assert np.isclose(fishact.parse._resample_array(x, 20),
                      np.array([90.])).all()
    assert np.isclose(fishact.parse._resample_array(x, 5),
                      np.array([10., 35.])).all()
    assert np.isclose(fishact.parse._resample_array(x, 3),
                      np.array([3., 12., 21., 27.])).all()

    x[-3] = np.nan
    assert np.isclose(fishact.parse._resample_array(x, 5)[0], 10.0) \
                and np.isnan(fishact.parse._resample_array(x, 5)[1])


def test_resample_segment():
    df = pd.DataFrame({'a': np.arange(10),
                       'b': np.arange(10, 20),
                       'c': np.arange(10, dtype=float),
                       'd': np.arange(10, 20, dtype=float)})

    re_df = fishact.parse._resample_segment(df, 5, ['c'])
    re_df = re_df.reindex_axis(sorted(re_df.columns), axis=1)
    correct_df = pd.DataFrame({'a': [0, 5],
                               'b': [10, 15],
                               'c': [10., 35.],
                               'd': [10., 15.]})
    assert_frame_equal(re_df, correct_df)

    re_df = fishact.parse._resample_segment(df, 5, ['c', 'd'])
    re_df = re_df.reindex_axis(sorted(re_df.columns), axis=1)
    correct_df = pd.DataFrame({'a': [0, 5],
                               'b': [10, 15],
                               'c': [10., 35.],
                               'd': [60., 85.]})
    assert_frame_equal(re_df, correct_df)

    re_df = fishact.parse._resample_segment(df, 3, ['c'])
    re_df = re_df.reindex_axis(sorted(re_df.columns), axis=1)
    correct_df = pd.DataFrame({'a': [0, 3, 6, 9],
                               'b': [10, 13, 16, 19],
                               'c': [3., 12., 21., 27.],
                               'd': [10., 13., 16., 19.]})
    assert_frame_equal(re_df, correct_df)

    re_df = fishact.parse._resample_segment(df, 3, ['c', 'd'])
    re_df = re_df.reindex_axis(sorted(re_df.columns), axis=1)
    correct_df = pd.DataFrame({'a': [0, 3, 6, 9],
                               'b': [10, 13, 16, 19],
                               'c': [3., 12., 21., 27.],
                               'd': [33., 42., 51., 57.]})
    assert_frame_equal(re_df, correct_df)

def test_resample():
    df = pd.DataFrame(
        {'fish': np.concatenate((np.ones(10), 2*np.ones(10))).astype(int),
         'exp_time': np.concatenate((np.arange(10),
                                     np.arange(10))).astype(float),
         'exp_ind': np.concatenate((np.arange(10), np.arange(10))).astype(int),
         'activity': np.concatenate((np.arange(10),
                                     np.arange(10, 20))).astype(float),
         'sleep': np.ones(20, dtype=float),
         'light': [True]*5 + [False]*5 + [True]*5 + [False]*5,
         'day': [5]*10 + [6]*10,
         'genotype': ['wt']*20,
         'time': pd.to_datetime(['2017-03-30 14:00:00',
                                 '2017-03-30 14:01:00',
                                 '2017-03-30 14:02:00',
                                 '2017-03-30 14:03:00',
                                 '2017-03-30 14:04:00',
                                 '2017-03-30 14:05:00',
                                 '2017-03-30 14:06:00',
                                 '2017-03-30 14:07:00',
                                 '2017-03-30 14:08:00',
                                 '2017-03-30 14:09:00']*2)})

    re_df = fishact.parse.resample(df, 5, signal=['activity', 'sleep'],
                                   quiet=True)
    re_df = re_df.reindex_axis(sorted(re_df.columns), axis=1)
    correct_df = pd.DataFrame(
        {'activity': np.array([10., 35., 60., 85.]),
         'day': [5, 5, 6, 6],
         'fish': np.array([1, 1, 2, 2], dtype=int),
         'genotype': ['wt']*4,
         'light': [True, False, True, False],
         'sleep': np.array([5., 5., 5., 5.]),
         'time': pd.to_datetime(['2017-03-30 14:00:00',
                                 '2017-03-30 14:05:00']*2),
         'exp_time': np.array([0., 5., 0., 5.]),
         'exp_ind': np.array([0, 5, 0, 5], dtype=int)})
    assert_frame_equal(re_df, correct_df)

    re_df = fishact.parse.resample(df, 3, quiet=True)
    re_df = re_df.reindex_axis(sorted(re_df.columns), axis=1)
    correct_df = pd.DataFrame(
        {'activity': np.array([3., 10.5, 18., 25.5, 33., 40.5, 48., 55.5]),
         'day': [5, 5, 5, 5, 6, 6, 6, 6],
         'fish': np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=int),
         'genotype': ['wt']*8,
         'light': [True, True, False, False, True, True, False, False],
         'sleep': np.array([3., 3., 3., 3., 3., 3., 3., 3.]),
         'time': pd.to_datetime(['2017-03-30 14:00:00',
                                 '2017-03-30 14:03:00',
                                 '2017-03-30 14:05:00',
                                 '2017-03-30 14:08:00']*2),
         'exp_time': np.array([0., 3., 5., 8., 0., 3., 5., 8.]),
         'exp_ind': np.array([0, 3, 5, 8, 0, 3, 5, 8], dtype=int)})
    assert_frame_equal(re_df, correct_df)


def test_tidy_data():
    # Test that it will not overwrite existing file
    with pytest.raises(RuntimeError) as excinfo:
        fishact.parse.tidy_data('test.csv', 'test_geno.txt', 'test.csv')
    excinfo.match("Cannot overwrite input file.")

    with pytest.raises(RuntimeError) as excinfo:
        fishact.parse.tidy_data('test.csv', 'test_geno.txt', 'test_geno.txt')
    excinfo.match("Cannot overwrite input file.")

    with pytest.raises(RuntimeError) as excinfo:
        fishact.parse.tidy_data('test.csv', 'test_geno.txt',
                                'tests/empty_file_for_tests.csv')
    excinfo.match("empty_file_for_tests.csv already exists, not overwriting.")

    ## TO DO: integration test: make sure output CSV is as expected.
