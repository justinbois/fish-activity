import collections

try:
    import tqdm
except:
    pass

import numpy as np
import pandas as pd
import numba


def _compute_bouts(df, rest=True):
    """
    Compute bout lengths for either sleep or active bouts.

    Parameters
    ----------
    df : pandas DataFrame
        Tidy DataFrame, as outputted by fishact.parse.load_activity(),
        containing the following columns:
        - activity: The activity as given by the instrument, based
          on the `middur` columns of the inputted data set. This
          column may be called 'middur' dependinfg on the `rename`
          kwarg.
        - time: time in proper datetime format, based on the `sttime`
          column of the inputted data file
        - sleep : 1 if fish is asleep (activity = 0), and 0 otherwise.
          This is convenient for computing sleep when resampling.
        - fish: ID of the fish
        - genotype: genotype of the fish
        - exp_time: Experimental time, based on the `start` column of
            the inputted data file
        - exp_ind: an index for the experimental time. Because of some
          errors in the acquisition, sometimes the times do not
          perfectly line up. exp_ind is just the index of the
          measurement. This is needed for computing averages over
          fish at each time point.
        - light: True if the light is on.
        - day: The day in the life of the fish. The day begins with
          `lights_on`.

    Returns
    -------
    output : pandas DataFrame
        Tidy DataFrame, with the following columns:
        - fish: ID of the fish
        - genotype: genotype of the fish
        - day_start: Start day of bout
        - day_end: End day of bout
        - light_start: True if light is on at start of bout.
        - light_end: True if light is on at end of bout.
        - bout_start_exp: Time of start of bout since the
            beginning of the experiment (usually in units of hours).
        - bout_end_exp: Time of end of bout since the
            beginning of the experiment (usually in units of hours).
        - bout_start_clock: Wall clock time of start of bout.
        - bout_end_clock: Wall clock time of the end of bout.
        - bout_length: Length of the bout in same units as
            experimental time.

    Notes
    -----
    .. We do not consider bouts at the beginning or end of an
       experiment.

    .. This calculation is currently very slow because we do not
       take advantage of ordering of the time series and index
       pandas DataFrames very often and keep appending DataFrames.
    """

    # Output DataFrame columns
    cols = collections.OrderedDict(
           {'day_start': int,
            'day_end': int,
            'light_start': bool,
            'light_end': bool,
            'bout_start_exp': float,
            'bout_end_exp': float,
            'bout_start_clock': '<M8[ns]',
            'bout_end_clock': '<M8[ns]',
            'bout_length': float})

    # Get Boolean NumPy array for sleep
    sleep = df['sleep'].values.astype(bool)

    # Compute where switching between sleep and awake
    switches = np.where(np.diff(sleep))[0] + 1

    # Make sure there is a bout to include
    if len(switches) < 2:
        return pd.DataFrame(columns=cols.keys())

    # Is fish starting out and ending asleep?
    start_asleep = bool(df['sleep'].iloc[0])
    end_asleep = bool(df['sleep'].iloc[-1])

    # Determine which switches to iterate over
    # Equiv. to range(int(not(start_asleep ^ rest)), len(switches-1), 2)
    if start_asleep:
        if rest:
            i = np.arange(1, len(switches)-1, 2, dtype=int)
        else:
            i = np.arange(0, len(switches)-1, 2, dtype=int)
    else:
        if rest:
            i = np.arange(0, len(switches)-1, 2, dtype=int)
        else:
            i = np.arange(1, len(switches)-1, 2, dtype=int)

    # Pull out indices of switches
    c = ['day', 'light', 'time', 'exp_time']
    df_switch = df.loc[df.index[switches], c].copy().reset_index(drop=True)

    # Build output array
    df_out = pd.DataFrame(columns=cols.keys())
    df_out['day_start'] = df_switch.loc[i, 'day'].values
    df_out['day_end'] = df_switch.loc[i+1, 'day'].values
    df_out['light_start'] = df_switch.loc[i, 'light'].values
    df_out['light_end'] = df_switch.loc[i+1, 'light'].values
    df_out['bout_start_exp'] = df_switch.loc[i, 'exp_time'].values
    df_out['bout_end_exp'] = df_switch.loc[i+1, 'exp_time'].values
    df_out['bout_start_clock'] = df_switch.loc[i, 'time'].values
    df_out['bout_end_clock'] = df_switch.loc[i+1, 'time'].values
    df_out['bout_length'] = df_switch.loc[i+1, 'exp_time'].values - \
                df_switch.loc[i, 'exp_time'].values

    # Ensure data types
    for col, dtype in cols.items():
        df_out[col] = df_out[col].astype(dtype)

    return df_out


def _sleep_latency(df):
    """
    Compute sleep latency.

    Parameters
    ----------
    df : pandas DataFrame
        Minimally with columns 'sleep' and 'exp_time'. Finds the first
        minute of sleep.

    Returns
    -------
    output : float
        sleep latency in units of experimental time (hours)

    Notes
    -----
    .. Sleep latency is defined as the length of time between the first
       not-sleeping to sleeping transition and the first awake minute
       after a light switching event. If there is no not-sleeping to
       sleeping transition before the next light switching event, the
       latency is NaN.
    .. The inputted DataFrame contains a single record that starts at
       the light switching event and ends at the last time point before
       the next light switching event.
    """
    # When was the fish awake?
    awake_mins = df[df['sleep']==0]

    # If always awake, return NaN
    if len(awake_mins) == 0:
        return np.nan

    # First awake minute
    first_awake_min = awake_mins['exp_time'].min()

    # Relevant sleep minutes
    sleep_mins = df.loc[(df['sleep']==1) & (df['exp_time'] > first_awake_min)]

    # If never asleep after first awake minute, return NaN
    if len(sleep_mins) == 0:
        return np.nan

    return sleep_mins['exp_time'].min() - first_awake_min


def bouts(df, rest=True, quiet=False):
    """
    Compute bouts for rest of activity.

    Parameters
    ----------
    df : pandas DataFrame
        Tidy DataFrame, as outputted by fishact.parse.load_activity(),
        containing the following columns:
        - activity: The activity as given by the instrument, based
          on the `middur` columns of the inputted data set. This
          column may be called 'middur' dependinfg on the `rename`
          kwarg.
        - time: time in proper datetime format, based on the `sttime`
          column of the inputted data file
        - sleep : 1 if fish is asleep (activity = 0), and 0 otherwise.
          This is convenient for computing sleep when resampling.
        - fish: ID of the fish
        - genotype: genotype of the fish
        - exp_time: Experimental time, based on the `start` column of
            the inputted data file
        - exp_ind: an index for the experimental time. Because of some
          errors in the acquisition, sometimes the times do not
          perfectly line up. exp_ind is just the index of the
          measurement. This is needed for computing averages over
          fish at each time point.
        - light: True if the light is on.
        - day: The day in the life of the fish. The day begins with
          `lights_on`.
    rest : bool, default True
        True if rest bouts are being computed. False if active
        bouts are being computed.
    quiet : bool, default False
        If True, do not show progress bar.

    Returns
    -------
    output : pandas DataFrame
        Tidy DataFrame, with the following columns:
        - fish: ID of the fish
        - genotype: genotype of the fish
        - day_start: Start day of bout
        - day_end: End day of bout
        - light_start: True if light is on at start of bout.
        - light_end: True if light is on at end of bout.
        - bout_start_exp: Time of start of bout since the
            beginning of the experiment (usually in units of hours).
        - bout_end_exp: Time of end of bout since the
            beginning of the experiment (usually in units of hours).
        - bout_start_clock: Wall clock time of start of bout.
        - bout_end_clock: Wall clock time of the end of bout.
        - bout_length: Length of the bout in same units as
            experimental time.

    Notes
    -----
    .. We do not consider bouts at the beginning or end of an
       experiment.
    """
    # Output DataFrame
    cols = collections.OrderedDict(
           {'fish': int,
            'genotype': str,
            'day_start': int,
            'day_end': int,
            'light_start': bool,
            'light_end': bool,
            'bout_start_exp': float,
            'bout_end_exp': float,
            'bout_start_clock': '<M8[ns]',
            'bout_end_clock': '<M8[ns]',
            'bout_length': float})
    df_out = pd.DataFrame(columns=[key for key in cols])

    # Set up progress bar
    if not quiet:
        print('Performing bout calculation....')
        try:
            iterator = tqdm.tqdm(df['fish'].unique())
        except:
            iterator = df['fish'].unique()
    else:
        iterator = df['fish'].unique()

    # Loop through fish and populate output DataFrame
    for fish in iterator:
        df_fish = df.loc[df['fish']==fish, :]
        df_bout = _compute_bouts(df_fish, rest=rest)
        df_bout['fish'] = fish
        df_bout['genotype'] = df_fish['genotype'].iloc[0]
        df_out = df_out.append(df_bout, ignore_index=True)

    # Ensure data types
    # for col, dtype in cols.items():
    #     df_out[col] = df_out[col].astype(dtype)

    return df_out


def daily_summary(df):
    """
    Make a summary DataFrame of fish activity and sleep

    Parameters
    ----------
    df : pandas DataFrame
        Tidy DataFrame, as outputted by fishact.parse.load_activity(),
        containing the following columns:
        - activity: The activity as given by the instrument, based
          on the `middur` columns of the inputted data set. This
          column may be called 'middur' dependinfg on the `rename`
          kwarg.
        - time: time in proper datetime format, based on the `sttime`
          column of the inputted data file
        - sleep : 1 if fish is asleep (activity = 0), and 0 otherwise.
          This is convenient for computing sleep when resampling.
        - fish: ID of the fish
        - genotype: genotype of the fish
        - exp_time: Experimental time, based on the `start` column of
            the inputted data file
        - exp_ind: an index for the experimental time. Because of some
          errors in the acquisition, sometimes the times do not
          perfectly line up. exp_ind is just the index of the
          measurement. This is needed for computing averages over
          fish at each time point.
        - light: True if the light is on.
        - day: The day in the life of the fish. The day begins with
          `lights_on`.

    Returns
    -------
    output : pandas DataFrame
        Tidy DataFrame with columns
        - fish: ID of the fish
        - genotype: genotype of the fish
        - day: The day in the life of the fish. The day begins with
          `lights_on`.
        - light: True if the light is on.
        - activity: Total seconds of activity in time period
        - sleep: Total minues of sleep in time period
    """
    gb = df.groupby(['fish', 'genotype', 'day', 'light'])
    df_sum = gb['activity', 'sleep'].sum()
    df_sum['latency'] = gb['sleep', 'exp_time'].apply(_sleep_latency)
    return df_sum.reset_index()


def _column_tup_to_str(ind):
    """
    Convert tuple of MultiIndex to string.

    Parameters
    ----------
    ind : tuple
        ind[0]: either 'sleep' or 'activity'
        ind[1]: int that is the day number
        ind[2]: bool, True being light, False being dark

    Returns
    -------
    output : str
        Conversion to a single string represnting info in tuple.
        E.g., ('activity', 6, True) gets converted to
        'total seconds of activity in day 6'.
    """
    if ind[0] == 'activity':
        string = 'total seconds of activity in '
    elif ind[0] == 'sleep':
        string = 'total minutes of sleep in '
    elif ind[0] == 'latency':
        string = 'minutes of sleep latency in '
    else:
        raise RuntimeError('%s is invalid MultiIndex' % ind[0])

    if ind[2]:
        return string + 'day ' + str(ind[1])
    else:
        return string + 'night ' + str(ind[1])


def write_daily_summary(df, outfile):
    """
    Write a CSV file with summary of daily statistics.
    """
    # Make sure all columns are there
    for col in ['activity', 'sleep', 'fish', 'genotype', 'light', 'day']:
        if col not in df.columns:
            raise RuntimeError('%s missing from input DataFrame' % col)

    # Compute summary stats
    df_sum = daily_summary(df)
    df_sum['latency'] *= 60

    # Pivot
    df_sum = pd.pivot_table(df_sum, index=['fish', 'genotype'],
                            values=['activity', 'sleep', 'latency'],
                            columns=['day', 'light'])

    # Set column names and sort for activity, day, day/night
    df_sum.columns.set_names(['signal', 'day', 'light'], inplace=True)
    df_sum.sort_index(axis=1, level=['signal', 'day', 'light'],
                      ascending=[True, True, False], inplace=True)

    # Sort by genotype and then fish ID
    df_sum.sort_index(axis=0, level=['genotype', 'fish'], inplace=True)

    # Rename the column headings from the MultiIndex
    df_sum.columns = [_column_tup_to_str(ind) for ind in df_sum.columns]

    # Write summary CSV
    df_sum.to_csv(outfile, index=False, float_format='%.4f')
