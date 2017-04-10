import numpy as np
import pandas as pd
import numba

def _compute_bouts(df, rest=True):
    """
    Compute bout lengths for either sleep or active bouts.

    Parameters
    ----------
    df : pandas DataFrame
        Minimally with columns 'sleep', 'day', 'light', 'exp_time', 'time'.

    Returns
    -------
    output : float
        sleep latency in units of experimental time (hours)

    Notes
    -----
    .. We do not consider bouts at the beginning or end of an
       experiment.
    """

    # Output DataFrame
    cols = {'day_start': int,
            'day_end': int,
            'light_start': bool,
            'light_end': bool,
            'bout_start_exp': float,
            'bout_end_exp': float,
            'bout_start_clock': '<M8[ns]',
            'bout_end_clock': '<M8[ns]',
            'bout_length': float}
    df_out = pd.DataFrame(columns=[key for key in cols])

    # Get Boolean NumPy array for sleep
    sleep = df['sleep'].values.astype(bool)

    # Compute where switching between sleep and awake
    switches = np.where(np.diff(sleep))[0] + 1

    # Make sure there is a bout to include
    if len(switches) < 2:
        return df_out

    # Is fish starting out and ending asleep?
    start_asleep = bool(df['sleep'].iloc[0])
    end_asleep = bool(df['sleep'].iloc[-1])

    # Determine which switches to iterate over
    # Equiv. to range(int(not(start_asleep ^ rest)), len(switches-1), 2)
    if start_asleep:
        if rest:
            iterator = range(1, len(switches)-1, 2)
        else:
            iterator = range(0, len(switches)-1, 2)
    else:
        if rest:
            iterator = range(0, len(switches)-1, 2)
        else:
            iterator = range(1, len(switches)-1, 2)

    # Build DataFrame
    for i in iterator:
        ind1 = df.index[switches[i]]
        ind2 = df.index[switches[i+1]]
        new_row = {'day_start': df.loc[ind1, 'day'],
                   'day_end': df.loc[ind2, 'day'],
                   'light_start': df.loc[ind1, 'light'],
                   'light_end': df.loc[ind2, 'light'],
                   'bout_start_exp': df.loc[ind1, 'exp_time'],
                   'bout_end_exp': df.loc[ind2, 'exp_time'],
                   'bout_start_clock': df.loc[ind1, 'time'],
                   'bout_end_clock': df.loc[ind2, 'time'],
            'bout_length': df.loc[ind2, 'exp_time'] - df.loc[ind1, 'exp_time']}
        df_out = df_out.append(new_row, ignore_index=True)

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
        Minimally with columns 'sleep' and 'exp_time'.

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


def total_activity_sleep(df):
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
    return gb['activity', 'sleep'].sum().reset_index()



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
    else:
        raise RuntimeError('%s is invalid MultiIndex' % ind[0])

    if ind[2]:
        return string + 'day ' + str(ind[1])
    else:
        return string + 'night ' + str(ind[1])


def write_total_activity_sleep(df, outfile):
    """
    Write a CSV file with summary of
    """
    # Make sure all columns are there
    for col in ['activity', 'sleep', 'fish', 'genotype', 'light', 'day']:
        if col not in df.columns:
            raise RuntimeError('%s missing from input DataFrame' % col)

    # Make new DataFrame with sum of activity and sleep for each fish
    sum_df = pd.pivot_table(df, index=['fish', 'genotype'],
                            values=['activity', 'sleep'],
                            columns=['day', 'light'], aggfunc=np.sum)

    # Set column names and sort for activity, day, day/night
    sum_df.columns.set_names(['signal', 'day', 'light'], inplace=True)
    sum_df.sort_index(axis=1, level=['signal', 'day', 'light'],
                      ascending=[True, True, False], inplace=True)

    # Sort by genotype and then fish ID
    sum_df.sort_index(axis=0, level=['genotype', 'fish'], inplace=True)

    # Rename the column headings from the MultiIndex
    sum_df.columns = [_column_tup_to_str(ind) for ind in sum_df.columns]

    return sum_df
