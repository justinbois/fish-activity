import numpy as np
import pandas as pd
import numba

def _sleep_latency(df):
    sleep_mins = df[df['sleep']==1]
    if len(sleep_mins) == 0:
        return np.nan
    return df.loc[sleep_mins.index[0], 'exp_time'] - df['exp_time'].iloc[0]

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
    def tup_to_str(ind):
        if ind[0] == 'activity':
            string = 'total seconds of activity in '
        elif ind[0] == 'sleep':
            string = 'total minutes of sleep in '
        if ind[2]:
            return string + 'day ' + str(ind[1])
        else:
            return string + 'night ' + str(ind[1])

    sum_df.columns = [tup_to_str(ind) for ind in sum_df.columns]

    return sum_df
