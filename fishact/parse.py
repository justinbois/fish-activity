import datetime

import numpy as np
import pandas as pd


def tidy_data(activity_name, genotype_name, out_name, lights_on, lights_off,
              day_in_the_life, resample_win=1, extra_cols=[],
              rename={'middur': 'activity'}):
    """
    Load in activity data and write tidy data file.

    Parameters
    ----------
    fname : string
        CSV file containing the activity data. This is a conversion
        to CSV of the Excel file that comes off the instrument.
    genotype_fname : string
        File containing genotype information. This is in standard
        Prober lab format, with tab delimited file.
        - First row discarded
        - Second row contains genotypes. String is only kept up to
          the last space because they typically appear something like
          'tph2-/- (n=20)', and we do not need the ' (n=20)'.
        - Subsequent rows containg wells in the 96 well plate
          corresponding to each genotype.
    out_name : string
        Name of file to write tidy DataFrame to.
    lights_on : string or datetime.time instance
        The time where lights come on each day, e.g., '9:00:00'.
    lights_off: string or datetime.time instance
        The time where lights go off each day, e.g., '23:00:00'.
    day_in_the_life : int
        The day in the life of the embryos when data acquisition
        started.
    resample_win : int, default 1
        Size of resampling window.
    extra_cols : list, default []
        List of extra columns to keep from the input file, e.g.,
        ['frect', 'fredur']. By default, only time, fish ID, and
        activity as measured by 'middur' is kept.
    rename : dict, default {'middur': 'activity'}
        Dictionary for renaming column headings.

    Notes
    -----
    .. Writes a tidy data set with columns:
        - activity: The activity over the time interval
        - time: time in proper datetime format
        - fish: ID of the fish
        - genotype: genotype of the fish
        - zeit: Zeitgeber time
        - zeit_ind: an index for the Zeitgeber time. Because of some
          errors in the acquisition, sometimes the times do not
          perfectly line up. zeit_ind is just the index of the
          measurement. This is needed for computing averages over
          fish at each time point.
        - light: True if the light is on.
        - day: The day in the life of the fish
    .. The column 'zeit' contains the Zeitgeber time, and is
       calculated from the time stamps in the original data
       set, *not* from the 'start' and 'end' columns.
    """
    df = load_data(activity_name, genotype_name, lights_on, lights_off,
                   day_in_the_life, extra_cols=extra_cols, rename=rename)
    df = resample(df, resample_win)
    df.to_csv(out_name, index=False)
    return None


def load_gtype(fname):
    """
    Read genotype file into tidy DataFrame

    Parameters
    ----------
    fname : string
        File containing genotype information. This is in standard
        Prober lab format, with tab delimited file.
        - First row discarded
        - Second row contains genotypes. String is only kept up to
          the last space because they typically appear something like
          'tph2-/- (n=20)', and we do not need the ' (n=20)'.
        - Subsequent rows containg wells in the 96 well plate
          corresponding to each genotype.

    Returns
    -------
    df : pandas DataFrame
        Tidy DataFrame with columns:
        - fish: ID of fish
        - genotype: genotype of fish
    """
    # Read file
    df = pd.read_csv(fname, delimiter='\t', comment='#', header=[0, 1])

    # Reset the columns to be the second level of indexing
    df.columns = df.columns.get_level_values(1)

    # Only keep genotype up to last space because sometimes has n
    df.columns = [col[:col.rfind(' ')] if col.rfind(' ') > 0 else col
                  for col in df.columns]

    # Melt the DataFrame
    df = pd.melt(df, var_name='genotype', value_name='fish').dropna()

    # Reset the index
    df = df.reset_index(drop=True)

    # Make sure data type is integer
    df.loc[:,'fish'] = df.loc[:, 'fish'].astype(int)

    return df


def load_data(fname, genotype_fname, lights_on, lights_off, day_in_the_life,
              extra_cols=[], rename={'middur': 'activity'}):
    """
    Load in activity CSV file to tidy DateFrame

    Parameters
    ----------
    fname : string
        CSV file containing the activity data. This is a conversion
        to CSV of the Excel file that comes off the instrument.
    genotype_fname : string
        File containing genotype information. This is in standard
        Prober lab format, with tab delimited file.
        - First row discarded
        - Second row contains genotypes. String is only kept up to
          the last space because they typically appear something like
          'tph2-/- (n=20)', and we do not need the ' (n=20)'.
        - Subsequent rows containg wells in the 96 well plate
          corresponding to each genotype.
    lights_on : string or datetime.time instance
        The time where lights come on each day, e.g., '9:00:00'.
    lights_off: string or datetime.time instance
        The time where lights go off each day, e.g., '23:00:00'.
    day_in_the_life : int
        The day in the life of the embryos when data acquisition
        started.
    extra_cols : list, default []
        List of extra columns to keep from the input file, e.g.,
        ['frect', 'fredur']. By default, only time, fish ID, and
        activity as measured by 'middur' is kept.
    rename : dict, default {'middur': 'activity'}
        Dictionary for renaming column headings.

    Returns
    -------
    df : pandas DataFrame
        Tidy DataFrame with columns:
        - activity: The activity as given by the instrument, based
          on the `middur` columns of the inputted data set
        - time: time in proper datetime format, based on the `sttime`
          column of the inputted data file
        - fish: ID of the fish
        - genotype: genotype of the fish
        - zeit: Zeitgeber time, based on the `start` column of the
          inputted data file
        - zeit_ind: an index for the Zeitgeber time. Because of some
          errors in the acquisition, sometimes the times do not
          perfectly line up. zeit_ind is just the index of the
          measurement. This is needed for computing averages over
          fish at each time point.
        - light: True if the light is on.
        - day: The day in the life of the fish

    Notes
    -----
    .. The column 'zeit' contains the Zeitgeber time, and is
       calculated from the time stamps in the original data
       set, *not* from the 'start' and 'end' columns.
    """

    # Convert lightson and lightsoff to datetime.time objects
    if type(lights_on) != datetime.time:
        lights_on = pd.to_datetime(lights_on).time()
    if type(lights_off) != datetime.time:
        lights_off = pd.to_datetime(lights_off).time()

    # Get genotype information
    df_gt = load_gtype(genotype_fname)

    # Determine which columns to read in
    if extra_cols is None:
        extra_cols = []
    cols = ['start', 'location', 'stdate', 'sttime', 'middur']
    new_cols = list(set(extra_cols) - set(cols))
    usecols = cols + new_cols

    # Read file
    df = pd.read_csv(fname, usecols=usecols)

    # Convert location to well number (just drop 'c' in front)
    df = df.rename(columns={'location': 'fish'})
    df['fish'] = df['fish'].str.extract('(\d+)', expand=False).astype(int)

    # Only keep fish that we have genotypes for
    df = df.loc[df['fish'].isin(df_gt['fish']), :]

    # Store the genotypes
    fish_lookup = {fish: df_gt.loc[df_gt['fish']==fish, 'genotype'].values[0]
                          for fish in df_gt['fish']}
    df['genotype'] = df['fish'].apply(lambda x: fish_lookup[x])

    # Convert date and time to a time stamp
    df['time'] = pd.to_datetime(df['stdate'] + df['sttime'],
                                format='%d/%m/%Y%H:%M:%S')

    # Get earliest time point
    t_min = pd.DatetimeIndex(df['time']).min()

    # Get Zeitgeber time in units of hours
    df['zeit'] = df['start'] / 3600

    # Determine light or dark
    clock = pd.DatetimeIndex(df['time']).time
    df['light'] = np.logical_and(clock >= lights_on, clock < lights_off)

    # Which day it is (remember, day goes lights on to lights on)
    df['day'] = pd.DatetimeIndex(
        df['time'] - datetime.datetime.combine(t_min.date(), lights_on)).day \
                + day_in_the_life - 1

    # Sort by fish and zeit
    df = df.sort_values(by=['fish', 'zeit']).reset_index(drop=True)

    # Set up zeit indices
    for fish in df['fish'].unique():
        df.loc[df['fish']==fish, 'zeit_ind'] = np.arange(
                                                    np.sum(df['fish']==fish))
    df['zeit_ind'] = df['zeit_ind'].astype(int)

    # Return everything if we don't want to delete anything
    if 'sttime' not in extra_cols:
        usecols.remove('sttime')
    if 'stdate' not in extra_cols:
        usecols.remove('stdate')
    if 'start' not in extra_cols:
        usecols.remove('start')
    usecols.remove('location')

    cols = usecols + ['time', 'fish', 'genotype', 'zeit', 'zeit_ind',
                      'light', 'day']
    df = df[cols]

    # Rename columns
    if rename is not None:
        df = df.rename(columns=rename)

    return df


def load_perl_processed_activity(activity_file, df_gt):
    """
    Load activity data into tidy DataFrame
    """
    df = pd.read_csv(activity_file, delimiter='\t', comment='#', header=[0, 1])

    # Make list of columns (use type conversion to allow list concatenation)
    df.columns = list(df.columns.get_level_values(1)[:2]) \
                                + list(df.columns.get_level_values(0)[2:])

    # Columns we want to drop
    cols_to_drop = df.columns[df.columns.str.contains('Unnamed')]
    df = df.drop(cols_to_drop, axis=1)

    # Start and end times are also dispensible
    df = df.drop(['start', 'end'], axis=1)

    # Find columns to drop (fish that do not have assigned genotypes)
    cols_to_drop = []
    for col in df.columns:
        if 'FISH' in col and int(col.lstrip('FISH')) not in df_gt['fish'].values:
                cols_to_drop.append(col)

    # Drop 'em!
    df = df.drop(cols_to_drop, axis=1)

    # Add a column for whether or not it is light
    df['light'] = pd.Series(df.CLOCK < 14.0, index=df.index)

    # Find where the lights switch from off to on.
    dark_to_light = np.where(np.diff(df['light'].astype(np.int)) == 1)[0]

    # Initialize array with day numbers
    day = np.zeros_like(df['light'], dtype=np.int)

    # Loop through transitions to set day numbers
    for i in range(len(dark_to_light) - 1):
        day[dark_to_light[i]+1:dark_to_light[i+1]+1] = i + 1
    day[dark_to_light[-1]+1:] = len(dark_to_light)

    # Insert the day numnber into DataFrame
    df['day'] = pd.Series(day, index=df.index)

    # Build ziet and put it in the DataFrame
    zeit = 24.0 * df['day'] + df['CLOCK']
    df['zeit'] = pd.Series(zeit, index=df.index)

    # Build list of genotypes
    genotypes = []

    # Check each column, put None for non-FISH column
    for col in df.columns:
        if 'FISH' in col:
            fish_id = int(col.lstrip('FISH'))
            genotypes.append(df_gt.genotype[df_gt.fish==fish_id].iloc[0])
        else:
            genotypes.append(None)

    df.columns = pd.MultiIndex.from_arrays((genotypes, df.columns),
                                        names=['genotype', 'variable'])

    # Value variables are the ones with FISH
    col_1 = df.columns.get_level_values(1)
    value_vars = list(df.columns[col_1.str.contains('FISH')])

    # ID vars are the non-FISH entries
    id_vars = list(df.columns[~col_1.str.contains('FISH')])

    # Perform the melt
    df = pd.melt(df, value_vars=value_vars, id_vars=id_vars,
                 value_name='activity', var_name=['genotype', 'fish'])

    # Rename any column that is a tuple
    for col in df.columns:
        if type(col) is tuple:
            df.rename(columns={col: col[1]}, inplace=True)

    # Make fish IDs integer
    df['fish'] = df['fish'].apply(lambda x: int(x.lstrip('FISH')))

    return df


def resample(df, ind_win):
    """
    Resample the DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with pertinent data. Must have columns 'time',
        'fish', 'genotype', 'day', 'light', 'zeit'.

    Returns
    -------
    output : pandas DataFrame
        Resampled DataFrame.

    Notes
    -----
    .. Assumes that the signal is aligned with the
       *left* of the time interval. I.e., if df['zeit'] = [0, 1, 2],
       the values of df['activity'] are assumed to be aggregated over
       time intervals 0 to 1, 1 to 2, and 2 to 3. The same is true
       for the outputted resampled array.
    """
    # Make a copy so as to leave original unperturbed
    df_in = df.copy()

    # Sort the DataFrame by fish and then zeit
    df_in = df_in.sort_values(by=['fish', 'zeit']).reset_index(drop=True)

    # If no resampling is necessary
    if ind_win == 1:
        return df_in

    # Extract  light
    light = df_in.loc[df_in['fish']==df_in['fish'].unique()[0], 'light'].values

    # Find first light switching event
    if light[0]:
        where_false = np.where(~light)[0]
        if len(where_false) == 0:
            first_ind = 0
        else:
            first_ind = where_false[0]
    else:
        where_true = np.where(light)[0]
        if len(where_true) == 0:
            first_ind = 0
        else:
            first_ind = where_true[0]

    # Determine start index for averaging
    if first_ind < ind_win:
        start_ind = first_ind
    else:
        start_ind = first_ind % ind_win

    # Make GroupBy object
    df_gb = df_in.groupby('fish')['activity']

    # Compute rolling sum (result is stored at right end of window)
    s = df_gb.rolling(window=ind_win).sum().reset_index(level=0, drop='fish')

    # Columns to keep in output DataFrame
    new_cols = ['time', 'fish', 'genotype', 'day', 'light', 'zeit']

    # Inds to keep
    inds = np.array([])
    win_inds = np.array([])
    for fish in df_in.fish.unique():
        start = df_in.loc[df_in.fish==fish, :].index[0] \
                                            + start_ind + ind_win - 1
        stop = df_in.loc[df_in.fish==fish, :].index[-1] + 1
        new_inds = np.arange(start, stop, ind_win)
        inds = np.concatenate((inds, new_inds - ind_win + 1))
        win_inds = np.concatenate((win_inds, new_inds))

    # Zeit indices
    n_fish = len(df_in.fish.unique())
    zeit_ind = list(range(int(len(inds) // n_fish))) * n_fish

    # New DataFrame
    df_resampled = df_in.loc[inds, new_cols].reset_index(drop=True)
    df_resampled['activity'] = s[win_inds].values
    df_resampled['zeit_ind'] = zeit_ind

    return df_resampled
