import datetime

try:
    import tqdm
except:
    pass

import numpy as np
import pandas as pd
import numba


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
        - expt_time: Experimental time
        - exp_ind: an index for the experimental time. Because of some
          errors in the acquisition, sometimes the times do not
          perfectly line up. exp_ind is just the index of the
          measurement. This is needed for computing averages over
          fish at each time point.
        - light: True if the light is on.
        - day: The day in the life of the fish
    """
    df = load_data(activity_name, genotype_name, lights_on, lights_off,
                   day_in_the_life, extra_cols=extra_cols, rename=rename)
    df = resample(df, resample_win)
    df.to_csv(out_name, index=False)
    return None


def load_gtype(fname, delimiter='\t', comment='#', double_header=True,
               rstrip=True):
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
    delimiter : string, default '\t'
        Delimiter of file
    comment : string, default '#'
        Test that begins and comment line in the file
    double_header : bool, default True
        If True, the file has a two-line header. The first line
        is ignored and the second is kept as a header, possibly
        with stripping using the `rstrip` argument. This is
        typical of the current Prober lab format.
    rstrip : bool, default True
        If True, strip out all text in genotype name to the right of
        the last space. This is because the genotype files typically
        have headers like 'wt (n=22)', and the '(n=22)' is useless.

    Returns
    -------
    df : pandas DataFrame
        Tidy DataFrame with columns:
        - fish: ID of fish
        - genotype: genotype of fish
    """
    # Read in the file
    if double_header:
        df = pd.read_csv(fname, delimiter=delimiter, comment=comment,
                         header=[0, 1])

        # Reset the columns to be the second level of indexing
        df.columns = df.columns.get_level_values(1)
    else:
        df = pd.read_csv(fname, delimiter=delimiter, comment=comment)

    # Only keep genotype up to last space because sometimes has n
    if rstrip:
        df.columns = [col[:col.rfind(' ')] if col.rfind(' ') > 0 else col
                            for col in df.columns]

    # Melt the DataFrame
    df = pd.melt(df, var_name='genotype', value_name='fish').dropna()

    # Reset the index
    df = df.reset_index(drop=True)

    # Make sure data type is integer
    df.loc[:,'fish'] = df.loc[:,'fish'].astype(int)

    return df


def load_activity(fname, genotype_fname, lights_on, lights_off, day_in_the_life,
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
        - exp_time: Experimental time, based on the `start` column of
            the inputted data file
        - exp_ind: an index for the experimental time. Because of some
          errors in the acquisition, sometimes the times do not
          perfectly line up. exp_ind is just the index of the
          measurement. This is needed for computing averages over
          fish at each time point.
        - light: True if the light is on.
        - day: The day in the life of the fish
    """

    # Convert lights_on and lights_off to datetime.time objects
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

    # Get experimental time in units of hours
    df['exp_time'] = df['start'] / 3600

    # Determine light or dark
    clock = pd.DatetimeIndex(df['time']).time
    df['light'] = np.logical_and(clock >= lights_on, clock < lights_off)

    # Which day it is (remember, day goes lights on to lights on)
    df['day'] = pd.DatetimeIndex(
        df['time'] - datetime.datetime.combine(t_min.date(), lights_on)).day \
                + day_in_the_life - 1

    # Sort by fish and exp_time
    df = df.sort_values(by=['fish', 'exp_time']).reset_index(drop=True)

    # Set up exp_time indices
    for fish in df['fish'].unique():
        df.loc[df['fish']==fish, 'exp_ind'] = np.arange(
                                                    np.sum(df['fish']==fish))
    df['exp_ind'] = df['exp_ind'].astype(int)

    # Return everything if we don't want to delete anything
    if 'sttime' not in extra_cols:
        usecols.remove('sttime')
    if 'stdate' not in extra_cols:
        usecols.remove('stdate')
    if 'start' not in extra_cols:
        usecols.remove('start')
    usecols.remove('location')

    cols = usecols + ['time', 'fish', 'genotype', 'exp_time', 'exp_ind',
                      'light', 'day']
    df = df[cols]

    # Rename columns
    if rename is not None:
        df = df.rename(columns=rename)

    return df


def load_perl_processed_activity(fname, genotype_fname, lights_off=14.0,
                                 day_in_the_life=5):
    """
    Load activity data into tidy DataFrame from Prober lab Perl script.

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
    lights_off : float, default 14.0
        The time where lights come on each day according to the
        Zeitgeber time, in units of hours.
    day_in_the_life : int
        The day in the life of the embryos when data acquisition
        started.

    Returns
    -------
    df : pandas DataFrame
        Tidy DataFrame with columns:
        - activity: The activity as given by the instrument, based
          on the `middur` columns of the inputted data set
        - fish: ID of the fish
        - genotype: genotype of the fish
        - exp_time: Experimental time, based on the start of the
          experiment.
        - exp_ind: an index for the experimental time. Because of some
          errors in the acquisition, sometimes the times do not
          perfectly line up. exp_ind is just the index of the
          measurement. This is needed for computing averages over
          fish at each time point.
        - zeit : Zeitgeber time
        - light: True if the light is on.
        - day: The day in the life of the fish
    """

    if lights_off < 0 or lights_off >= 24:
        raise RuntimeError('Invalid lights_off.')

    # Load the genotypes file
    df_gt = load_gtype(genotype_fname)

    # Load in the data set
    df = pd.read_csv(fname, delimiter='\t', comment='#', header=[0, 1])

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
    df['light'] = pd.Series(df.CLOCK < lights_off, index=df.index)

    # Find where the lights switch from off to on.
    dark_to_light = np.where(np.diff(df['light'].astype(np.int)) == 1)[0]

    # Initialize array with day numbers
    day = np.zeros_like(df['light'], dtype=np.int)

    # Loop through transitions to set day numbers
    for i in range(len(dark_to_light) - 1):
        day[dark_to_light[i]+1:dark_to_light[i+1]+1] = i + 1
    day[dark_to_light[-1]+1:] = len(dark_to_light)

    # Insert the day numnber into DataFrame
    df['day'] = pd.Series(day, index=df.index) + day_in_the_life

    # Build exp_time and put it in the DataFrame
    exp_time = 24.0 * df['day'] + df['CLOCK'] - df['CLOCK'][0]
    df['exp_time'] = pd.Series(exp_time, index=df.index)

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

    # Rename CLOCK to zeit
    df = df.rename(columns={'CLOCK': 'zeit'})

    # Set up exp_time indices
    for fish in df['fish'].unique():
        df.loc[df['fish']==fish, 'exp_ind'] = np.arange(
                                                    np.sum(df['fish']==fish))
    df['exp_ind'] = df['exp_ind'].astype(int)

    return df


@numba.jit(nopython=True)
def _resample_array(x, ind_win):
    """
    Resample a NumPy array.

    Parameters
    ----------
    x : ndarray
        Array to resample with summing.
    ind_win : int
        Width of window to de resampling.

    Returns
    -------
    output : ndarray
        resampled array.
    """
    if len(x) == 0:
        raise RuntimeError('`x` must be nonempty.')

    # If the resampling window exceeds length of the array
    if len(x) < ind_win:
        return np.array([np.mean(x) * ind_win])

    # Perform the resampling
    if len(x) % ind_win == 0:
        re_x = np.empty(len(x) // ind_win)
        for i in range(len(x) // ind_win):
            re_x[i] = np.sum(x[i*ind_win:(i+1)*ind_win])
    else:
        re_x = np.empty(len(x) // ind_win + 1)
        for i in range(len(x) // ind_win):
            re_x[i] = np.sum(x[i*ind_win:(i+1)*ind_win])
        re_x[-1] = np.mean(x[ind_win*(len(x)//ind_win):]) * ind_win

    return re_x


def _resample_segment(df, ind_win, signal):
    """
    Resample a single sorted sequential time course of a DataFrame.
    """
    # Convert signal to list
    if type(signal) not in [list, tuple]:
        signal = [signal]

    # Make DataFrame to hold resampled data
    inds = df.index[::ind_win]
    cols = [col for col in df.columns if col not in signal]
    re_df = df.loc[inds, cols].reset_index(drop=True)

    # Resample signal
    for col in signal:
        re_df[col] = _resample_array(df[col].values, ind_win)

    return re_df



def resample(df, ind_win, signal=['activity'], quiet=False):
    """
    Resample the DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with pertinent data. Must have columns 'time',
        'fish', 'genotype', 'day', 'light', 'exp_time'.
    ind_win : int
        Window for resampling, in units of indices.
    signal : list
        List of columns in the DataFrame to resample. These are
        the signals, e.g., ['activity', 'midct'], to resample.
    quiet : bool, default False
        If True, status output to the screen is silenced.

    Returns
    -------
    output : pandas DataFrame
        Resampled DataFrame.

    Notes
    -----
    .. Assumes that the signal is aligned with the
       *left* of the time interval. I.e., if df['exp_time'] = [0, 1, 2],
       the values of df['activity'] are assumed to be aggregated over
       time intervals 0 to 1, 1 to 2, and 2 to 3. The same is true
       for the outputted resampled array.
    """
    # Make a copy so as to leave original unperturbed
    df_in = df.copy()

    # Sort the DataFrame by fish and then exp_time
    df_in = df_in.sort_values(by=['fish', 'exp_time']).reset_index(drop=True)

    # If no resampling is necessary
    if ind_win == 1:
        return df_in

    # Set up output DataFrame
    df_out = pd.DataFrame(columns=df_in.columns)

    if not quiet:
        print('Performing resampling....')
        try:
            iterator = tqdm.tqdm(df_in['fish'].unique())
        except:
            iterator = df_in['fish'].unique()
    else:
        iterator = df_in['fish'].unique()

    for fish in iterator:
        # Slice out entry for fish
        df_fish = df_in.loc[
                    df_in['fish']==fish, :].copy().reset_index(drop=True)

        # Find indices where light switches
        df_fish['switch'] = df_fish['light'].diff()
        df_fish.loc[df_fish.index[0], 'switch'] = True
        inds = df_fish.where(df_fish['switch']).dropna().index

        # Resample data for each segment
        for i, ind in enumerate(inds[:-1]):
            new_df = _resample_segment(
                            df_fish.loc[ind:inds[i+1]-1, :], ind_win, signal)
            new_df = new_df.drop('switch', 1)
            df_out = df_out.append(new_df, ignore_index=True)
        new_df = _resample_segment(df_fish.loc[inds[-1]:, :], ind_win, signal)
        new_df = new_df.drop('switch', 1)
        df_out = df_out.append(new_df, ignore_index=True)

    # Make sure the data types are ok
    for col in df_in:
        df_out[col] = df_out[col].astype(df_in[col].dtype)

    return df_out
