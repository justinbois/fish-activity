import csv
import datetime
import os
import warnings

try:
    import tqdm
except:
    pass

import numpy as np
import pandas as pd
import numba


def _sniff_file_info(fname, comment='#', check_header=True, quiet=False):
    """
    Infer number of header rows and delimiter of a file.

    Parameters
    ----------
    fname : string
        CSV file containing the genotype information.
    comment : string, default '#'
        Character that starts a comment row.
    check_header : bool, default True
        If True, check number of header rows, assuming a row
        that begins with a non-digit character is header.
    quiet : bool, default False
        If True, suppress output to screen.

    Returns
    -------
    n_header : int or None
        Number of header rows. None is retured if `check_header`
        is False.
    delimiter : str
        Inferred delimiter
    line : str
        The first line of data in the file.

    Notes
    -----
    .. Valid delimiters are: ['\t', ',', ';', '|', ' ']
    """

    valid_delimiters = ['\t', ',', ';', '|', ' ']

    with open(fname, 'r') as f:
        # Read through comments
        line = f.readline()
        while line != '' and line[0] == comment:
            line = f.readline()

        # Read through header, counting rows
        if check_header:
            n_header = 0
            while line != '' and (not line[0].isdigit()):
                line = f.readline()
                n_header += 1
        else:
            n_header = None

        if line == '':
            delimiter = None
            if not quiet:
                print('Unable to determine delimiter, returning None')
        else:
            # If no tab, comma, ;, |, or space, assume single entry per column
            if not any(d in line for d in valid_delimiters):
                delimiter = None
                if not quiet:
                    print('Unable to determine delimiter, returning None')
            else:
                delimiter = csv.Sniffer().sniff(line).delimiter

    # Return number of header rows and delimiter
    return n_header, delimiter, line


def tidy_data(fname, genotype_fname, out_fname, lights_on='9:00:00',
              lights_off='23:00:00', day_in_the_life=4,
              wake_threshold=0.1, extra_cols=[],
              rename={'middur': 'activity'}, comment='#',
              gtype_double_header=None, gtype_rstrip=False, resample_win=1,
              resample_signal=['activity', 'sleep']):
    """
    Load in activity data and write tidy data file with possibly 
    resampled data.

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
    out_fname : string
        Name of file to write tidy DataFrame to.
    lights_on : string or datetime.time instance, default '9:00:00'
        The time where lights come on each day, e.g., '9:00:00'.
    lights_off: string or datetime.time, or None, default '23:00:00'
        The time where lights go off each day, e.g., '23:00:00'.
        If None, the 'light' column is all True, meaning we are not
        keeping track of lighting.
    day_in_the_life : int, default 4
        The day in the life of the embryos when data acquisition
        started.
    wake_threshold : float, default 0.1
        Threshold number of seconds per minute that the fish moved
        to be considered awake.
    extra_cols : list, default []
        List of extra columns to keep from the input file, e.g.,
        ['frect', 'fredur']. By default, only time, fish ID, and
        activity as measured by 'middur' is kept.
    rename : dict, default {'middur': 'activity'}
        Dictionary for renaming column headings.
    comment : string, default '#'
        Test that begins and comment line in the file
    gtype_double_header : bool or None, default None
        If True, the file has a two-line header. The first line
        is ignored and the second is kept as a header, possibly
        with stripping using the `rstrip` argument. If False, assume
        a single header row. If None, infer the header, giving a
        warning if a double header is inferred.
    gtype_rstrip : bool, default True
        If True, strip out all text in genotype name to the right of
        the last space. This is because the genotype files typically
        have headers like 'wt (n=22)', and the '(n=22)' is useless.
    resample_win : int, default 1
        Size of resampling window in units of indices.
    resample_signal : list, default ['activity', 'sleep']
        Which signals to resample.

    Notes
    -----
    .. Writes a tidy data set with columns:
        - activity: The activity as given by the instrument, based
          on the `middur` columns of the inputted data set. This
          column may be called 'middur' depending on the `rename`
          kwarg.
        - time: time in proper datetime format, based on the `sttime`
          column of the inputted data file
        - sleep : 1 if fish is asleep (activity = 0), and 0 otherwise.
          This is convenient for computing sleep when resampling.
        - location: ID of the location of the animal. This is often
          renamed to `fish`, but not by default.
        - genotype: genotype of the fish
        - zeit: The Zeitgeber time, based off of the clock time, not
          the experimental time. Zeitgeber time zero is specified with 
          the `zeitgeber_0` kwarg, or alternatively with the 
          `zeitgeber_0_day` and `zeitgeber_0_time` kwargs.
        - zeit_ind: Index of the measured Zeitgeber time. Because of 
          some errors in the acquisition, sometimes the times do not
          perfectly line up. This is needed for computing averages over
          locations at each time point.
        - exp_time: Experimental time, based on the `start` column of
          the inputted data file
        - exp_ind: an index for the experimental time. Because of some
          errors in the acquisition, sometimes the times do not
          perfectly line up. exp_ind is just the index of the
          measurement. This is needed for computing averages over
          fish at each time point.
        - acquisition: Number associated with which acquisition the data
          are coming from. If the experimenter restarts acquisition,
          this number would change.
        - light: True if the light is on.
        - day: The day in the life of the fish. The day begins with
          `lights_on`.
    """
    if out_fname in [fname, genotype_fname]:
        raise RuntimeError('Cowardly refusing to overwrite input file.')

    if os.path.isfile(out_fname):
        raise RuntimeError(out_fname 
                + ' already exists, cowardly refusing to overwrite.')

    df = load_activity(
        fname, genotype_fname, lights_on=lights_on,
        lights_off=lights_off, day_in_the_life=day_in_the_life,
        wake_threshold=wake_threshold, extra_cols=extra_cols,
        rename=rename, comment=comment, 
        gtype_double_header=gtype_double_header, gtype_rstrip=gtype_rstrip)

    if 'location' in rename:
        loc_name = rename['location']
    else:
        loc_name = 'location'

    df = resample(df, resample_win, signal=resample_signal, loc_name=loc_name)
    df.to_csv(out_fname, index=False)
    return None


def load_gtype(fname, comment='#', double_header=None, rstrip=False,
               quiet=False):
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
    comment : string, default '#'
        Test that begins and comment line in the file
    double_header : bool or None, default None
        If True, the file has a two-line header. The first line
        is ignored and the second is kept as a header, possibly
        with stripping using the `rstrip` argument. If False, assume
        a single header row. If None, infer the header, giving a
        warning if a double header is inferred.
    rstrip : bool, default True
        If True, strip out all text in genotype name to the right of
        the last space. This is because the genotype files typically
        have headers like 'wt (n=22)', and the '(n=22)' is useless.
    quiet : bool, default False
        If True, suppress output to screen.

    Returns
    -------
    df : pandas DataFrame
        Tidy DataFrame with columns:
        - location: ID of location
        - genotype: genotype of animal at location
    """

    # Sniff file info
    n_header, delimiter, _ = _sniff_file_info(fname, check_header=True,
                                              comment=comment, quiet=True)
    if double_header is None:
        if n_header == 2:
            double_header = True
            if not quiet:
                warnings.warn('Inferring two header rows.', RuntimeWarning)

    if double_header:
        df = pd.read_csv(fname, comment=comment, header=[0, 1],
                         delimiter=delimiter)

        # Reset the columns to be the second level of indexing
        df.columns = df.columns.get_level_values(1)
    else:
        df = pd.read_csv(fname, comment=comment, delimiter=delimiter)

    # Only keep genotype up to last space because sometimes has n
    if rstrip:
        df.columns = [col[:col.rfind(' ')] if col.rfind(' ') > 0 else col
                            for col in df.columns]

    # Melt the DataFrame
    df = pd.melt(df, var_name='genotype', value_name='location').dropna()

    # Reset the index
    df = df.reset_index(drop=True)

    # Make sure data type is integer
    df.loc[:,'location'] = df.loc[:,'location'].astype(int)

    return df


def load_activity(fname, genotype_fname, instrument=-9999, trial=-9999,
                  lights_on='9:00:00', lights_off='23:00:00', 
                  day_in_the_life=4, zeitgeber_0=None, zeitgeber_0_day=5,
                  zeitgeber_0_time=None, wake_threshold=0.1, extra_cols=[],
                  rename={'middur': 'activity'}, comment='#',
                  gtype_double_header=None, gtype_rstrip=False):
    """
    Load in activity CSV file to tidy DateFrame

    Parameters
    ----------
    fname : str, or list or tuple or strings
        If a string, the CSV file containing the activity data. This is
        a conversion to CSV of the Excel file that comes off the
        instrument. If a list or tuple, each entry contains a CSV file 
        for a single experiment. The data in these files are stitched
        together.
    genotype_fname : str
        File containing genotype information. This is in standard
        Prober lab format, with tab delimited file.
        - First row discarded
        - Second row contains genotypes. String is only kept up to
          the last space because they typically appear something like
          'tph2-/- (n=20)', and we do not need the ' (n=20)'.
        - Subsequent rows containing wells in the 96 well plate
          corresponding to each genotype.
    instrument: str of int
        Name of instrument used to make measurement. If not specified,
        the 'instrument' column is populated with -9999 as a 
        placeholder.
    trial: str or int, default -9999
        Trial number of measurement. If not specified, the 'trial'
        column is populated with -9999 as a placeholder.
    lights_on : string or datetime.time instance, default '9:00:00'
        The time where lights come on each day, e.g., '9:00:00'.
    lights_off: string or datetime.time, or None, default '23:00:00'
        The time where lights go off each day, e.g., '23:00:00'.
        If None, the 'light' column is all True, meaning we are not
        keeping track of lighting.
    day_in_the_life : int, default 4
        The day in the life of the embryos when data acquisition
        started.
    zeitgeber_0 : datetime instance, default None
        If not None, gives the date and time of Zeitgeber time zero. 
    zeitgeber_0_day : int, default 5
        The day in the life of the embryos where Zeitgeber time zero is.
        Ignored if `zeitgeber_0` is not None.
    zeigeber_0_time : str, or None (default)
        String representing time of Zeitgeber time zero. If None,
        defaults to the value of `lights_on`. Ignored if `zeitgeber_0` 
        is not None.
    wake_threshold : float, default 0.1
        Threshold number of seconds per minute that the fish moved
        to be considered awake.
    extra_cols : list, default []
        List of extra columns to keep from the input file, e.g.,
        ['frect', 'fredur']. By default, only time, fish ID, and
        activity as measured by 'middur' is kept.
    rename : dict, default {'middur': 'activity', 'location': 'fish'}
        Dictionary for renaming column headings.
    comment : string, default '#'
        Test that begins and comment line in the file
    gtype_double_header : bool or None, default None
        If True, the file has a two-line header. The first line
        is ignored and the second is kept as a header, possibly
        with stripping using the `rstrip` argument. If False, assume
        a single header row. If None, infer the header, giving a
        warning if a double header is inferred.
    gtype_rstrip : bool, default True
        If True, strip out all text in genotype name to the right of
        the last space. This is because the genotype files typically
        have headers like 'wt (n=22)', and the '(n=22)' is useless.

    Returns
    -------
    df : pandas DataFrame
        Tidy DataFrame with columns:
        - activity: The activity as given by the instrument, based
          on the `middur` columns of the inputted data set. This
          column may be called 'middur' depending on the `rename`
          kwarg.
        - time: time in proper datetime format, based on the `sttime`
          column of the inputted data file
        - sleep : 1 if fish is asleep (activity = 0), and 0 otherwise.
          This is convenient for computing sleep when resampling.
        - location: ID of the location of the animal. This is often
          renamed to `fish`, but not by default.
        - genotype: genotype of the fish
        - zeit: The Zeitgeber time, based off of the clock time, not
          the experimental time. Zeitgeber time zero is specified with 
          the `zeitgeber_0` kwarg, or alternatively with the 
          `zeitgeber_0_day` and `zeitgeber_0_time` kwargs.
        - zeit_ind: Index of the measured Zeitgeber time. Because of 
          some errors in the acquisition, sometimes the times do not
          perfectly line up. This is needed for computing averages over
          locations at each time point.
        - exp_time: Experimental time, based on the `start` column of
          the inputted data file
        - exp_ind: an index for the experimental time. Because of some
          errors in the acquisition, sometimes the times do not
          perfectly line up. exp_ind is just the index of the
          measurement. This is needed for computing averages over
          fish at each time point.
        - acquisition: Number associated with which acquisition the data
          are coming from. If the experimenter restarts acquisition,
          this number would change.
        - instrument: Name of instrument used to acquire the data. If no
          instrument is known, this is populated with NaNs.
        - trial: Name of trial of data acquisition.  If no trial is 
          known, this is populated with NaNs.
        - light: True if the light is on.
        - day: The day in the life of the fish. The day begins with
          `lights_on`.

    Notes
    -----
    .. If `lights_off` is `None`, this means we ignore the lighting,
       but we still want to know what day it is. Specification of
       `lights_on` says what wall clock time specifies the start of
       a day.
    """


    # Get genotype information
    df_gt = load_gtype(genotype_fname, comment=comment,
                       double_header=gtype_double_header, rstrip=gtype_rstrip)

    # Read in DataFrames
    if type(fname) == str:
        fname = [fname]

    # Read in DataFrames
    df = pd.concat([_load_single_activity_file(
                        filename, 
                        df_gt,
                        extra_cols=extra_cols, 
                        comment=comment,
                        acquisition=ac+1)
                    for ac, filename in enumerate(fname)])

    # Columns to use
    usecols = list(df.columns)

    # Sort by location and then time
    df = df.sort_values(['location', 'time']).reset_index(drop=True)

    # Convert lights_on to datetime
    if type(lights_on) != datetime.time:
        lights_on = pd.to_datetime(lights_on).time()
    if type(lights_off) != datetime.time and lights_off is not None:
        lights_off = pd.to_datetime(lights_off).time()

    # Convert zeitgeber_0 to datetime object
    if zeitgeber_0 is not None and type(zeitgeber_0) == str:
        zeitgeber_0 = pd.to_datetime(zeitgeber_0)

    # Determine light or dark
    if lights_off is None:
        df['light'] = [True] * len(df)
    else:
        clock = pd.DatetimeIndex(df['time']).time
        df['light'] = np.logical_and(clock >= lights_on, clock < lights_off)

    # Get earliest time point
    t_min = pd.DatetimeIndex(df['time']).min()

    # Which day it is (day goes lights on to lights on)
    df['day'] = pd.DatetimeIndex(
        df['time'] - datetime.datetime.combine(t_min.date(), lights_on)).day \
                + day_in_the_life - 1

    # Compute zeitgeber_0
    if zeitgeber_0 is None:
        times = df.loc[
            (df['day']==zeitgeber_0_day) & (df['light'] == True), 
            'time']
        if len(times) == 0:
            raise RuntimeError(
                    'Unable to find Zeitgeber_0. Check `day_in_the_life` and '
                  + 'zeitgeber_0_day` inputs.')
        zeit_date = times.min().date()
        zeitgeber_0 = pd.to_datetime(str(zeit_date) + ' ' + str(lights_on))

    # Add Zeitgeber time
    df['zeit'] = (df['time'] - zeitgeber_0).dt.total_seconds() / 3600

    # Set up exp_time indices
    for loc in df['location'].unique():
        df.loc[df['location']==loc, 'exp_ind'] = np.arange(
                                            np.sum(df['location']==loc))
    df['exp_ind'] = df['exp_ind'].astype(int)

    # Infer time interval in units of hours (almost always 1/60)
    dt = np.diff(df.loc[df['location'] == df['location'].unique()[0], 'time'])
    dt = np.median(dt.astype(float) / 3600e9)

    # Add zeit indices
    df['zeit_ind'] = (np.round(df['zeit'] / dt)).astype(int)

    # Only use columns we want
    if 'sttime' not in extra_cols:
        usecols.remove('sttime')
    if 'stdate' not in extra_cols:
        usecols.remove('stdate')
    if 'start' not in extra_cols:
        usecols.remove('start')

    cols = usecols + ['zeit', 'zeit_ind', 'exp_ind', 'light', 'day']
    df = df[cols]

    # Compute sleep
    df['sleep'] = (df['middur'] < wake_threshold).astype(int)

    # Get experimental time in units of hours (DEPRECATED)
    # df['exp_time'] = df['start'] / 3600

    # Rename columns
    if rename is not None:
        df = df.rename(columns=rename)

    # Fill in trial and instrument information
    df['instrument'] = [instrument] * len(df)
    df['trial'] = [trial] * len(df)

    return df


def merge_experiments(dfs, instrument_trial=None):
    """
    Merge DataFrames from multiple experiments into one.

    Parameters
    ----------
    dfs : list of DataFrames
        Each DataFrame is as would be returned by load_activity() and
        has columns
        - activity: The activity as given by the instrument, based
          on the `middur` columns of the inputted data set. This
          column may be called 'middur' depending on the `rename`
          kwarg.
        - time: time in proper datetime format, based on the `sttime`
          column of the inputted data file
        - sleep : 1 if fish is asleep (activity = 0), and 0 otherwise.
          This is convenient for computing sleep when resampling.
        - location: ID of the location of the animal. This is often
          renamed to `fish`, but not by default.
        - genotype: genotype of the fish
        - zeit: The Zeitgeber time, based off of the clock time, not
          the experimental time. Zeitgeber time zero is specified with 
          the `zeitgeber_0` kwarg, or alternatively with the 
          `zeitgeber_0_day` and `zeitgeber_0_time` kwargs.
        - zeit_ind: Index of the measured Zeitgeber time. Because of 
          some errors in the acquisition, sometimes the times do not
          perfectly line up. This is needed for computing averages over
          locations at each time point.
        - exp_time: Experimental time, based on the `start` column of
          the inputted data file
        - exp_ind: an index for the experimental time. Because of some
          errors in the acquisition, sometimes the times do not
          perfectly line up. exp_ind is just the index of the
          measurement. This is needed for computing averages over
          fish at each time point.
        - acquisition: Number associated with which acquisition the data
          are coming from. If the experimenter restarts acquisition,
          this number would change.
        - instrument: Name of instrument used to acquire the data. If no
          instrument is known, this is populated with NaNs.
        - trial: Name of trial of data acquisition.  If no trial is 
          known, this is populated with NaNs.
        - light: True if the light is on.
        - day: The day in the life of the fish. The day begins with
          `lights_on`.
    instrument_trial : list of 2-tuples
        A list containing the name (or number) of instrument and trial 
        used in acquiring the respective DataFrames. The length of the
        list must be the same as the length of `dfs`. If None, the 
        instrument and trial columns are taken from the inputted 
        DataFrames. If a given entry is None, then values from
        the `instrument` and `trial` columns of the inputted DataFrame 
        are used. In this case, *all* inputted DataFrames must have
        `instrument` and `trial` columns fully populated with proper
        values (not NaN's nor the placeholder -9999).

    Notes
    -----
    .. This will give unexpected results if the sampling frequency of
       the data sets is different at all.
    """

    if instrument_trial is not None and len(instrument_trial) != len(dfs):
        raise RuntimeError('Must have same number of entries in '
                                + '`instrument_trial` as in `dfs`.')

        if len(instrument_trial) != len(set(instrument_trial)):
            raise RuntimeError('instrument, trial pairs are not all unique.')

    # Make sure all DataFrames have the same columns
    cols = list(sorted(dfs[0].columns))
    if len(dfs) > 1:
        for df in dfs[1:]:
            if not np.all(list(sorted(df.columns)) == cols):
                raise RuntimeError(
                            'Not all DataFrames have the same columns.')

    # Make sure all instrument/trial pairs are unique
    if instrument_trial is None:
        inst_trial_list = []
        for df in dfs:
            inst_trial_list += instrument_trial_pairs(df)
        if (-9999, -9999) in inst_trial_list:
            raise RuntimeError('Not all DataFrames have populated '
                                + 'instrument and trial columns.')
        if len(inst_trial_list) != len(set(inst_trial_list)):
            raise RuntimeError(
                'Nonunique instrument/trial pairs in inputted DataFrames.')

    # Make outputted DataFrame
    df_out = pd.DataFrame(columns=cols)

    # Populate new DataFrame
    for i, df in enumerate(dfs):
        if instrument_trial is not None:
            if not (df['instrument'] == -9999).all():
                warning.warn('Overwriting instrument and trial columns.')
            df['instrument'] = [instrument_trial[i][0]] * len(df)
            df['trial'] = [instrument_trial[i][1]] * len(df)
        df_out = df_out.append(df, ignore_index=True)

    # Sort the DataFrame
    df_out = df_out.sort_values(by=['instrument', 'trial', 'zeit'])

    return df_out


def _load_single_activity_file(
        fname, 
        df_gt,
        extra_cols=[],
        comment='#',
        acquisition=1):
    """
    Load in activity CSV file to tidy DateFrame

    Parameters
    ----------
    fname : string
        The CSV file containing the activity data. This is
        a conversion to CSV of the Excel file that comes off the
        instrument.
    df : pandas DataFrame
        Tidy DataFrame with columns:
        - location: ID of location
        - genotype: genotype of of animal at location
    extra_cols : list, default []
        List of extra columns to keep from the input file, e.g.,
        ['frect', 'fredur']. By default, only time, location ID, and
        activity as measured by 'middur' is kept.
    comment : string, default '#'
        Test that begins and comment line in the file

    Returns
    -------
    df : pandas DataFrame
        Tidy DataFrame with columns:
        - activity: The activity as given by the instrument, based
          on the `middur` columns of the inputted data set. This
          column may be called 'middur' depending on the `rename`
          kwarg.
        - time: time in proper datetime format, based on the `sttime`
          column of the inputted data file
        - location: ID of the organism
        - acquisition: Number associated with which acquisition the data
          are coming from. If the experimenter restarts acquisition,
          this number would change.
        - genotype: genotype of the animal in the location

    Notes
    -----
    .. If `lights_off` is `None`, this means we ignore the lighting,
       but we still want to know what day it is. Specification of
       `lights_on` says what wall clock time specifies the start of
       a day.
    """

    # Determine which columns to read in
    if extra_cols is None:
        extra_cols = []
    cols = ['start', 'location', 'stdate', 'sttime', 'middur']
    new_cols = list(set(extra_cols) - set(cols))
    usecols = cols + new_cols

    # Sniff out the delimiter, see how many headers, check file not empty
    _, delimiter, _ = _sniff_file_info(fname, check_header=False,
                                       comment=comment, quiet=True)

    # Read file
    df = pd.read_csv(fname, usecols=usecols, comment=comment,
                     delimiter=delimiter)

    # Detect if it's the new file format, and the convert fish to integer
    if '-' in df['location'].iloc[0]:
        df['location'] = (df['location']
                            .apply(lambda x: x[x.rfind('-')+1:])
                            .astype(int))
    else:
        df['location'] = (df['location']
                            .str.extract('(\d+)', expand=False)
                            .astype(int))

    # Only keep fish that we have genotypes for
    df = df.loc[df['location'].isin(df_gt['location']), :]

    # Store the genotypes
    loc_lookup = {loc: df_gt.loc[df_gt['location']==loc, 'genotype']
                             .values[0]
                        for loc in df_gt['location']}
    df['genotype'] = df['location'].apply(lambda x: loc_lookup[x])

    # Convert date and time to a time stamp
    df['time'] = pd.to_datetime(df['stdate'] + df['sttime'],
                                format='%d/%m/%Y%H:%M:%S')

    # Add the acquisition number
    df['acquisition'] = acquisition * np.ones(len(df), dtype=int)

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
        Width of window to do resampling.

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


def resample(df, ind_win, signal=['activity', 'sleep'], loc_name='location',
             quiet=False):
    """
    Resample the DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with pertinent data. Must have columns 'time',
        'fish', 'genotype', 'day', 'light', 'zeit'.
    ind_win : int
        Window for resampling, in units of indices.
    signal : list, default ['activity', 'sleep']
        List of columns in the DataFrame to resample. These are
        the signals, e.g., ['activity', 'midct'], to resample.
    loc_name : str
        Name of column containing the "location," i.e., animal location.
        'fish' is a common entry.
    quiet : bool, default False
        If True, status output to the screen is silenced.

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

    # Convert the signal columns to floats
    for sig in signal:
        df_in[sig] = df_in[sig].astype(float)

    # Sort the DataFrame by instrument, trial, location and then zeit
    df_in = ( df_in.sort_values(by=['instrument', 'trial', loc_name, 'zeit'])
                               .reset_index(drop=True))

    # If no resampling is necessary
    if ind_win == 1:
        return df_in

    # Set up output DataFrame
    df_out = pd.DataFrame(columns=df_in.columns)

    # Set up iterator to go through instrument/trial/location triples
    iterator = [
            (r['instrument'], r['trial'], r[loc_name]) 
             for _, r in df_in.groupby(['instrument', 'trial', loc_name])
                               .size()
                               .reset_index()
                               .iterrows()]
    if not quiet:
        print('Performing resampling....')
        try:
            iterator = tqdm.tqdm(iterator)
        except:
            pass

    for inst, trial, loc in iterator:
        # Slice out entry for loc
        inds = (  (df['instrument'] == inst) 
                & (df['trial'] == trial) 
                & (df[loc_name] == loc))
        df_loc = df_in.loc[inds, :].copy().reset_index(drop=True)

        # Find indices where light or acquisition switches
        df_loc['switch'] = (  df_loc['light'].diff().astype(bool) 
                            | df_loc['acquisition'].diff())
        df_loc.loc[df_loc.index[0], 'switch'] = True
        inds = df_loc.where(df_loc['switch']).dropna().index

        # Resample data for each segment
        for i, ind in enumerate(inds[:-1]):
            new_df = _resample_segment(
                            df_loc.loc[ind:inds[i+1]-1, :], ind_win, signal)
            new_df = new_df.drop('switch', 1)
            df_out = df_out.append(new_df, ignore_index=True)
        new_df = _resample_segment(df_loc.loc[inds[-1]:, :], ind_win, signal)
        new_df = new_df.drop('switch', 1)
        df_out = df_out.append(new_df, ignore_index=True)

    # Make sure the data types are ok
    for col in df_in:
        df_out[col] = df_out[col].astype(df_in[col].dtype)

    return df_out


def instrument_trial_pairs(df):
    """
    Extract a list of all unique instrument/trial pairs.
    """
    df_iter = df.groupby(['instrument', 'trial']).size().reset_index()
    return [(r['instrument'], r['trial']) for _, r in df_iter.iterrows()]


def load_perl_processed_activity(fname, genotype_fname, lights_off=14.0,
                                 wake_threshold=0.1, day_in_the_life=4):
    """
    Load activity data into tidy DataFrame from Prober lab Perl script.
    This is not actively maintained and is likely to be deprecated.

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
    wake_threshold : float, default 0.1
        Threshold number of seconds per minute that the fish moved
        to be considered awake.
    day_in_the_life : int
        The day in the life of the embryos when data acquisition
        started.

    Returns
    -------
    df : pandas DataFrame
        Tidy DataFrame with columns:
        - activity: The activity as given by the instrument, based
          on the `middur` columns of the inputted data set
        - sleep : 1 if fish is asleep (activity = 0), and 0 otherwise.
          This is convenient for computing sleep when resampling.
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

    # Insert the day number into DataFrame
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

    # Compute sleep
    df['sleep'] = (df['activity'] < wake_threshold).astype(int)

    return df
