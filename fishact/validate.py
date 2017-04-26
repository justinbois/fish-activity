import csv
import numpy as np
import pandas as pd

from . import parse


def test_genotype_file(fname, quiet=False):
    """
    Validate a genotype input file.

    Parameters
    ----------
    fname : string
        Name of file containing genotype file.
    quiet : bool, default False
        If True, print problems with data set to screen.

    Returns
    -------
    output: bool
        True if genotype file passed test. False otherwise.
    """

    # Keep tabs on number of failures.
    n_fail = 0

    if not quiet:
        print()

    # Sniff out the delimiter, see how many headers, check file not empty
    n_header, delimiter, line = parse._sniff_file_info(
                    fname, check_header=True, comment='#', quiet=False)

    # Check headers
    if n_header >= 3:
        if not quiet:
            print('ERROR: Genotype file possibly uses wrong comment character.')
            print('\n***GENOTYPE VALIDATION FAILED***\n')
        return 1
    elif n_header == 2:
        if not quiet:
            print('Warning: Genotype file probably has two header rows.\n')
        n_fail += 1
        header = [0, 1]
    elif n_header == 0:
        if not quiet:
            print('ERROR: Genotype file is probably missing a header.')
        n_fail += 1
        header = 'infer'
    else:
        header = 'infer'

    # Make sure there is some data
    if line == '':
        if not quiet:
            print('ERROR: Genotype file has no data,\n')
            print('***GENOTYPE VALIDATION FAILED***\n')
        return n_fail + 1

    # Check comma delimiting
    if delimiter != ',':
        print('ERROR: Genotype file is not comma delimited.\n')
        n_fail += 1

    # Read file
    df = pd.read_csv(fname, comment='#', header=header, delimiter=delimiter)

    # Reset the columns to be the second level of indexing
    if header == [0, 1]:
        df.columns = df.columns.get_level_values(1)

    # Make sure there is an omit column
    if 'omit' not in df.columns:
        if not quiet:
            print('ERROR: No `omit` column in genotype file.\n')

    # Make sure columns (genotypes) are unique
    dups = df.columns.get_duplicates()
    if len(dups) > 0:
        if not quiet:
            print('ERROR: Duplicated genotypes:', dups, '\n')
        n_fail += 1

    # Melt the DataFrame
    df = pd.melt(df, var_name='genotype', value_name='fish').dropna()

    # Make sure it is non-empty
    if len(df) == 0:
        if not quiet:
            print('ERROR: No fish are assigned genotypes.\n')
        n_fail += 1

    # Reset the index
    df = df.reset_index(drop=True)

    # Make sure data type is integer
    df.loc[:,'fish'] = df.loc[:,'fish'].astype(int)

    # Check for duplicate fish
    if df['fish'].duplicated().sum() > 0:
        if not quiet:
            print('ERROR: Fish were duplicates:',
                  list(df.loc[df['fish'].duplicated(), 'fish'].unique()))
            print('\n')
        n_fail += 1

    if n_fail > 0:
        if not quiet:
            print('***GENOTYPE VALIDATION FAILED***\n')
    else:
        if not quiet:
            print('Genotype validation passed.\n')

    return n_fail == 0


def test_activity_file(fname, genotype_fname, quiet=False):
    """
    Validate an activity input file.

    Parameters
    ----------
    fname : string
        Name of file containing genotype file.
    quiet : bool, default False
        If True, print problems with data set to screen.

    Returns
    -------
    output: bool
        True if activity file passed test. False otherwise.
    """

    if not quiet:
        print()

    # Keep tabs on number of failures.
    n_fail = 0

    # Sniff out the delimiter, see how many headers, check file not empty
    _, delimiter, line = parse._sniff_file_info(fname, check_header=False,
                                                comment='#', quiet=False)

    if delimiter != ',':
        if not quiet:
            print('ERROR: Activity file is not comma delimited.\n')
        n_fail += 1

    if line == '':
        if not quiet:
            print('ERROR: Activity file has no data.\n')
        n_fail += 1

    # Read in data file
    df = pd.read_csv(fname)

    # Make sure columns are correct
    cols = ['location', 'animal', 'user', 'sn', 'an', 'datatype', 'start',
            'end', 'startreason', 'endreason', 'frect', 'fredur', 'midct',
            'middur', 'burct', 'burdur', 'stdate', 'sttime']
    if len(df.columns) != len(cols):
        if not quiet:
            print('ERROR: Wrong number of columns in DataFrame.\n')
        n_fail += 1

    if set(df.columns) - set(cols) != set():
        if not quiet:
            print('ERROR: Columns present that should not be:',
                  set(df.columns) - set(cols), '\n')
        n_fail += 1

    if set(cols) - set(df.columns) != set():
        if not quiet:
            print('ERROR: Columns absent that should be there:',
                  set(cols) - set(df.columns), '\n')
        n_fail += 1

    # Check for missing data
    total_nan = df.isnull().sum().sum()
    if total_nan > 0:
        if not quiet:
            if total_nan > 100:
                print('ERROR: More than 100 missing data entries.\n')
            else:
                df_null = df.isnull().unstack()
                print('ERROR: Missing data:')
                print(df_null[df_null])
                print('\n')
        n_fail += 1

    # Check to make sure all values that should be nonnegative are
    if (df['start'] < 0).sum() > 0:
        if not quiet:
            print('ERROR: Some negative `start` times.\n')
        n_fail += 1
    if (df['end'] < 0).sum() > 0:
        if not quiet:
            print('ERROR: Some negative `start` times.\n')
        n_fail += 1
    if (df['frect'] < 0).sum() > 0:
        if not quiet:
            print('ERROR: Some negative `frect` values.\n')
        n_fail += 1
    if (df['fredur'] < 0).sum() > 0:
        if not quiet:
            print('ERROR: Some negative `fredur` values.\n')
        n_fail += 1
    if (df['midct'] < 0).sum() > 0:
        if not quiet:
            print('ERROR: Some negative `midct` values.\n')
        n_fail += 1
    if (df['middur'] < 0).sum() > 0:
        if not quiet:
            print('ERROR: Some negative `middur` values.\n')
        n_fail += 1
    if (df['burct'] < 0).sum() > 0:
        if not quiet:
            print('ERROR: Some negative `burct` values.\n')
        n_fail += 1
    if (df['burdur'] < 0).sum() > 0:
        if not quiet:
            print('ERROR: Some negative `burdur` values.\n')
        n_fail += 1

    # Check that start times are sequential
    if (np.diff(df['start'].unique()) < 0).sum():
        if not quiet:
            print('ERROR: Nonsequential `start` values.\n')
        n_fail += 1

    # Make sure all end times greater than their start times
    if (df['end'] <= df['start']).sum() > 0:
        if not quiet:
            print('ERROR: Some `end` times occur before their `start` times.\n')
        n_fail += 1

    # Check that start and stop times of intervals are kosher
    time_int = (df['end'] - df['start']).median()
    good_int = np.isclose(df['end'] - df['start'], time_int)
    if np.sum(good_int) > 0:
        if not quiet:
            df_bad = df.loc[~good_int, ['start', 'end']]
            max_bad = (df_bad['end'] - df_bad['start']).max()
            min_bad = (df_bad['end'] - df_bad['start']).min()
            print('ERROR: Bad time intervals based on `start` and `end`.')
            print('            Standard interval:', time_int)
            print('                 Max interval:', max_bad)
            print('                 Min interval:', min_bad)
            print('      Number of bad intervals:', len(df_bad))
            print('    Fraction of intervals bad:', len(df_bad) / len(df))
            print('\n')
        n_fail += 1

    # Check that clock time matches start times
    df['time'] = pd.to_datetime(df['stdate'] + df['sttime'],
                                format='%d/%m/%Y%H:%M:%S')
    df['time_start'] = pd.to_numeric(df['time'] - df['time'].min()) / 1e9
    t_diff = df['time_start'] - df['start']
    if not np.isclose(t_diff.max(), 0):
        if not quiet:
            print('ERROR: `sttime` and `start` do not match.')
            print('    Maximum `sttime` - `start`:', t_diff.max())
            print('    Minimum `sttime` - `start`:',
                                    t_diff[np.abs(t_diff)>0].min())
            print('       Number of discrepancies:',
                  (~np.isclose(t_diff, 0)).sum())
            print('     Fraction of intervals bad:',
                  (~np.isclose(t_diff, 0)).sum() / len(df))
            print('\n')
        n_fail += 1

    # Count how many unique fish there are
    n_fish = len(df['location'].unique())

    # Convert location to well number (just drop 'c' in front)
    df = df.rename(columns={'location': 'fish'})
    df['fish'] = df['fish'].str.extract('(\d+)', expand=False).astype(int)

    # Make sure all fish are accounted for in genotype file
    try:
        df_g = parse.load_gtype(genotype_fname, quiet=True)
        test_gtypes = True
    except:
        print('ERROR: Cannot open genotype file.\n')
        n_fail += 1
        test_gtypes = False

    if test_gtypes:
        g_set = set(df_g['fish'].unique())
        a_set = set(df['fish'].unique())

        set_diff = a_set - g_set
        if set_diff != set():
            n_fail += 1
            print('ERROR: Fish [ ', end='')
            for x in sorted(list(set_diff)):
                print(x, end=' ')
            print('] in activity file but not in genotype file.')
            if 'omit' in df_g.columns:
                print()
            else:
                print('Possibly due to no `omit` column in genotype file.\n')

        set_diff = g_set - a_set
        if set_diff != set():
            n_fail += 1
            print('ERROR: Fish [ ', end='')
            for x in sorted(list(set_diff)):
                print(x, end=' ')
            print('] in genotype file but not in activity file.\n')

    if n_fail > 0:
        if not quiet:
            print('***ACTIVITY VALIDATION FAILED***\n')
    else:
        if not quiet:
            print('Activity validation passed.\n')

    return n_fail == 0
