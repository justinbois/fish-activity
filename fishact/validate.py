import numpy as np
import pandas as pd

from . import parse

def test_genotype_file(fname):

        df = data_parser.load_gtype(fname)

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
