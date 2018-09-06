import pandas as pd

class Transfer:

    # transfer *.dat to *.csv

    def __init__(self):
        self.origin_path = 'Data/{}'

    def process(self):
        print('Transfer *.dat to *.csv...')
        f = pd.read_table(self.origin_path.format('ratings.dat'), sep='::', engine='python',
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        f.to_csv(self.origin_path.format('ratings.csv'), index=False)
        print('Finished.')