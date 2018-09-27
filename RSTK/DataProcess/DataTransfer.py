import pandas as pd

class DataTransfer:

    # transfer *.dat to *.csv

    def __init__(self):
        self.origin_path = 'Data/{}'

    def transfer(self):
        print('Transfer *.dat to *.csv...')
        f = pd.read_table(self.origin_path.format('ratings.dat'), sep='::', engine='python',
                          names=['Us  erID', 'MovieID', 'Rating', 'Timestamp'])
        f.to_csv(self.origin_path.format('ratings.csv'), index=False)
        print('Finished.')