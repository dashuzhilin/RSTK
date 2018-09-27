import pandas as pd
import random

class DataSplit(object):
    # split data for training and testing
    def __init__(self, input_file=None):
        # self.origin_path = 'Data/{}'
        self.input_file = input_file

    def split(self, M, k, seed):

        data = []
        train = []
        test = []

        print('Generate train set and test set...')

        # f = open(self.origin_path.format('ratings.csv'), 'r')
        # f.readline()
        # f = open(self.origin_path.format('u.data'), 'r')
        f = open(self.input_file)
        line = f.readline()
        while line:
            # line_split = line.split(',')
            line_split = line.split('\t')
            data.append((int(line_split[0]), int(line_split[1]), int(line_split[2])))
            line = f.readline()
        f.close()

        random.seed(seed)
        for user, item, rating in data:
            if random.randint(0, M) == k:
                test.append((user, item, rating))
            else:
                train.append((user, item, rating))

        return train, test