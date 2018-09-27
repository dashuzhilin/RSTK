# coding=utf-8

import os
from collections import defaultdict
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

class CrossValidation(object):
    def __init__(self, source, model, targets, n_folds=5, sep='\t',
                 write_sep='\t',  write_mode='w', metrics=None,
                 random_seed=None, as_binary=False, binary_col=2):

        self.source = source
        self.model = model
        self.targets = targets
        self.n_folds = n_folds
        self.sep = sep
        self.metrics = metrics
        self.random_seed = random_seed

        # write data into folds
        self.write_sep = write_sep
        self.write_mode = write_mode

        # internal vars
        self.results = defaultdict(list)

        # process data
        self.as_binary = as_binary
        self.binary_col = binary_col

    def compute(self):
        print("Data source: %s \nAlgorithm name: %s \nNumber of folds: %d\n" % (self.source,
                                                                                  self.model.model_name,
                                                                                  self.n_folds))

        # check the validation of input data file
        try:
            open(self.source)
        except TypeError:
            raise TypeError("File cannot be empty or file is invalid: " + str(self.source))

        # read with pandas
        df = pd.read_csv(self.source, sep=self.sep, header=None)
        # set value to 1 by row
        if self.as_binary:
            df.iloc[:, self.binary_col] = 1
        # sort by user id and item id
        self.df = df.sort_values(by=[0, 1])

        # create folds
        if self.targets is not None:
            self.targets += "folds/"
            if not os.path.exists(self.targets):
                os.mkdir(self.targets)

            for n in range(self.n_folds):
                if not os.path.exists(self.targets + str(n)):
                    os.mkdir(self.targets + str(n))

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        trained_model = list(kfold.split(self.df))

        # write to files
        if self.targets is not None:
            fold = 0
            for train_index, test_index in trained_model:
                if self.targets is not None:
                    target_train = self.targets + str(fold) + '/train.dat'
                    target_test = self.targets + str(fold) + '/test.dat'

                    df_train = self.df.ix[train_index]
                    df_test = self.df.ix[test_index]

                    df_train.sort_values(by=[0, 1]).to_csv(target_train, sep=self.write_sep, mode=self.write_mode,
                                                           header=None, index=False)
                    df_test.sort_values(by=[0, 1]).to_csv(target_test, sep=self.write_sep, mode=self.write_mode,
                                                          header=None, index=False)
                    fold += 1

        # validate each fold and get metrics
        for k in range(self.n_folds):
            train_file = self.targets + '%d/train.dat' % k
            test_file = self.targets + '%d/test.dat' % k

            self.model.train_file = train_file
            self.model.test_file = test_file

            self.model.run()

            if self.metrics is None:
                self.metrics = self.model.evaluation.keys()

            for metric in self.metrics:
                self.results[metric.upper()].append(self.model.evaluation[metric.upper()])

        # compute mean and standard deviation
        mean = defaultdict(dict)
        std = defaultdict(dict)

        for metric in self.metrics:
            mean[metric.upper()] = np.mean(self.results[metric.upper()])
            std[metric.upper()] = np.std(self.results[metric.upper()])

        evaluation_mean = 'Mean: '
        evaluation_std = 'STD: '
        for metrics in self.metrics:
            evaluation_mean += "%s: %.6f " % (metrics.upper(), mean[metrics.upper()])
            evaluation_std += "%s: %.6f " % (metrics.upper(), std[metrics.upper()])
        print(evaluation_mean)
        print(evaluation_std)


