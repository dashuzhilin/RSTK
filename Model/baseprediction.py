# coding=utf-8
""""
    This class is base for rating prediction algorithms.

"""

# © 2018. Case Recommender (MIT License)

from scipy.spatial.distance import squareform, pdist
import numpy as np


from Evaluation.ratingprediction import RatingPredictionEvaluation
from Utils.extra_functions import print_header
from DataProcess.processdata import ReadFile, WriteFile

__author__ = 'Arthur Fortes <fortes.arthur@gmail.com>'


class BaseRatingPrediction(object):
    def __init__(self, train_file, test_file, output_file=None, similarity_metric='cosine', sep='\t',
                 output_sep='\t'):
        """
         This class is base for all rating prediction algorithms. Inherits the class Recommender
         and implements / adds common methods and attributes for rating prediction approaches.

        :param train_file: File which contains the train set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type train_file: str

        :param test_file: File which contains the test set. This file needs to have at least 3 columns
        (user item feedback_value).
        :type test_file: str, default None

        :param output_file: File with dir to write the final predictions
        :type output_file: str, default None

        :param similarity_metric:
        :type similarity_metric: str, default cosine

        :param sep: Delimiter for input files
        :type sep: str, default '\t'

        :param output_sep: Delimiter for output file
        :type output_sep: str, default '\t'

        """

        self.train_file = train_file
        self.test_file = test_file
        self.similarity_metric = similarity_metric
        self.output_file = output_file
        self.sep = sep
        self.output_sep = output_sep

        # internal vars
        self.item_to_item_id = {}
        self.item_id_to_item = {}
        self.user_to_user_id = {}
        self.user_id_to_user = {}
        self.train_set = None
        self.test_set = None
        self.users = None
        self.items = None
        self.matrix = None
        self.evaluation_results = None
        self.recommender_name = None
        self.extra_info_header = None
        self.predictions = []

    def read_files(self):
        """
        Method to initialize recommender algorithm.

        """

        self.train_set = ReadFile(self.train_file, sep=self.sep).read()

        if self.test_file is not None:
            self.test_set = ReadFile(self.test_file, sep=self.sep).read()
            self.users = sorted(set(list(self.train_set['users']) + list(self.test_set['users'])))
            self.items = sorted(set(list(self.train_set['items']) + list(self.test_set['items'])))
        else:
            self.users = self.train_set['users']
            self.items = self.train_set['items']

        for i, item in enumerate(self.items):
            self.item_to_item_id.update({item: i})
            self.item_id_to_item.update({i: item})
        for u, user in enumerate(self.users):
            self.user_to_user_id.update({user: u})
            self.user_id_to_user.update({u: user})

    def create_matrix(self):
        """
        Method to create a feedback matrix

        """

        self.matrix = np.zeros((len(self.users), len(self.items)))

        for user in self.train_set['users']:
            for item in self.train_set['feedback'][user]:
                self.matrix[self.user_to_user_id[user]][self.item_to_item_id[item]] = \
                    self.train_set['feedback'][user][item]

    def compute_similarity(self, transpose=False):
        """
        Method to compute a similarity matrix from original df_matrix

        :param transpose: If True, calculate the similarity in a transpose matrix
        :type transpose: bool, default False

        """

        # Calculate distance matrix
        if transpose:
            similarity_matrix = np.float32(squareform(pdist(self.matrix.T, self.similarity_metric)))
        else:
            similarity_matrix = np.float32(squareform(pdist(self.matrix, self.similarity_metric)))

        # Remove NaNs
        similarity_matrix[np.isnan(similarity_matrix)] = 1.0
        # transform distances in similarities. Values in matrix range from 0-1
        similarity_matrix = (similarity_matrix.max() - similarity_matrix) / similarity_matrix.max()

        return similarity_matrix

    def evaluate(self, metrics, verbose=True, as_table=False, table_sep='\t'):
        """
        Method to evaluate the final ranking

        :param metrics: List of evaluation metrics
        :type metrics: list, default ('MAE', 'RMSE')

        :param verbose: Print the evaluation results
        :type verbose: bool, default True

        :param as_table: Print the evaluation results as table
        :type as_table: bool, default False

        :param table_sep: Delimiter for print results (only work with verbose=True and as_table=True)
        :type table_sep: str, default '\t'

        """

        self.evaluation_results = {}

        if metrics is None:
            metrics = list(['MAE', 'RMSE'])

        results = RatingPredictionEvaluation(verbose=verbose, as_table=as_table, table_sep=table_sep, metrics=metrics
                                             ).evaluate_recommender(predictions=self.predictions,
                                                                    test_set=self.test_set)

        for metric in metrics:
            self.evaluation_results[metric.upper()] = results[metric.upper()]

    def write_predictions(self):
        """
        Method to write final ranking

        """

        if self.output_file is not None:
            WriteFile(self.output_file, data=self.predictions, sep=self.sep).write()

    def compute(self, verbose=True):
        """
        Method to run the recommender algorithm

        :param verbose: Print the information about recommender
        :type verbose: bool, default True

        """

        # read files
        self.read_files()

        # initialize empty predictions (Don't remove: important to Cross Validation)
        self.predictions = []

        if verbose:
            test_info = None

            main_info = {
                'title': 'Rating Prediction > ' + self.recommender_name,
                'n_users': len(self.train_set['users']),
                'n_items': len(self.train_set['items']),
                'n_interactions': self.train_set['number_interactions'],
                'sparsity': self.train_set['sparsity']
                    }

            if self.test_file is not None:
                test_info = {
                    'n_users': len(self.test_set['users']),
                    'n_items': len(self.test_set['items']),
                    'n_interactions': self.test_set['number_interactions'],
                    'sparsity': self.test_set['sparsity']
                }

            print_header(main_info, test_info)
