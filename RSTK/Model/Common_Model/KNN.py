import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import defaultdict
import random


class KNN(object):
    def precision_at_k(self, ranking, k):
        assert k >= 1
        ranking = np.asarray(ranking)[:k] != 0
        if ranking.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(ranking)

    def average_precision(self, ranking):
        ranking = np.asarray(ranking) != 0
        out = [self.precision_at_k(ranking, k + 1) for k in range(ranking.size) if ranking[k]]
        if not out:
            return 0.
        return np.mean(out)

    def mean_average_precision(self, ranking):
        return np.mean([self.average_precision(r) for r in ranking])

    def ndcg_at_k(self, ranking):
        ranking = np.asfarray(ranking)
        r_ideal = np.asfarray(sorted(ranking, reverse=True))
        dcg_ideal = r_ideal[0] + np.sum(r_ideal[1:] / np.log2(np.arange(2, r_ideal.size + 1)))
        dcg_ranking = ranking[0] + np.sum(ranking[1:] / np.log2(np.arange(2, ranking.size + 1)))

        return dcg_ranking / dcg_ideal


class UserKNN(KNN):
    def __init__(self, train_file=None, test_file=None, similarity_metric="cosine", k_neighbors=None, reg_bi=10, reg_bu=15,
                 sep='\t', output_sep='\t'):
        self.train_file = train_file
        self.test_file = test_file
        self.similarity_metric = similarity_metric
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
        self.evaluation = None
        self.predictions = []

        self.reg_bi = reg_bi
        self.reg_bu = reg_bu

        # internal vars
        self.number_users = None
        self.number_items = None
        self.bu = {}
        self.bi = {}
        self.bui = {}

        self.model_name = 'UserKNN'
        self.k_neighbors = k_neighbors

        # internal vars
        self.su_matrix = None
        self.users_id_viewed_item = None

    def compute_similarity(self, transpose=False):
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

    def init_model(self):
        """
        Method to initialize the model. Compute similarity matrix based on user (user x user)

        """

        self.number_users = len(self.users)
        self.number_items = len(self.items)
        self.matrix = np.zeros((self.number_users, self.number_items))
        for user in self.train_set['users']:
            for item in self.train_set['feedback'][user]:
                self.matrix[self.user_to_user_id[user]][self.item_to_item_id[item]] = \
                    self.train_set['feedback'][user][item]

        self.users_id_viewed_item = {}

        # Set the value for k
        if self.k_neighbors is None:
            self.k_neighbors = int(np.sqrt(len(self.users)))

        self.su_matrix = self.compute_similarity(transpose=False)

        # Map the users which seen an item with their respective ids
        for item in self.items:
            for user in self.train_set['users_viewed_item'].get(item, []):
                self.users_id_viewed_item.setdefault(item, []).append(self.user_to_user_id[user])

    def predict_scores(self, user, unpredicted_items):
        """
        rui = bui + (sum((rvi - bvi) * sim(u,v)) / sum(sim(u,v)))

        """

        u_id = self.user_to_user_id[user]
        predictions = []

        for item in unpredicted_items:
            neighbors = []
            rui = 0
            sim_sum = 0
            for user_v_id in self.users_id_viewed_item.get(item, []):
                user_v = self.user_id_to_user[user_v_id]
                neighbors.append((user_v, self.su_matrix[u_id, user_v_id], self.train_set['feedback'][user_v][item]))
            neighbors = sorted(neighbors, key=lambda x: -x[1])

            if neighbors:
                for triple in neighbors[:self.k_neighbors]:
                    rui += (triple[2] - self.bui[triple[0]][item]) * triple[1] if triple[1] != 0 else 0.001
                    sim_sum += triple[1] if triple[1] != 0 else 0.001

                rui = self.bui[user][item] + (rui / sim_sum)

            else:
                rui = self.bui[user][item]

            # normalize the ratings based on the highest and lowest value.
            if rui > self.train_set["max_value"]:
                rui = self.train_set["max_value"]
            if rui < self.train_set["min_value"]:
                rui = self.train_set["min_value"]

            predictions.append((user, item, rui))

        return sorted(predictions, key=lambda x: x[1])

    def run(self, metrics=None):
        # read files
        self.train_set = ReadFile(self.train_file, sep=self.sep).read()
        self.test_set = ReadFile(self.test_file, sep=self.sep).read()
        self.users = sorted(set(list(self.train_set['users']) + list(self.test_set['users'])))
        self.items = sorted(set(list(self.train_set['items']) + list(self.test_set['items'])))

        for i, item in enumerate(self.items):
            self.item_to_item_id.update({item: i})
            self.item_id_to_item.update({i: item})
        for u, user in enumerate(self.users):
            self.user_to_user_id.update({user: u})
            self.user_id_to_user.update({u: user})

        # initialize empty predictions (Don't remove: important to Cross Validation)
        self.predictions = []
        self.init_model()

        # compute baseline
        self.bu = {}
        self.bi = {}
        self.bui = {}

        for i in range(10):

            # bi = (rui - mi - bu) / (regBi + number of interactions)
            self.bi = dict()
            for item in self.items:
                count = 0

                for user in self.train_set['users_viewed_item'].get(item, []):
                    self.bi[item] = self.bi.get(item, 0) + float(self.train_set['feedback'][user].get(item, 0)) - \
                                    self.train_set['mean_value'] - self.bu.get(user, 0)
                    count += 1

                if count > 1:
                    self.bi[item] = float(self.bi[item]) / float(self.reg_bi + count)
                elif count == 0:
                    self.bi[item] = self.train_set['mean_value']

            # bu = (rui - mi - bi) / (regBu + number of interactions)
            self.bu = dict()
            for user in self.users:
                count = 0
                for item in self.train_set['items_seen_by_user'].get(user, []):
                    self.bu[user] = self.bu.get(user, 0) + float(self.train_set['feedback'][user].get(item, 0)) - \
                                    self.train_set['mean_value'] - self.bi.get(item, 0)
                    count += 1
                if count > 1:
                    self.bu[user] = float(self.bu[user]) / float(self.reg_bu + count)
                elif count == 0:
                    self.bu[user] = self.train_set['mean_value']

        # bui = mi + bu + bi
        for user in self.users:
            for item in self.items:
                self.bui.setdefault(user, {}).update(
                    {item: self.train_set['mean_value'] + self.bu.get(user, 0) + self.bi.get(item, 0)})

        del self.bu
        del self.bi

        # predict each rating by knn
        for user in self.users:
            if len(self.train_set['feedback'].get(user, [])) != 0:
                self.predictions += self.predict_scores(user, self.test_set['items_seen_by_user'].get(user, []))

        if self.test_file is not None:
            self.evaluation = {}

            if metrics is None:
                metrics = list(['MAE', 'RMSE'])

            predictions_dict = {}

            for sample in self.predictions:
                predictions_dict.setdefault(sample[0], {}).update({sample[1]: sample[2]})

            eval_results = {}
            predictions_list = []
            test_list = []

            for user in predictions_dict:
                for item in predictions_dict[user]:
                    rui_predict = predictions_dict[user][item]
                    rui_test = self.test_set["feedback"].get(user, {}).get(item, np.nan)
                    if not np.isnan(rui_test):
                        predictions_list.append(rui_predict)
                        test_list.append(float(rui_test))

            eval_results.update({
                'MAE': round(mean_absolute_error(test_list, predictions_list), 6),
                'RMSE': round(np.sqrt(mean_squared_error(test_list, predictions_list)), 6)
            })

            results = eval_results

            for metric in metrics:
                self.evaluation[metric.upper()] = results[metric.upper()]


class ItemKNN(KNN):
    def __init__(self, train_file=None, test_file=None, similarity_metric="cosine", k_neighbors=None, reg_bi=10, reg_bu=15,
                 sep='\t', output_sep='\t'):
        self.train_file = train_file
        self.test_file = test_file
        self.similarity_metric = similarity_metric
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
        self.evaluation = None
        self.predictions = []

        self.reg_bi = reg_bi
        self.reg_bu = reg_bu

        # internal vars
        self.number_users = None
        self.number_items = None
        self.bu = {}
        self.bi = {}
        self.bui = {}

        self.model_name = 'ItemKNN'
        self.k_neighbors = k_neighbors

        # internal vars
        self.si_matrix = None
        self.similar_items = None

    def compute_similarity(self, transpose=False):
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

    def init_model(self):
        """
        Method to initialize the model. Compute similarity matrix based on user (user x user)

        """

        self.number_users = len(self.users)
        self.number_items = len(self.items)
        self.matrix = np.zeros((self.number_users, self.number_items))
        for user in self.train_set['users']:
            for item in self.train_set['feedback'][user]:
                self.matrix[self.user_to_user_id[user]][self.item_to_item_id[item]] = \
                    self.train_set['feedback'][user][item]

        self.similar_items = defaultdict(list)

        # Set the value for k
        if self.k_neighbors is None:
            self.k_neighbors = int(np.sqrt(len(self.items)))

        self.si_matrix = self.compute_similarity(transpose=True)

        for i_id, item in enumerate(self.items):
            self.similar_items[i_id] = sorted(range(len(self.si_matrix[i_id])),
                                              key=lambda k: -self.si_matrix[i_id][k])[1:self.k_neighbors + 1]

    def predict_scores(self, user, unpredicted_items):
        predictions = []

        for item in unpredicted_items:
            neighbors = []
            rui = 0
            sim_sum = 0
            item_id = self.item_to_item_id[item]
            for iter in self.train_set['items_seen_by_user'][user]:
                neighbors.append((iter, self.si_matrix[item_id, self.item_to_item_id[iter]], self.train_set['feedback'][user][iter]))

            neighbors = sorted(neighbors, key=lambda x: -x[1])[::self.k_neighbors]
            if neighbors:
                for triple in neighbors:
                    rui += (triple[2] - self.bui[user][triple[0]]) * triple[1] if triple[1] != 0 else 0.001
                    sim_sum += triple[1] if triple[1] != 0 else 0.001

                rui = self.bui[user][item] + (rui / sim_sum)

            else:
                rui = self.bui[user][item]

            # normalize the ratings based on the highest and lowest value.
            if rui > self.train_set["max_value"]:
                rui = self.train_set["max_value"]
            if rui < self.train_set["min_value"]:
                rui = self.train_set["min_value"]

            predictions.append((user, item, rui))

        return sorted(predictions, key=lambda x: x[1])

    def run(self, metrics=None):
        # read files
        self.train_set = ReadFile(self.train_file, sep=self.sep).read()
        self.test_set = ReadFile(self.test_file, sep=self.sep).read()
        self.users = sorted(set(list(self.train_set['users']) + list(self.test_set['users'])))
        self.items = sorted(set(list(self.train_set['items']) + list(self.test_set['items'])))

        for i, item in enumerate(self.items):
            self.item_to_item_id.update({item: i})
            self.item_id_to_item.update({i: item})
        for u, user in enumerate(self.users):
            self.user_to_user_id.update({user: u})
            self.user_id_to_user.update({u: user})

        # initialize empty predictions (Don't remove: important to Cross Validation)
        self.predictions = []
        self.init_model()

        # compute baseline
        self.bu = {}
        self.bi = {}
        self.bui = {}

        for i in range(10):

            # bi = (rui - mi - bu) / (regBi + number of interactions)
            self.bi = dict()
            for item in self.items:
                count = 0

                for user in self.train_set['users_viewed_item'].get(item, []):
                    self.bi[item] = self.bi.get(item, 0) + float(self.train_set['feedback'][user].get(item, 0)) - \
                                    self.train_set['mean_value'] - self.bu.get(user, 0)
                    count += 1

                if count > 1:
                    self.bi[item] = float(self.bi[item]) / float(self.reg_bi + count)
                elif count == 0:
                    self.bi[item] = self.train_set['mean_value']

            # bu = (rui - mi - bi) / (regBu + number of interactions)
            self.bu = dict()
            for user in self.users:
                count = 0
                for item in self.train_set['items_seen_by_user'].get(user, []):
                    self.bu[user] = self.bu.get(user, 0) + float(self.train_set['feedback'][user].get(item, 0)) - \
                                    self.train_set['mean_value'] - self.bi.get(item, 0)
                    count += 1
                if count > 1:
                    self.bu[user] = float(self.bu[user]) / float(self.reg_bu + count)
                elif count == 0:
                    self.bu[user] = self.train_set['mean_value']

        # bui = mi + bu + bi
        for user in self.users:
            for item in self.items:
                self.bui.setdefault(user, {}).update(
                    {item: self.train_set['mean_value'] + self.bu.get(user, 0) + self.bi.get(item, 0)})

        del self.bu
        del self.bi

        # predict each rating by knn
        for user in self.users:
            if len(self.train_set['feedback'].get(user, [])) != 0:
                self.predictions += self.predict_scores(user, self.test_set['items_seen_by_user'].get(user, []))

        if self.test_file is not None:
            self.evaluation = {}

            if metrics is None:
                metrics = list(['MAE', 'RMSE'])

            predictions_dict = {}

            for sample in self.predictions:
                predictions_dict.setdefault(sample[0], {}).update({sample[1]: sample[2]})

            eval_results = {}
            predictions_list = []
            test_list = []

            for user in predictions_dict:
                for item in predictions_dict[user]:
                    rui_predict = predictions_dict[user][item]
                    rui_test = self.test_set["feedback"].get(user, {}).get(item, np.nan)
                    if not np.isnan(rui_test):
                        predictions_list.append(rui_predict)
                        test_list.append(float(rui_test))

            eval_results.update({
                'MAE': round(mean_absolute_error(test_list, predictions_list), 6),
                'RMSE': round(np.sqrt(mean_squared_error(test_list, predictions_list)), 6)
            })

            results = eval_results

            for metric in metrics:
                self.evaluation[metric.upper()] = results[metric.upper()]


class ReadFile(object):
    def __init__(self, input_file, sep='\t', header=None, names=None, as_binary=False, binary_col=2):

        self.input_file = input_file
        self.sep = sep
        self.header = header
        self.names = names
        self.as_binary = as_binary
        self.binary_col = binary_col

        try:
            open(self.input_file)
        except TypeError:
            raise TypeError("File cannot be empty or file is invalid: " + str(self.input_file))

    def read(self):

        list_users = set()
        list_items = set()
        list_feedback = []
        dict_feedback = {}
        items_unobserved = {}
        items_seen_by_user = {}
        users_viewed_item = {}
        mean_value = 0
        number_interactions = 0

        with open(self.input_file) as infile:
            for line in infile:
                if line.strip():
                    inline = line.split(self.sep)
                    if len(inline) == 1:
                        raise TypeError("Error: Separator type is invalid!")
                    user, item, value = int(inline[0]), int(inline[1]), float(inline[2])
                    dict_feedback.setdefault(user, {}).update({item: 1.0 if self.as_binary else value})
                    items_seen_by_user.setdefault(user, set()).add(item)
                    users_viewed_item.setdefault(item, set()).add(user)
                    list_users.add(user)
                    list_items.add(item)
                    mean_value += 1.0 if self.as_binary else value
                    list_feedback.append(1.0 if self.as_binary else value)
                    number_interactions += 1

        mean_value /= float(number_interactions)

        list_users = sorted(list(list_users))
        list_items = sorted(list(list_items))

        # Create a dictionary with unobserved items for each user / Map user with its respective id
        for user in list_users:
            items_unobserved[user] = list(set(list_items) - set(items_seen_by_user[user]))

        # Calculate the sparsity of the set: N / (nu * ni)
        sparsity = (1 - (number_interactions / float(len(list_users) * len(list_items)))) * 100

        dict_file = {
            'feedback': dict_feedback,
            'users': list_users,
            'items': list_items,
            'sparsity': sparsity,
            'number_interactions': number_interactions,
            'users_viewed_item': users_viewed_item,
            'items_unobserved': items_unobserved,
            'items_seen_by_user': items_seen_by_user,
            'mean_value': mean_value,
            'max_value': max(list_feedback),
            'min_value': min(list_feedback),
        }

        return dict_file

