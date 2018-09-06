import numpy as np
from ..Model import FM
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer


# Read in data
def load_data(filename):
    data = []
    data = []
    ratings = []
    users = set()
    items = set()
    with open(filename) as f:
        for line in f:
            user, item, rating, ts = line.split('\t')
            data.append({"userId": str(user), "itemId": str(item)})
            ratings.append(float(rating))
            users.add(user)
            items.add(item)
    return data, np.array(ratings)


def run():
    # Load data
    data, ratings = load_data("./Data/u.data")
    # Transform feature
    v = DictVectorizer()
    X = v.fit_transform(data).toarray()
    # 5-fold cross validation
    kf = KFold(n_splits=5)
    errors = []
    i = 1
    for train_index, test_index in kf.split(X):
        # Fit
        fm = FM()
        train_X = X[train_index]
        train_y = ratings[train_index]
        fm.fit(train_X, train_y)
        # Predict
        test_X = X[test_index]
        test_y = ratings[test_index]
        test_pred = fm.predict(test_X)
        # RMSE
        error = np.sqrt(mean_squared_error(test_y, test_pred))
        errors.append(error)
        print('Fold %d: %f' % (i, error))
        i += 1
    print('Mean: %f' % np.mean(errors))
