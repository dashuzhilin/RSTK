import numpy as np


class FM:

    def __init__(self):
        # Create parameter
        self.bias = None
        self.weight = None
        self.factor = None

    def fit(self, X: list, Y: list,
            n_factors: int = 10,
            n_epoch: int = 20,
            learning_rate: float = 0.005):
        assert len(np.shape(X)) == 2
        # Initialize parameter
        n_features = np.shape(X)[1]
        self.bias = 0
        self.weight = np.zeros([n_features, 1])
        self.factor = np.zeros([n_features, n_factors])
        # SGD
        n_train = np.shape(X)[0]
        Y = np.asarray(Y).reshape([-1, 1])
        for ep in range(n_epoch):
            for _ in range(n_train):
                # Select random sample
                i = np.random.randint(n_train, size=1)
                x = X[i]
                y = Y[i]
                error = np.squeeze(self.predict(x) - np.asarray(y))
                # Update bias
                self.bias -= error * learning_rate
                # Update weight
                self.weight -= x.T * error * learning_rate
                # Update factor
                temp = np.dot(x, self.factor)
                self.factor -= (np.dot(x.T, temp) - self.factor * np.square(x).T) * error * learning_rate

    def predict(self, X: list) -> list:
        assert len(np.shape(X)) == 2
        n_test = np.shape(X)[0]
        # Add bias
        y = np.ones([n_test, 1]) * self.bias
        # Add weight
        y += np.dot(X, self.weight)
        # Add latent factor
        y += np.sum(np.square(np.dot(X, self.factor)) - np.dot(np.square(X), np.square(self.factor)), axis=1, keepdims=True) * 0.5
        return y
