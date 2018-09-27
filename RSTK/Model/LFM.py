import random
import math
from ..DataProcess.DataIO import DataIO
from ..Evaluation.LFM_Evaluator import LFMEvaluator


class LFM(object):
    def __init__(self, train_set=None, test_set=None, metrics=list(['MAE', 'RMSE']), nfactor=20, steps=10, learn_rate=0.02, reg=0.02):
        self.train_file = train_set
        self.test_file = test_set
        self.nfactor = nfactor
        self.steps = steps
        self.learn_rate = learn_rate
        self.reg = reg

        # internal vars
        self.P = dict()
        self.Q = dict()
        self.bu = dict()
        self.bi = dict()
        self.mu = 0
        self.model_name = 'LFM'
        self.metrics = metrics
        self.evaluation = None

    def train(self):
        train = DataIO(input_file=self.train_file).read()
        for u, i, rui in train:
            self.bu[u] = 0
            self.bi[i] = 0
            if u not in self.P:
                self.P[u] = [random.random() / math.sqrt(self.nfactor) for x in range(0, self.nfactor)]
            if i not in self.Q:
                self.Q[i] = [random.random() / math.sqrt(self.nfactor) for x in range(0, self.nfactor)]

        # print('Training LFM.. ')
        for step in range(0, self.steps):
            for u, i, rui in train:
                # compute predict value
                ret = self.mu + self.bu[u] + self.bi[i]
                ret += sum(self.P[u][k] * self.Q[i][k] for k in range(0, len(self.P[u])))
                pui = ret
                # update parameters
                self.bu[u] += self.learn_rate * ((rui - pui) - self.reg * self.bu[u])
                self.bi[i] += self.learn_rate * ((rui - pui) - self.reg * self.bi[i])
                for k in range(0, self.nfactor):
                    self.P[u][k] += self.learn_rate * (self.Q[i][k]*(rui - pui) - self.reg*self.P[u][k])
                    self.Q[i][k] += self.learn_rate * (self.P[u][k]*(rui - pui) - self.reg*self.Q[i][k])
            self.learn_rate *= 0.9

    def test(self):
        # print('Testing LFM.. ')
        test = DataIO(input_file=self.test_file).read()
        rmse, mae = LFMEvaluator(test_set=test, p=self.P, q=self.Q, mu=self.mu, bi=self.bi, bu=self.bu).rating()
        print('RMSE: ', rmse)
        self.evaluation = {}
        self.evaluation.update({
            'MAE': mae,
            'RMSE': rmse
        })
        # for metric in self.metrics:
        #     self.evaluation_results[metric.upper()] = results[metric.upper()]
        return rmse

    def run(self):
        self.train()
        self.test()
