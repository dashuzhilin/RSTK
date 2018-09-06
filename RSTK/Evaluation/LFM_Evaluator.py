import math


class LFMEvaluator(object):
    def __init__(self, test_set=None, p=None, q=None, mu=0, bu=None, bi=None):
        self.test_set = test_set
        self.P = p
        self.Q = q
        self.bu = bu
        self.bi = bi
        self.mu = mu

    def rating(self):
        rmse = 0
        mae = 0
        num = 0
        for u, i, rui in self.test_set:
            if ((u in self.P) and (i in self.Q)):
                ret = self.mu + self.bu[u] + self.bi[i]
                ret += sum(self.P[u][k] * self.Q[i][k] for k in range(0, len(self.P[u])))
                pui = ret
                rmse += (rui - pui) * (rui - pui)
                mae += math.sqrt((rui - pui) * (rui - pui))
                num += 1
        rmse = math.sqrt(rmse / num)
        mae = mae/num
        return rmse, mae



