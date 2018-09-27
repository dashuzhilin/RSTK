from collections import namedtuple
import numpy as np


def rmse(predictions, verbose=True):
    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse = np.mean([float((true_r - est) ** 2)
                   for (_, _, true_r, est, _) in predictions])
    rmse_ = np.sqrt(mse)

    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse_))

    return rmse_

class Prediction(namedtuple('Prediction',
                            ['uid', 'iid', 'r_ui', 'est', 'details'])):
    __slots__ = ()  # for memory saving purpose.

    def __str__(self):
        s = 'user: {uid:<10} '.format(uid=self.uid)
        s += 'item: {iid:<10} '.format(iid=self.iid)
        if self.r_ui is not None:
            s += 'r_ui = {r_ui:1.2f}   '.format(r_ui=self.r_ui)
        else:
            s += 'r_ui = None   '
        s += 'est = {est:1.2f}   '.format(est=self.est)
        s += str(self.details)

        return s


