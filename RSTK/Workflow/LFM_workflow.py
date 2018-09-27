# -*- coding: utf-8 -*-
import time
import os

from ..DataProcess.DataSplit import DataSplit
from ..Model.LFM import LFM
from ..Utils.Crossvalidate_Util import CrossValidation

db = 'Data/u.data'
folds_path = 'Data/'
tr = 'Data/folds/0/train.dat'
te = 'Data/folds/0/test.dat'

def run():
    assert os.path.exists('Data/u.data'), \
        'File not exists in path, please add data file first.'
    print('Start LFM..')
    # start = time.time()
    #
    # train, test = Split().split(10, 1, 1)
    # LFM(train, 20, 10, 0.02, 0.02, P, Q, bu, bi, mu)
    # print('RMSE : ', ModelRmse(train, test, P, Q, bu, bi, mu))
    #
    # print('LFM Cost time: %f' % (time.time() - start))

    model = LFM()
    CrossValidation(source=db, model=model, targets=folds_path, n_folds=5).compute()
    LFM(tr, te).run()
