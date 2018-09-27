# -*- coding: utf-8 -*-
import os
from ..Model import UserKNN
from ..Model import ItemKNN
from ..Utils.Crossvalidate_Util import CrossValidation

db = 'Data/u.data'
folds_path = 'Data/'
tr = 'Data/folds/0/train.dat'
te = 'Data/folds/0/test.dat'

def run():
    assert os.path.exists('Data/u.data'), \
        'File not exists in path, please add data file first.'
    print('Start KNN..')

    model = UserKNN()
    CrossValidation(source=db, model=model, targets=folds_path, n_folds=5).compute()
    UserKNN(tr, te).run()

    # model = ItemKNN()
    # CrossValidation(source=db, model=model, targets=folds_path, n_folds=5).compute()
    # ItemKNN(tr, te).compute()