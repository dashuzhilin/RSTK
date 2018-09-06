from ..Model import SVD
from ..Evaluation import rmse
from ..Evaluation import Prediction

from surprise import dataset
from surprise import Reader
from surprise.model_selection import train_test_split

def run():
    reader= Reader(line_format='user item rating timestamp',sep=',')

    data=dataset.Dataset.load_from_file('./Data/ratings1.csv',reader=reader)

    trainset, testset= train_test_split(data,test_size=.25)

    algo=SVD()
    algo.fit(trainset)

    predictions=algo.test(testset)
    rmse(predictions)

