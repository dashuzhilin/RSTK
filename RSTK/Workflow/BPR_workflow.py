# -*- coding: UTF-8 -*-
from ..Model.BPR import *
import os
import tensorflow as tf

def pre_user_favorite(values,user_num):
    #  0号用户对这个用户对所有电影的预测评分
    session1 = tf.Session()
    u1_dim = tf.expand_dims(values[0][user_num],0)
    u1_all = tf.matmul(u1_dim, values[1],transpose_b=True)
    result = session1.run(u1_all)
    print (result)

    print("以下是给用户" + str(user_num) + "的推荐：")
    p = np.squeeze(result)
    p[np.argsort(p)[:-5]] = 0
    for index in range(len(p)):
        if p[index] != 0:
            print("item_id:{}  score:{:.4f}".format(index, p[index]))

def load_train_value():
    if os.path.exists("../Data/bpr_user_emb_w.csv"):
        print('exist')
        values0 = np.loadtxt(open("../Data/bpr_user_emb_w.csv"))
        values1 = np.loadtxt(open("../Data/bpr_item_emb_h.csv"))
        values = np.array([values0,values1])
        return values
    print('not exist')
    return None

def run():
    path = "./Data/u.data"
    BPR_train(path)
    values = load_train_value()
    pre_user_favorite(values,5)

if __name__ == '__main__':
    run()
