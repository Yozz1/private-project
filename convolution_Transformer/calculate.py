import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font",family='YouYuan')
import math

from sklearn.preprocessing import MinMaxScaler

data1 = pd.read_csv('C:\\Users\\user\\Desktop\\zkl_ori\\RNN_for_engine\\RNN_result_FD004.csv', header=0)
data1 = np.array(data1)


score = []
s = 0
for i in range(data1.shape[0]):
    score.append((data1[i][0] - data1[i][1]) * 120)

# print(score)
score = np.array(score).reshape((-1, 1))
for i in range(score.shape[0]):
    if score[i][0] < 0:
        s += math.exp(-score[i][0] / 13) - 1
    else:
        s += math.exp(score[i][0] / 10) - 1

print('score = ', s)


mina = []
for i in range(data1.shape[0]):
    mina.append(((data1[i][0]  - data1[i][1]) * 120) ** 2)

mina = np.array(mina)
RMSE = np.sqrt(np.sum(mina) / mina.shape[0])

print('RMSE = ', RMSE)










