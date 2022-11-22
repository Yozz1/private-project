import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font",family='YouYuan')
import math

from sklearn.preprocessing import MinMaxScaler

data1 = pd.read_csv('C:\\Users\\user\\Desktop\\zkl_ori\\RNN_for_engine\\LSTM_result_val_FD004.csv', header=None)
data1 = np.array(data1)
data1 = np.array(sorted(data1, key=lambda x: x[1], reverse=False))

trans = MinMaxScaler()

label = data1[:, [1]] * 120
pred_lstm = data1[:, [0]] * 120

data2 = pd.read_csv('val_FD004_64h_32head_64q_MLP_0.1647.csv', header=0)
data2 = np.array(data2)
data2 = np.array(sorted(data2, key=lambda x: x[1], reverse=False))


pred_transformer = data2[:, [0]] * 120


# label = trans.fit_transform(label, feature_range =(0, 120))

data_final = np.column_stack((label, pred_lstm))
data_final = np.column_stack((data_final, pred_transformer))
data_final = pd.DataFrame(data_final)
data_final.to_csv('FD004对比结果.csv', header=None, index=None)

plt.plot(label, label='RUL', linewidth = 1.5, color='green')
plt.plot(pred_lstm, label='lstm预测值', linewidth = 1.5, color='blue')
plt.plot(pred_transformer, label='transformer预测值', linewidth =1.5, color='red')
plt.title('对比结果')
plt.legend()
plt.show()





