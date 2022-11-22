import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font",family='YouYuan')
import math

def RMSE(data):
    RMSE = 0
    for i in range(data.shape[0]):
        RMSE += (data[i][0] - data[i][1])**2
    RMSE = math.sqrt(RMSE / data.shape[0])

    return RMSE

def huatu():
    data = pd.read_csv('label_train_all.csv', header=0)
    data = np.array(data)[:300,:]
    print('训练集： ', RMSE(data))
    list1_1 = np.array(sorted(data, key=lambda x: x[1], reverse=True))
    list1_2 = data

    data = pd.read_csv('label_test1.csv', header=0)
    data = np.array(data)
    print('验证集： ', RMSE(data))
    list2_1 = np.array(sorted(data, key=lambda x: x[1], reverse=False))
    # p = pd.DataFrame(list2_1)
    # p.to_csv('val_2_1_24feature_1layers_64h_32head_64q_MLP_0.1273.csv', header=None, index=None)
    list2_2 = data


    data = pd.read_csv('label_test2.csv', header=0)
    data = np.array(data)
    print('测试集： ', RMSE(data))
    list3_1 = np.array(sorted(data, key=lambda x: x[1], reverse=True))
    p = pd.DataFrame(list3_1)
    p.to_csv('测试集顺序.csv', header=None, index=None)
    list3_2 = data





    plt.subplot(3, 2, 1)
    plt.plot(list1_1[:, [0]], label='预测值', linewidth = 2, color='blue')
    plt.plot(list1_1[:, [1]], label='真实值', linewidth = 2, color='red')
    plt.title('训练顺序')

    plt.subplot(3, 2, 2)
    plt.plot(list1_2[:, [0]], label='预测值', linewidth = 1.5, color='blue')
    plt.plot(list1_2[:, [1]], label='真实值', linewidth = 1.5, color='red')
    plt.title('训练乱序')

    plt.subplot(3, 2, 3)
    plt.plot(list2_1[:, [0]], label='预测值', linewidth = 1.5, color='blue')
    plt.plot(list2_1[:, [1]], label='真实值', linewidth = 1.5, color='red')
    plt.title('测试顺序①')

    plt.subplot(3, 2, 4)
    plt.plot(list2_2[:, [0]], label='预测值', linewidth = 1.5, color='blue')
    plt.plot(list2_2[:, [1]], label='真实值', linewidth = 1.5, color='red')
    plt.title('测试乱序①')

    plt.subplot(3, 2, 5)
    plt.plot(list3_1[:, [0]], label='预测值', linewidth = 1.5, color='blue')
    plt.plot(list3_1[:, [1]], label='真实值', linewidth = 1.5, color='red')
    plt.title('测试顺序②')

    plt.subplot(3, 2, 6)
    plt.plot(list3_2[:, [0]], label='预测值', linewidth = 1.5, color='blue')
    plt.plot(list3_2[:, [1]], label='真实值', linewidth = 1.5, color='red')
    plt.title('测试乱序②')


    plt.legend()
    plt.show()

huatu()