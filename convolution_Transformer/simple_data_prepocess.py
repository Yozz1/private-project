import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils import data as Data
import torch


def re_data(data, up):
    y = data[:, [-1]].reshape((-1, 1))
    for i in range(y.shape[0]):
        if y[i][0] >= up:
            y[i][0] = up
    data = np.delete(data, -1, axis=1)
    data = np.column_stack((data, y))

    return data


def read_simple(path1='C:\\Users\\user\\Desktop\\zkl_ori\\zkl\\CMAPSSData\\train_FD001_change.csv',
                path2='C:\\Users\\user\\Desktop\\zkl_ori\\zkl\\CMAPSSData\\test_FD001_change.csv',
                path3='C:\\Users\\user\\Desktop\\zkl_ori\\zkl\\CMAPSSData\\val_FD001_change.csv',
                     featrue = 24,s=1, batch_size=2000, shuffle=False):
    #归一化
    trans_label = MinMaxScaler()
    #训练集
    data_train = pd.read_csv(path1, header=None)

    data_train.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                        's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                        's18', 's19', 's20', 's21', 'RUL']
    data_train = np.array(data_train)
    #测试集
    data_test = pd.read_csv(path2, header=None)
    data_test = np.array(data_test) [ : ,:]
    #验证集
    data_val = pd.read_csv(path3, header=None)


    data_val = np.array(data_val)

    # 处理
    if featrue == 13:
        data_train = data_train[:, [6, 7, 8, 11, 12, 13, 15, 16, 17, 19, 21, 24, 25, 26]]
        data_test = data_test[:, [6, 7, 8, 11, 12, 13, 15, 16, 17, 19, 21, 24, 25, 26]]
        data_val = data_val[:, [4, 5, 6, 9, 10, 11, 13, 14, 15, 17, 19, 22, 23, 24]]
    else:
        data_train = data_train[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]
        data_test = data_test[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]
        data_val = data_val[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]






    data_train = re_data(data_train, 120)
    data_test = re_data(data_test, 120)
    data_val = re_data(data_val, 120)
    # data_test = pd.DataFrame(data_test)
    # data_train = pd.DataFrame(data_train)
    # data_test.columns = [ 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
    #                    's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
    #                    's18', 's19', 's20', 's21', 'RUL']
    # data_train.columns = [ 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
    #                    's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
    #                    's18', 's19', 's20', 's21', 'RUL']

    train_x = (data_train[:, : -1] * 2 - 1 ).reshape(-1, s, featrue)
    test_x = (data_test[:, : -1] * 2 - 1).reshape(-1, s, featrue)
    val_x = (data_val[:, : -1]* 2 - 1).reshape(-1, s, featrue)

    # train_x = (data_train[:, : -1] ).reshape(-1, s, featrue)
    # test_x = (data_test[:, : -1] ).reshape(-1, s, featrue)
    # val_x = (data_val[:, : -1] ).reshape(-1, s, featrue)


    train_y = data_train[:, [-1]].reshape(-1, 1)
    test_y = data_test[:, [-1]].reshape(-1, 1)
    val_y = data_val[:, [-1]].reshape(-1, 1)

    train_y = trans_label.fit_transform(train_y).reshape(-1, )
    test_y = trans_label.transform(test_y).reshape(-1, )
    val_y = trans_label.transform(val_y).reshape(-1, )

    train_data = Data.TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y), torch.FloatTensor(train_y))
    train_loader = Data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    test_data = Data.TensorDataset(torch.FloatTensor(test_x), torch.FloatTensor(test_y), torch.FloatTensor(test_y))
    test_loader = Data.DataLoader(
        test_data, batch_size=13096, shuffle=True)

    val_data = Data.TensorDataset(torch.FloatTensor(val_x), torch.FloatTensor(val_y), torch.FloatTensor(val_y))
    val_loader = Data.DataLoader(
        val_data, batch_size=100, shuffle=True)


    return train_loader, test_loader, val_loader




    # return train_data, test_data, val_data



if __name__ == '__main__':
    read_simple()