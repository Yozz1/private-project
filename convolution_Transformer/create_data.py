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


def read_create_data(path='C:\\Users\\user\\Desktop\\zkl_ori\\zkl\\train_nasa_raw.csv', featrue = 23,s=1, batch_size=2000, shuffle=False):

    min_max = MinMaxScaler()

    # datap = pd.read_csv('val.csv', index=None, header=None)
    # data = np.array(datap)

    #训练集
    data = pd.read_csv(path, header=None)
    data = np.array(data)
    # data_v = np.array(data[53243:53443, :])
    # print(data_v.shape)

    #测试集1
    datap = pd.read_csv('test_set.csv', header=None)
    datap = np.array(datap)
    # data = np.vstack((data, datap))


    data_m = pd.read_csv('test_set_2.csv', header=None)
    data_m = np.array(data_m)

    data = re_data(data, 125)
    datap = re_data(datap, 125)
    data_m = re_data(data_m, 125)

    # x_train = min_max.fit_transform(data)

    data = min_max.fit_transform(data)
    datap = min_max.transform(datap)
    data = np.vstack((data, datap))

    data = data[:,[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]
    #测试集2
    data_m = min_max.transform(data_m)
    data_m = data_m[:,[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]
    val_x = data_m[:,:-1]
    val_y = data_m[:,[-1]]
    val_x =np.float32(val_x).reshape(-1, s, featrue)
    val_y = np.float32(val_y).reshape(-1, s)

    val_y = np.mean(val_y, axis=1)
    val_y = np.float32(val_y)[:, None].reshape(-1,)





    feat = data[:-datap.shape[0], :-1]
    feat_v = data[-datap.shape[0]:, :-1]
    feat = np.float32(feat)
    feat_v = np.float32(feat_v)

    feat = feat.reshape(-1, s, featrue)
    feat_v = feat_v.reshape(-1, s, featrue)



    label = data[:-datap.shape[0], [-1]]
    label_v = data[-datap.shape[0]:, [-1]]
    label = label.reshape(-1, s)
    label_v = label_v.reshape(-1, s)

    label = np.mean(label, axis=1)
    label = np.float32(label)[:, None]

    label_v = np.mean(label_v, axis=1)
    label_v = np.float32(label_v)[:, None]
    # label = min_max.fit_transform(label)

    label = label.reshape(-1, )
    label_v = label_v.reshape(-1, )


    train_x = feat
    train_y = label

    test_x = feat_v
    test_y = label_v

    val_x = val_x
    val_y = val_y


    train_data = Data.TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y), torch.FloatTensor(train_y))
    train_data = Data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    test_data = Data.TensorDataset(torch.FloatTensor(test_x), torch.FloatTensor(test_y), torch.FloatTensor(test_y))
    test_data = Data.DataLoader(
        test_data, batch_size=16, shuffle=True, drop_last=True)

    val_data = Data.TensorDataset(torch.FloatTensor(val_x), torch.FloatTensor(val_y), torch.FloatTensor(val_y))
    val_data = Data.DataLoader(
        val_data, batch_size=16, shuffle=True, drop_last=True)

    return train_data, test_data, val_data



if __name__ == '__main__':
    read_create_data()