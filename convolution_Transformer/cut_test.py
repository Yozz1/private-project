import numpy as np
import pandas as pd

test = pd.read_csv('./CMAPSSData/test_FD004_change.csv', header=None)


t = ['RUL']
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names + t
test.columns = col_names
group = test.groupby(by="unit_nr")
x = group.get_group(1).iloc[[-1], 2:]
x = np.array(x)
print(x)


list = []
for i in range(1, 249):
    for j in range(25):
        list.append(group.get_group(i).iloc[-1][j + 2])

list = np.array(list).reshape(-1, 25)

list = pd.DataFrame(list)
# print(list)
# print(list)

list.to_csv('./CMAPSSData/val_FD004_change.csv', header=None, index=None)

# test = np.array(test)

# index = 0

# for i in range(test.shape[0]):
#     if