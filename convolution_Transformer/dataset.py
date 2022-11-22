import pickle
import pandas as pd
import sys
sys.getdefaultencoding()
import pickle
import numpy as np
np.set_printoptions(threshold=1000000000000000)
path = 'C:\\Users\\Administrator\\Desktop\\张含冰\\transferlearning-master\\code\\deep\\adarnn\\PRSA_Data_1.pkl'
file = open(path,'rb')
inf = pickle.load(file,encoding='iso-8859-1')       #读取pkl文件的内容
print(inf)
#fr.close()
inf=str(inf)
obj_path = 'C:\\Users\\Administrator\\Desktop\\张含冰\\transferlearning-master\\code\\deep\\adarnn\\1.txt'
ft = open(obj_path, 'w')
ft.write(inf)


