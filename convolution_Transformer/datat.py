import pickle

path='PRSA_Data_1.pkl'
f=open(path,'rb')
data=pickle.load(f)
print(data)
print(len(data))