import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy_indexed as npi

dire='data/cifar-10-batches-py/' # load cifa data
file= ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5',
       'test_batch']

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



for j,f in enumerate(file) :
    Dicti= unpickle(dire+f)
    data=Dicti[b'data']
    label= Dicti[b'labels']
    label = np.array(label).reshape(-1,1)
    if j is 0 :
        load= np.append(label,data,axis=1)
    else:
        ds=np.append(label,data,axis=1)
        load= np.append(load,ds,axis=0)
        
        
a=load

a= np.append(a[:,0].reshape(-1,1),a,axis=1)

# List=a[:,0].tolist()
# for j in range(10):
#     idx = [i for i in range(len(List)) if List[i] == j] 

#     print(j,len(idx))

q=npi.group_by(a[:, 0]).split(a[:, 1:])

np.save('data/all_cifar.npy',q) #save data in a 'data' directory

print("data saved ")
