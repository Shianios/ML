# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 19:31:44 2018

@author: Loizos
"""

import numpy as np
import Convolution as Conv
import matplotlib.pyplot as plt

def loadMNIST(prefix, folder):
    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(folder + "/" + prefix + '-images.idx3-ubyte', dtype = 'ubyte')
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
    data = data[nMetaDataBytes:].astype(dtype = 'float32').reshape([nImages, width, height])
    labels = np.fromfile(folder + "/" + prefix + '-labels.idx1-ubyte', dtype = 'ubyte')[2 * intType.itemsize:]

    return data, labels

trainingImages, trainingLabels = loadMNIST("train", "C:/Users/Loizos/Desktop/Git/SOM/data")
testImages, testLabels = loadMNIST("t10k", "C:/Users/Loizos/Desktop/Git/SOM/data")

training_data = np.empty((28,28,trainingImages.shape[0]))
print(trainingLabels.shape,trainingImages.shape)
print(training_data.shape)
for i in range (0,trainingImages.shape[0]):
    training_data[:,:,i] = trainingImages[i,:,:]

number = 7   
num_pos = []
for i in range (0,trainingLabels.shape[0]):
    if trainingLabels[i] == number:
        num_pos.append(i)

num_ave_arr = np.zeros((28,28))
for i in (num_pos):
    num_ave_arr += (training_data[:,:,i]) 

#five_ave_arr =  np.floor((five_ave_arr)/(len(five_pos)))   
print(np.min(num_ave_arr),np.max(num_ave_arr))
num_ave_arr = np.multiply(np.divide(np.subtract(num_ave_arr,np.min(num_ave_arr)),np.subtract(np.max(num_ave_arr),np.min(num_ave_arr))),255.0)
print(np.min(num_ave_arr),np.max(num_ave_arr))
plt.figure(figsize = (2,2))
plt.title('original')
plt.imshow(num_ave_arr, cmap=plt.get_cmap('gray'))
plt.show()

Conv_1 = Conv.Convolution(num_ave_arr,**{"opp":"Convol", "filter":"Laplace_op", "strid":1, "pading":1}).compute()
plt.figure(figsize = (2,2))
plt.title('Convolution')
plt.imshow(Conv_1, cmap=plt.get_cmap('gray'))
plt.show()
print(Conv_1.shape)

Pool_1 = Conv.Convolution(Conv_1,**{"opp":"Pooling", "Pool_op":"Average", "size":2, "strid":2, "pading":0}).compute()
plt.figure(figsize = (2,2))
plt.title('Pooling Average')
plt.imshow(Pool_1, cmap=plt.get_cmap('gray'))
plt.show()
print(Pool_1.shape)

Pool_1 = Conv.Convolution(Conv_1,**{"opp":"Pooling", "Pool_op":"Max", "size":2, "strid":2, "pading":0}).compute()
plt.figure(figsize = (2,2))
plt.title('Pooling Max')
plt.imshow(Pool_1, cmap=plt.get_cmap('gray'))
plt.show()
print(Pool_1.shape)