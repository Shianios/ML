# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 21:34:57 2018

@author: Loizos Shianios
"""

import numpy as np
#import matplotlib.pyplot as plt

class Filters:
    def __init__ (self,data,**kwargs):
        self.kwargs = kwargs
        self.filter_select = {"median" : Filters.median(self,data),
                              "Sobel" : Filters.Sobel(self,data)}
    
    def compute(self):
        return self.filter_select[self.kwargs.get('filter')]
    
    def median (self,data):
        d = data.copy()
        d_shape = data.shape
        for i in range (0,len(d_shape)-1):
            for j in range (0,d_shape[i]-4):
                for k in range (0,d_shape[i+1]-4):
                    aux = data[j:(j+3),k:(k+3)]
                    sort_ = np.sort(aux, axis=None)
                    element = sort_[4]
                    d[j+1,k+1] = element
        return d
    
    def Sobel (self,data):
        d = data.copy()
        d_shape = data.shape
        a = 1.
        b = 2.
        Sob_filter_x = np.array([[-a,0.,a],[-b,0.,b],[-a,0.,a]])
        Sob_filter_y = np.array([[-a,-b,-a],[0.,0.,0.],[a,b,a]])
        for i in range (0,len(d_shape)-1):
            #print('i',i)
            for j in range (1,d_shape[i]-1):
                for k in range (1,d_shape[i+1]-1):
                    aux = data[(j-1):(j+2),(k-1):(k+2)] 
                    element_x = np.einsum('ij,ij',Sob_filter_x,aux)
                    element_y = np.einsum('ij,ij',Sob_filter_y,aux)
                    element = np.sqrt(element_x**2 + element_y**2)
                    d[j,k] = element
                    #print(d) 
        return d

'''
def loadMNIST(prefix, folder):
    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(folder + "/" + prefix + '-images.idx3-ubyte', dtype = 'ubyte')
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
    data = data[nMetaDataBytes:].astype(dtype = 'float32').reshape([nImages, width, height])
    labels = np.fromfile(folder + "/" + prefix + '-labels.idx1-ubyte', dtype = 'ubyte')[2 * intType.itemsize:]

    return data, labels

trainingImages, trainingLabels = loadMNIST("train", "C:/Users/Loizos/Desktop/assignment/SOM/data")
testImages, testLabels = loadMNIST("t10k", "C:/Users/Loizos/Desktop/assignment/SOM/data")

training_data = np.empty((28,28,trainingImages.shape[0]))
print(trainingLabels.shape,trainingImages.shape)
print(training_data.shape)
for i in range (0,trainingImages.shape[0]):
    training_data[:,:,i] = trainingImages[i,:,:]
    
five_pos = []
for i in range (0,trainingLabels.shape[0]):
    if trainingLabels[i] == 7:
        five_pos.append(i)

five_ave_arr = np.zeros((28,28))
for i in (five_pos):
    five_ave_arr += (training_data[:,:,i]) 

#five_ave_arr =  np.floor((five_ave_arr)/(len(five_pos)))   
print(np.min(five_ave_arr),np.max(five_ave_arr))
five_ave_arr = np.multiply(np.divide(np.subtract(five_ave_arr,np.min(five_ave_arr)),np.subtract(np.max(five_ave_arr),np.min(five_ave_arr))),255.0)
print(np.min(five_ave_arr),np.max(five_ave_arr))
plt.figure(figsize = (2,2))
plt.title('original')
plt.imshow(five_ave_arr, cmap=plt.get_cmap('gray'))
plt.show()

d_filtered = Filters(five_ave_arr,**{"filter":"median"}).compute()
#d_filtered = np.multiply(d_filtered,d_filtered)
d_filtered += np.abs(np.min(d_filtered))
print(np.min(d_filtered),np.max(d_filtered))
d_filtered = np.multiply(np.divide(np.subtract(d_filtered,np.min(d_filtered)),np.subtract(np.max(d_filtered),np.min(d_filtered))),255.0)
print(np.min(d_filtered),np.max(d_filtered))
plt.figure(figsize = (2,2))
plt.title('median filter')
plt.imshow(d_filtered, cmap=plt.get_cmap('gray'))
plt.show()

d_add = np.multiply(five_ave_arr,d_filtered)
print(np.min(d_add),np.max(d_add))
d_add = np.multiply(np.divide(np.subtract(d_add,np.min(d_add)),np.subtract(np.max(d_add),np.min(d_add))),255.0)
print(np.min(d_add),np.max(d_add))
plt.figure(figsize = (2,2))
plt.title('original mult by median')
plt.imshow(d_add, cmap=plt.get_cmap('gray'))
plt.show()

d_filt_2 = Filters(d_filtered ,**{"filter":"Sobel"}).compute()
d_filt_2 += np.abs(np.min(d_filt_2))
d_filt_2 = np.sqrt(d_filt_2)
d_filt_2 *= -1.0
print(np.min(d_filt_2),np.max(d_filt_2))
d_filt_2 = np.multiply(np.divide(np.subtract(d_filt_2,np.min(d_filt_2)),np.subtract(np.max(d_filt_2),np.min(d_filt_2))),255.0)
print(np.min(d_filt_2),np.max(d_filt_2))
plt.figure(figsize = (2,2))
plt.title('Sobel filter')
plt.imshow(d_filt_2, cmap=plt.get_cmap('gray'))
plt.show()

d_filt_3 = np.multiply(d_filt_2,d_filtered)
d_filt_3 += np.abs(np.min(d_filt_3))
print(np.min(d_filt_3),np.max(d_filt_3))
d_filt_3 = np.multiply(np.divide(np.subtract(d_filt_3,np.min(d_filt_3)),np.subtract(np.max(d_filt_3),np.min(d_filt_3))),255.0)
print(np.min(d_filt_3),np.max(d_filt_3))
plt.figure(figsize = (2,2))
plt.title('Median mult by Sobel')
plt.imshow(d_filt_3, cmap=plt.get_cmap('gray'))
plt.show()

d_fit_4 = np.multiply(d_filt_2,five_ave_arr)
d_fit_4 += np.abs(np.min(d_fit_4))
print(np.min(d_fit_4),np.max(d_fit_4))
d_fit_4 = np.multiply(np.divide(np.subtract(d_fit_4,np.min(d_fit_4)),np.subtract(np.max(d_fit_4),np.min(d_fit_4))),255.0)
print(np.min(d_fit_4),np.max(d_fit_4))
plt.figure(figsize = (2,2))
plt.title('Sobel mult by original')
plt.imshow(d_fit_4, cmap=plt.get_cmap('gray'))
plt.show()

#d_fin = np.multiply(np.multiply(d_filt_2,d_filtered),five_ave_arr)
#d_fin = np.multiply(np.add(d_filt_2,five_ave_arr),d_filtered)
d_fin = np.multiply(five_ave_arr,d_filt_3)
#d_fin = np.multiply(d_fin,d_fin)
print(np.min(d_fin),np.max(d_fin))
d_fin = np.multiply(np.divide(np.subtract(d_fin,np.min(d_fin)),np.subtract(np.max(d_fin),np.min(d_fin))),255.0)
print(np.min(d_fin),np.max(d_fin))
plt.figure(figsize = (2,2))
plt.title('Original mult by median multiply by Sobel')
plt.imshow(d_fin, cmap=plt.get_cmap('gray'))
plt.show()
'''