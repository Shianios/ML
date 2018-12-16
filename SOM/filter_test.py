# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 21:34:57 2018

@author: Loizos Shianios
"""

import numpy as np
#import matplotlib.pyplot as plt

class Filters:
    # Filters leave the size of the data matrix unchanged. For now all operations are restricted to 2D data.
    def __init__ (self,data,**kwargs):
        self.kwargs = kwargs
        self.aux_dic = {}
        self.data = data
        if 'size' in kwargs:
            if kwargs['size'] > 0:
                aux_size = kwargs['size']
                self.kwargs['size'] = aux_size + int(2.*(float(aux_size)/2. - np.floor(float(aux_size)/2.))-1.0)  
                # The above line will map all even numbers to the previews odd and odd left unchanged. We restric filters dimensions to odd numbers
                self.kwargs.update({'mid': int((self.size**2 - 1)/2)})
            else:
                self.kwargs['size'] = 3
        else: 
            self.kwargs.update({'size': 3})
            self.kwargs.update({'mid': 4})
  
    def compute (self):
        return getattr(self, self.kwargs.get('filter'))(self.data,**self.kwargs)
    
    def Median_op (self,data,**kwargs):
        mid = kwargs['mid']
        sort_ = np.sort(data, axis=None)
        element = sort_[mid]
        return np.median(data)
      
    def Median (self,data,**kwargs):
        size = kwargs['size']
        d = data.copy()
        d_shape = data.shape
        ind = int(np.floor(size/2))
        for i in range (0,len(d_shape)-1):
            for j in range (0,d_shape[i]-size+1):
                for k in range (0,d_shape[i+1]-size+1):
                    aux = data[j:(j+size),k:(k+size)]
                    d[j+ind,k+ind] = self.Median_op(aux,**kwargs)
        return d
    
    def Sobel_op (self,data,**kwargs):
        if 'Sob_filter_x' not in self.aux_dic: # To avoid redeffining the Sobel matrices in a loop
            a = 1.
            b = 2.
            Sob_filter_x = np.array([[a,0.,-a],[b,0.,-b],[a,0.,-a]])
            Sob_filter_y = Sob_filter_x.T
            self.aux_dic.update({'Sob_filter_x' : Sob_filter_x})
            self.aux_dic.update({'Sob_filter_y' : Sob_filter_y})
            del a,b,Sob_filter_x,Sob_filter_y
            
        element_x = np.einsum('ij,ij',self.aux_dic['Sob_filter_x'],data)
        element_y = np.einsum('ij,ij',self.aux_dic['Sob_filter_y'],data)
        element = np.sqrt(element_x**2 + element_y**2)
        return element
    
    def Sobel (self,data,**kwargs):
        d = data.copy()
        d_shape = data.shape
        size = 3    # For now we retrict Sobel to 3x3 matrix
        ind = int(np.floor(size/2))
        for i in range (0,len(d_shape)-1):
            for j in range (0,d_shape[i]-size):
                for k in range (0,d_shape[i+1]-size):
                    aux = data[j:(j+size),k:(k+size)] 
                    d[j+ind,k+ind] = self.Sobel_op(aux,**kwargs)          
        return d
    
    def Laplace_op (self,data,**kwargs):
        if 'Laplace_matrix' not in self.aux_dic: # To avoid redeffining the Sobel matrices in a loop
            a = 1.
            b = 4.
            Laplace = np.array([[0.,a,0.],[a,-b,a],[0.,a,0.]])
            self.aux_dic.update({'Laplace_matrix' : Laplace})
            del a,b,Laplace
            
        element = np.einsum('ij,ij',self.aux_dic['Laplace_matrix'],data)
        return element
    
    def Laplace (self,data,**kwargs):
        d = data.copy()
        d_shape = data.shape
        size = 3    # For now we retrict Laplace to 3x3 matrix
        ind = int(np.floor(size/2))
        for i in range (0,len(d_shape)-1):
            for j in range (0,d_shape[i]-size):
                for k in range (0,d_shape[i+1]-size):
                    aux = data[j:(j+size),k:(k+size)] 
                    d[j+ind,k+ind] = self.Laplace_op(aux,**kwargs)          
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

trainingImages, trainingLabels = loadMNIST("train", "C:/Users/Loizos/Desktop/Git/SOM/data")
testImages, testLabels = loadMNIST("t10k", "C:/Users/Loizos/Desktop/Git/SOM/data")

training_data = np.empty((28,28,trainingImages.shape[0]))
print(trainingLabels.shape,trainingImages.shape)
print(training_data.shape)
for i in range (0,trainingImages.shape[0]):
    training_data[:,:,i] = trainingImages[i,:,:]

number = 8   
num_pos = []
for i in range (0,trainingLabels.shape[0]):
    if trainingLabels[i] == number:
        num_pos.append(i)

num_ave_arr = np.zeros((28,28))
for i in (num_pos):
    num_ave_arr += (training_data[:,:,i]) 

#five_ave_arr =  np.floor((five_ave_arr)/(len(five_pos)))   
num_ave_arr = np.multiply(np.divide(np.subtract(num_ave_arr,np.min(num_ave_arr)),np.subtract(np.max(num_ave_arr),np.min(num_ave_arr))),255.0)
plt.figure(figsize = (2,2))
plt.title('original')
plt.imshow(num_ave_arr, cmap=plt.get_cmap('gray'))
plt.show()

Laplace_filt = Filters(num_ave_arr,**{"filter":"Laplace"}).compute()
Laplace_filt += np.abs(np.min(Laplace_filt))
#Laplace_filt = np.sqrt(Laplace_filt)
Laplace_filt *= -1.
print(np.min(Laplace_filt),np.max(Laplace_filt))
Laplace_filt = np.multiply(np.divide(np.subtract(Laplace_filt,np.min(Laplace_filt)),np.subtract(np.max(Laplace_filt),np.min(Laplace_filt))),255.0)
print(np.min(Laplace_filt),np.max(Laplace_filt))
plt.figure(figsize = (2,2))
plt.title('Laplace filter')
plt.imshow(Laplace_filt, cmap=plt.get_cmap('gray'))
plt.show()

d_mult = np.multiply(num_ave_arr,Laplace_filt)
print(np.min(d_mult),np.max(d_mult))
d_mult = np.multiply(np.divide(np.subtract(d_mult,np.min(d_mult)),np.subtract(np.max(d_mult),np.min(d_mult))),255.0)
print(np.min(d_mult),np.max(d_mult))
plt.figure(figsize = (2,2))
plt.title('original mult by Laplace')
plt.imshow(d_mult, cmap=plt.get_cmap('gray'))
plt.show()


d_filtered = Filters(num_ave_arr,**{"filter":"median"}).compute()
#d_filtered = np.multiply(d_filtered,d_filtered)
d_filtered += np.abs(np.min(d_filtered))
print(np.min(d_filtered),np.max(d_filtered))
d_filtered = np.multiply(np.divide(np.subtract(d_filtered,np.min(d_filtered)),np.subtract(np.max(d_filtered),np.min(d_filtered))),255.0)
print(np.min(d_filtered),np.max(d_filtered))
plt.figure(figsize = (2,2))
plt.title('median filter')
plt.imshow(d_filtered, cmap=plt.get_cmap('gray'))
plt.show()

d_add = np.multiply(num_ave_arr,d_filtered)
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

d_fit_4 = np.multiply(d_filt_2,num_ave_arr)
d_fit_4 += np.abs(np.min(d_fit_4))
print(np.min(d_fit_4),np.max(d_fit_4))
d_fit_4 = np.multiply(np.divide(np.subtract(d_fit_4,np.min(d_fit_4)),np.subtract(np.max(d_fit_4),np.min(d_fit_4))),255.0)
print(np.min(d_fit_4),np.max(d_fit_4))
plt.figure(figsize = (2,2))
plt.title('Sobel mult by original')
plt.imshow(d_fit_4, cmap=plt.get_cmap('gray'))
plt.show()

d_fin = np.multiply(num_ave_arr,d_filt_3)
print(np.min(d_fin),np.max(d_fin))
d_fin = np.multiply(np.divide(np.subtract(d_fin,np.min(d_fin)),np.subtract(np.max(d_fin),np.min(d_fin))),255.0)
print(np.min(d_fin),np.max(d_fin))
plt.figure(figsize = (2,2))
plt.title('Original mult by median multiply by Sobel')
plt.imshow(d_fin, cmap=plt.get_cmap('gray'))
plt.show()


d_fin_2 = np.multiply(d_mult,d_filtered)
print(np.min(d_fin_2),np.max(d_fin_2))
d_fin_2 = np.multiply(np.divide(np.subtract(d_fin_2,np.min(d_fin_2)),np.subtract(np.max(d_fin_2),np.min(d_fin_2))),255.0)
print(np.min(d_fin_2),np.max(d_fin_2))
plt.figure(figsize = (2,2))
plt.title('Original mult by median multiply by Laplace')
plt.imshow(d_fin_2, cmap=plt.get_cmap('gray'))
plt.show()

fin = np.add(d_fin,d_fin_2)
print(np.min(fin),np.max(fin))
fin = np.multiply(np.divide(np.subtract(fin,np.min(fin)),np.subtract(np.max(fin),np.min(fin))),255.0)
print(np.min(fin),np.max(fin))
plt.figure(figsize = (2,2))
plt.title('Last two added')
plt.imshow(fin, cmap=plt.get_cmap('gray'))
plt.show()
'''