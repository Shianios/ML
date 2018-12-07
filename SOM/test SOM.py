# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 12:58:32 2018

@author: Loizos Shianios
"""

import SOM
import numpy as np
import matplotlib.pyplot as plt

r = 4                           # Redius of cell selection around wining neuron.
l_rate = 0.9                    # Learning rate must be less than 1.0                 
epochs = 50
Dimensions = [60]               # This are the dimensions of the feature map, e.g. 10 by 10 gid.  
                                # We can have a tensor of rank upto 26, e.g. a 5th rank tensor [20,5,30,40,10]
Dims_boundary_cond = ['o']      # Dimensions can be either open 'o' or close 'c', i.e loops
                                # A 2nd rank tensor (i.e. 2D array) with both dims close generates a 
                                # torus (doughnut), not a shpere. The default is set to open.
train = 1                       # 0 loads pretrained map and skips training

''' 
training_data = np.random.uniform(-0.5,0.5,(3,10))

Last index of shape is for the individual entries of data. The remaining 
indices are based on the data structure. E.g. a set of 10 vectors with each
vector having 3 entries will have a shape (3,10). A set of 50 matrices
with dims [5,6] will have a shape (5,6,50). A set of 20 grey scale 
images with dims 100 by 80 pixels will have shape (100,80,20). The valeus
will typically range from 0 to 1, denoting the grey scale. A set of 50
RGB images with dims 640 by 480 pixels will have a shape (640,480,3,50).
The third element of the shape with valeu 3 is for each color range (i.e.
red,green,blue ranging from 0 to 255). A set of 100 videos shot in
CMYK color mode with 6000 frames each, with dims 1024 by 1024 pixels will 
have a shape of (1024,1024,4,6000,100). Note though, that the first 4 indices
can be scrambled e.g (4,1024,6000,1024,100), but the last must always be
the number of elements in the data set. The clustering is curried over
this number. Also it is best to use a convinient ordering for the remaining indices. 
'''

# To load MNIST data set
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

# To load pretrained map and skip training
def loadpretrainSOM(name, folder, shp):
    W = np.empty((shp))
    for i in range (0, shp[0]):
        file = folder + "/" + name + str(i) + '.txt'
        W[i,...] = np.loadtxt(file)
    return W

# Reshape data set to fit SOM structure
training_data = np.empty((28,28,trainingImages.shape[0]))
test_data = np.empty((28,28,testImages.shape[0]))
for i in range (0,trainingImages.shape[0]):
    training_data[:,:,i] = trainingImages[i,:,:]
for i in range (0,testImages.shape[0]):
    test_data[:,:,i] = testImages[i,:,:]
del trainingImages, testImages

# Create SOM
SOM_1 = SOM.SOM(training_data,trainingLabels,l_rate,r,epochs,Dimensions,Dims_boundary_cond)

if train==0 : # If we have already trained SOM we can load the map
    SOM_1.W = loadpretrainSOM('results', 'C:/Users/Loizos/Desktop/assignment/SOM', SOM_1.W.shape)
else:   # Else train and save the map for future use.
    SOM_1.train()
    for i in range (0,SOM_1.W.shape[0]):
        str_name = 'results' + str(i) + '.txt'
        np.savetxt(str_name,SOM_1.W[i])
        
# We need to use 'supervision' to assign the labels to the created clusters
SOM_1.assign_label()

# Test the algorithm using the testing data
scores = np.zeros((2),dtype = np.uint16)
for i in range (0,len(testLabels)):
    if i%100==0: print(i)    
    class_ = SOM_1.classify(test_data[...,i])
    if class_ == testLabels[i]: scores[0]+=1
    else: scores[1]+=1
print(scores)

# Plot the graph of learning rate.
if train != 0 : 
    plt.figure(0)
    plt.subplot(111)
    plt.plot(SOM_1.l_rate_d)
'''
# Print the clusters created
plt.figure(figsize = (2,2))
for i in range (1,SOM_1.W.shape[0]+1):
    plt.figure(i)
    plt.subplot(111)
    plt.imshow(SOM_1.W[i], cmap=plt.get_cmap('gray'))
    plt.show()
'''

'''
for i in range (0,SOM_1.W.shape[0]):
    print(SOM_1.dictlist[i])
    print(SOM_1.dictlist[i]['max'])
'''





