#import itertools
import numpy as np

dictlist = [dict() for x in range(0,3)]  

X1 = np.arange(12)
X2 = np.arange(12)
X3 = np.arange(-5,5)
X = np.concatenate((X1,X2,X3))
y = np.arange(3)

for x in (X):
    if x in dictlist[y[x%3]]: dictlist[y[x%3]][x]+=1
    else: dictlist[y[x%3]].update({x : 1})
    print(dictlist)
    print('#####################')

for i in range (0,3):
    print(dictlist[i])
    