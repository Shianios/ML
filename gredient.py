# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:46:34 2019

@author: Loizos
"""
# This is not the grad class. We check SymPy's  functionality. There seams to be 
# a problem with lambdify


import numpy as np
import sympy as sy


w = np.arange(60.).reshape(5,3,4)
h = np.arange(24.).reshape((3,4,2))
b = np.arange(10.).reshape((5,2))
t = np.arange(10.).reshape((5,2))

ind_w = ['a0','b0','c0']
ind_h = ['b0','c0','d0']
ind_b = ['a0','d0']
ind_z = ['a1','b2','c2']

W = sy.IndexedBase('W', shape=w.shape)
H = sy.IndexedBase('H', shape=h.shape)
B = sy.IndexedBase('B', shape=b.shape)
T = sy.IndexedBase('T', shape=t.shape)

O = sy.sin(sy.tensorproduct(W[ind_w],H[ind_h]) + B[ind_b])
E = (T[ind_b] - O)**2
g = sy.diff(E,W[ind_z])
print('############## O ##############')
print(O)
print()
print(sy.get_indices(O))
print()
print(sy.get_contraction_structure(O))
print('###############################')
print('############## E ##############')
print(E)
print()
print(sy.get_indices(E))
print()
print(sy.get_contraction_structure(E))
print('###############################')
print('############## g ##############')
print(g)
print()
print(sy.get_indices(g))
print()
print(sy.get_contraction_structure(g))

print('#################################################')
print("############## open g's dictionary ##############")
k = list(sy.get_contraction_structure(g).keys())
ind = 1 
for i in k:
    print(str(ind)+':',i)
    ind+=1
print()
print(sy.get_contraction_structure(g)[list(sy.get_contraction_structure(g).keys())[0]])
print()
print(sy.get_contraction_structure(g)[list(sy.get_contraction_structure(g).keys())[1]])
print('#########################################')

'''
f = sy.lambdify((sy.DenseNDimArray('X'),sy.DenseNDimArray('Y'),sy.DenseNDimArray('B')),O.doit(),'numpy')

f(w,h,b)
'''



