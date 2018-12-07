# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 18:05:33 2018

@author: Loizos Shianios
"""
import numpy as np
#import pandas as pd

class act_funcs:
    def __init__(self,data,**kwargs):
        self.kwargs = kwargs
        
        if 'order' in kwargs: order = kwargs['order']
        else: order = 0
        if 'params' in kwargs: kwargs1 = kwargs.get('params')
        else: kwargs1 = {}
        if 'func' in kwargs: pass
        else: self.kwargs = {"func": 'sigmoid'}
            
        self.funcs_ = {"sigmoid" : act_funcs.sigm_(self,data,order,**kwargs1),
                  "tanh": act_funcs.tanh_(self,data,order,**kwargs1),
                  "gauss": act_funcs.gauss_(self,data,order,**kwargs1),
                  "inv_gauss": act_funcs.inv_gauss_(self,data,order,**kwargs1)}
    
    def compute(self):
        return self.funcs_[self.kwargs.get('func')]
    
    # Functions meathods
    def sigm_(self,data,order,**kwargs1):     
        if not kwargs1:
            s = a = 1.
            m = 0.
        else:
            if 's' in kwargs1: s = kwargs1.get('s')
            else: s = 1.
            if 'm' in kwargs1: m = kwargs1.get('m')
            else: m = 0.
            if 'a' in kwargs1: a = kwargs1.get('a')
            else: a = 1.
        
        if order == 0:
            Exp = np.exp(np.add(m,np.multiply(-a,data)))
            data = np.divide(s,np.add(1.,Exp))
            del Exp
        elif order == 1:
            Exp = np.exp(np.add(m,np.multiply(-a,data)))
            Exp1 = np.add(1.,Exp)
            data = np.divide(np.multiply(a*s,Exp),np.multiply(Exp1,Exp1))
            del Exp, Exp1
        else:
            print('not implimented yet')
        del s,m,a
        return data
            
    def tanh_(self,data,order,**kwargs1):
        if not kwargs1:
            s = a = 1.
            m = 0.
        else:
            if 's' in kwargs1: s = kwargs1.get('s')
            else: s = 1.
            if 'm' in kwargs1: m = kwargs1.get('m')
            else: m = 0.
            if 'a' in kwargs1: a = kwargs1.get('a') 
            else: a = 1.
            
        if order == 0:
            d = np.add(m,np.multiply(a,data))
            data = np.multiply(s,np.tanh(d))
            del d
        elif order == 1:
            d = np.add(m,np.multiply(a,data))
            tanh = np.tanh(d)
            data = np.multiply(s,np.subtract(1.,np.multiply(tanh,tanh)))
            del d, tanh
        del s,m,a
        return data
            
    def gauss_(self,data,order,**kwargs1):
        if not kwargs1:
            s = a = 1.
            m = 0.
        else:
            if 's' in kwargs1: s = kwargs1.get('s')
            else: s = 1.
            if 'm' in kwargs1: m = kwargs1.get('m')
            else: m = 0.
            if 'a' in kwargs1: a = kwargs1.get('a') 
            else: a = 1.
            
        if order == 0:
            d = np.subtract(data,m)
            d = np.multiply(d,d)
            d = np.divide(d,-a)
            data = np.multiply(s,np.exp(d))
            del d
        elif order == 1:
            d = np.subtract(data,m)
            Exp = np.exp(np.divide(np.multiply(d,d),-a))
            data = np.divide(np.multiply(-2*a*s,np.multiply(Exp,d)),a)
            del d, Exp
        del s,m,a
        return data
            
    def inv_gauss_(self,data,order,**kwargs1):
        if not kwargs1:
            s = a = 1.
            m = 0.
        else:
            if 's' in kwargs1: s = kwargs1.get('s')
            else: s = 1.
            if 'm' in kwargs1: m = kwargs1.get('m')
            else: m = 0.
            if 'a' in kwargs1: a = kwargs1.get('a') 
            else: a = 1.
            
        if order == 0:
            d = np.subtract(data,m)
            d = np.multiply(d,d)
            d = np.divide(d,-a)
            data = np.multiply(s,np.subtract(1.,np.exp(d)))
            del d
        elif order == 1:
            d = np.subtract(data,m)
            Exp = np.exp(np.divide(np.multiply(d,d),-a))
            data = np.divide(np.multiply(2*a*s,np.multiply(Exp,d)),a)
            del d, Exp
        del s,m,a
        return data
    
'''
kwargs = {"func":'inv_gauss',"order":0,"params":{"s":1.,"m":0.,"a":1.}}

data_ex = np.ones((5,2))
data_1 = 1.
ex1 = act_funcs(data_ex,**kwargs).compute()
print(ex1,type(ex1))
print('######################')
print(data_ex)
'''

