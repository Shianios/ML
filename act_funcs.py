# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 18:05:33 2018

@author: Loizos Shianios
"""
import numpy as np
#import pandas as pd

class act_funcs:
    def __init__(self,data,**kwargs):
        if 'func' not in kwargs: kwargs['func'] = 'sigmoid'
        self.kwargs = kwargs
        self.data = data
    
    def compute(self):
        return getattr(self, self.kwargs.get('func'))(self.data,**self.kwargs)

    def param_extract_contineus(func):  # For the contineus act funcs parameters have the same form.
        def extract(*args,**kwargs):           
            if 'order' not in kwargs:
                kwargs['order'] = 0
            if 'params' in kwargs:
                params = kwargs['params']
                del kwargs['params']
            try:
                params
            except NameError:
                kwargs['s'] = 1.
                kwargs['a'] = 1.
                kwargs['m'] = 0.
            else: # Maybe one or two of the parameters is defined in the params, but not all.
                if 's' in params: kwargs['s'] = params.get('s')
                else: kwargs['s'] = 1.
                if 'm' in params: kwargs['m'] = params.get('m')
                else: kwargs['m'] = 0.
                if 'a' in params: kwargs['a'] = params.get('a')
                else: kwargs['a'] = 1.
            result = func(*args,**kwargs)
            return result      
        return extract
    
    @staticmethod
    @param_extract_contineus
    def sigmoid(data,**kwargs):
        if kwargs['order'] == 0:
            Exp = np.exp(np.add(kwargs['m'],np.multiply(-1.*kwargs['a'],data)))
            data = np.divide(kwargs['s'],np.add(1.,Exp))
            del Exp
        elif kwargs['order'] == 1:
            Exp = np.exp(np.add(kwargs['m'],np.multiply(-kwargs['a'],data)))
            Exp1 = np.add(1.,Exp)
            data = np.divide(np.multiply(kwargs['a']*kwargs['s'],Exp),np.multiply(Exp1,Exp1))
            del Exp, Exp1
        else:
            print('not implimented yet')
        return data
    
    @staticmethod
    @param_extract_contineus
    def tanh (data,**kwargs):         
        if kwargs['order'] == 0:
            d = np.add(kwargs['m'],np.multiply(kwargs['a'],data))
            data = np.multiply(kwargs['s'],np.tanh(d))
            del d
        elif kwargs['order'] == 1:
            d = np.add(kwargs['m'],np.multiply(kwargs['a'],data))
            tanh = np.tanh(d)
            data = np.multiply(kwargs['s'],np.subtract(1.,np.multiply(tanh,tanh)))
            del d, tanh
        return data
    
    @staticmethod
    @param_extract_contineus        
    def gauss (data,**kwargs):
        if kwargs['order'] == 0:
            d = np.subtract(data,kwargs['m'])
            d = np.multiply(d,d)
            d = np.divide(d,-1.*kwargs['a'])
            data = np.multiply(kwargs['s'],np.exp(d))
            del d
        elif kwargs['order'] == 1:
            d = np.subtract(data,kwargs['m'])
            Exp = np.exp(np.divide(np.multiply(d,d),-1.*kwargs['a']))
            data = np.divide(np.multiply(-2.*kwargs['a']*kwargs['s'],np.multiply(Exp,d)),kwargs['a'])
            del d, Exp
        return data
    
    @staticmethod
    @param_extract_contineus        
    def inv_gauss (data,**kwargs):          
        if kwargs['order'] == 0:
            d = np.subtract(data,kwargs['m'])
            d = np.multiply(d,d)
            d = np.divide(d,-1.*kwargs['a'])
            data = np.multiply(kwargs['s'],np.subtract(1.,np.exp(d)))
            del d
        elif kwargs['order'] == 1:
            d = np.subtract(data,kwargs['m'])
            Exp = np.exp(np.divide(np.multiply(d,d),-1.*kwargs['a']))
            data = np.divide(np.multiply(2.*kwargs['a']*kwargs['s'],np.multiply(Exp,d)),kwargs['a'])
            del d, Exp
        return data
    
'''
# Test
d = np.arange(5)
res1 = act_funcs.sigmoid(d)
res2 = act_funcs(d,**{'func':'sigmoid'}).compute()
res3 = act_funcs(d).compute()

Exp = np.exp(np.add(0.,np.multiply(-1.*1.,d)))
res4 = np.divide(1.,np.add(1.,Exp))

print(res1)
print(res2)
print(res3)
print(res4)
'''




