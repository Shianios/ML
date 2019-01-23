# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 18:05:33 2018

@author: Loizos Shianios
"""
import numpy as np

class act_funcs:
    def __init__(self,data,**kwargs):
        self.kwargs = kwargs
        self.data = data
    
    def compute(self):
        return getattr(self, self.kwargs.get('func'))(self.data,**self.kwargs)
    
    # Functions meathods
    def sigmoid (self,data,**kwargs):  
        order = int(kwargs['order'])
        params = kwargs['params']
        if not params:
            s = a = 1.
            m = 0.
        else:
            if 's' in params: s = params.get('s')
            else: s = 1.
            if 'm' in params: m = params.get('m')
            else: m = 0.
            if 'a' in params: a = params.get('a')
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
            
    def tanh (self,data,**kwargs):
        order = int(kwargs['order'])
        params = kwargs['params']
        if not params:
            s = a = 1.
            m = 0.
        else:
            if 's' in params: s = params.get('s')
            else: s = 1.
            if 'm' in params: m = params.get('m')
            else: m = 0.
            if 'a' in params: a = params.get('a')
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
            
    def gauss (self,data,**kwargs):
        order = int(kwargs['order'])
        params = kwargs['params']
        if not params:
            s = a = 1.
            m = 0.
        else:
            if 's' in params: s = params.get('s')
            else: s = 1.
            if 'm' in params: m = params.get('m')
            else: m = 0.
            if 'a' in params: a = params.get('a')
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
            
    def inv_gauss (self,data,**kwargs):
        order = int(kwargs['order'])
        params = kwargs['params']
        if not params:
            s = a = 1.
            m = 0.
        else:
            if 's' in params: s = params.get('s')
            else: s = 1.
            if 'm' in params: m = params.get('m')
            else: m = 0.
            if 'a' in params: a = params.get('a')
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
    

