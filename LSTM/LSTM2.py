# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 19:47:01 2018

@author: Loizos
"""

import act_funcs as act_funcs
import numpy as np

class LSTM2:
    def __init__ (self, inpt, in_size, out_size, mem_size, **kwargs):
        self.input = inpt
        self.cif = np.zeros((mem_size,1))
        self.C = np.zeros((mem_size,1))
        self.L = np.zeros((mem_size,1))
        self.S = np.zeros((mem_size,1))
        self.O = np.zeros((out_size,1))
        self.out = np.zeros((out_size,1))
        self.W_c = LSTM2.initialize(mem_size,in_size+out_size)
        d = out_size + in_size + 2*mem_size
        self.W_cif = LSTM2.initialize(mem_size,d)
        self.W_o = LSTM2.initialize(out_size,d)
        del d
        self.W_p = np.full((out_size,2*mem_size),1.0)
        self.kwargs_cif = {"func": kwargs['func_cif'],"order":0,"params":kwargs['params_cif']}
        self.kwargs_in = {"func": kwargs['func_in'],"order":0,"params":kwargs['params_in']}
        self.kwargs_out = {"func": kwargs['func_out'],"order":0,"params":kwargs['params_out']} 
        self.kwargs_p = {"func": kwargs['func_p'],"order":0,"params":kwargs['params_p']} 
       
    def feed_forward(self):
        self.x1 = np.concatenate((self.out,self.input))
        self.x2 = np.concatenate((self.x1,self.L,self.S))
        self.cif = act_funcs.act_funcs(np.matmul(self.W_cif,self.x2),**self.kwargs_cif).compute() 
        self.C = act_funcs.act_funcs(np.matmul(self.W_c,self.x1),**self.kwargs_in).compute()
        self.L = np.add(np.multiply(self.cif,self.L), np.multiply(np.subtract(1.,self.cif), self.C))
        self.S = np.add(np.multiply(np.subtract(1.,self.cif),self.S), np.multiply(self.cif, self.C))
        self.x3 = np.concatenate((self.x1,self.L,self.S))
        self.O = act_funcs.act_funcs(np.matmul(self.W_o,self.x3),**self.kwargs_out).compute()
        self.x4 = np.concatenate((self.L,self.S))
        self.out = np.multiply(self.O,act_funcs.act_funcs(np.matmul(self.W_p,self.x4),**self.kwargs_p).compute())
        
    @staticmethod
    def initialize(*shape): 
        #return np.full((shape),0.5)
        #np.random.seed(5)
        return np.random.normal(0.0, 0.5, shape)
    
    