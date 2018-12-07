# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 19:08:29 2018

@author: Loizos
"""
import numpy as np
import LSTM2 as LSTM2

kwargs = {"func_cif": 'gauss' , "params_cif": {"s":1.,"m":0.,"a":1.}, 
          "func_in": 'tanh' , "params_in": {"s":1.,"m":0.,"a":1.},
          "func_out": 'tanh' ,"params_out": {"s":1.,"m":0.,"a":1.},
          "func_p": 'inv_gauss' ,"params_p": {"s":1.,"m":0.,"a":1.}}

mem_size = 500
out_size = 10
in_size = 50
in_data = np.divide( np.arange( -in_size + np.floor(in_size/2),np.floor(in_size/2) ).reshape(in_size,1) ,5.)
print(in_data)

N1 = LSTM2.LSTM2(in_data,in_size,out_size,mem_size,**kwargs)
N1.feed_forward()
print('out',N1.out)
del in_data
in_data = np.divide(np.arange(-in_size,0).reshape(in_size,1),5.)
print(in_data)
N1.input = in_data
N1.feed_forward()
print('out',N1.out)
N2 = LSTM2.LSTM2(in_data,in_size,out_size,mem_size,**kwargs)
N2.feed_forward()
print('out 2',N2.out)
