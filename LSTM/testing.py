# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 19:08:29 2018

@author: Loizos
"""
import numpy as np
import LSTM2 as LSTM2
import matplotlib.pyplot as plt

kwargs = {"func_cif": 'gauss' , "params_cif": {"s":1.,"m":0.,"a":1.}, 
          "func_in": 'tanh' , "params_in": {"s":1.,"m":0.,"a":1.},
          "func_out": 'tanh' ,"params_out": {"s":1.,"m":0.,"a":1.},
          "func_p": 'gauss' ,"params_p": {"s":1.,"m":0.,"a":1.}}

mem_size = 10
out_size = 1
in_size = 1
#in_data = np.divide( np.arange( -in_size + np.floor(in_size/2),np.floor(in_size/2) ).reshape(in_size,1) ,5.)
# sin wave example data seq_length points
seq_length = 100
data_time_steps = np.linspace(2, 10, seq_length + 1)
data_ex = np.sin(data_time_steps)
data_ex.resize((seq_length + 1,1))
in_data = np.asarray(data_ex)
out = np.zeros((seq_length+1,1))
N1 = LSTM2.LSTM2(in_data[0],in_size,out_size,mem_size,**kwargs)
for i in range(1,seq_length):
    N1.feed_forward()
    out[i] = N1.out
    N1.input = in_data[i]
  
results = np.concatenate((in_data,out), axis=1)    
#print('out',results)

plt.figure(1)
plt.plot(in_data)
plt.plot(out)