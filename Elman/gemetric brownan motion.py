import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (10,16)

def Brownian(m, v):
    b = np.zeros((len(v)))
    for i in range (0,len(v)):
        b[i] = np.random.normal(m[i], v[i])
    W = np.cumsum(b)                                            # brownian path
    return W, b

def drift(x,w): #Use mean
    d = np.zeros((len(x)))
    for i in range(1,len(x)):
        if i < w:
            d[i] = np.mean(x[:i])
        else:
            d[i] = np.mean(x[(i-w):i])
    #d =  (i) / ((N-i)*N)
    return d

def volatility (x,w): #Use standard deviation
    v = np.zeros((len(x)))
    for i in range (1,len(x)):
        if i <= w:
            v[i] = np.std(x[:i])
        else:
            v[i] = np.std(x[(i-w):i])
    return v

################### AAPL STOCK DATA (my data) #################################

'''def scalling(data, out_sample, d_min, d_max):
    s_min = np.zeros((data.shape[1]))
    s_max = np.zeros((data.shape[1]))
    for i in range (0,data.shape[1]):
        s_min[i] = min(data[:-out_sample,i])
        s_max[i] = max(data[:-out_sample,i])
    b = (d_max - d_min) / (s_max - s_min)
    a = (d_min / b) - s_min
    
    for i in range (0, data.shape[0]):
        for j in range(0, data.shape[1]):
            in_data[i,j] = (data[i,j] + a[j]) * b[j]
    return in_data, a, b'''

test_data = 0.0
training_data = [1]
#data_range_min = -1.0
#data_range_max = 1.0
#win = int(20)

df = pd.read_csv('/Users/Loizos/Desktop/work/FTSP/AAPL(3y).csv')
data = df.values[:,:].copy()
del df
 
#trim data to get open high low close and volume
in_data = np.zeros((data.shape[0],5))
in_data[:,:-1] = np.asarray(data[:,1:5], dtype = np.float64)
in_data[:,4] = np.asarray(data[:,6], dtype = np.float64)
del data
#scale values
out_sample = int(test_data * in_data.shape[0])
if out_sample <= 1: out_sample = 2

#in_data, A, B = scalling(in_data, out_sample, data_range_min, data_range_max)

Training_length = in_data.shape[0] - out_sample 
Window = [5,10,20,40,60,80,120,160,200]

returns = in_data[1:-(out_sample-1),training_data[0]] - in_data[:-out_sample,training_data[0]]
#print('Returns',returns)
R_Drift = np.zeros((len(Window),Training_length))
R_Volatility = np.zeros((len(Window),Training_length))
dB = np.zeros((len(Window),Training_length))
repetitions = int(10)
val = np.zeros((len(Window),Training_length,repetitions))

#Drift = drift(in_data[:Training_length,training_data[0]], Window)
#Volatility = volatility(in_data[:Training_length,training_data[0]], Window)
    
# We use a scale factor to keep the models invriant when scaling the data
data_range = max(in_data[:-out_sample,training_data[0]]) - min(in_data[:-out_sample,training_data[0]])
data_range = np.log10(data_range)
scale_factor = -(2**(-data_range)) * np.sqrt(np.exp(1))
del data_range

print ('m =',scale_factor)
for k in range (0,repetitions):

    for i in range (0, len(Window)):
        R_Drift[i,:] = drift(returns, Window[i])
        R_Volatility[i,:] = volatility(returns, Window[i])
        dB[i,:] = Brownian(R_Drift[i], R_Volatility[i])[1]

    for j in range (0, len(Window)):
        val[j,0,k] = in_data[0,training_data[0]]
        for i in range (0, Training_length-1):
            #if i % (Window[j]*1) == 0 :
            #    val[j,i+1,k] = in_data[i,training_data[0]]
            #else:
            #    val[j,i+1,k] = (val[j,i,k] + R_Drift[j,i-1]  + scale_factor * R_Volatility[j,i-1] * dB[j,i-1])
            val[j,i+1,k] = (in_data[i,training_data[0]] + R_Drift[j,i-1]  + scale_factor * R_Volatility[j,i-1] * dB[j,i-1])
   
mean_val = np.zeros((len(Window),Training_length))
for j in range (0, len(Window)):
    for i in range (0, Training_length):
        mean_val[j,i] = np.mean(val[j,i,:])

Mean = np.zeros(Training_length)
for i in range (0, Training_length):
   Mean[i] = np.mean(mean_val[:,i])

error = (in_data[:-out_sample,training_data[0]] - Mean) #/ in_data[:-out_sample,training_data[0]]

plt.figure(1)
plt.subplot(211)
plt.plot(in_data[:-out_sample,training_data[0]], label ='actual')
plt.plot(Mean, label ='Mean')
#for i in range (0, len(Window)):
#    plt.plot(mean_val[i],'--', label = str(Window[i])+(' MA'))
plt.legend(loc = 'upper left')

plt.subplot(212)
plt.plot(error)
print('Error mean:', np.mean(error),'Error Std:',np.std(error))
print('Abs Error mean:',np.mean(np.abs(error)),'Abs Error Std:',np.std(np.abs(error)))

###############################################################################
