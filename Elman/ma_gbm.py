import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12,30)

def Brownian(m, v):
    b = np.zeros((len(v)))
    for i in range (0,len(v)):
        if i==0:
            b[i] = np.random.normal(m[i], 0.01)
        else:
            b[i] = np.random.normal(m[i], np.absolute(v[i]))
    #W = np.cumsum(b)                                          # brownian path
    return b

def drift(x,w): #Use mean
    d = np.zeros((len(x)))
    for i in range(0,len(x)):
        if i == 0:
            d[i] = x[i]
        elif i <= w:
            d[i] = np.mean(x[:i])
        else:
            d[i] = np.mean(x[(i-w):i])
    #d =  (i) / ((N-i)*N)
    return d

def volatility (x,w): #Use standard deviation
    v = np.zeros((len(x)))
    for i in range (1,len(x)):
        if i == 1:
            v[i] = (x[i] - ((x[i]+x[i-1])/2))**2
        elif i <= w:
            v[i] = np.std(x[:i])
            if i ==1:print('check',v[i])
        else:
            v[i] = np.std(x[(i-w):i])
    return v

################### AAPL STOCK DATA (my data) #################################
training_data = [0]
trn_data_labels = ['open','high','low','close','volume']
df = pd.read_csv('/Users/Loizos/Desktop/FTSP/AAPL(3y).csv')
data = df.values[:,:].copy()
del df
 
#trim data to get open high low close and volume
in_data = np.zeros((data.shape[0],5))
in_data[:,:-1] = np.asarray(data[:,1:5], dtype = np.float64)
in_data[:,4] = np.asarray(data[:,6], dtype = np.float64)
del data

Training_length = in_data.shape[0]
Window = [120,250,500]

returns = np.zeros((len(training_data),Training_length-1))

for i in range (0, len(training_data)):
    returns[i,:] = np.ediff1d(in_data[:,training_data[i]])
    
R_Drift = np.zeros((len(training_data),len(Window),Training_length-1))
R_Volatility = np.zeros((len(training_data),len(Window),Training_length-1))
dB = np.zeros((len(training_data),len(Window),Training_length-1))
repetitions = int(20)
val = np.zeros((len(training_data),len(Window),Training_length-1,repetitions))
    
# We use a scale factor to keep the models invriant when scaling the data
data_range = np.zeros((len(training_data)))
for i in range (0,len(training_data)):
    data_range[i] = max(in_data[:,training_data[i]]) - min(in_data[:,training_data[i]])
    data_range[i] = np.log10(data_range[i])
scale_factor = np.zeros((len(training_data)))
for i in range (0,len(training_data)):
    scale_factor[i] = -(10**(-data_range[i])) * np.sqrt(np.exp(1))
del data_range

#print ('m =',scale_factor)
for l in range (0, len(training_data)):
    for k in range (0,repetitions):

        for i in range (0, len(Window)):
            R_Drift[l,i,:] = drift(returns[l], Window[i])
            R_Volatility[l,i,:] = volatility(returns[l], Window[i])
            dB[l,i,:] = Brownian(R_Drift[l,i], R_Volatility[l,i])

        for j in range (0, len(Window)):
            #val[l,j,0,k] = in_data[0,training_data[0]]
            for i in range (0, Training_length-1):
                #if i % (Window[j]*1) == 0 :
                #    val[j,i+1,k] = in_data[i,training_data[0]]
                #else:
                #    val[j,i+1,k] = (val[j,i,k] + R_Drift[j,i-1]  + scale_factor * R_Volatility[j,i-1] * dB[j,i-1])
                val[l,j,i,k] = in_data[i+1,training_data[l]] + R_Drift[l,j,i]  + scale_factor[l] * R_Volatility[l,j,i] * dB[l,j,i]
   
mean_val = np.zeros((len(training_data),len(Window),Training_length))
for k in range (0,len(training_data)):
    for j in range (0, len(Window)):
        for i in range (0, Training_length-1):
            mean_val[k,j,i] = np.mean(val[k,j,i,:])

Mean = np.zeros((len(training_data),Training_length-1))
for j in range (0,len(training_data)):
    for i in range (0, Training_length-1):
        Mean[j,i] = np.mean(mean_val[j,:,i])
        
error = np.zeros((len(training_data),Training_length-1))
for i in range (0,len(training_data)):
    error[i,:] = (in_data[1:,training_data[i]] - Mean[i,:].T) 

plt.figure(1)
plt.subplot(211)
for i in range (0,len(training_data)):
    plt.plot(in_data[:,training_data[i]], label ='actual '+str(trn_data_labels[training_data[i]]))
    plt.plot(Mean[i],'--', label ='Prediction '+str(trn_data_labels[training_data[i]]))
#for i in range (0, len(Window)):
#    plt.plot(mean_val[i],'--', label = str(Window[i])+(' MA'))
plt.legend(loc = 'upper left')

'''plt.subplot(212)
for i in range (0,len(training_data)):
    plt.plot(error[i])'''
for i in range (0,len(training_data)):
    print('Error mean '+str(trn_data_labels[training_data[i]]) +' :', np.mean(error[i,50:]),'Error Std:',np.std(error[i,50:]))
    print('Abs Error mean '+str(trn_data_labels[training_data[i]]) +' :',np.mean(np.abs(error[i,50:])),'Abs Error Std:',np.std(np.abs(error[i,50:])))

'''plt.subplot(212)
plt.scatter(error[2],error[0],label = 'High against open error')
plt.scatter(error[2],error[1],label = 'Low against open error')
plt.legend(loc = 'upper left')'''
plt.subplot(212)
plt.plot(error[0,50:])
#plt.plot(error[0])


###############################################################################
