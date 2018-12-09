import numpy as np
import pandas as pd
import csv
import time
import preconditioned_stochastic_gradient_descent_Copy as psgd 

# Elmann Recurrent Neural Network. Specificaly design to process time series data. May eneter an arbitary number of time series as input
# and produce any number of predicted time series.

class ElmanRNN: # Each reccurrent unit is fed from one hidden neuron but each recurrent unit feeds all hidden neurons
    def __init__(self, data, data_size, num_input, num_hidden, num_classes, win, target, min_error, learning_rate, max_iter,scale_min, scale_max):
        # !!!Note: There is no need to declaire all this self. instances. In this way we made avery parameter of the network and its training
        # meathods available to every function instead of passing what is need to where and when is needed. At this point we care to code
        # the algorithm as quickly as possible, but we need to fix this in future version.
        
        self.data = data
        self.data_size = data_size
        self.data_points = data.shape[0]
        self.num_input = num_input
        self.w_in = ElmanRNN.initialize(num_input, num_hidden)
        self.dw_in = np.zeros((num_input, num_hidden))
        self.num_hidden = num_hidden
        self.recurrent = np.zeros((1, num_hidden))
        self.old_recurrent = np.zeros((1, num_hidden))
        self.hidden = np.zeros((1, num_hidden))
        self.w_h = ElmanRNN.initialize(1, num_hidden)
        self.dw_h = np.zeros((1, num_hidden))
        self.num_classes = num_classes
        self.w_out = ElmanRNN.initialize(num_hidden, num_classes)
        self.dw_out = np.zeros((num_hidden, num_classes))
        self.out = np.zeros((1, num_classes))
        self.target = np.asarray(target)
        self.min_error = min_error
        self.learning_rate = learning_rate
        self.errors = np.zeros((data_size - 1, num_classes))
        self.stef_coeff = np.zeros((data_size-1, num_classes))
        self.Error = 1.0
        self.max_iter = max_iter
        self.step = 0
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.win = win
        #Scalling Factors to be used when scalling back to original values out of the network.
        self.A = np.zeros((self.num_input))
        self.B = np.zeros((self.num_input))
   
    def scalling(self, d_min, d_max):
        s_min = np.zeros((self.num_input))
        s_max = np.zeros((self.num_input))
        in_data = np.zeros((self.data_points,self.num_input))
        s_min[2] = min(self.data[:,2])*0.9
        s_max[1] = max(self.data[:,1])*1.1
        for i in range (0,self.num_input-1):
            s_min[i] = s_min[2]
            s_max[i] = s_max[1]
        s_min[4] = min(self.data[:,4])*0.9
        s_max[4] = max(self.data[:,4])*1.1
        for i in range (0,self.num_input):
            self.B[i] = (d_max - d_min) / (s_max[i] - s_min[i])
            self.A[i] = (d_min / self.B[i]) - s_min[i]
        for i in range (0, self.data_points):
            for j in range(0, self.num_input):
                in_data[i,j] = (self.data[i,j] + self.A[j]) * self.B[j]
        del s_min, s_max
        return in_data
     
    def feedforward(self, inp):
        self.hidden[0,:] = np.matmul(inp, self.w_in) + np.sum(self.recurrent)
        for i in range (0, self.num_hidden):
            #self.hidden[0, i] = self.sigmoid(self.hidden[0, i])                                     #Sigmoid activation function
            self.hidden[0, i] = np.tanh(self.hidden[0,i])                                              #Tanh activation function
        self.old_recurrent = self.recurrent.copy()
        self.recurrent[0, :] = np.multiply(self.hidden, self.w_h)
        self.out[0, :] = np.matmul(self.hidden, self.w_out)
            
    def stef(self):         #Stochastic Time Effective Function     
        #Drift function
        Drift = np.zeros((self.data_points))
        for i in range (0, self.data_points):
            Drift[i] = i / ((self.data_points - i) * self.data_points)
        
        #Volatility function
        Volatility = np.zeros((self.data_points, self.num_classes))
        for i in range (0,self.num_classes):
            for j in range(0,self.data_points):
                if j == 0:
                    Volatility[0,i] = np.abs(self.data[0,i])
                elif j <= self.win:
                    Volatility[j,i] = np.std(self.data[:j,i])
                else :
                    Volatility[j,i] = np.std(self.data[(j-self.win):j,i])
        
        
        # Construct Brownian path from returns, i.e. use mean and std of returns for random distribution
        r = 1
        if r == 1:
            returns = np.zeros((self.data_points, self.num_classes))
            for i in range (0, self.num_classes):
                for j in range (1, self.data_points):
                    returns[j,i] = self.data[j,self.target[i]] - self.data[j-1,self.target[i]]
        elif r == 2:
            returns = np.zeros((self.data_points, self.num_classes))
            for i in range (0, self.num_classes):
                returns[:,i] = self.data[:,self.target[i]]
        
        mean = np.zeros((self.data_points, self.num_classes))
        sigma = np.zeros((self.data_points, self.num_classes))
        #np.random.seed(1)
        dB = np.zeros((self.data_points, self.num_classes))
        for i in range (0, self.num_classes):
            for j in range (0, self.data_points):
                if j == 0:
                    mean[0,i] = returns[0,i]
                    sigma[0,i] = np.abs(returns[0,i])
                elif j <= self.win:
                    mean[j,i] = np.mean(returns[:(j+1),i])
                    sigma[j,i] = np.std(returns[:(j+1),i])
                else:
                    mean[j,i] = np.mean(returns[(j-self.win):(j+1),i])
                    sigma[j,i] = np.std(returns[(j-self.win):(j+1),i])
                dB[j,i] = np.random.normal(mean[j,i], sigma[j,i])
        #number = int(input('Enter a number: '))
        Volatility = Volatility * dB      
        for i in range (0, self.num_classes):
            Volatility[:,i] = np.cumsum(Volatility[:,i])
        
        for i in range (0, self.num_classes):
            for j in range (0, self.data_points-1):
                #self.stef_coeff[j,i] = (np.abs(np.tanh(returns[j,i]))+1) * np.exp(Drift[j]  + Volatility[j,i])
                self.stef_coeff[j,i] = (1/self.B[self.target[i]]) * np.exp(Drift[j]  + Volatility[j,i])
                
        del Drift, Volatility, mean, sigma, dB, returns
        
    def err_calc(self, iter):           #Actual errors multipplied by stef coefficient
        for i in range (0 ,self.num_classes):
            self.errors[iter, i] = np.subtract(self.data[iter+1, self.target[i]], self.out[0,i])
# Two training methods, first one uses standard Stochastic Gredient Discent (SGD). Second uses Precontition Stochastc Gredient Discent (PSGD)
# In general PSGD is 3 to 4 times slower than SGD. But it guarantees solution, no error explotion and most of times greater accuracy
# in a given number of steps. Both uses real time recurrent learning (RTRL), i.e. derivatives are calculated from first data pointto the last.
# In both we use mini butch training, i.e. one data point at each training instance. Grads function is used for PSGD, weights_update_RTRL
# is used for SGD. One of the two is used for training. The choice is yours.
# !!!NOTE: Both meathods are not functions of the recurrent network, they should be moved out of the class in future versions.

    def wheights_update_RTRL(self, iter):
        #Update input to hidden matrix
        for j in range (0, self.num_hidden):
            for i in range (0, self.num_input):
                for k in range (0, self.num_classes):
                    #self.w_in[i,j] += 2 * self.learning_rate * self.errors[iter,k] * self.w_out[j,k] * self.dif_sigmoid(self.hidden[0,j]) * self.data[iter, i]                          #For sigmoid activation function
                    self.w_in[i,j] += self.learning_rate * (self.stef_coeff[iter,k] * self.errors[iter,k] * self.w_out[j,k] * (1/(np.cosh(np.arctanh(self.hidden[0,j])))**2) * self.data[iter, i])            #For tanh activation function

        # Update hidden to reccurrent unit matrix
        for j in range (0, self.num_hidden):
            for k in range (0, self.num_classes):
                #self.w_h[0,j] += self.learning_rate * self.errors[iter,k] * self.w_out[j,k] * self.dif_sigmoid(self.hidden[0,j]) * self.old_recurrent[0, j]                        #For sigmoid activation function
                self.w_h[0,j] += self.learning_rate * (self.stef_coeff[iter,k] * self.errors[iter,k] * self.w_out[j,k] * (1/(np.cosh(np.arctanh(self.hidden[0,j])))**2) * self.old_recurrent[0, j])      #For tanh activation function

        # Update hidden to out matrix
        for i in range (0, self.num_classes):    
            for j in range (0, self.num_hidden):    
                self.w_out[j,i] += self.learning_rate * self.stef_coeff[iter,i] * self.errors[iter,i] * self.hidden[0,j]                                                   

    def train_RTRL_online(self):            #online stands for mini butch size 1.
        self.step = 0
        self.data= self.scalling(self.scale_min,self.scale_max).copy()
        self.stef()
        #Each time we call training we set recurrent units to zero
        while self.Error > self.min_error and self.step < self.max_iter:
            self.recurrent = np.zeros((1, self.num_hidden))
            self.old_recurrent = np.zeros((1, self.num_hidden))
            for i in range (0, self.data_points - 1):
                self.feedforward(self.data[i])
                self.err_calc(i)
                self.wheights_update_RTRL(i)
            self.Error = (np.sum(np.multiply(self.errors , self.errors)))/(self.errors.shape[0])
            print (self.step, ': Error -', self.Error)
            self.step +=1
        #print (self.step, ': Error -', self.Error)

    def grads(self, iter):              #Derivatives of Error in rispect to weight matrices
        #input to hidden matrix
        for j in range (0, self.num_hidden):
            for i in range (0, self.num_input):
                for k in range (0, self.num_classes):
                    #self.w_in[i,j] += 2 * self.learning_rate * self.errors[iter,k] * self.w_out[j,k] * self.dif_sigmoid(self.hidden[0,j]) * self.data[iter, i]                          #For sigmoid activation function
                    self.dw_in[i,j] = -(self.stef_coeff[iter,k] * self.errors[iter,k] * self.w_out[j,k] * (1/(np.cosh(np.arctanh(self.hidden[0,j])))**2) * self.data[iter, i])                                 #For tanh activation function

        #hidden to reccurrent unit matrix
        for j in range (0, self.num_hidden):
            for k in range (0, self.num_classes):
                #self.w_h[0,j] += self.learning_rate * self.errors[iter,k] * self.w_out[j,k] * self.dif_sigmoid(self.hidden[0,j]) * self.old_recurrent[0, j]                        #For sigmoid activation function
                self.dw_h[0,j] = -(self.stef_coeff[iter,k] * self.errors[iter,k] * self.w_out[j,k] * (1/(np.cosh(np.arctanh(self.hidden[0,j])))**2) * self.old_recurrent[0, j])                            #For tanh activation function

        #hidden to out matrix
        for i in range (0, self.num_classes):    
            for j in range (0, self.num_hidden):    
                self.dw_out[j,i] = -self.stef_coeff[iter,i] * self.errors[iter,i] * self.hidden[0,j]                                               

        out = np.vstack((self.dw_in.reshape((-1,self.num_hidden)), self.dw_h.reshape((-1,self.num_hidden)), (self.dw_out.T).reshape((-1,self.num_hidden)))) 
        return (out)
    
    def train_PSGD(self):
        self.step = 0
        self.data = self.scalling(self.scale_min,self.scale_max)
        #Each time we call training we set recurrent units to zero
        while self.Error > self.min_error and self.step < self.max_iter:
            self.stef()
            self.recurrent = np.zeros((1, self.num_hidden))
            self.old_recurrent = np.zeros((1, self.num_hidden))
            Q = np.eye((self.num_input + self.num_classes + 1) * self.num_hidden)
            for i in range (0, self.data_points-1):
                ini_recurrent = self.recurrent.copy()
                ini_old_recurrent = self.old_recurrent.copy()
                #carry one feed forward and store parameters for later usage within current taining epoch
                self.feedforward(self.data[i])
                self.err_calc(i)
                old_grad = self.grads(i).copy()
                after_recurrent = self.recurrent.copy()
                after_old_recurrent = self.old_recurrent.copy()
                after_errors = self.errors.copy()
                
                # no need to update Q in every iteration. So I introduce Q_update_gap
                # When Q_update_gap >> 1, PSGD becomes SGD 
                Q_update_gap = max(int(np.floor(np.log10(i+1.0))), 1) 
                if i % Q_update_gap == 0:
                    # calculate a perturbed gradient
                    sqrt_eps = np.sqrt(np.finfo('float32').eps)
                    delta_theta = np.random.normal(0.0, sqrt_eps, self.num_input * self.num_hidden)
                    delta_theta = delta_theta.reshape((-1,self.num_hidden))
                    delta_theta = np.vstack((delta_theta, (np.random.normal(0.0, sqrt_eps, self.num_hidden)).reshape(-1,self.num_hidden)))
                    delta_theta = np.vstack((delta_theta, (np.random.normal(0.0, sqrt_eps, self.num_hidden * self.num_classes).reshape(-1,self.num_hidden))))
                    #add small random pertubation delta theta to weights
                    for j in range (0, delta_theta.shape[0]):
                        if j < self.num_input:
                            self.w_in[j,:] += delta_theta[j,:]
                        elif j >= self.num_input and j < (self.num_input + 1):
                            self.w_h[j-self.num_input] += delta_theta[j,:]
                        elif j >= self.num_input + 1 and j < (self.num_input + 1 + self.num_classes) :
                            self.w_out[:,j - self.num_input - 1] += delta_theta[j,:].T
                        else:
                            print('Error: j out of range')
                    #set recurrent units to initial values i.e. before initial pertubation
                    self.recurrent = ini_recurrent
                    self.old_recurrent = ini_old_recurrent
                    self.feedforward(self.data[i])
                    self.err_calc(i)
                    delta_grad = self.grads(i) - old_grad
                    #set weights back to values before pertubation
                    for j in range (0, delta_theta.shape[0]):
                        if j < self.num_input:
                            self.w_in[j,:] -= delta_theta[j,:]
                        elif j >= self.num_input and j < (self.num_input + 1):
                            self.w_h[j-self.num_input] -= delta_theta[j,:]
                        elif j >= self.num_input + 1 and j < (self.num_input + 1 + self.num_classes) :
                            self.w_out[:,j - self.num_input - 1] -= delta_theta[j,:].T
                        else:
                            print('Error: j out of range')
                    #set recurrent units to values before pertubation
                    self.recurrent = after_recurrent.copy()
                    self.old_recurrent = after_old_recurrent.copy()
                    # update the preconditioner 
                    Q = psgd.update_precond_dense(Q, delta_theta, delta_grad)       
        
                # update weights; propose to clip the preconditioned gradient
                pre_grad = psgd.precond_grad_dense(Q, old_grad)
                for j in range (0, pre_grad.shape[0]):
                    if j < self.num_input:
                        self.w_in[j,:] -= self.learning_rate*(pre_grad[j,:] / max(1.0, np.sqrt(np.sum(pre_grad[j,:]*pre_grad[j,:]))))
                    elif j >= self.num_input and j < (self.num_input + 1):
                        self.w_h[j-self.num_input] -= self.learning_rate*(pre_grad[j,:] / max(1.0, np.sqrt(np.sum(pre_grad[j,:]*pre_grad[j,:]))))
                    elif j >= self.num_input + 1 and j < (self.num_input + 1 + self.num_classes) :
                        self.w_out[:,j - self.num_input - 1] -= self.learning_rate*(pre_grad[j,:].T / max(1.0, np.sqrt(np.sum(pre_grad[j,:]*pre_grad[j,:]))))
                    else:
                        print('Error: j out of range')
                self.errors = after_errors
                #number = int(input('Enter a number: '))
            self.Error = (np.sum(np.multiply(self.errors , self.errors)))/(self.errors.shape[0])
            if self.step %10 ==0 :print (self.step, ': Error -', self.Error)
            self.step += 1
        #print (self.step, ': Error -', self.Error)
        
    #Meathods to initialize random matrices
    #@staticmethod
    #def initialize(*shape): return 0.5 * np.random.uniform(-1., 1., shape)
    @staticmethod
    def initialize(*shape): 
        #np.random.seed(5)
        return np.random.normal(0.0, 0.4, shape)
    
    # Meathods to use sigmoid as activation function in hidden layer 
    #(sigmoid is not a member function of any python library so we need to define it and its derivative)
    #@staticmethod
    #def sigmoid (x): return 1/(1 + np.exp(-x))
    #@staticmethod
    #def dif_sigmoid (x): return np.exp(-x)/((1 + np.exp(-x))**2)

#####################################################################################################

#Data and Network Parameters

test_data = 0.333                       #The percentage size of testing data. It is subtracted from the total data to create the training data set and testing set                        
training_data = [1]                     #Always a list. More than just one number can be entered to traine on multiple points in the same network and return a vector. Use [0] for sigle input data series.
I = 1                                   #Iteratotr use to repeat experiments. To change a parameter comand must be inserted in the loop
Results = np.zeros((I, 4))              #Use to store results and performance of repeated same experiments.
data_range_min = -1.00                  #use to rescale data in range min - max. Usually values range from -1 to 1. No need to be symetric around 0.
data_range_max = 1.00                   #Another usual option is 0 to 1. min always smaller than max

#Network parameters
Min_Error = 0.0001                      #Minimum accepted error
l_rate = 0.003                          #learning rate
Window = 10                             #Window used for calculating mean and sigma of normal distribution in brownian paths
Max_steps = 300                         #Maximum learning iterations
No_hidden = 11                          #Number of hidden layears
s = 2                                   #Switch to choose between 1 = RTRL(SGD) or 2 = PSGD
new_Max_steps = 30                      #use to set max learning iterations in training during out of sample testing.
new_l_rate = 0.001                      #use to set learning rate in training during out of sample testing.


#use n to choose which data to be used 
# (1: sin wave with 2000 data points, 2 same sin wave as 1 but with noise(1/3 of range is actualy noise),
#  3: sin wave with fewer data points(it can be set to any number), 4: random time series, 5: Apple's stock price time series)
n = 5

if n == 1 :
    #sin wave with no noise 2000 points
    with open('/Users/Loizos/Desktop/work/FTSP/sin_no_noise.csv') as f1:
        sin_w = []
        for line in f1:
            line = line.split() # to deal with blank 
            if line:            # lines (ie skip them)
                line = [float(i) for i in line]
                sin_w.append(line)
    f1.close()
    in_data = np.asarray(sin_w)
    out_sample = int(test_data * in_data.shape[0])
    
elif n ==2:
    #sin wave with noise 2000 points
    with open('/Users/Loizos/Desktop/sin.csv') as f1:
        sin_w_n = []
        for line in f1:
            line = line.split() # to deal with blank 
            if line:            # lines (ie skip them)
                line = [float(i) for i in line]
                sin_w_n.append(line)
    f1.close()
    in_data = np.asarray(sin_w_n)
    out_sample = int(test_data * in_data.shape[0])

elif n == 3:
    #sin wave example data seq_length points
    seq_length = 100
    data_time_steps = np.linspace(2, 10, seq_length + 1)
    data_ex = np.sin(data_time_steps)
    data_ex.resize((seq_length + 1, 1))
    in_data = np.asarray(data_ex)
    out_sample = int(test_data * in_data.shape[0])

elif n == 4:
    # random data
    dfile =  open('/Users/Loizos/Desktop/work/FTSP/q-test_results.txt', 'r')

    in_dtable = []
    for eachLine in dfile:
        in_line = list()
        numbers = eachLine.split()
        if numbers:
            number = [float(number) for number in numbers]
            in_line.append(number)
        in_dtable.append(in_line) 
        del in_line
    dfile.close()

    data = np.asarray(in_dtable)
    in_data = np.reshape(data, (data.shape[0],data.shape[2]))
    del in_dtable
    del data
    
elif n==5:
    #Select data
    z = 1
    if z == 1:
        df = pd.read_csv('/Users/Loizos/Desktop/work/FTSP/AAPL(3y).csv')
        f = ',AAPL(3y)'
    elif z == 2:
        df = pd.read_csv('/Users/Loizos/Desktop/work/FTSP/^N225.csv',na_values=[' null',' nan'])
        f = ',Nikki225(8y)'
    elif z == 3:
        f = ',BAC(3y)'
        df = pd.read_csv('/Users/Loizos/Desktop/work/FTSP/BAC.csv')
    
    #To select starting day
    #df = df.loc[df['Date'] >= '2015-12-31']
    
    #To remove dates and adj. close
    cols = [c for c in df.columns if (c != 'Adj Close' and c!= 'Date')]
    df = df[cols]
    # To drop any rows with nans or infinite values
    Headers = list(df.columns.values)
    for x in Headers:
        df = df[np.isfinite(df[str(x)])]
    # To choose weather to use row data returns or log returns
    d=1
    if d==1:
        in_data = df.values.copy()
        del df
    elif d==2:
        data = df.values.copy()
        in_data = np.zeros((data.shape[0]-1,data.shape[1]))
        for i in range (0,data.shape[0]-1):
            in_data[i,:] = data[i+1,:] - data[i,:]
        del df
    elif d==3:
        data = df.values.copy()
        in_data = np.zeros((data.shape[0]-1,data.shape[1]))
        for i in range (0,data.shape[0]-1):
            in_data[i,:] = np.log(data[i+1,:]/data[i,:])
        del df
    
    out_sample = int(test_data * in_data.shape[0])
    print('Data size', in_data.shape[0],'out_sample', out_sample)
    #number = int(input('Enter a number: '))
else :
    print("Re ilithie vale enan arithmo pou to 1 mexri to 5")
    

##################################################################################################

for a in range (0, I):
    N1 = ElmanRNN(in_data[:-out_sample,:], in_data.shape[0] - out_sample,in_data.shape[1], No_hidden, len(training_data),Window, training_data, Min_Error, l_rate, Max_steps,data_range_min ,data_range_max)
    # To select whether to insert weights from previous test or to export current tests weights

    r = 2
    
    if r == 1 : # To import weights from previous run
        N1.max_iter = 10      #If already trained weights are imported no need to go through full retraining
        df = pd.read_csv('/Users/Loizos/Desktop/work/FTSP/weights/ElmanRNN_RTRL(weights,11,0.01,-1.0,1.0,BAC).csv')
        for i in range (0, N1.num_input):
            in_line = df.values[i]
            N1.w_in[i,:] = [x for i,x in enumerate(in_line) if i!=0]
        for i in range (0, 1):
            in_line = df.values[i+N1.num_input]
            N1.w_h[i,:] = [x for i,x in enumerate(in_line) if i!=0]
        for i in range (0, N1.num_classes):
            in_line = df.values[i+N1.num_input+1]
            N1.w_out[:,i] = np.asarray([x for i,x in enumerate(in_line) if i!=0]).T
        #number = int(input('Enter a number: '))

    print ('#################### Commence training: ', a +1, ' ####################')
    ini_time = time.time()
    if s == 1:
        N1.train_RTRL_online()
    elif s ==2:
        N1.train_PSGD()
    else:
        while s !=1 or s !=2:
            s = int(input('Enter 1 or 2 VLAKA! : '))
    #number = int(input('Enter a number: '))
    Results[a,3] = (time.time() - ini_time)
    Results[a,0] = a
    Results[a,1] = N1.Error
    Results[a,2] = N1.step
    #del N1

print('Total Time:', np.sum(Results[:,3]))
print('Total Error:', np.sum(Results[:,1]))
print('Average Error:', np.sum(Results[:,1]/Results.shape[0]))

'''
file_name = '/Users/Loizos/Desktop/work/FTSP/ElmanRNN_'
if s == 1:
    file_name += 'RTRL('
elif s == 2:
    file_name += 'PSGD('
file_name += str(No_hidden)
file_name += ','
file_name += str(l_rate)
file_name += ','
file_name += str(data_range_min)
file_name +=','
file_name += str(data_range_max)
file_name += ')'

with open('%s.csv' % file_name, 'w') as csvfile:
        fieldnames = ['Iteration', 'Error', 'Step', 'Duration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range (0, I):
            writer.writerow({'Iteration': Results[i,0], 'Error': Results[i,1], 'Step': Results[i,2], 'Duration': Results[i,3]})
        writer = csv.writer(csvfile)
        writer.writerow(['Total Time', np.sum(Results[:,3])])
        writer.writerow(['Average Error', np.sum(Results[:,1]/Results.shape[0])])
csvfile.close()
'''

list1 = list(range(1,N1.w_in.shape[0]+1))
list2 = list(range(1,N1.w_in.shape[1]+1))
W_in = pd.DataFrame(N1.w_in, index = list1, columns = list2)
list1 = list(range(1,N1.w_h.shape[0]+1))
list2 = list(range(1,N1.w_h.shape[1]+1))
W_h = pd.DataFrame(N1.w_h, index = list1, columns = list2)
list1 = list(range(1,N1.w_out.T.shape[0]+1))
list2 = list(range(1,N1.w_out.T.shape[1]+1))
W_out = pd.DataFrame(N1.w_out.T, index = list1, columns = list2)
Ws = pd.concat([W_in, W_h, W_out])

file_name = '/Users/Loizos/Desktop/FTSP/weights/(BAC)ElmanRNN_'
if s == 1:
    file_name += 'RTRL('
elif s == 2:
    file_name += 'PSGD('
file_name += 'weights, trn_d_pnts('
file_name += str(in_data.shape[0] - out_sample)
file_name += '), test_pnts('
file_name += str(out_sample)
file_name += '), No_h('
file_name += str(No_hidden)
file_name += '), l_r('
file_name += str(l_rate)
file_name += '), d_range('
file_name += str(data_range_min)
file_name +='_'
file_name += str(data_range_max)
file_name +='), steps('
file_name += str(Max_steps)
file_name += '), window('
file_name += str(Window)
file_name += '), train_data'
file_name += str(training_data)
file_name += f
if d == 2:
    file_name +='R'
elif d == 3:
    file_name +='Ln(R)'
file_name += '}'
Ws.to_csv('%s.csv' % file_name)

del W_in, W_h, W_out, Ws

##################################### Out of sample test #################################################

print('Comence out of sample testing')
del Results
Results = np.zeros((out_sample,len(training_data),3))
N1.max_iter = new_Max_steps
N1.learning_rate = new_l_rate
for a in range (0, out_sample):
    N1.Error = 1.0
    N1.feedforward(N1.data[N1.data_points-1])  
    print(a+1, '/', out_sample, 'Feed:')
    if d == 1:
        for i in range (0,len(training_data)):
            print((N1.data[N1.data_points - 1,training_data[i]] / N1.B[training_data[i]]) - N1.A[training_data[i]])
        for i in range (0,len(training_data)):
            Results[a,i,0] = (N1.out[0,i] / N1.B[training_data[i]]) - N1.A[training_data[i]]
            Results[a,i,1] = in_data[in_data.shape[0] - out_sample + a, training_data[i]]
            Results[a,i,2] = Results[a,i,1] - Results[a,i,0]
    elif d ==2:
        for i in range (0,len(training_data)):
            print((N1.data[N1.data_points - 1,training_data[i]] / N1.B[training_data[i]]) - N1.A[training_data[i]], data[data.shape[0] - out_sample + a-1, training_data[i]])
        for i in range (0,len(training_data)):
            Results[a,i,0] = (N1.out[0,i] / N1.B[training_data[i]]) - N1.A[training_data[i]] + data[data.shape[0] - out_sample + a-1, training_data[i]]
            Results[a,i,1] = data[data.shape[0] - out_sample + a, training_data[i]]
            Results[a,i,2] = Results[a,i,1] - Results[a,i,0]
    elif d ==3:
        for i in range (0,len(training_data)):
            print((N1.data[N1.data_points - 1,training_data[i]] / N1.B[training_data[i]]) - N1.A[training_data[i]], data[data.shape[0] - out_sample + a-1, training_data[i]])
        for i in range (0,len(training_data)):
            Results[a,i,0] = np.exp((N1.out[0,i] / N1.B[training_data[i]]) - N1.A[training_data[i]]) + data[data.shape[0] - out_sample + a-1, training_data[i]]
            Results[a,i,1] = data[data.shape[0] - out_sample + a, training_data[i]]
            Results[a,i,2] = Results[a,i,1] - Results[a,i,0] 
    
    print(': Results:',Results[a])
    if a < (out_sample-1):
        N1.data = in_data[(a+int(out_sample/2)):-(out_sample-a-1),:].copy()
        N1.data_points = N1.data.shape[0]
    else:
        print('Last point')
        N1.data = in_data[(a+int(out_sample/2)):,:].copy()
        N1.data_points = N1.data.shape[0]
    if s == 1:
        N1.train_RTRL_online()
    elif s ==2:
        N1.train_PSGD()
    else:
        while s !=1 or s !=2:
            s = int(input('Enter 1 or 2 VLAKA! : '))
    #number = int(input('Enter a number: '))
    print('#############################')

file_name = '/Users/Loizos/Desktop/work/FTSP/results/ElmanRNN_'
if s == 1:
    file_name += 'RTRL{'
elif s == 2:
    file_name += 'PSGD{'
file_name += 'out_sample_test, trn_pnts('
file_name += str(in_data.shape[0] - out_sample)
file_name += '), test_pnts('
file_name += str(out_sample)
file_name += '), No_h('
file_name += str(No_hidden)
file_name += '), l_r('
file_name += str(l_rate)
file_name += '), d_range('
file_name += str(data_range_min)
file_name +='_'
file_name += str(data_range_max)
file_name +='), steps('
file_name += str(Max_steps)
file_name += '), window('
file_name += str(Window)
file_name += '), trn_data'
file_name += str(training_data)
file_name += f
if d == 2:
    file_name +='R'
elif d == 3:
    file_name +='Ln(R)'
file_name += '}'
with open('%s.csv' % file_name , 'w') as csvfile:
    fieldnames = ['Prediction', 'Actual', 'Error']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range (0, Results.shape[0]):
        writer.writerow({'Prediction': Results[i,:,0], 'Actual': Results[i,:,1], 'Error': Results[i,:,2]})
csvfile.close()

#Just to wake you up after the long wait
import winsound
duration = 500  # millisecond
freq = 440  # Hz
winsound.Beep(freq, duration)
duration = 500  # millisecond
freq = 640  # Hz
winsound.Beep(freq, duration)
duration = 500  # millisecond
freq = 440  # Hz
winsound.Beep(freq, duration)
    