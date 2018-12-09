"""
Demo code for preconditioned SGD on solving the RNN benchmark XOR problem
Long term memories are required to solve this problem successfully  
"""
import numpy as np
import matplotlib.pyplot as plt
import preconditioned_stochastic_gradient_descent_Copy as psgd 
import torch
from torch.autograd import Variable

# parameter settings
batch_size, test_seq_len0 = 100, 20
dim_in, dim_out = 2, 1
dim_hidden = 30

# generate training data for the XOR problem
def get_xor_batch( seq_len0 ):
    seq_len = round(seq_len0 + 0.1*np.random.rand()*seq_len0)
    x = np.zeros([batch_size, seq_len, dim_in])
    y = np.zeros([batch_size, dim_out])
    for i in range(batch_size):
        x[i,:,0] = np.random.choice([-1.0, 1.0], seq_len)

        i1 = int(np.floor(np.random.rand()*0.1*seq_len))
        i2 = int(np.floor(np.random.rand()*0.4*seq_len + 0.1*seq_len))             
        x[i, i1, 1] = 1.0
        x[i, i2, 1] = 1.0
        if x[i,i1,0] == x[i,i2,0]:
            y[i] = -1.0 # lable 0
        else:
            y[i] = 1.0  # lable 1
            
    #tranpose x to dimensions: sequence_length * batch_size * dimension_input  
    return np.transpose(x, [1, 0, 2]), y


# generate a random orthogonal matrix for recurrent matrix initialization 
def get_rand_orth( dim ):
    temp = np.random.normal(size=[dim, dim])
    q, _ = np.linalg.qr(temp)
    return q


# return loss of a vanilla RNN model with parameter theta=(W1, W2) and inputs=(x, y)
def rnn_loss(theta, x, y):
    W1 = theta[:(dim_in+dim_hidden+1)*dim_hidden].view(dim_in+dim_hidden+1, dim_hidden)
    W2 = theta[(dim_in+dim_hidden+1)*dim_hidden:].view(dim_hidden+1, dim_out)
    ones = Variable(torch.ones(batch_size, 1))

    h = Variable(torch.zeros(batch_size, dim_hidden))
    for xt in x:
        net_in = Variable(torch.FloatTensor(xt))
        h = torch.tanh( torch.cat((net_in, h, ones), dim=1).mm(W1) )
        
    net_out = torch.cat((h, ones), dim=1).mm(W2)
    score = Variable(torch.FloatTensor(y)) * net_out
    return (-score + torch.log(1.0 + torch.exp(score))).mean()


# initialize the RNN weights and preconditioner
theta0 = np.concatenate(( 
            np.random.normal(loc=0.0, scale=0.1, size=[dim_in*dim_hidden]), 
            get_rand_orth(dim_hidden).flatten(),
            np.zeros(shape=[dim_hidden]), 
            np.random.normal(loc=0.0, scale=0.1, size=[dim_hidden*dim_out]),
            np.zeros(shape=[dim_out])))
theta = Variable(torch.FloatTensor(theta0), requires_grad=True)
dim_theta = len(theta)
Q = np.eye(dim_theta)

# begin the iteration here
Loss = []
for i in range(1000000):
    x, y = get_xor_batch(test_seq_len0)
    
    # calculate the gradient
    loss = rnn_loss(theta, x, y)
    loss.backward()
    grad = theta.grad.data.numpy().copy() # make a copy; otherwise, grad will change when theta changes
    theta.grad.data.zero_()    
    # no need to update Q in every iteration. So I introduce Q_update_gap
    # When Q_update_gap >> 1, PSGD becomes SGD  
    Q_update_gap = max(int(np.floor(np.log10(i+1.0))), 1) 
    if i % Q_update_gap == 0:
        # calculate a perturbed gradient
        sqrt_eps = np.sqrt(np.finfo('float32').eps)
        delta_theta = np.random.normal(0.0, sqrt_eps, dim_theta)
        loss = rnn_loss(theta + Variable(torch.FloatTensor(delta_theta)), x, y)
        loss.backward()
        delta_grad = theta.grad.data.numpy() - grad
        theta.grad.data.zero_()
        
        # update the preconditioner 
        Q = psgd.update_precond_dense(Q, delta_theta, delta_grad)
        
    # update theta.data; propose to clip the preconditioned gradient
    pre_grad = psgd.precond_grad_dense(Q, grad)
    theta.data -= 0.01*torch.FloatTensor( 
            pre_grad/max(1.0, np.sqrt(np.sum(pre_grad*pre_grad))) )
    
    Loss.append(loss.data.numpy()[0])
    if i%100 == 0:
        print('Loss: ', Loss[-1])    
    if Loss[-1] < 0.01:
        break

plt.plot(Loss)