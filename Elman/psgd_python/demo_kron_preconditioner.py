"""
Demo code for preconditioned SGD on solving the RNN benchmark XOR problem
Long term memories are required to solve this problem successfully  
"""
import numpy as np
import matplotlib.pyplot as plt
import preconditioned_stochastic_gradient_descent as psgd 
import torch
from torch.autograd import Variable

# parameter settings
batch_size, test_seq_len0 = 100, 30
dim_in, dim_out = 2, 1
dim_hidden = 50


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


# return loss of a vanilla RNN model with parameters=(W1, W2) and inputs=(x, y)
def rnn_loss(W1, W2, x, y):
    ones = Variable(torch.ones(batch_size, 1))

    h = Variable(torch.zeros(batch_size, dim_hidden))
    for xt in x:
        net_in = Variable(torch.FloatTensor(xt))
        h = torch.tanh( torch.cat((net_in, h, ones), dim=1).mm(W1) )
        
    net_out = torch.cat((h, ones), dim=1).mm(W2)
    score = Variable(torch.FloatTensor(y)) * net_out
    return (-score + torch.log(1.0 + torch.exp(score))).mean()


# initialize the RNN weights and preconditioner
W1_np = np.concatenate((np.random.normal(loc=0.0, scale=0.1, size=[dim_in, dim_hidden]),
                        get_rand_orth(dim_hidden),
                        np.zeros([1, dim_hidden])), axis=0)
W2_np = np.concatenate((np.random.normal(loc=0.0, scale=0.1, size=[dim_hidden, dim_out]),
                        np.zeros([1, dim_out])), axis=0)
W1 = Variable(torch.FloatTensor(W1_np), requires_grad=True)
W2 = Variable(torch.FloatTensor(W2_np), requires_grad=True)
Q1_left, Q1_right = np.eye(W1.size()[0]), np.eye(W1.size()[1])
Q2_left, Q2_right = np.eye(W2.size()[0]), np.eye(W2.size()[1])


# begin the iteration here
Loss = []
for i in range(1000000):
    x, y = get_xor_batch(test_seq_len0)
    
    # calculate the gradient
    loss = rnn_loss(W1, W2, x, y)
    loss.backward()
    grad_W1 = W1.grad.data.numpy().copy() # make a copy; otherwise, it will change when W1 changes
    grad_W2 = W2.grad.data.numpy().copy()
    W1.grad.data.zero_()
    W2.grad.data.zero_()
    
    # no need to update Q in every iteration. So I introduce Q_update_gap
    # When Q_update_gap >> 1, PSGD becomes SGD  
    Q_update_gap = max(int(np.floor(np.log10(i+1.0))), 1)
    if i % Q_update_gap == 0:
        # calculate a perturbed gradient
        sqrt_eps = np.sqrt(np.finfo('float32').eps)
        delta_W1 = np.random.normal(0.0, sqrt_eps, tuple(W1.size()))
        delta_W2 = np.random.normal(0.0, sqrt_eps, tuple(W2.size()))
        loss = rnn_loss(W1 + Variable(torch.FloatTensor(delta_W1)),
                        W2 + Variable(torch.FloatTensor(delta_W2)), x, y)
        loss.backward()
        delta_grad_W1 = W1.grad.data.numpy() - grad_W1
        delta_grad_W2 = W2.grad.data.numpy() - grad_W2
        W1.grad.data.zero_()
        W2.grad.data.zero_()
        
        # update the preconditioner 
        Q1_left, Q1_right = psgd.update_precond_kron(Q1_left, Q1_right, delta_W1, delta_grad_W1)
        Q2_left, Q2_right = psgd.update_precond_kron(Q2_left, Q2_right, delta_W2, delta_grad_W2)
        
    # update theta.data
    W1.data -= 0.01*torch.FloatTensor( psgd.precond_grad_kron(Q1_left, Q1_right, grad_W1) )
    W2.data -= 0.01*torch.FloatTensor( psgd.precond_grad_kron(Q2_left, Q2_right, grad_W2) )
    
    Loss.append(loss.data.numpy()[0])
    if i%100 == 0:
        print('Loss: ', Loss[-1])     
    if Loss[-1] < 0.01:
        break
    
plt.plot(Loss)