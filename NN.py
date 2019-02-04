# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 14:33:58 2019

@author: Loizos Shianios
"""

import numpy as np
import act_funcs as act_funcs
import op_dict_construct as dict_con

class FCN: # Fully Connected Network
    # In the construction of the network we create two dictionaries. One to store all layers, weights and biases,
    # and one to store all the operations that give the results of a layer. Also we create a list of indices
    # needed in Einstein sumation.
    def __init__ (self,data,prnt,*args,**kwargs):
        self.kwargs = kwargs
        self.Layers = {}
        self.indices = []
        self.op_dict = {}
        N_dims = self.dim_extract(data,args)
        self.Network_constr(N_dims)
        self.Op_constr()
        if prnt is True:self.Print_Arch(N_dims)
        del N_dims
        
    def dim_extract(self,data,args):    
        # Get shape of the input layer from data. The first index of data is treated as 
        # the indexing of individual data entries.
        tub = ()
        for i in range(1,len(data.shape)):
            tub += (data.shape[i],)
        if len(tub) < 1: N_dims = ((1,),)
        else: N_dims = (tub,)
        # Extract the shape of each hidden layer from args and store as tuples in N_dims. 
        # If args are not passed, the default is set to have an output layer of a single element.
        if not args:
            args = ['1']
            
        l_dims = ()
        num = ''

        for i in range(len(args)): 
            for j in (args[i]):
                if str.isdigit(j):
                    num += j
                else:
                    l_dims = l_dims + (int(num),)
                    num = ''
            if num: 
                l_dims = l_dims + (int(num),)
                num = ''
            N_dims = N_dims + (l_dims,)
            l_dims = ()      
        del num, l_dims, tub
        return N_dims
    
    def Network_constr(self,N_dims):
        # Construct Hidden layers from inputed dimensions, Weights to connect all Layers, Biases
        # and indices strings to be used in Einstein Summation during feed forward opperation.

        letters = "abcdefghijklmnopqrstuvwxyz"
        Letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        initialize = lambda low, high, shape : np.random.uniform(-np.sqrt(6/low), np.sqrt(6/high), shape)
            # For the initialization we use the propossed ranges as in: http://deeplearning.net/tutorial/mlp.html#mlp
            # When we will allow the network to have other act. functions we will pass an additional argument 
            # to multiply the valeus.
        
        self.L_Count = len(N_dims)-1 # To be used latter in feed forward. See coment in feed_forward
        for i in range(0,self.L_Count):
            W_dims = N_dims[i+1] + N_dims[i]
            self.Layers["H{0}".format(i)] = np.zeros(N_dims[i])
            self.Layers["W{0}".format(i)] = initialize(np.prod(N_dims[i]),np.prod(N_dims[i+1]),W_dims)
            self.Layers["B{0}".format(i)] = initialize(np.prod(N_dims[i]),np.prod(N_dims[i+1]),N_dims[i+1])
            self.indeces_construct(Letters,letters,len(W_dims),len(N_dims[i]))
        self.Layers["H{0}".format(i+1)] = np.zeros(N_dims[i+1])     # Create output layer.
        self.Layers["H0"][:] = data[0]    # First network layer is the input layer so we pass to it the first data entry.
        del letters, Letters, initialize, W_dims

    def Op_constr(self):
        # Construct the Network equations as strings and store to a dictionary. These are to be used in SymPy 
        # to derive the gredients of the network symbolicly, and use in gredient discent meathod during training.
        # For now we are missing the addition of the activation function.
        
        op_construct = dict_con.dict_construct() # Create an instance of switch class.
        for i in list(self.Layers.keys()):
            arg = i[0]
            arg += '_' + str("H{0}".format(int(i[1])+1) in self.op_dict) # Deterine if the Hi key has already been added in op_dict
            n = int(i[1:]) + 1
            op_construct.construct(arg, self.op_dict, n)
        if self.op_dict and not self.op_dict[list(self.op_dict.keys())[-1]]: # Check if we created an extra H layer in op_
            del self.op_dict[list(self.op_dict.keys())[-1]]
        # Once the dictionary is complite we add to the operations the activation fuction of each layer.
        '''
            For now we only have sigm as activation function. This will chance.
        '''
        for i in self.op_dict.keys():
            self.op_dict[i] = 'sigm*' + self.op_dict[i]
        del op_construct
        
    
            
    def indeces_construct(self,Letters,letters,A,a):
        # The outer IF statment is used so that our indices are consistent as we move between layers.
        # The resulted indexed structure of the first layer will be used as input for the next layer,
        # so it must contineu curring the same indices. This is most needed when we compute the gradient 
        # of the network, so that the resulted gradients have the correct indices. The nested IF statment
        # is used to swap between lower case and capitalised letters. Fortunatly numpy's einstein summation
        # function is case sensitive.
        if len(self.indices)<1:
            ind_a = letters[:a]
            ind_A_aux = Letters[:(A-a)]
            ind_A = ind_A_aux + ind_a
            ind = ind_A + ',' + ind_a + '->' + ind_A_aux
            self.indices.append(ind)
        else:
            ind_a = self.indices[-1][self.indices[-1].find('>')+1:]
            if ind_a[0] in letters:
                ind_A_aux = Letters[:(A-a)]
            else:
                ind_A_aux = letters[:(A-a)]
            ind_A = ind_A_aux + ind_a
            ind = ind_A + ',' + ind_a + '->' + ind_A_aux
            self.indices.append(ind)
        del ind_a, ind_A, ind_A_aux, ind
    
    def feed_forward(self):
        # When counting the layers we omitted to count the output layer. This is because we start updating the hidden layers
        # from 1 while in the loop of feed_forward the indexing starts from 0. Just observe that H1 = W0*H0 + B0. 
        # We could start from 1 and include the output layer in the counting but in the loop we would have to carry three 
        # subtractions instead of one addition; Hi = W[i-1]*H[i-1] + B[i-1]. 
        for i in range (0,self.L_Count):
            '''
                For now we have a fix activation function for all layers. This will change latter on.
            '''
            self.Layers["H{0}".format(i+1)][:] = act_funcs.act_funcs(
                    np.einsum(self.indices[i],self.Layers["W{0}".format(i)],self.Layers["H{0}".format(i)]) + self.Layers["B{0}".format(i)],
                    **{"func":'sigmoid',"order":0,"params":{}}).compute()
     
    def Print_Arch (self,N_dims):
        # Print network Hidden layers dims, the complite expression of the output, layer shapes and indices 
        # for Einstein summation. Note if we have a rank 2 tensor operatedwith a rank 1 tensor (vector) 
        # the operation is the usual matrix multipication.
        print()
        print("Network's Hidden Layers' dims:")
        print(N_dims)
        print()
        print("Network's operations:")
        print(self.op_dict)
        print()
        print("Network's Layers, Weights and Biases shapes:")
        for i in (self.Layers.keys()):
            print(i,'shape:', self.Layers[i].shape)   
        print()
        print('Einstein summation indices:')
        for i in range (len(self.indices)):
            print(self.indices[i])
        print()
        print('Network constructed')
        print('############################################')
        print()
        
    #def grads(self):
        

# Pass network layer architecture as a list of strings. Final element is the shape of the output. 
# If only one element is passed in args, that element is the shape of the output. If args is 
# empty or not passed to the Network the network will output a single scallar. Note layer dimensions
# can be separated by any character or symbol other than intigers, #, ' or " (depenting on which one
# you ue to enclose the string). So 12:5 and 12.5 and 12>5 and 12a5 and 12 5 will result in the same layer shape.
   
args = ['5:4','15','5:4:10:3','6:5']
#args = ['3','5']
#args = ['3']
#args = []

data = np.arange(300).reshape(10,6,5)
Network = FCN(data,True,*args)
Network.feed_forward()
print(Network.Layers[list(Network.Layers.keys())[-1]]) # Print output layer
