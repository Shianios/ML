# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:46:34 2019

@author: Loizos Shianios
"""
# This is not the final grad class.

import numpy as np
import sympy as sy
import re

class grads:
    def __init__(self,op_dict,indices,shapes,*g_elmnts):
        self.ind_dict = {}
        self.operants = {}
        self.act_funcs = {}
        self.layers_net = {}
        self.h_layers = {}
        self.grads_dict = {}
        
        # Create all tensors of the network
        for i in Shapes.keys():
            operant = sy.IndexedBase(i,shape=shapes[i])
            self.operants[i] = operant
        
        # Create all index structures. Create the first layer outside of the loop
        # and the remaining inside a loop. This is because the numbering of the 
        # indices is unconstrained for the first layer, but it is constrained for the
        # rest. The H layers, must continue caring the same index structure, as the
        # uncontracted structure of the previous layer, and the W contracted structure
        # must be the same as the H layer structure.
        index1 = indices[0].find(',')
        dict_el = []
        for j in indices[0][:index1] : dict_el.append(j+str(0))
        self.ind_dict["W{0}".format(0)] = dict_el
        index2 = indices[0].find('-')
        dict_el = []
        for j in indices[0][index1+1:index2] : dict_el.append(j+str(0))
        self.ind_dict["H{0}".format(0)] = dict_el
        dict_el = []
        
        for i in range(1,len(indices)):
            # Get the uncontructed structure of previus W tensor.
            h_ind = self.ind_dict["H{0}".format(i-1)]
            w_ind = self.ind_dict["W{0}".format(i-1)]
            unco_ind = [x for x in w_ind if x not in h_ind]
            self.ind_dict["H{0}".format(i)] = unco_ind
            # Find length of uncostruction structure for current W tensor.
            index1 = indices[i].find(',') - len(self.ind_dict["H{0}".format(i)])
            dict_el = []
            for j in indices[i][:index1] : dict_el.append(j+str(i))
            self.ind_dict["W{0}".format(i)] = dict_el + self.ind_dict["H{0}".format(i)]
        # Create the output layer index structure from the uncontracted structure of the last W.
        h_ind = self.ind_dict["H{0}".format(i)]
        w_ind = self.ind_dict["W{0}".format(i)]
        unco_ind = [x for x in w_ind if x not in h_ind]
        self.ind_dict["H{0}".format(i+1)] = unco_ind
        # Create Bias Tensors. We do not create them in previous loop. First we create a list
        # of there keys. This is because we, might have chosen not to include biases
        # to all layers, or even any at all.
        B_key_list = [x for x in Shapes.keys() if x[0] == 'B']
        for i in B_key_list:
            j = str(int(i[1:])+1)
            self.ind_dict[i] = self.ind_dict['H'+j]
        
        # Get the operation string of each layer convert it to string code and use eval to create sympy elements
        for i in op_dict.keys():
            ind_n = str(int(i[1:])-1) # Get the numbering of the layers
            equation = ''
            oprnt_pos = [m.start() for m in re.finditer(ind_n, op_dict[i])]
            func = op_dict[i][:oprnt_pos[0]-2]
            # For now it seems unnesesary to have a dictionary for the act funcs.
            # But if we allow them to have variable parameters this might be useful.
            self.act_funcs[i] = sy.Function(func) 
            
            for ind in range(len(oprnt_pos)-1):
                op = op_dict[i][oprnt_pos[ind]+1:oprnt_pos[ind+1]-1]
                if op == '*': 
                    if equation:
                        oprnts = [op_dict[i][oprnt_pos[ind]-1] + ind_n]
                        oprnts.append(op_dict[i][oprnt_pos[ind+1]-1] + ind_n)
                        equation += 'sy.tensorproduct(' + 'self.operants[' + "'" + oprnts[0] + "'" + '][' + 'self.ind_dict[' + "'" + oprnts[0] + "'" + ']],'
                        equation +=  'self.operants[' + "'" + oprnts[1] + "'" + '][' + 'self.ind_dict[' + "'" + oprnts[1] + "'" + ']])'
                    else:
                        oprnts = [op_dict[i][oprnt_pos[ind]-1] + ind_n]
                        oprnts.append(op_dict[i][oprnt_pos[ind+1]-1] + ind_n)
                        equation += 'sy.tensorproduct(' + 'self.operants[' + "'" + oprnts[ind] + "'" + '][' + 'self.ind_dict[' + "'" + oprnts[ind] + "'" + ']],'
                        equation +=  'self.operants[' + "'" + oprnts[ind+1] + "'" + '][' + 'self.ind_dict[' + "'" + oprnts[ind+1] + "'" + ']])'
                    
                elif op == '+':
                    if equation:
                        oprnts = [op_dict[i][oprnt_pos[ind+1]-1] + ind_n]
                        equation += '+' + 'self.operants[' + "'" + oprnts[0] + "'" + '][self.ind_dict[' + "'" + oprnts[0] + "'" + ']]'
                    else:
                        oprnts = [op_dict[i][oprnt_pos[ind]-1] + ind_n]
                        equation += 'self.operants[' + "'" + oprnts[0] + "'" + '][self.ind_dict[' + "'" + oprnts[0] + "'" + ']]' + '+'
            
            self.layers_net[i] = eval(equation)
            self.h_layers[i] = self.act_funcs[i](self.layers_net[i])
        
        # Create the error term
        Target = sy.IndexedBase('T',shape=shapes[list(shapes.keys())[-1]])
        Err = Target[self.ind_dict[list(self.ind_dict.keys())[-1]]] - self.h_layers[list(self.h_layers.keys())[-1]]
        E = Err**2 # In future we will allow both L1 and L2 errors, as well as user defined.
        
        # Differentiate the error with respect to each element we asked for its gradient, through the g_elements list.
        # If list is not passed we compute the gradient of all operants exept the H terms.
        if g_elmnts:
            h_keys = list(self.h_layers.keys())
        else:
            g_elmnts = list(self.operants.keys())
            for j in reversed(g_elmnts):
                if j[0]=='H' : g_elmnts.remove(j)
            h_keys = list(self.h_layers.keys())
            
        for k in (h_keys[:-1]):
            sub = self.operants[k][self.ind_dict[k]]
            E = E.subs(sub,self.h_layers[k])
        for i in g_elmnts:
            dx = self.operants[i][self.ind_dict[i]]
            G = sy.diff(E,dx)
            
            # Here we substitute back H terms H_net terms and the Err term, to symblify.
            for k in (h_keys):
                sub = self.h_layers[k]
                G = G.subs(sub,self.operants[k][self.ind_dict[k]])
            for k in (h_keys):
                sub = self.layers_net[k]
                G = G.subs(sub,sy.IndexedBase((k+'_net'),shape=shapes[k])[self.ind_dict[k]])
            Err = Err.subs(self.h_layers[h_keys[-1]],self.operants[h_keys[-1]][self.ind_dict[h_keys[-1]]])
            G = G.subs(Err,sy.IndexedBase('Err',shape=shapes[h_keys[-1]])[self.ind_dict[h_keys[-1]]])
            self.grads_dict[i] = G
        
        # We substitute the act. func. derivative terms. For this we will convert the 
        # equations to python strings and manipulate.
        for k in self.grads_dict.keys():
            op_exp = sy.printing.str.sstrrepr(self.grads_dict[k])
            #print(k,':',op_exp)
            #print()
            sub_pos = [m.start() for m in re.finditer('Subs', op_exp)]
            sub_exp = ''

            for i in reversed(sub_pos):
                H_pos = op_exp.find('H',i)
                delim = op_exp.find('_',H_pos)
                H_key = op_exp[H_pos:delim]
                ind_end = op_exp.find(']',H_pos)
                func = sy.printing.str.sstrrepr(self.act_funcs[H_key])
                oprnt = op_exp[H_pos:ind_end+1]
                sub_exp = '*' + func + '/' + oprnt + sub_exp
            op_exp = op_exp[:sub_pos[0]-1] + sub_exp
            #print(k,':',op_exp)
            #print('----------')
            self.grads_dict[k] = op_exp
        #print('##########')
        
        # Finally, we said that the act funcs have parameters. The grads will have terms from this parameters
        # We need to compute the actual derivative of the activation function and get those terms. For that
        # we need to parse the actual expression of the act func into sympy and differentiate it. 
              
        print('Operants:',self.operants) 
        print()
        print('Indices:',self.ind_dict)
        print()
        print('Activation Functions:',self.act_funcs)
        print()
        print('Layer Net input:',self.layers_net)
        print()
        print('Layer ops:',self.h_layers)
        print()        
        print('Err = ',Err)
        print()
        print('E = ',E)
        print()
        print('########## Grads ##########')
        for i in self.grads_dict.keys():
            print(i,':',self.grads_dict[i])
            print()
            
# ----------------- END OF CLASS ----------------- #
            
# Example input
w0 = np.arange(450.).reshape((3,5,6,5))
h0 = np.arange(30.).reshape((6,5))
b0 = np.arange(15.).reshape((3,5))
w1 = np.arange(225.).reshape((5,3,3,5))
h1 = np.arange(15.).reshape((3,5))
b1 = np.arange(15.).reshape((5,3))
t = np.arange(15.).reshape((5,3))
Net_op_dic = {'H1': 'sigm(W0*H0+B0)', 'H2': 'sigm(W1*H1+B1)'}
Indices = ['ABab,ab->AB','abAB,AB->ab']
Shapes = {'H0':(6,5),'W0':(3,5,6,5),'B0':(3,5),'W1':(5,3,3,5),'H1':(3,5),'B1':(5,3),'H2':(5,3)}
g_elements = ['W0','B0','W1','B1']
         
gradient = grads(Net_op_dic,Indices,Shapes,*g_elements)



# In the class we will convert the following code to meathods to be called and
# get the same results. 
'''            
ind_w1 = ['a0','b0','c0']
ind_w1_2 = ['a01','b01','c01']
ind_h1 = ['b0','c0']
ind_b1 = ['a0']
ind_w2 = ['a1','a0']
ind_h2 = ['a0',]
ind_b2 = ['a1']
ind_t = ['a1']
kron_del_ind = []

W1 = sy.IndexedBase('W1', shape=w0.shape)
H1 = sy.IndexedBase('H1', shape=h0.shape)
B1 = sy.IndexedBase('B1', shape=b0.shape)
W2 = sy.IndexedBase('W2', shape=w1.shape)
H2 = sy.IndexedBase('H2', shape=h1.shape)
B2 = sy.IndexedBase('B2', shape=b1.shape)
T = sy.IndexedBase('T', shape=t.shape)
act_func = sy.Function('sigm')

H2_net = sy.tensorproduct(W1[ind_w1],H1[ind_h1]) + B1[ind_b1]
H2 = act_func(H2_net)
H3_net = sy.tensorproduct(W2[ind_w2],H2) + B2[ind_b2]
H3 = act_func(H3_net)
Err = (T[ind_t] - H3)
E = Err**2
G = sy.diff(E,W1[ind_w1_2])

H2_net_exp = sy.printing.str.sstrrepr(H2_net)
print('H2_net:',H2_net_exp)
print()
H2_exp = sy.printing.str.sstrrepr(H2)
print('H2:',H2_exp)
print()
H3_net_exp = sy.printing.str.sstrrepr(H3_net)
print('H3_net:',H3_net_exp)
print()
H3_exp = sy.printing.str.sstrrepr(H3)
print('H3:',H3_exp)
print()
H3_exp = H3_exp.replace(H2_exp,'H2')
print('H3:',H3_exp)
print()

Err_exp = sy.printing.str.sstrrepr(Err)
print('Err:',Err_exp,'indices:',sy.get_indices(Err))
print()
E_exp = sy.printing.str.sstrrepr(E)
print('E:',E_exp)
print()
G_exp = sy.printing.str.sstrrepr(G)
print('G:',G_exp)
print()
op = G_exp.replace(Err_exp,'Err')
print('OP:', op)
print()
op = op.replace(H3_net_exp,'H3_net')
print('Op:', op)
print()
op = op.replace(H2_net_exp,'H2_net')
print('Op:', op)
print()
while op.find('Subs')>0:
    str_ind = op.find('Subs')
    str_ind_end = op.find('Subs',str_ind+4)
    sub_term_ind = op.find('H',str_ind)
    sub_term = op[sub_term_ind:op.find(',',sub_term_ind)]
    print(str_ind,str_ind_end)
    op = op[:str_ind] + sy.printing.str.sstrrepr(act_func) + '(' + sub_term +',-1)' + op[str_ind_end-1:]
    print('Op:', op)
print()
while op.find('*KroneckerDelta')>0:
    str_ind = op.find('*KroneckerDelta')
    str_ind = op.find('(',str_ind)
    str_ind_dif = op.find(')',str_ind) - str_ind
    kron_del_ind.append(op[str_ind+1:(str_ind+str_ind_dif)])
    op = op.replace('*KroneckerDelta('+ kron_del_ind[len(kron_del_ind)-1] + ')' ,'')
    print('Op:', op)
    print()
print('Indices to swap:',kron_del_ind)
print('............................')
'''