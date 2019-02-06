# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:46:34 2019

@author: Loizos Shianios
"""
# This is not the final grad class.

import numpy as np
import sympy as sy
import re
#from sympy.parsing.sympy_parser import parse_expr # Hard to parse tensorial expressions.
#from sympy import Idx


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

class grads:
    def __init__(self,op_dict,indices,shapes):
        self.ind_dict = {}
        self.operants = {}
        self.act_funcs = {}
        self.layers_net = {}
        self.h_layers = {}
        for i in Shapes.keys():
            operant = sy.IndexedBase(i,shape=shapes[i])
            self.operants[i] = operant
        print('operants:',self.operants) 
        print()
        
        for i in range(len(indices)):
            index1 = indices[i].find(',')
            dict_el = []
            for j in indices[i][:index1]:dict_el.append(j+str(i))
            self.ind_dict["W{0}".format(i)] = dict_el
            index2 = indices[i].find('-')
            dict_el = []
            for j in indices[i][index1+1:index2]:dict_el.append(j+str(i))
            self.ind_dict["H{0}".format(i)] = dict_el
            dict_el = []
            for j in indices[i][index2+2:]:dict_el.append(j+str(i))
            self.ind_dict["B{0}".format(i)] = dict_el
        self.ind_dict["H{0}".format(i+1)]=self.ind_dict["B{0}".format(i)]
        print('Indices:',self.ind_dict)
        print()
        
        for i in op_dict.keys():
            ind_n = str(int(i[1:])-1) # Get the numbering of the layers
            equation = ''
            oprnt_pos = [m.start() for m in re.finditer(ind_n, op_dict[i])]
            func = op_dict[i][:oprnt_pos[0]-2]
            
            self.act_funcs[i] = sy.Function(func) # For now it seems unnesesary to have a dictionary for the act funcs.
                                                      # But if we allow them to have variable parameters this might be useful.
            
            for ind in range(len(oprnt_pos)-1):
                op = op_dict[i][oprnt_pos[ind]+1:oprnt_pos[ind+1]-1]
                if op == '*': 
                    oprnts = [op_dict[i][oprnt_pos[ind]-1] + ind_n]
                    oprnts.append(op_dict[i][oprnt_pos[ind+1]-1] + ind_n)
                    equation += 'sy.tensorproduct(' + 'self.operants[' + "'" + oprnts[ind] + "'" + '][' + 'self.ind_dict[' + "'" + oprnts[ind] + "'" + ']],'
                    equation +=  'self.operants[' + "'" + oprnts[ind+1] + "'" + '][' + 'self.ind_dict[' + "'" + oprnts[ind+1] + "'" + ']])'
                    
                elif op == '+':
                    if equation:
                        oprnts = [op_dict[i][oprnt_pos[ind+1]-1] + ind_n]
                        equation += '+' + 'self.operants[' + "'" + oprnts[0] + "'" + '][self.ind_dict[' + "'" + oprnts[0] + "'" + ']]'
                    else:
                        oprnts = [op_dict[i][oprnt_pos[ind+1]-1] + ind_n]
                        equation += 'self.operants[' + "'" + oprnts[0] + "'" + '][self.ind_dict[' + "'" + oprnts[0] + "'" + ']]' + '+'
            
            self.layers_net[i] = eval(equation)
            
            self.h_layers[i] = self.act_funcs[i](self.layers_net[i])
            '''
            print(func)
            print(op_dict[i])
            print(oprnt_pos)
            print('EG:',i,equation)
            print('Net:',self.layers_net[i] )
            '''
        print('Activation Functions:',self.act_funcs )
        print()
        print('Layer Net input:',self.layers_net )
        print()
        print('Hidden layer ops:',self.h_layers )
        print()
        
gradient = grads(Net_op_dic,Indices,Shapes)


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

# Testing lambdify modul. not going to be part of the class.

'''
print()
print(sy.printing.lambdarepr.lambdarepr(G))
print()
print(print_python(G))
print()
print(sy.printing.pycode(G))
'''

'''
print('############## O ##############')
print(O)
print()
print(type(O.doit()))
print()
print(tuple(sy.get_indices(O)))
print()
print(sy.get_contraction_structure(O))
print('###############################')
print('############## E ##############')
print(E)
print()
print(sy.get_indices(E))
print()
print(sy.get_contraction_structure(E))
print('###############################')
print('############## g ##############')
print(G)
print()
print(sy.get_indices(G))
print()
print(sy.get_contraction_structure(G))

print('#################################################')
print("############## open g's contraction strc dictionary ##############")
k = list(sy.get_contraction_structure(G).keys())
ind = 1 
for i in k:
    print(str(ind)+':',i)
    ind+=1

print()
print(sy.get_contraction_structure(G)[list(sy.get_contraction_structure(G).keys())[0]])
print()
print(sy.get_contraction_structure(G)[list(sy.get_contraction_structure(G).keys())[1]])
print('#########################################')

ind = ind_w + list(set(ind_h) - set(ind_w)) 
ind_val = list(w.shape) + list(set(list(h.shape)) - set(list(w.shape)))
#ind_val[:] = [i - 1 for i in ind_val]
dims_ = {}
for i in range(len(ind)):
    dims_[ind[i]] = ind_val[i]
    
print(ind,tuple(ind_val))
print(ind_w,ind_h)
print(dims_)

mod = [{'Mul':np.einsum},'numpy']
f = sy.lambdify((W,H), O.doit(), modules = 'numpy', dummify=False)

#f = theano_function([W,H], [O],  dims= {'W':2,'H':2})
ind_range_w = [np.mgrid[0:ind_val[0]],np.mgrid[0:ind_val[1]]]
ind_range_h = [np.mgrid[0:ind_val[2]],np.mgrid[0:ind_val[0]]]
print(ind_range_w)
print(ind_range_h)
print()
#print(f(w,h))

f = sy.lambdify((sy.DeferredVector('W'),sy.DeferredVector('H'),'a0','b0','c0','d0'),O.doit(),'numpy')
    
#f = sy.lambdify((sy.DenseNDimArray('X'),sy.DenseNDimArray('Y'),sy.DenseNDimArray('B')),O.doit(),'numpy')

print(f(w,h,4,2,3,1))
'''