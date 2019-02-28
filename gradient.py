# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:46:34 2019

@author: Loizos Shianios
"""
# This is not the final grad class.

import sympy as sy
import re

class grads:
    def __init__(self,op_dict,indices,shapes,prnt,*g_elmnts):
        self.grads_dict = {}
        # We create the dictionaries, to have the tearms stored when we need to substitute in expressions.
        self.ind_dict = {}
        self.ind_dict_einsum = {}
        self.operants = {}
        self.act_funcs = {}
        self.layers_net = {}
        self.h_layers = {}
        
        # Create all tensors of the network
        for i in shapes.keys():
            operant = sy.IndexedBase(i,shape=shapes[i])
            self.operants[i] = operant
        self.operants['Err'] = sy.IndexedBase('Err',shape=shapes[list(shapes.keys())[-1]])
        
        self.ind_strc(indices,shapes)
        self.operations(op_dict)
        Err, E = self.Error(shapes)
        self.Diff(Err,E,shapes,*g_elmnts)
        if prnt is True : self.Print(Err,E)
        
    def get_diffs(self):
        return self.grads_dict
    def get_indices(self):
        return self.ind_dict_einsum
        
    def ind_strc(self,indices,shapes):
        # Create all index structures. Create the first layer outside of the loop
        # and the remaining inside a loop. This is because the numbering of the 
        # indices is unconstrained for the first layer, but it is constrained for the
        # rest. The H layers, must continue caring the same index structure, as the
        # uncontracted structure of the previous layer, and the W contracted structure
        # must be the same as the H layer structure. Simultaniously we create the 
        # index structures for einsum to be used in network class. 
        
        index1 = indices[0].find(',')
        dict_el = []
        for j in indices[0][:index1] : dict_el.append(j+str(0))
        self.ind_dict['W0'] = dict_el
        self.ind_dict_einsum['W0'] = indices[0][:index1]
        index2 = indices[0].find('-')
        dict_el = []
        for j in indices[0][index1+1:index2] : dict_el.append(j+str(0))
        self.ind_dict['H0'] = dict_el
        self.ind_dict_einsum['H0'] = indices[0][index1+1:index2] 
        dict_el = []
        
        for i in range(1,len(indices)):
            # Get the uncondructed structure of previus W tensor.
            h_ind = self.ind_dict["H{0}".format(i-1)]
            w_ind = self.ind_dict["W{0}".format(i-1)]
            unco_ind = [x for x in w_ind if x not in h_ind]
            self.ind_dict["H{0}".format(i)] = unco_ind
            unco_ind_einsum = ''
            for j in unco_ind: unco_ind_einsum += j[0]
            self.ind_dict_einsum ["H{0}".format(i)] = unco_ind_einsum

            # Find length of uncondructed structure for current W tensor.
            index1 = indices[i].find(',') - len(self.ind_dict["H{0}".format(i)])
            dict_el = []
            for j in indices[i][:index1] : dict_el.append(j+str(i))
            self.ind_dict["W{0}".format(i)] = dict_el + self.ind_dict["H{0}".format(i)]
            self.ind_dict_einsum["W{0}".format(i)] = indices[i][:index1] +  self.ind_dict_einsum["H{0}".format(i)]
            
        # Create the output layer and Err index structure from the uncontracted structure of the last W.
        layer_no = len(indices)
        h_ind = self.ind_dict["H{0}".format(layer_no-1)]
        w_ind = self.ind_dict["W{0}".format(layer_no-1)]
        unco_ind = [x for x in w_ind if x not in h_ind]
        self.ind_dict["H{0}".format(layer_no)] = unco_ind
        unco_ind_einsum = ''
        for j in unco_ind : unco_ind_einsum += j[0]
        self.ind_dict_einsum["H{0}".format(layer_no)] = unco_ind_einsum
        self.ind_dict['Err'] = unco_ind
        del layer_no
        
        # Create Bias Tensors. We do not create them in previous loop. First we create a list
        # of there keys. This is because we, might have chosen not to include biases
        # to all layers, or even any at all.
        B_key_list = [x for x in shapes.keys() if x[0] == 'B']
        for i in B_key_list:
            j = str(int(i[1:])+1)
            self.ind_dict[i] = self.ind_dict['H'+j]
            self.ind_dict_einsum[i] = self.ind_dict_einsum['H'+j]
            
    def operations(self,op_dict):
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
            
    def Error(self,shapes):
        # Create the error term
        Target = sy.IndexedBase('T',shape=shapes[list(shapes.keys())[-1]])
        Err = Target[self.ind_dict[list(self.ind_dict.keys())[-1]]] - self.h_layers[list(self.h_layers.keys())[-1]]
        E = Err**2 # In future we will allow both L1 and L2 errors, as well as user defined.
        return Err, E
    
    def Diff(self,Err,E,shapes,*g_elmnts):
        # Differentiate the error with respect to each element we asked for its gradient, through the g_elements list.
        # If list is not passed we compute the gradient of all operants exept the H terms and Err.
        if g_elmnts:
            h_keys = list(self.h_layers.keys())
        else:
            g_elmnts = list(self.operants.keys())
            for j in reversed(g_elmnts):
                if j[0]=='H' or j=='Err' : g_elmnts.remove(j)
            h_keys = list(self.h_layers.keys())
        
        # Substitute all terms in Error term. 
        for k in reversed(h_keys[:-1]):
            sub = self.operants[k][self.ind_dict[k]]
            E = E.subs(sub,self.h_layers[k])
            
        # Differetiate  
        for i in g_elmnts:
            dx = self.operants[i][self.ind_dict[i]]
            G = sy.diff(E,dx)
            # Here we substitute back H terms H_net terms the Err term and supress the indices, to symblify.
            # Note: The full expression of a single layer network could be sigm(W0[A,B,c]*H0[A,B]+B[c]). When 
            # substituting terms we need to substitute W0[A,B,c] with W0. That is why in the G.sub we have
            # the term self.operants[k][self.ind_dict[k]]; self.operants[k] gives the W0 and [self.ind_dict[k]]
            # gives the indices e.g. [A,B,c].
            for k in (h_keys):
                sub = self.h_layers[k]
                G = G.subs(sub,self.operants[k][self.ind_dict[k]])
            for k in (h_keys):
                sub = self.layers_net[k]
                G = G.subs(sub,sy.IndexedBase((k+'_'),shape=shapes[k])) # We create temporarily the _net terms as we do not have them stored.
            Err = Err.subs(self.h_layers[h_keys[-1]],self.operants[h_keys[-1]][self.ind_dict[h_keys[-1]]])
            G = G.subs(Err,sy.IndexedBase('Err',shape=shapes[h_keys[-1]])[self.ind_dict[h_keys[-1]]])
            for k in list(self.operants.keys()):
                G = G.subs(self.operants[k][self.ind_dict[k]],self.operants[k])
            self.grads_dict[i] = G

        # We substitute the act. func. and implicite derivative terms. For this we will convert the 
        # equations to python strings and manipulate.
        for k in self.grads_dict.keys():
            op_exp = sy.printing.str.sstrrepr(self.grads_dict[k])
            sub_pos = [m.start() for m in re.finditer('Subs', op_exp)]
            sub_exp = ''

            for i in reversed(sub_pos):
                H_pos = op_exp.find('H',i)
                if op_exp[H_pos+1]=='0':    # No need for substitution of the input layer H0
                    pass
                else:
                    delim = op_exp.find('_',H_pos)
                    H_key = op_exp[H_pos:delim]
                    # To avoid getting erronious keys we carry a check that only the layer letter and numbering
                    # is included in the key.
                    H_key_aux = H_key[0]
                    j=1
                    while j < len(H_key) and H_key[j].isdigit():
                        H_key_aux += H_key[j]
                        j +=1
                    H_key =  H_key_aux
                    ind_end = op_exp.find(',',H_pos)

                    func = sy.printing.str.sstrrepr(self.act_funcs[H_key])
                    oprnt = op_exp[H_pos:ind_end]
                    sub_exp = '@d1_' + func + '|' + oprnt + sub_exp # Use @ to designate element wise multiplication.
            op_exp = op_exp[:sub_pos[0]-1] + sub_exp
            self.grads_dict[k] = op_exp

        # Finally, we said that the act funcs have parameters. The grads will have terms from this parameters
        # We need to compute the actual derivative of the activation function and get those terms. For that
        # we need to parse the actual expression of the act func into sympy and differentiate it. 
        ''' 
            TO DO
        '''
        
        # substitude any scalar multiplications represented by * and not by @. Remember we might get terms from
        # the activation functions.
        delim = ['*','@']
        for k in self.grads_dict.keys():
            op_pos = 0
            # Find an operation in the string weather it is * or @. Find the terms around it. The
            # second tearm will be the rest of the equation from the position of the operation to 
            # the end. We find in this new string the next operation and deduct the operant coresponding
            # to the first operation. Check if both terms are tensors or not. If not and the operation
            # is * change it to @.
            while op_pos >= 0: # while we get valeus other than -1 we contineu searching the string.
                op_s = 0
                old_pos = op_pos
                op_pos = 0
                
                while op_s < len(delim)-1 and op_pos <= 0: 
                    op_pos = self.grads_dict[k].find(delim[op_s],old_pos+1)
                    op_s +=1
                term1 = self.grads_dict[k][old_pos:op_pos]
                if term1[0] in delim: term1 = term1[1:]
                term2 = self.grads_dict[k][op_pos+1:]
                stop = -1
                s = 0
                while s < len(delim)-1 and stop < 0:
                    stop = term2.find(delim[s])
                    s += 1
                term2 = term2[:stop]
                
                if term1 in self.operants.keys() and term2 in self.operants.keys():
                    pass
                else:
                    self.grads_dict[k] = self.grads_dict[k][:op_s+1] + '@' + self.grads_dict[k][op_s+2:]
                '''
                print('delim',op_s,'old pos',old_pos,'new pos',op_pos)
                print('Terms to process=',term1,'|',term2)
                print('s:',s,'stop:', stop)
                print('Terms to consider=', term1,'|',term2)
                print('----------')
            print('########## Next key ##########')
            '''
        
    def Print(self,Err,E):
        print('########## Grad class dictionaries ##########')
        print()
        print('Operants:',self.operants) 
        print()
        print('Indices:',self.ind_dict)
        print()
        print('Indices einsum:',self.ind_dict_einsum)
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
        print('---------- Grads ----------')
        for i in self.grads_dict.keys():
            print(i,':',self.grads_dict[i])
            print()
        print('########## End of Grad Dicts ##########')
           
# ----------------- END OF CLASS ----------------- #
        
'''
import numpy as np       
# Example input for testing
W0 = np.arange(450.).reshape((3,5,6,5))
H0 = np.arange(30.).reshape((6,5))
B0 = np.arange(15.).reshape((3,5))
W1 = np.arange(225.).reshape((5,3,3,5))
H1 = np.arange(15.).reshape((3,5))
B1 = np.arange(15.).reshape((5,3))
H2 = np.arange(15.).reshape((5,3))
T = np.arange(15.).reshape((5,3))
Net_op_dic = {'H1': 'sigm(W0*H0+B0)', 'H2': 'sigm(W1*H1+B1)'}
Indices = ['ABab,ab->AB','abAB,AB->ab']
Shapes = {'H0':(6,5),'W0':(3,5,6,5),'B0':(3,5),'W1':(5,3,3,5),'H1':(3,5),'B1':(5,3),'H2':(5,3)}
g_elements = []
         
gradient = grads(Net_op_dic,Indices,Shapes,False,*g_elements)
Gs = gradient.get_diffs()
for i in Gs.keys():
    print(i, ':', Gs[i], '\n')
'''
