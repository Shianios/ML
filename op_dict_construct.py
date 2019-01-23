# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 20:29:36 2019

@author: Loizos Shianios
"""

# The switch was constructed to work for any order of layers creation. 

class dict_construct(object):
    def construct (self, arg, dict_, n):
        method_name = str(arg)
        return getattr(self, method_name, lambda dict_, n: print("Invalid arguments in operation dictionary construction"))(dict_, n) 
    
    def W_True (self, dict_, n):
        if dict_["H{0}".format(n)] and dict_["H{0}".format(n)][-1] == '+':
            dict_["H{0}".format(n)] = '(' + 'W'+ str(n-1) +'*' + 'H' + str(n-1) + dict_["H{0}".format(n)]
        else:
           dict_["H{0}".format(n)] = '(' + 'W'+ str(n-1) +'*' + 'H' + str(n-1) + '+' + dict_["H{0}".format(n)] 
        
    def W_False (self, dict_, n):
        dict_["H{0}".format(n)] = '(' + 'W' + str(n-1) + '*' + 'H' + str(n-1)
        
    def H_True (self, dict_, n):
        pass
    
    def H_False (self, dict_, n):
        dict_["H{0}".format(n)] = ''
        
    def B_True (self, dict_, n):
        if dict_["H{0}".format(n)] and dict_["H{0}".format(n)][-1] == '+':
            dict_["H{0}".format(n)] += 'B'+ str(n-1) + ')'
        else:
            dict_["H{0}".format(n)] += '+' + 'B'+ str(n-1) + ')'
            
    def B_False (self, dict_, n):
        dict_["H{0}".format(n)] = 'B'+ str(n-1) + ')'