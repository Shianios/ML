# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 21:34:57 2018

@author: Loizos Shianios
"""

import numpy as np

class Filters:
    # Filters leave the size of the data matrix unchanged. For now all operations are restricted to 2D data.
    def __init__ (self,data,**kwargs):
        self.kwargs = kwargs
        self.aux_dic = {}
        self.data = data
        if 'size' in kwargs:
            if kwargs['size'] > 0:
                aux_size = kwargs['size']
                self.kwargs['size'] = aux_size + int(2.*(float(aux_size)/2. - np.floor(float(aux_size)/2.))-1.0)  
                # The above line will map all even numbers to the previews odd and odd left unchanged. We restric filters dimensions to odd numbers
            else:
                self.kwargs['size'] = 3
        else: 
            self.kwargs.update({'size': 3})
  
    def compute (self):
        return getattr(self, self.kwargs.get('filter'))(self.data,**self.kwargs)
    
    def Median_op (self,data,**kwargs):
        return np.median(data)
      
    def Median (self,data,**kwargs):
        size = kwargs['size']
        d = data.copy()
        d_shape = data.shape
        ind = int(np.floor(size/2))
        for i in range (0,len(d_shape)-1):
            for j in range (0,d_shape[i]-size+1):
                for k in range (0,d_shape[i+1]-size+1):
                    aux = data[j:(j+size),k:(k+size)]
                    d[j+ind,k+ind] = self.Median_op(aux,**kwargs)
        return d
    
    def Sobel_op (self,data,**kwargs):
        if 'Sob_filter_x' not in self.aux_dic: # To avoid redeffining the Sobel matrices in a loop
            a = 1.
            b = 2.
            Sob_filter_x = np.array([[a,0.,-a],[b,0.,-b],[a,0.,-a]])
            Sob_filter_y = Sob_filter_x.T
            self.aux_dic.update({'Sob_filter_x' : Sob_filter_x})
            self.aux_dic.update({'Sob_filter_y' : Sob_filter_y})
            del a,b,Sob_filter_x,Sob_filter_y
            
        element_x = np.einsum('ij,ij',self.aux_dic['Sob_filter_x'],data)
        element_y = np.einsum('ij,ij',self.aux_dic['Sob_filter_y'],data)
        element = np.sqrt(element_x**2 + element_y**2)
        return element
    
    def Sobel (self,data,**kwargs):
        d = data.copy()
        d_shape = data.shape
        size = 3    # For now we retrict Sobel to 3x3 matrix
        ind = int(np.floor(size/2))
        for i in range (0,len(d_shape)-1):
            for j in range (0,d_shape[i]-size):
                for k in range (0,d_shape[i+1]-size):
                    aux = data[j:(j+size),k:(k+size)] 
                    d[j+ind,k+ind] = self.Sobel_op(aux,**kwargs)          
        return d
    
    def Laplace_op (self,data,**kwargs):
        if 'Laplace_matrix' not in self.aux_dic: # To avoid redeffining the Laplace matrix in a loop
            a = 1.
            b = 4.
            Laplace = np.array([[0.,a,0.],[a,-b,a],[0.,a,0.]])
            self.aux_dic.update({'Laplace_matrix' : Laplace})
            del a,b,Laplace
            
        element = np.einsum('ij,ij',self.aux_dic['Laplace_matrix'],data)
        return element
    
    def Laplace (self,data,**kwargs):
        d = data.copy()
        d_shape = data.shape
        size = 3    # For now we retrict Laplace to 3x3 matrix
        ind = int(np.floor(size/2))
        for i in range (0,len(d_shape)-1):
            for j in range (0,d_shape[i]-size):
                for k in range (0,d_shape[i+1]-size):
                    aux = data[j:(j+size),k:(k+size)] 
                    d[j+ind,k+ind] = self.Laplace_op(aux,**kwargs)          
        return d

