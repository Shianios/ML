# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:26:37 2018

@author: Loizos Shianios
"""

import numpy as np
import filters as Filters

class Convolution:
    def __init__ (self,data,**kwargs):
        self.aux_dic = {}
        self.kwargs = kwargs
        self.data = data
    
    def compute (self):
        return getattr(self, self.kwargs.get('opp'))(self.data,**self.kwargs)
    
    def Convol (self,data,**kwargs):
        if 'size' in kwargs:
            if kwargs['size'] > 0:
                size = int(kwargs['size'])
            else: size = 3
        else: size = 3
        
        if 'strid' in kwargs: 
            if kwargs['strid'] >= 0:
                strid = int(kwargs['strid'])
            else : strid = 1
        else : strid = 1
        
        if 'pading' in kwargs: 
            if kwargs['pading'] >= 0:
                pading = int(kwargs['pading'])
            else: pading = 0
        else: pading = 0
    
        if 'filter' not in kwargs: kwargs.update({'filter':'Sobel_op'})
        
        shp = [int( np.floor((data.shape[0]+2*pading-size)/strid +1) ), int( np.floor((data.shape[1]+2*pading-size)/strid +1) )]
        d = np.zeros((shp),np.float64)
        del shp
        
        if pading > 0:
            aux_data = data.copy()
            data = np.zeros((aux_data.shape[0]+2*pading , aux_data.shape[1]+2*pading))
            data[pading:aux_data.shape[0]+pading , pading:aux_data.shape[1]+pading] = aux_data
            del aux_data
            
        for j in range (0,data.shape[0]-size,strid):
            for k in range (0,data.shape[1]-size,strid):
                aux = data[j:(j+size),k:(k+size)] 
                l = int(j/strid)
                m = int(k/strid)
                d[l,m] = Filters.Filters(aux,**{"filter":kwargs['filter']}).compute()          
        return d
    
    def Pooling(self,data,**kwargs):
        if 'size' in kwargs:
            if kwargs['size'] > 0:
                size = int(kwargs['size'])
            else: size = 3
        else: size = 3
        
        if 'strid' in kwargs: 
            if kwargs['strid'] >= 0:
                strid = int(kwargs['strid'])
            else : strid = 1
        else : strid = 1
        
        if 'pading' in kwargs: 
            if kwargs['pading'] >= 0:
                pading = int(kwargs['pading'])
            else: pading = 0
        else: pading = 0
        
        if 'pool_op' not in kwargs: kwargs.update({'pool_op':'Max'})
        
        shp = [int( np.floor((data.shape[0]+2*pading-size)/strid +1) ), int( np.floor((data.shape[1]+2*pading-size)/strid +1) )]
        d = np.zeros((shp),np.float64)
        del shp
        
        if pading > 0:
            aux_data = data.copy()
            data = np.zeros((aux_data.shape[0]+2*pading , aux_data.shape[1]+2*pading))
            data[pading:aux_data.shape[0]+pading , pading:aux_data.shape[1]+pading] = aux_data
            del aux_data
        
        for j in range (0,data.shape[0]-size,strid):
            for k in range (0,data.shape[1]-size,strid):
                aux = data[j:(j+size),k:(k+size)] 
                l = int(j/strid)
                m = int(k/strid)
                d[l,m] = getattr(self, kwargs.get('Pool_op'))(aux,**kwargs)          
        return d
    
    def Max(self,data,**kwargs):
        return np.max(data)
    
    def Average(self,data,**kwargs):
        return np.mean(data)
