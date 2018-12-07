# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 12:58:32 2018

@author: Loizos Shianios
"""
import itertools
import numpy as np
import filters as filters

class SOM:
    def __init__(self,data,labels,l_rate,r,epochs,dims,bound_cond):
        self.data = data
        self.labels = labels
        self.l_rate_0 = l_rate
        self.radius = r
        self.dimensions = dims
        self.bound_cond = bound_cond
        self.epochs = epochs
        #construct the Weight matrix
        shp =[]
        for i in range (len(data.shape)-1):
            shp.append(data.shape[i])
        self.W = np.random.uniform(0.0,0.5,[np.product(dims),*shp])
        del shp
        # construct neihgborhood coordinates. If the radius change they are reconstructed.
        self.list_coord = self.coord_gen(self.radius,len(self.dimensions)).copy()

        # construct indices string for einstain summation functions
        indices_W ="z"
        indices_p =""
        letters = "abcdefghijklmnopqrstuvwxy" # This limits us to upto a 26th rank tensor for W
        for i in range (0,len(data.shape)-1):
            indices_W += letters[i]
            indices_p +=letters[i]
        self.sum_indices = indices_W + ',' + indices_p
        self.p_len_indices = indices_p + ',' + indices_p
        del letters, indices_W, indices_p
        
        # List of dictionaries to store the labels corresponding to a cell of the map
        self.dictlist = [dict() for x in range(0,self.W.shape[0])]  
    
    # construct coordinates (i,j,...) of neighborhood. Indices range from -r to r
    # and the number of indices is equal to d. If the Manhattan distance (i.e if
    # the sum of the absolute valeus of the coordinates) is greater than r,
    # the coordinate is rejected. For example if d=3 and r=1 the coordinate
    # (-1,1,0) has distance 2 which is greater than 1 and it is rejected.
    def coord_gen(self,r,d):
        list_p = []
        list_aux = []
        
        for i in range (0,2*r+1):
            list_aux += [-r+i]
        list_p = list(itertools.product(list_aux, repeat=d))

        for p_vec in reversed (list_p):
            p_len = 0
            for j in range (0,len(p_vec)):
                p_len += abs(p_vec[j])
            if p_len > r:
                list_p.remove(p_vec)
        del list_aux
        return list_p
    
    # construct string of neighbooring neurons coordinates and retrieve numbering
    def Neighb_func(self,win,dims,coords,conds):
        # First find the winning neuron's coordinates
        neighb_neurns = list()
        win_neur_coord = np.zeros((len(dims)),int)
        remainder = win # renaming, just for convinience
        
        for i in range (0,len(dims)-1):
            product = 1
            # for loop to get the product of all dimensions' lengths lower than the current dimension i
            for j in range (0,len(dims)-i-1):
                product *= dims[j]
            coord = min(dims[len(dims)-1-i],int((remainder-1)/product))
            remainder = int(remainder - coord*product)
            win_neur_coord[len(dims)-i-1] = coord
        win_neur_coord[0] = remainder
        if 'coord' in locals(): del coord
        
        # Then add all neihgborhood coordinates, and check for the given boundary conditions
        coords_len = len(coords)-1
        for i in range (0,len(coords)):
            coord_x = win_neur_coord + coords[coords_len-i]
            coord_x[0] -= 1 # subtract one as first dim numbering starts from 1 not 0
            check = 0
            for j in range (0,len(coord_x)):
                if j < len(conds):
                    if conds[j] == 'c':
                        '''if coord_x[j] < 0:
                            coord_x[j] = dims[j] + coord_x[j]
                        elif coord_x[j] > dims[j]-1:
                            coord_x[j] = coord_x[j] - dims[j]'''
                        # The above code gives the same result as the one below with a single if.
                        if coord_x[j] < 0 or coord_x[j] > dims[j]-1: coord_x[j] = abs(dims[j] - abs(coord_x[j]))
                    else:
                        if coord_x[j] < 0 or coord_x[j] > dims[j]-1: 
                            check = 1                       
                else: # If boundary conditions are less than no. of dimensions
                      # we treat the remaining dimensions as open.
                    if coord_x[j] < 0 or coord_x[j] > dims[j]-1: 
                        check = 1
            # Convert coord_x to the neuron index.
            if check == 0:
                coord_x[0] +=1 # add back the 1 we subtracted earlier
                neighbr = 0
                for k in range (1,len(coord_x)): # start from 1 becasue first dim is not multiplied by any valeu.
                    neighbr = coord_x[k]*dims[k-1]
                neighbr += coord_x[0]
                neighb_neurns.append(neighbr)

        return neighb_neurns
    def win_neuron(self,W,data):
        diff = W[:,...] - data[...]
        dist = np.empty((diff.shape[0]))
        for ind in range(0,dist.shape[0]):
            dist[ind] = np.sqrt(np.einsum(self.p_len_indices, diff[ind,...], diff[ind,...]))
        min_val = np.amin(dist)
        win_neuron = np.nonzero(dist == min_val)[0] # this will return the index of the first matching element as single element array
        return win_neuron
    
    def train(self):
        l_rate = self.l_rate_0
        c = 0.5
        levy_coef = np.sqrt(c/(2*np.pi))*self.l_rate_0
        self.l_rate_d = []
        for k in range (1,self.epochs+1):
            if k%5==0:
                if self.radius > 0:
                    check_r = self.radius
                    self.radius = max(0,self.radius-1)
                    if self.radius != check_r:
                        self.list_coord = self.coord_gen(self.radius,len(self.dimensions))
            if k%5==0: # Apply filtering to the clusters.
                for i in range (0, self.W.shape[0]):
                    d_median = filters.Filters(self.W[i,...],**{"filter":"median"}).compute()
                    d_median += np.abs(np.min(d_median))
                    d_median = np.multiply(np.divide(np.subtract(d_median,np.min(d_median)),np.subtract(np.max(d_median),np.min(d_median))),255.0)
                    d_edge = filters.Filters(d_median,**{"filter":"Sobel"}).compute()
                    d_edge += np.abs(np.min(d_edge))
                    d_edge = np.sqrt(d_edge)
                    d_edge *= -1.0
                    d_edge = np.multiply(np.divide(np.subtract(d_edge,np.min(d_edge)),np.subtract(np.max(d_edge),np.min(d_edge))),255.0)
                    d_filt = np.multiply(d_median,d_edge)
                    d_filt += np.abs(np.min(d_filt))
                    d_filt = np.multiply(np.divide(np.subtract(d_filt,np.min(d_filt)),np.subtract(np.max(d_filt),np.min(d_filt))),255.0)
                    
                    self.W[i,...] = np.multiply(d_filt,self.W[i,...])
                    self.W[i,...] += np.abs(np.min(self.W[i,...]))
                    self.W[i,...] = np.multiply(np.divide(np.subtract(self.W[i,...],np.min(self.W[i,...])),np.subtract(np.max(self.W[i,...]),np.min(self.W[i,...]))),255.0)
                
            #l_rate = min(0.8,l_rate*1.4)
            #l_rate = self.l_rate_0*(np.exp(float(k-1)/(-20.0)))
            x = float(k)/10.
            l_rate = levy_coef * (np.exp(c/(-2.* x**2)))/(x)**(3)
            self.l_rate_d.append(l_rate)
            for i in range (0, self.data.shape[len(self.data.shape)-1]): # no need to follow sequential selection but we do for simplicity
                win_neuron = self.win_neuron(self.W,self.data[...,i])
                neigh_neur = self.Neighb_func(win_neuron[0]+1,self.dimensions,self.list_coord,self.bound_cond) 
                # In above line +1 because we number neurons from 1 not zero.
                for j in range (0,len(neigh_neur)):
                    self.W[neigh_neur[j]-1,...] = (1.0-l_rate) * self.W[neigh_neur[j]-1,...] + l_rate*(self.data[...,i])
            
            if k%5==0: print(str(k)+',','radius:', str(self.radius)+',', 'learning rate:', l_rate)     
        
    def assign_label(self):
        for i in range (0, self.data.shape[len(self.data.shape)-1]):
            win_neuron = self.win_neuron(self.W,self.data[...,i])
            if self.labels[i] in self.dictlist[win_neuron[0]] : 
                self.dictlist[win_neuron[0]][self.labels[i]]+=1
            else:
                self.dictlist[win_neuron[0]].update({self.labels[i] : 1})
        # Get the label with the maximum number
        for i in range (0,self.W.shape[0]):
            max_key = max(self.dictlist[i], key=self.dictlist[i].get)
            self.dictlist[i].update({'max' : max_key})
            
    def classify(self,data):
        win_neuron = self.win_neuron(self.W,data)
        key  = self.dictlist[win_neuron[0]]['max']
        return key
        