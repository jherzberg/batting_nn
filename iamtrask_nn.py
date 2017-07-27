# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 17:22:26 2017

@author: Josh

Mini NN
"""
# dependencies
import numpy as np
import time

# Activation Function: Sigmoid
def nonlin(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))

# one layer net
def one_layer(its):
    # Data
    x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

    # Labels
    y = np.array([[0,1,0,1]]).T
    
    # Random Init Matrix
    syn0 = np.random.random((3,1))
    
    print(np.dot(x, syn0)) # Check weights are random at beginning
    
    for i in range(its):
        l0 = x # forward prop
        l1 = nonlin(np.dot(l0, syn0))
        
        l1_error = y - l1 #loss
        
        if i%1000 == 0: # print loss
            print(l1_error.mean())
            
        l1_delta = l1_error * nonlin(l1, True) #back prop
        # update
        syn0 += np.dot(l0.T, l1_delta) # update
    
    print(l1)

def two_layer(its):
    #Data
    x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]], dtype=np.float16)
    
    # Labels
    y = np.array([[0,1,1,0]], dtype=np.float16).T
    
    # Weight matrices
    syn0 = np.random.normal(0,.1,(3,4)).astype('f')
    syn1 = np.random.normal(0, .1, (4,1)).astype('f')
    
    for i in range(its):
        l0 = x # forward prop
        l1 = nonlin(np.dot(l0, syn0))
        l2 = nonlin(np.dot(l1, syn1))
        
        loss = y - l2 #loss
        
        if i%(its/10) == 0: # print loss
            print(loss.mean())
            
        l2_delta = loss * nonlin(l2, True)
        dydl2 = l2_delta.dot(syn1.T)
        l1_delta = dydl2*nonlin(l1, True)
         
        # update
        syn1 += np.dot(l1.T, l2_delta)
        syn0 += np.dot(l0.T, l1_delta)
        
    print("C: ", nonlin(np.dot(nonlin(np.dot(np.array([[1,0,0]]), syn0), False), syn1), False))
    print("C: ", nonlin(np.dot(nonlin(np.dot(np.array([[0,1,0]]), syn0), False), syn1), False))
    print("F: ", nonlin(np.dot(nonlin(np.dot(np.array([[1,1,0]]), syn0), False), syn1), False))

start = time.time()
two_layer(100000)
biteight = time.time() - start
print( time.time() - start)










































