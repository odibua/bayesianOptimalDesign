# -*- coding: utf-8 -*-
#Test functions for optimizaton algorithms
"""
Created on Thu Jun 22 06:45:00 2017

@author: odibua
"""
import numpy as np

def sphereFunc(x):
     f = np.sum(x**2);
#    dim = len(np.shape(x));
#    if (dim<=1):
#        n=1;
#        f = x**2
#    else: 
#        f = np.sum(x**2,axis=0);
     return(f)
    
def rosenbrockFunc(x):
    #f= np.sum(100*(x[1:,0:]-x[0:-1,0:]**2)**2+(x[0:-1,0:]-1)**2,axis=0);
    #print(np.array(x))
    #print("f ",100*(x[1:,0:]-x[0:-1,0:]**2)**2+(x[0:-1,0:]-1)**2)
    f= np.sum(100*(x[1:]-x[0:-1]**2)**2+(x[0:-1]-1)**2,axis=0);
    #print("f ",f)
    return(1./f)

def ackleysFunc(x):
    #f= np.sum(100*(x[1:,0:]-x[0:-1,0:]**2)**2+(x[0:-1,0:]-1)**2,axis=0);
    #print(np.array(x))
    #print("f ",100*(x[1:,0:]-x[0:-1,0:]**2)**2+(x[0:-1,0:]-1)**2)
    D=len(x)
    #print(D)
    f= -20*np.exp(-0.2*np.sqrt((1./D)*np.sum(x[0:]**2)))-np.exp((1./D)*np.sum(np.cos(2*np.pi*x[0:])))+20+np.exp(1);
    #print("f ",f)
    return(f)
    
def rastriginFunc(x):
    A = 10;
    dim = len(np.shape(x));
#    print("rastrigin")
#    print(dim);
#    print(np.shape(x))
    n=np.shape(x)[0];
    f = A*n+np.sum(x**2-A*np.cos(2*np.pi*x),axis=0);
#    if (dim<=1):
#        n=1;
#        f = A*n+(x**2-A*np.cos(2*np.pi*x));
#    else: 
#        n=np.shape(x)[0];
#        f = A*n+np.sum(x**2-A*np.cos(2*np.pi*x),axis=0);
    
    return(f)

def mcCormicFunc(x):
    f = np.sin(x[0]+x[1])+(x[0]-x[1])**2-1.5*x[0]+2.5*x[1]+1;
    return(f)
    
def bealeFunc(x):
    f = (1.5-x[0]+x[0]*x[1])**2+(2.25-x[0]+x[0]*x[1]**2)**2+(2.625-x[0]+x[0]*x[1]**3)**2;
    return(f)
    
def holderTableFunc(x):
    f = -np.abs(np.sin(x[0])*np.cos(x[1])*np.exp(np.abs(1-np.sqrt(x[0]**2+x[1]**2)/np.pi)));
    return(f)