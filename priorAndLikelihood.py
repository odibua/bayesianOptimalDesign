#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:03:25 2017

@author: odibua
"""

from pyDOE import *
import numpy as np
from scipy.stats import multivariate_normal

def priorUniform(lb,ub):
    def generateSamples(n):        
        #lb = np.array(lb); ub=np.array(ub);
        uniformSamples=lhs(len(lb),samples=n);
        theta = [(ub-lb)*uniformSamples[j]+lb for j in range(len(uniformSamples))]        
        return theta
    return generateSamples
        
#def priorUniform(n,lb,ub):
#    lb = np.array(lb); ub=np.array(ub);
#    uniformSamples=lhs(len(lb),samples=n);
#    theta = [(ub-lb)*uniformSamples[j]+lb for j in range(len(uniformSamples))]
#    
#    return(theta)

def combDriveMeas(combDriveClass,d,nd,theta,variance):
    y=[];
    nThet=len(theta); N=len(d);
    for j in range(0,N,nd):
        if (j<=(N-nd)):
            function=combDriveClass.classicCircuitModel(d[j],d[j+1])
            y.append(function(theta[0],theta[1])+np.random.normal(0,np.sqrt(variance)))
    return np.array(y)

def testFunctionMeas(designFunction,d,nd,theta,rv):
    y=[]; G=[]
    #print(d)
    nThet=len(theta); N=len(d);
    for j in range(0,N,nd): 
        if (j<=(N-nd)):
            function=designFunction(d[j])
            GTemp=function(theta[j])
            #print("GTemp ",GTemp,"theta",theta[j])
            #function(theta[0])
#            G.extend(function(theta[0]))
#            y.extend(function(theta[0])+np.random.normal(0,np.sqrt(variance)))
            G.extend(GTemp)
            y.extend(GTemp+rv.rvs())
            #print("G,y",G[j],y[j])
            #y.extend(function(theta[0])+np.random.normal(0,np.sqrt(variance)))
    #print("A",y,G)
    #print("A",y[0])
    
    return (np.array(y),np.array(G))