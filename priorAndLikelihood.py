#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:03:25 2017

@author: Ohi Dibua
Function that contains nested functions which generate prior parameter samples
"""

from pyDOE import *
import numpy as np
from scipy.stats import multivariate_normal

def priorUniform(lb,ub):
    def generateSamples(n):        
        #lb = np.array(lb); ub=np.array(ub);
        #uniformSamples=np.random.uniform(lb,ub,n);#lhs(len(lb),samples=n);
        uniformSamples=lhs(len(lb),samples=n);
        theta = [(ub-lb)*uniformSamples[j]+lb for j in range(len(uniformSamples))]        
        return theta
    return generateSamples
        

