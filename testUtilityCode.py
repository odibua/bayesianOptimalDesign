#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 19:02:15 2017

@author: odibua
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from utilityFunc import KLUtility

from priorAndLikelihood import priorUniform
from priorAndLikelihood import combDriveMeas
from priorAndLikelihood import testFunctionMeas

from combDriveModel import calcTimeConstantOxideOnly
from combDriveModel import combDriveModels
from combDriveModel import testFuncClass

from scipy.stats import multivariate_normal
from scipy.stats import bernoulli

from testFuncs import sphereFunc
from testFuncs import mcCormicFunc
from testFuncs import ackleysFunc
from testFuncs import bealeFunc
 
forwardModelClass=testFuncClass().testFunction; 
thetaStart=0.4373; thetaEnd=1.0
lb=np.array([thetaStart]); ub=np.array([thetaEnd]);
nSamples=int(1e3); nSimultaneous=2; designDimension=1;
genPriorSamples = priorUniform(lb,ub);
variance=(1e-4);
mu=[0]*nSimultaneous; cov = [[0]*nSimultaneous for j in xrange(nSimultaneous)]; 
for j in xrange(nSimultaneous):
    cov[j][j]=variance 
distribution=multivariate_normal(mu,cov)
KL = KLUtility(forwardModelClass,distribution,genPriorSamples,nSamples,nSimultaneous,designDimension)
d=np.array([0.8,1.0])
U=KL(d)
print(U)