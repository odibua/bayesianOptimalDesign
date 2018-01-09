#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 19:02:15 2017

@author: Ohi Dibua
Code for testing the correctness of the utility function to be used for Bayesian Optimal 
through particle swarm optimization. The different types of particle swarm otimization are
outlined in the psoMethods file. The function tested is based on the first two examples [1], 
and the correct results for the optimal design points are:
    Given:
        designDimension=1, nSimultaneous=1, thetaStart=0 thetaEnd=1.0, dOpt = 1.0 and U = 3.36
        designDimension=1, nSimultaneous=2, thetaStart=0 thetaEnd=1.0, dOpt = (0.2,1.0) or (1.0,0.2) and U=3.7
        designDimension=1, nSimultaneous=2, thetaStart=0 thetaEnd=0.4373, dOpt = (0.2,0.2) and U=2.7
        designDimension=1, nSimultaneous=2, thetaStart=0.4373 thetaEnd=1.0, dOpt = (1.0,1.0) and U=3.6
"""
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from utilityFunc import KLUtility
from utilityFunc import evaluateFitnessFunctions

from priorAndLikelihood import priorUniform
from priorAndLikelihood import combDriveMeas
from priorAndLikelihood import testFunctionMeas

from combDriveModel import testFuncClass

from scipy.stats import multivariate_normal
from scipy.stats import bernoulli

from psoMethods import PSO
from psoParticles import psoParticle
from npsoInterpFuncs import npsoInterpFunc 
from npsoParticles import npsoParticle 
methodPSO
 
forwardModelClass=testFuncClass().testFunction; 
thetaStart=0; thetaEnd=1.0
lb=np.array([thetaStart]); ub=np.array([thetaEnd]);
nSamples=int(1e2); nSimultaneous=1; designDimension=1;
genPriorSamples = priorUniform(lb,ub);
variance=(1e-4);
mu=[0]*nSimultaneous; cov = [[0]*nSimultaneous for j in xrange(nSimultaneous)]; 
for j in xrange(nSimultaneous):
    cov[j][j]=variance 
distribution=multivariate_normal(mu,cov)
KL = KLUtility(forwardModelClass,distribution,genPriorSamples,nSamples,nSimultaneous,designDimension)
#d=np.array([0.8,1.0])
#U=KL(d)
#print(U)

##Define bounds 
posMin = np.array([0.0]*nSimultaneous);
posMax = np.array([1.0]*nSimultaneous);
velMin = posMin;
velMax = posMax;
#Define PSO Parameters
numParticles=40;
neighborSize = 2#NPSO Parameter
w=1.0;
tol=1e-3;
numIters=100#nFeatures*15;
numEvalState=2;
kappa = 0.5;
mult=1;
c1=2.0
c2 = c1*mult;
constrict=1.0
optimType='Max';
#
##Call PSO class
pso=PSO();

#Execute standard PSO
if (methodPSO=='PSO'):
    output=pso.executePSO(c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,psoParticle,optimType,numEvalState,KL,evaluateFitnessFunctions)
elif (methodPSO=='GCPSO'):
    #Execute constrict PSO
    c1=2.05; c2=c1;
    output=pso.executeGCPSO(constrict,c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,psoParticle,optimType,numEvalState,funcUsed,evaluateFitnessFunctions)
elif (methodPSO=='NPSO'):
    #Execute NPSO
    output=pso.executeNPSO(neighborSize,w,posMin,posMax,velMin,velMax,numIters,numParticles,npsoParticle,optimType,numEvalState,funcUsed,evaluateFitnessFunctions,npsoInterpFunc)

