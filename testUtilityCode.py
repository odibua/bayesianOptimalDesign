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
        designDimension=1, nSimultaneous=1, thetaStart=0 thetaEnd=0.4373, dOpt = 0.2 and U = 2.4
        designDimension=1, nSimultaneous=1, thetaStart=0.4373 thetaEnd=1.0, dOpt = 1.0 and U = 3.3
        designDimension=1, nSimultaneous=2, thetaStart=0 thetaEnd=1.0, dOpt = (0.2,1.0) or (1.0,0.2) and U=3.7
        designDimension=1, nSimultaneous=2, thetaStart=0 thetaEnd=0.4373, dOpt = (0.2,0.2) and U=2.8
        designDimension=1, nSimultaneous=2, thetaStart=0.4373 thetaEnd=1.0, dOpt = (1.0,1.0) and U=3.6
        
Currently the number of samples is only 100. To get closer to the true results, 
it is necessary to increase the number of samples. The type of PSO used also has an impact on the correctness of solutions
"""
import sys
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from utilityFunc import KLUtility
from utilityFunc import evaluateFitnessFunctions

from priorAndLikelihood import priorUniform

from combDriveModel import testFuncClass

from scipy.stats import multivariate_normal
from scipy.stats import bernoulli

from psoMethods import PSO
from psoParticles import psoParticle
from npsoInterpFuncs import npsoInterpFunc 
from npsoParticles import npsoParticle 
methodString = ['PSO','GCPSO','NPSO'];
#methodUse=methodString[1]
methodUse=sys.argv[1]; thetaMin = float(sys.argv[2]); thetaMax = float(sys.argv[3]); nSamples=int(sys.argv[4])

if (methodUse not in methodString):
    sys.exit('PSO method does not exist. Input PSO,GCPSO,or NPSO')
elif (thetaMax < thetaMin or thetaMax>1 or thetaMin<0):
    sys.exit('Invalid paramter bounds, input bounds between 0 and 1')
elif (nSamples<=0):
     sys.exit('Invalid paramter bounds, input bounds between 0 and 1')
    


#Define prior
#thetaMin=0; thetaMax=0.4373
lb=np.array([thetaMin]); ub=np.array([thetaMax]);
#nSamples=int(1e2); 
genPriorSamples = priorUniform(lb,ub);
 
forwardModelClass=testFuncClass().testFunction; #Test model defined



#Define design parameter dimensions and its boundaries
nSimultaneous=2; designDimension=1; dLB=0; dUB=1.0

#Define distribution for the noise to be added to model
variance=(1e-4);
mu=[0]*nSimultaneous; cov = [[0]*nSimultaneous for j in xrange(nSimultaneous)]; 
for j in xrange(nSimultaneous):
    cov[j][j]=variance 
distribution=multivariate_normal(mu,cov) 

#Define Utility function that take design parameters as arguments after passing in 
KL = KLUtility(forwardModelClass,distribution,genPriorSamples,nSamples,nSimultaneous,designDimension)
#d=np.array([0.2,0.2])
#U=KL(d)
#print(U)
#sys.exit()
#Particle Swarm Optimization on Utility function
##Define bounds 
posMin = np.array([np.array([dLB]*designDimension)]*nSimultaneous);
posMax = np.array([np.array([dUB]*designDimension)]*nSimultaneous);
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
if (methodUse=='PSO'):
    output=pso.executePSO(c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,psoParticle,optimType,numEvalState,KL,evaluateFitnessFunctions)
elif (methodUse=='GCPSO'):
    #Execute constrict PSO
    c1=2.05; c2=c1;
    output=pso.executeGCPSO(constrict,c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,psoParticle,optimType,numEvalState,KL,evaluateFitnessFunctions)
elif (methodUse=='NPSO'):
    #Execute NPSO
    output=pso.executeNPSO(neighborSize,w,posMin,posMax,velMin,velMax,numIters,numParticles,npsoParticle,optimType,numEvalState,KL,evaluateFitnessFunctions,npsoInterpFunc)

