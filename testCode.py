#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:12:28 2017

@author: odibua
"""

import numpy as np
import random
from testFuncs import evaluateFitnessFunctions
from testFuncs import rastriginFunc
from testFuncs import sphereFunc
from testFuncs import mcCormicFunc
from testFuncs import bealeFunc
from testFuncs import holderTableFunc
from testFuncs import rosenbrockFunc
from testFuncs import ackleysFunc
import matplotlib.pyplot as plt
from psoMethods import PSO
from psoParticles import psoParticle
import copy
from npsoInterpFuncs import npsoInterpFunc 
from npsoParticles import npsoParticle 

testString =['sphere','ackleys','mccormic','beales','rastrigin']
testUse=testString[3]
nDim=10
if (testUse=='sphere'):
##Sphere function parameters
    posMin = np.array([-100]*nDim).astype(float);
    posMax = np.array([100]*nDim).astype(float);
    velMin = posMin;
    velMax = posMax;
    funcUsed = sphereFunc
elif (testUse=='ackleys'):
####Ackleys Function parameter inputs
    posMin = np.array([-32.768,-32.768]);
    posMax = np.array([32.768,32.768]);
    posMin = [-32.768,-32.768,];
    posMax = [32.768,32.768];
    velMin = posMin;
    velMax = posMax;
    funcUsed = ackleysFunc
elif (testUse=='mccormic'):
###McCormic Function parameter inputs
    posMin = np.array([-1.5,-3]);
    posMax = np.array([4.0,4.0]);
    posMin = [-1.5,-3];
    posMax = [4.0,4.0];
    velMin = posMin;
    velMax = posMax;
    funcUsed = mcCormicFunc
elif (testUse=='beales'):
###Beale's Function parameter inputs
    posMin = np.array([-4.5,-4.5]);
    posMax = np.array([4.5,4.5]);
    posMin = [-4.5,-4.5];
    posMax = [4.5,4.5];
    velMin = posMin;#np.array([-1,-1]);
    velMax = posMax;#np.array([1,1]);
    funcUsed = bealeFunc;
elif (testUse=='rastrigin'):
#####Rastrigin Function parameter inputs
    posMin = np.array([-5.12]*nDim);
    posMax = np.array([5.12]*nDim);
    #posMin = [-5.12,-5.12,-5.12];
    #posMax = [5.12,5.12,5.12];
    velMin = posMin;
    velMax = posMax;
    funcUsed = rastriginFunc

#Define PSO Parameters
numParticles=30;
neighborSize = 2#NPSO Parameter
w=1.0;
tol=1e-3;
numIters=100#nFeatures*15;
numEvalState=2;
kappa = 0.5;
mult=1;
c1=1.0
c2 = c1*mult;
constrict=1.0
optimType='Min';

#Call PSO class
pso=PSO();

#Execute standard PSO
#output=pso.executePSO(c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,psoParticle,optimType,numEvalState,funcUsed,evaluateFitnessFunctions)

#Execute constrict PSO
#c1=2.05; c2=c1;
#output=pso.executeGCPSO(constrict,c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,psoParticle,optimType,numEvalState,funcUsed,evaluateFitnessFunctions)

#Execute NPSO
output=pso.executeNPSO(neighborSize,w,posMin,posMax,velMin,velMax,numIters,numParticles,npsoParticle,optimType,numEvalState,funcUsed,evaluateFitnessFunctions,npsoInterpFunc)

