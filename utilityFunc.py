#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 20:36:45 2018

@author: odibua
This code is used for calculating the utility function of an input model using a
simulation based method. The utility function is formed by the Kullback-Leibler (KL)
divergence, and calculated as outlined in [1]

The KLUtility outer function takes the following arguments:
    model: Function that calculates model that is used in KL utility function. 
           It is a nested function that takes in first the design dimension, and then
           the parameters we will be performing inference on.
    distribution: The distribution the additive error to the model which will generate
                  measurement samples used in calculating the utility. 
    genPriorSamples: Generates nSamples of model parameters.
    nSample: Number of samples that will be generated based on the prior.
    nSimultaneous: Number of simultaneous experiments that will be run (yields number of inputs).
    designDimension: The number of elements in input dimension.
    
    Inner Utility function takes as an input, d, the design parameter
    
[1] X. Huan, Y. Marzouk, "Simulation-based optimal Bayesian experimental design for nonlinear systems,"
    Journal of Computational Physics, 2012
"""
import random
import numpy as np

#Evaluation function that is used for advancing particle swarm optimization.
def evaluateFitnessFunctions(optimType,currentState,localBestState,globalBestState=None):
    if (globalBestState==None):
        currentFitness=currentState[0];
        currentPos=currentState[1];
        
        localBestFitness=localBestState[0];
        localBestPos=localBestState[1];
        
        newLocalBool=0;
        if (optimType.lower()=='max'):
            if (currentFitness>localBestFitness  or localBestFitness==float("inf")):
                newLocalBool=1;
        elif (optimType.lower()=='min'):
            if (currentFitness<localBestFitness):
                newLocalBool=1;
        if (newLocalBool==1):
            localBestState=(currentFitness,currentPos)
        
        return localBestState
    
    elif (globalBestState is not None):
        localBestFitness=localBestState[0];
        localBestPos=localBestState[1];
        
        globalBestFitness=globalBestState[0];
        globalBestPos=globalBestState[1];
        
        newGlobalBool=0;
        if (optimType.lower()=='max'):
            if (localBestFitness>globalBestFitness or globalBestFitness==float("inf")):
                newGlobalBool=1;
        elif (optimType.lower()=='min'):
            if (localBestFitness<globalBestFitness  or globalBestFitness==float("inf")):
                newGlobalBool=1;
        if (newGlobalBool==1):
            print("Change global best fitness",localBestFitness,"input",localBestPos);
            globalBestState = (localBestFitness,localBestPos)
        
        return globalBestState

#Calculates the utility function based on the KL divergence. 
def KLUtility(model,distribution,genPriorSamples,nSamples,nSimultaneous,designDimension):
    def Utility(d):
        #Initialize arrays for true models and measurement errors
        Y=np.zeros((nSamples,nSimultaneous));
        G=np.zeros((nSamples,nSimultaneous));
        
        #Initialize arrays to store log-likelihood and log-entropy
        logLikelihood=np.zeros((nSamples,1))
        logEntropy=np.zeros((nSamples,1))
        
        #Generate prior samples of parameters
        theta=genPriorSamples(nSamples);
        
        #Create list of models based on simultaneous experiments
        dTild = np.reshape(d,(designDimension,nSimultaneous));
        modelList = [model(dTild[:,l]) for l in range(nSimultaneous)];

        #Generate true models, and noisy measurements. Calculate log-likelihood
        for j in range(nSamples):
            G[j,:] = np.transpose(np.array([modelList[l](theta[j]) for l in range(nSimultaneous)]));
            Y[j,:] = G[j,:] + distribution.rvs();
            logLikelihood[j] = np.log(distribution.pdf(Y[j,:]-G[j,:]));
 
        #Calcute log-entropy and the utility function
        for j in range(nSamples):
            logEntropy[j] = np.log(sum([distribution.pdf(Y[j,:]-G[k,:]) for k in range(nSamples)])/nSamples)
        U = sum(np.array(logLikelihood-logEntropy))/nSamples
        
        return U
    return Utility 