#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 20:36:45 2018

@author: odibua
"""
import random
import numpy as np
 
def KLUtility(model,distribution,genPriorSamples,nSamples,nSimultaneous,designDimension):
    def Utility(d):
        Y=np.zeros((nSamples,nSimultaneous));
        G=np.zeros((nSamples,nSimultaneous));
        logLikelihood=np.zeros((nSamples,1))
        logEntropy=np.zeros((nSamples,1))
        theta=genPriorSamples(nSamples);
        dTild = np.reshape(d,(designDimension,nSimultaneous));

        modelList = [model(dTild[:,l]) for l in range(nSimultaneous)];

        for j in range(nSamples):
            G[j,:] = np.transpose(np.array([modelList[l](theta[j]) for l in range(nSimultaneous)]));
            Y[j,:] = G[j,:] + distribution.rvs();
            logLikelihood[j] = np.log(distribution.pdf(Y[j,:]-G[j,:]));
 
        for j in range(nSamples):
            logEntropy[j] = np.log(sum([distribution.pdf(Y[j,:]-G[k,:]) for k in range(nSamples)])/nSamples)
        U = sum(np.array(logLikelihood-logEntropy))/nSamples
        
        return U
    return Utility 