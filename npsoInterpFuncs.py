#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 19:23:26 2017

@author: odibua
"""
import numpy as np
import copy

#Function that interpolates two solutions and updates the global best state in comparison to 
#them according to fitnessEvaluationFuncion
def npsoInterpFunc(optimType,JState,KState,xMin,xMax,globalBestState,fitnessFunction,fitnessEvaluationFunction):
    fXJ=JState[0]; XJ=JState[1];
    fXK=KState[0]; XK=KState[1];
    fXBest=globalBestState[0]; XBest=globalBestState[1];
    
    #First interpolation     
    X1 = [0.5*(XJ[k]**2-XBest[k]**2-XK[k]**2)*fXJ*fXK/
        ((XJ[k]-XK[k])*fXBest+(XK[k]-XBest[k])*fXJ
        +(XBest[k]-XJ[k])*fXK) for k in range(len(XJ))];  
    
    #Bound interpolation
    for k in range(len(XJ)):
        if (X1[k]<xMin[k]):
            X1[k]=xMin[k];
        elif (X1[k]>xMax[k]):
            X1[k]=xMax[k];
            
    fX1 = fitnessFunction(np.array(X1));
    X1State = (fX1,X1); 

    #Update global best state with respect to interpolated state using 
    #fitnessEvaluationFunction
    globalBestState=fitnessEvaluationFunction(optimType,[],X1State,globalBestState);
    fXBest=globalBestState[0]; XBest=globalBestState[1];
    
    #Second interpolation
    X2 = [0.5*((XJ[k]**2-XK[k]**2)*fXBest+(XK[k]**2-XBest[k]**2)*fXJ+(XBest[k]**2-XJ[k]**2)*fXK)
                /((XJ[k]-XK[k])*fXBest+(XK[k]-XBest[k])*fXJ+(XBest[k]-XJ[k])*fXK) for k in range(len(XJ))]
    
    #Bound interpolation
    for k in range(len(XJ)):
        if (X2[k]<xMin[k]):
            X2[k]=xMin[k];
        elif (X2[k]>xMax[k]):
            X2[k]=xMax[k];
            
    fX2 = fitnessFunction(np.array(X2));
    X2State = (fX2,X2);
    
    #Update global best state with respect to second interpolate
    #state using fitnessEvaluationFunction
    globalBestState=fitnessEvaluationFunction(optimType,[],X2State,globalBestState);
    
    return globalBestState