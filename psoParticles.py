#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 22:59:25 2017

@author: Ohi Dibua
This file contains a class that defines a particle used in a standard optimization 
problem solved by standard particle swarm optimization (PSO) or 
constricted particle swarm optimization (GCPSO). The optimization 
problems are of the form: 
    
    min f(x,y)=min f(x) or max f(x,y) = max f(x) where y are inputs with known values
     x          x           x            x
    
The particle class initializes the following arguments:
    optimType = optimization type ('Max' or 'Min')
    w: intertial constant (Recommended range is between 0.5 and 1.0)
    c1: cognitive constant (Recommended range is between 1.00 and 2.05)
    c2: social constant (Recommended range is between 1.00 and 2.05)
    posMin: minimum values of elements of x
    posMax: maximum values of elements of x
    velMin: minimum values of the veleocities ascribed to different elements of x
    velMax: maximum values of the velocities ascribed to different elements of x
    fitnessFunction = f(x) 
    fitnessEvaluationFunction: func(optimType,currentState,localBestState,globalBestState=None)
    Function that takes in two states that define proposed solutions and outputs 
    the solution that most meets the objective. If passing in the global best state,
    and empty list is passed in for localBestState. For typical PSO the states are state = (fitness,x)    
"""
import numpy as np
import copy
import time
import sys,os
import logging as logger
import sys, os
import time

class psoParticle():
    #Initialize parameters important for propagating particles and evaluating their fitness with respect
    #to fitness evaluation function
    def __init__(self,optimType,c1,c2,w,posMin,posMax,velMin,velMax,fitnessFunction,fitnessEvaluationFunction):
        #Initialize constants used to calculate velocity of particles and propagate their positions in input space
        self.c1=c1;
        self.c2=c2;
        self.w=w; 
        self.velMin=velMin; self.velMax=velMax;
        self.posMin=posMin; self.posMax=posMax;

        #Initiaize strings and functions related to evaluating fitness
        self.optimType=optimType;
        self.fitnessFunction=fitnessFunction;
        self.fitnessEvaluationFunction=fitnessEvaluationFunction;
        
        #Initialize positions and velocities of particles       
        self.pos=copy.deepcopy(posMax)
        self.vel=copy.deepcopy(velMax)           
        for k, item in enumerate(posMax):
            if (np.isscalar(self.pos[k])):
                r1=np.random.uniform(); r2=np.random.uniform();
                self.pos[k]=(posMax[k]-posMin[k])*r1+posMin[k];
                self.vel[k]=(velMax[k]-velMin[k])*r2+velMin[k];
            else:
                for l in range(len(self.pos[k])):
                    r1=np.random.uniform(); r2=np.random.uniform();
                    self.pos[k][l]=(posMax[k][l]-posMin[k][l])*r1+posMin[k][l];
                    self.vel[k][l]=(velMax[k][l]-velMin[k][l])*r2+velMin[k][l];
        
        #Define initial fitness as the personal best fitness of particles            
        self.localBestFitness=self.fitnessFunction(np.array(self.pos))
        self.localBestPos = copy.deepcopy(self.pos)

    #Update the input that yields best solution for local particle
    def updateLocalBest(self):
        #Evaluate fitness of current position and define current state and personal best state
        self.currentFitness=self.fitnessFunction(np.array(self.pos));        
        currentState = (self.currentFitness,self.pos);
        localBestState = (self.localBestFitness,self.localBestPos);  
        
        #Update personal best state according to fitness evaluation function
        localBestState = self.fitnessEvaluationFunction(self.optimType,currentState,localBestState);
        self.localBestFitness=localBestState[0]
        self.localBestPos=localBestState[1]  
        
        return localBestState 

    #Update velocity and position of particle according to their current position,
    #their personal best position, and most fit particle in swarm.
    #If constrict argument is not none then the constrict update equation is used
    def updateVelocityandPosition(self,globalBestState,t,constrict=None):
        posGlobalBest=globalBestState[1]
        #Obtain coefficients used in particle propagation for particular time step
        if not np.isscalar(self.w): 
            w=self.w[t];
        else: w=self.w;           
        if not np.isscalar(self.c1): 
            c1=self.c1[t];
        else: c1=self.c1;
        
        if not np.isscalar(self.c2): 
            c2=self.c2[t];
        else: c2=self.c2;
        if (constrict is not None):
            phi=c1+c2;
            constrFactor = 2.0/np.abs(2-phi-np.sqrt(phi**2-4*phi));
        else:
            constrFactor = 1.0
                
        #Stochastically update position and velocity (if there is no constrict, the velocity is bound by velMax
        #and velMin)
        for k, item in enumerate(self.vel):
            r1=np.random.uniform(); r2=np.random.uniform();
            if np.isscalar(self.pos[k]):
                cognitiveVel = r1*c1*(self.localBestPos[k]  - self.pos[k]);
                socialVel = r2*c2*(posGlobalBest[k] - self.pos[k]);
                self.vel[k] = constrFactor*(w*self.vel[k] + cognitiveVel + socialVel);
                if constrict is None:
                    self.vel[k] = np.min([np.max([self.vel[k],self.velMin[k]]),self.velMax[k]]);               
                self.pos[k] = self.pos[k] + self.vel[k];
                
                if self.pos[k]>=self.posMax[k] or self.pos[k]<=self.posMin[k]:
                    r1=np.random.uniform();
                    self.pos[k]=(self.posMax[k]-self.posMin[k])*r1+self.posMin[k];                    
            else:
                for l in range(len(self.pos[k])):
                    cognitiveVel = r1*c1*(self.localBestPos[k][l] - self.pos[k][l]);
                    socialVel = r2*c2*(posGlobalBest[k][l] - self.pos[k][l]);
                    self.vel[k][l] = constrFactor*(w*self.vel[k][l] + cognitiveVel + socialVel);
                    if constrict is None:
                        self.vel[k][l] = np.min([np.max([self.vel[k][l],self.velMin[k][l]]),self.velMax[k][l]]);
                    
                    self.pos[k][l] = self.pos[k][l] + self.vel[k][l];
                    
                    if (self.pos[k][l]>=self.posMax[k][l] or self.pos[k][l]<=self.posMin[k][l]):
                        r1=np.random.uniform(); 
                        self.pos[k][l]=(self.posMax[k][l]-self.posMin[k][l])*r1+self.posMin[k][l];   
    
    #Return the current state of particle
    def getCurrentState(self):
        currentState = (self.currentFitness,self.pos);
        return currentState
    
    #Return personal best state of particle
    def getlocalBestState(self):
        currentState = (self.currentFitness,self.pos);
        return currentState
################################################################################# 
