#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:32:25 2017

@author: Ohi Dibua

ClassPSO that defines different version of the particle swarm optimization algorithm for execution.
These algorithms are clasically used to solve typical optimization problems:
    
    min f(x,y)=min f(x) or max f(x,y) = max f(x) where y are inputs with known values
     x          x           x            x

However, this class contains flexibility that allows the solution of more complicated optimization
problems through the user-definition of the fitnessFunction and the fitnessEvaluationFunction. For 
more complex problems, the user can define their own particle class.
     
Within the PSO class, the versions of the optimization algorithm are as follows:
    
    executePSO: Standard particle swarm optimization (PSO) as described in [1]
            
        It takes as inputs:
                optimType = optimization type ('Max' or 'Min')
                w: intertial constant (Recommended range is between 0.5 and 1.0)
                c1: cognitive constant (Recommended range is between 1.00 and 2.05)
                c2: social constant (Recommended range is between 1.00 and 2.05)
                posMin: minimum values of elements of x
                posMax: maximum values of elements of x
                velMin: minimum values of the veleocities ascribed to different elements of x
                velMax: maximum values of the velocities ascribed to different elements of x
                numIters: maximum number of time steps taken in algorithm
                numParticles: number of particles in swarm (recommended range is between 20 and 50)
                particle: Class that defines the particles that will be propagated in the algorithm.
                This class updates the positions and local best
                optimType: optimization type ('Max' or 'Min')
                numEvalState: Number of states required to fully evaluate the fitness of a particle and define it's solution
                fitnessFunction = f(x) 
                fitnessEvaluationFunction: func(optimType,currentState,localBestState,globalBestState=None)
                Function that takes in two states that define proposed solutions and outputs 
                the solution that most meets the objective. If passing in the global best state,
                and empty list is passed in for localBestState. For typical PSO the states are state = (fitness,x)
            
    executeGCPSO: Constrict PSO as described in [1]. Variation of particle swarm optimization that updates velocities 
                  with a pre-factor such that the velocities never go unstable and do not need to be bounded [1]
        
        All inputs are the same as executePSO with the exception of:
                constrict: A float that indicates to the particle class that the constrict update velocity should be used
    
    executeNPSO: Non-parametric PSO. Variation of particle swarm optimization that updates velocities using the best position in a neighborhood of 
                 particles and that updates positon using both the velocity and the best position in the swarm. It also 
                 interpolates between solutions found by two random particles, and updates the global best if these
                 solutions are better [1]
                 
        It removes from the standard PSO the inputs:
            c1, c2
        and adds:
            neighborsize: Size of neighbor to be observed when choosing best neighborhood
            npsoInterpFunc: Function that contains interpolation between states of two particles and returns the
            interpolated state or the best swarm state according to the evaluation function   
 [1] Behesti et al., "Non-parametric particle swarm optimization for global optimization",Applied Soft. Computing. 2015                                 
"""
import numpy as np
import copy
import time
import sys,os
import logging as logger
import sys, os
import time

class PSO():
    #Standard particle swarm optimization function. It returns the swarm and the global best state
    def executePSO(self,c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,particle,optimType,numEvalState,fitnessFunction,fitnessEvaluationFunction):
        #Initialize swarm, global best state, and boolean indicating if the global best has changed
        swarm=[];
        globalBestState=[float("inf")]*numEvalState
        newGlobalBool=0;        
        for j in range(numParticles):
                swarm.append(particle(optimType,c1,c2,w,posMin,posMax,velMin,velMax,fitnessFunction,fitnessEvaluationFunction)); 
        
        #Repeat the process of propagating the particles in solution space
        #and updating the local and global best states
        for t in range(numIters):
            for j in range(numParticles): 
                localBestState = copy.deepcopy(swarm[j].updateLocalBest());
                globalBestState = fitnessEvaluationFunction(optimType.lower(),[],localBestState,globalBestState) 
            for j in range(numParticles):             
                swarm[j].updateVelocityandPosition(globalBestState,t);
        return (swarm,globalBestState)
    
    #Constrict particle swarm optimization function. Returns the swarm and global best state
    def executeGCPSO(self,constrict,c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,particle,optimType,numEvalState,fitnessFunction,fitnessEvaluationFunction):
        #Initialize swarm, global best state, and boolean indicating if the global best has changed
        swarm=[];
        globalBestState=[float("inf")]*numEvalState
        newGlobalBool=0;       
        for j in range(numParticles):
                swarm.append(particle(optimType,c1,c2,w,posMin,posMax,velMin,velMax,fitnessFunction,fitnessEvaluationFunction)); 
        
        #Repeat the process of propagating the particles in solution space
        #and updating the local and global best states
        for t in range(numIters):
            for j in range(numParticles): 
                localBestState = copy.deepcopy(swarm[j].updateLocalBest());
                globalBestState = fitnessEvaluationFunction(optimType.lower(),[],localBestState,globalBestState) 
            for j in range(numParticles):             
                swarm[j].updateVelocityandPosition(globalBestState,t,constrict);
        return (swarm,globalBestState)

    #Non-parametric particle swarm optimization. Returns the swarm and global best state     
    def executeNPSO(self,neighborSize,w,posMin,posMax,velMin,velMax,numIters,numParticles,particle,optimType,numEvalState,fitnessFunction,fitnessEvaluationFunction,npsoInterpFunc):
        #Initialize swarm, global best state, and boolean indicating if the global best has changed
        swarm=[];
        globalBestState=[float("inf")]*numEvalState
        newGlobalBool=0;        
        for j in range(numParticles):
                swarm.append(particle(optimType,w,posMin,posMax,velMin,velMax,fitnessFunction,fitnessEvaluationFunction));             

        #Repeat the process of propagating the particles in solution space
        #and updating the local and global best states
        for t in range(numIters): 
            for j in range(numParticles):  
                #Update local and global best state
                localBestState = copy.deepcopy(swarm[j].updateLocalBest());
                globalBestState = fitnessEvaluationFunction(optimType.lower(),[],localBestState,globalBestState)  
 
                #Find particle with best personal solution in neighborhood               
                neighborLocalBestList=[swarm[j].getLocalBestState()]; 
                for l in range(neighborSize):  
                        idxDeltL = j-(l+1); idxDeltU = j+(l+1);                         
                        if (idxDeltL < 0):
                            rem = np.remainder(idxDeltL,numParticles);
                            neighborLocalBestList.append(swarm[rem].getLocalBestState())
                        else:
                            neighborLocalBestList.append(swarm[idxDeltL].getLocalBestState())
                        if (idxDeltU>=numParticles): 
                            neighborLocalBestList.append(swarm[rem].getLocalBestState())
                        else:
                            neighborLocalBestList.append(swarm[idxDeltU].getLocalBestState()) 
                bestNeighborState=neighborLocalBestList[0];
                for l in range(neighborSize*2+1):
                    bestNeighborState=fitnessEvaluationFunction(optimType.lower(),bestNeighborState,neighborLocalBestList[l]) 

                #Update velocity and position based on best neighborhood particle, and particle in swarm
                swarm[j].updateVelocityandPosition(bestNeighborState,globalBestState,t) 
 
                #Randomly select two particles with states to interpolate and check if interpolated
                #state is better than globalBestState according to fitnessEvaluationFunc
                idxParticles=np.random.choice(numParticles,2,replace=False);
                JState = swarm[idxParticles[0]].getCurrState();
                KState = swarm[idxParticles[1]].getCurrState();                 
                globalBestState = npsoInterpFunc(optimType,JState,KState,posMin,posMax,globalBestState,fitnessFunction,fitnessEvaluationFunction)

        return (swarm,globalBestState)
            
            
            
        
    