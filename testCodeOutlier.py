#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:12:28 2017

@author: odibua
"""

import numpy as np
import random
from testFuncs import rastriginFunc
from testFuncs import sphereFunc
from testFuncs import mcCormicFunc
from testFuncs import bealeFunc
from testFuncs import holderTableFunc
from testFuncs import rosenbrockFunc
from testFuncs import ackleysFunc
from psoMethods import PSO
from psoMethods import psoParticle
from psoMethods import psoParticleCluster
from psoMethods import psoParticleClusterAndOutliers
from clusteringMethodsV2 import manipClusterPSO
#from mainClustering import gaussianKernel
from clusterFitnessEvaluationFunctions import constKpFitFunc 
from clusterFitnessEvaluationFunctions import variableKpFitCSMeasFunc
from clusterFitnessEvaluationFunctions import clusterFitnessEvaluationFunc
from clusterFitnessEvaluationFunctions import clusterOutlierFitnessEvaluationFunc
from clusterFitnessEvaluationFunctions import clusterOutlierFitness
import matplotlib.pyplot as plt
import copy
#from clusterVersion2 import manipClusterPSO
#from clusterVersion2 import gaussianKernel

# reading in all data into a NumPy array
#dataTypeArr =['ArtificialVarNoOut','ArtificialVarOut','WDBC','Shuttle']
dataTypeArr=['Shuttle']
for dataType in dataTypeArr:    
    if (dataType=='WDBC'):
        eps=0.01
        all_data = np.loadtxt(open("./wdbc_data.csv","r"),
                delimiter=",",
                skiprows=0,
                dtype=np.float64
                )
        
        # load class labels from column 1
        yData = all_data[:,0]
        # conversion of the class labels to integer-type array
        yData = yData.astype(np.int64, copy=False)
        
        # load the 13 features
        XData = all_data[:,1:]
        ##Downsample malignant
        idxBenign=np.where(yData==0)[0]
        idxMalign=np.where(yData==1)[0]
        idxMalign=idxMalign[np.random.choice(len(idxMalign),10)];
        yData = yData[np.concatenate((idxBenign,idxMalign)).astype(int)];
        XData = XData[np.concatenate((idxBenign,idxMalign)).astype(int),0:];
        
    ##############################################################################################
    elif (dataType=='Shuttle'):
        eps=0.01
        all_data = np.loadtxt(open("./shuttle_data.csv","r"),
                delimiter=",",
                skiprows=0,
                dtype=np.float64
                )    
        # load class labels from column 1
        yData = all_data[:,0]
        # conversion of the class labels to integer-type array
        yData = yData.astype(np.int64, copy=False)
        
        # load the 13 features
        XData = all_data[:,1:]
        #Downsample shuttle
        idxShuttle=np.where(yData!=4)[0]
        yData = yData[idxShuttle];
        XData = XData[idxShuttle,0:]
            

    ######################################################
    elif (dataType=='ArtificialVarNoOut'):
        eps=0.1;
        cov=np.array([[0.05, 0],[0,0.05]])
        XData=np.vstack([np.random.multivariate_normal([0,0],cov,10),np.random.multivariate_normal([2,2],cov,10),np.random.multivariate_normal([4,4],cov,10),np.random.multivariate_normal([8,8],cov,10)])
    elif (dataType=='ArtificialVarOut'):
        eps=0.05;
        cov=np.array([[0.05, 0],[0,0.05]])
        XData=np.vstack([np.random.multivariate_normal([0,0],cov,10),np.random.multivariate_normal([0.9,0.9],cov/100,1),np.random.multivariate_normal([2,2],cov,10),np.random.multivariate_normal([2.9,2.9],cov/100,1),np.random.multivariate_normal([4,4],cov,10),np.random.multivariate_normal([4.9,4.9],cov/100,1),np.random.multivariate_normal([8,8],cov,10)])
    #
    shapeData = np.shape(XData);
    nData=shapeData[0]; nFeatures=shapeData[1];
    featureMin=np.min(XData,axis=0);
    featureMax=np.max(XData,axis=0);
    XDataNormalized = copy.deepcopy(XData);
    for j in range(nData):
        for k in range(nFeatures):
            XDataNormalized[j][k]=(XData[j][k]-featureMin[k])/(featureMax[k]-featureMin[k]);
    featureMin=np.min(XDataNormalized,axis=0);
    featureMax=np.max(XDataNormalized,axis=0);
    if (dataType=='Shuttle'):
        nData=500;
        idx=np.random.choice(len(yData),nData,replace=False)
    else:
        idx=range(nData)
    
    X=XDataNormalized[idx,0:]#Normalized[idx,0:];

    #nbaData=np.load("PG2000-2001Data.npz");
    #X=nbaData["X"]
    #nData=np.shape(X)[0]
    #nFeatures=np.shape(X)[1]
    #featureMin=np.min(X,axis=0);
    #featureMax=np.max(X,axis=0);
    #XNormalized=copy.deepcopy(X);
    #for j in range(nData):
    #    for k in range(nFeatures):
    #        XNormalized[j][k]=(X[j][k]-featureMin[k])/(featureMax[k]-featureMin[k]);
    #X = XNormalized
    
    KpMin=1.0;
    KpMax=20.0;
    lMin=0.0;
    lMax=round(nData*0.1)*1.0#5.0;
    Kp=6;
    numEvalState=8#4;
    featureMin=np.min(X,axis=0);
    featureMax=np.max(X,axis=0);
    nFeatures=len(featureMin)
    posMin=[Kp];
    posMax=[Kp];
    
    #posMin=[KpMin];
    #posMax=[KpMax];
    #posMin.append(featureMin);
    #posMax.append(featureMax);
    #velMin=[-(KpMax-KpMin)/2.0];
    #velMax=[(KpMax-KpMin)/2.0];
    #velMin.append(posMin[1]/2.0); 
    #velMax.append(posMax[1]/2.0);
    
    posMin=[KpMin];
    posMax=[KpMax];
    posMin.append(lMin);
    posMax.append(lMax)
    posMin.append(featureMin);
    posMax.append(featureMax);
    velMin=[-(KpMax-KpMin)/2.0];
    velMax=[(KpMax-KpMin)/2.0];
    velMin.append(-(lMax-lMin)/2.0);
    velMax.append((lMax-lMin)/2.0);
    velMin.append(posMin[2]/2.0); 
    velMax.append(posMax[2]/2.0);
    
    
    numParticles = 50;
    numNeighbor = np.ceil(numParticles/5).astype(int);
    w=0.75;
    tol=1e-3;
    numIters=100#nFeatures*15;
    #c1 = np.linspace(0.3,2.0,numIters)
    #c2= np.linspace(2.0,3.0,numIters)
    kappa = 0.5;
    mult=1;
    c1=1.0
    c2 = c1*mult;
    optimType='Min';
    
    sigma=np.linalg.norm(featureMax-featureMin)/2.0;
    pso=PSO();
    
    
    random.seed(0)
    bestGlobalFitnessList=[]
    bestGlobalKpList=[]
    bestGlobalCentroidsList=[]
    bestGlobalNumEmptyClustersList=[]
    bestGloballList=[]
    bestGlobalOutliersList=[]
    weightsList=[]
    weightsOutliersList=[]
    nTrials=15;
    #eps=0.01
    evaluationFunc = clusterOutlierFitness(eps);
    print(dataType)
    for j in range(nTrials):
        print("Trial",j+1)
        output1=pso.executePSOCluster(c1,c2,w,posMin,posMax,velMin,velMax,numIters,numParticles,psoParticleClusterAndOutliers,optimType,X,numEvalState,variableKpFitCSMeasFunc,evaluationFunc,sigma)
        evalState=output1[1]
        bestGlobalFitnessList.append(evalState[0])
        bestGlobalKpList.append(evalState[1])
        bestGlobalCentroidsList.append(np.array(evalState[2]))
        bestGlobalNumEmptyClustersList.append(evalState[3])
        bestGloballList.append(evalState[4])
        bestGlobalOutliersList.append(np.array(evalState[5]))
        weightsList.append(np.array(evalState[6]))
        weightsOutliersList.append(np.array(evalState[7]))
    if (dataType=='Shuttle' or dataType=='WDBC'):
        y=yData[idx]
        np.savez(dataType+'Data'+'.npz',X=X,y=y,sigma=sigma,
        bestGlobalFitnessList=bestGlobalFitnessList,bestGlobalKpList=bestGlobalKpList,
        bestGlobalCentroidsList=bestGlobalCentroidsList,bestGlobalNumEmptyClustersList=bestGlobalNumEmptyClustersList,bestGloballList=bestGloballList,
        bestGlobalOutliersList=bestGlobalOutliersList,weightsList=weightsList,weightsOutliersList=weightsOutliersList)         
    else:
        np.savez(dataType+'Data'+'.npz',X=X,sigma=sigma,
        bestGlobalFitnessList=bestGlobalFitnessList,bestGlobalKpList=bestGlobalKpList,
        bestGlobalCentroidsList=bestGlobalCentroidsList,bestGlobalNumEmptyClustersList=bestGlobalNumEmptyClustersList,bestGloballList=bestGloballList,
        bestGlobalOutliersList=bestGlobalOutliersList,weightsList=weightsList,weightsOutliersList=weightsOutliersList) 

