#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:11:50 2017

@author: odibua
"""
import numpy as np
import matplotlib as mplib
import matplotlib.pyplot as plt
#import matplotlib.animation as anim
from scipy import integrate

class testFuncClass:
    def testFunction(self,d):
        def testFunctionEval(theta):
           
            y = (theta**3)*(d**2)+theta*np.exp(-np.abs(0.2-d));
           # print("d ",d,"y ",y,"theta ",theta)
            return y
        return testFunctionEval
    
def calcTimeConstantOxideOnly(c0NomIn,params):
    c0NomIn=(c0NomIn*1000.0)*params.NA
    #print c0NomIn
    lambda_d=np.sqrt((params.epsilonR*params.epsilon0*params.kB*params.T)/(2*(params.eCharge**2)*c0NomIn)) 
    c0NomIn=c0NomIn/(params.NA*1000)
    eps=lambda_d/params.L
    oxideLayer=1.0
    sternLayer=1.0
    doubleLayer=1.0
    bulkCapacitance=params.bulkCapacitance
            
    #Define Resistance(s) and Capacitance(s)
    C0=2*eps*doubleLayer #Dimensionless linear component electric double layer capacitor
    c0=1 #Dimensionless bulk concentration of single ion
    R=1./(2*c0) #Dimensionless initial resistance of bulk       
    Cox=(params.epsilonOx/params.epsilonR)*(lambda_d/params.lambda_Ox)*C0*oxideLayer  #Dimensionless capacitance of oxide
    Cstern=(params.epsilonStern/params.epsilonR)*(lambda_d/params.lambda_Stern)*C0*sternLayer #Dimensionless capacitance of Stern Layer
    CBulk=(params.epsilonR/params.epsilonR)*(lambda_d/params.L)*C0*bulkCapacitance
    
    tau=(Cox*R)*((params.L**2)/params.Di)
    return tau

class combDriveModels():    
    def classicCircuitModel(self,VRMS,omegaDimIn):
        def vikramModel(tau1,alpha):
            alpha=alpha*1e-2
            epsilonR=78
            COx=0.0154948325
            CBulk = 0.000138125364
            tau2=(tau1/COx)*CBulk
            f_Tau1=((0.5*omegaDimIn*tau1)**2)/(1+((0.5*omegaDimIn*tau1)**2))  
            
            displ=epsilonR*(alpha)*f_Tau1*(VRMS**2)  
            
            return displ
        return vikramModel
    
    def leakyDelectricWithStern(self,VRMS,omegaDimIn):
        def vikramModelRCDielStrnFitAlpha(combDriveParams,params,ROx,lambdaOx,CStern,tauBulk,alpha):
            alpha=alpha*1e-2;
            lambdaOx = lambdaOx*1e-9;
            epsilonOx = params.epsilonOx;
            epsilonBulk = params.epsilonR;
            epsilon0  = params.epsilon0;
            g = combDriveParams.d;
            
            Cox = epsilon0*epsilonOx/(lambdaOx);
            CBulk = (epsilon0*epsilonBulk)/g;
            RBulk = tauBulk/CBulk;
            
            ZOx = ROx/(1+omegaDimIn*ROx*Cox*1j);
            ZBulk = RBulk/(1+omegaDimIn*tauBulk*1j);
            ZStern = 1/(omegaDimIn*CStern*1j);
            Z = 2*(ZOx+ZStern)+ZBulk;

            fTau =abs(ZBulk/Z);
            displ=epsilonBulk*alpha*fTau*(VRMS**2);
            
            
            return displ
        return vikramModelRCDielStrnFitAlpha