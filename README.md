# Bayesian Optimal Design

This file contains code that will be used to, given a model, propose the optimal design points for experiments. 
In this case, optimality is defined by a utility function that is derived based on the Kullback-Leibler (KL) divergence
and is the design points that allow you to perform inferences on model parameters that yield the most information. 
The implementation of Bayesian Optimal Design is simulation based and follows the strategy outlined in [1]. The optimization is done using particle swarm optimiztion, which is commented upon more thoroughly in the code, and at https://github.com/odibua/Optimization/blob/master/particleSwarmOptimizationMethods/README.md

[1] X. Huan, Y. Marzouk, "Simulation-based optimal Bayesian experimental design for nonlinear systems,"
    Journal of Computational Physics, 2012. https://arxiv.org/abs/1108.4146
    
# Required dependencies

This code requires pyDOE and an updated version of scipy that allows the calling of scipy.stats.multivariate_normal. 
The ability of the code to converge is heavily dependent on how uniformly distributed the prior actually is

# Running Tests
The testUtilityFunction contains code that allows the testing of this, and the comments in that file yield the solutions that one should expect from testing this. The test can be run from the python command line argument with the following inputs

    methodUse=sys.argv[1]; thetaMin = float(sys.argv[2]); thetaMax = float(sys.argv[3]); nSamples=int(sys.argv[4])

The valid range for theta is between 0 and 1, and the valid strings for methodUse (which refers to PSO optimization) is in 
the string

    methodString = ['PSO','GCPSO','NPSO']
    
The design parameter and distribution of the noise are defined in the following block of code

    #Define design parameter dimensions and its boundaries
    nSimultaneous=2; designDimension=1; dLB=0; dUB=1.0

    #Define distribution for the noise to be added to model
    variance=(1e-4);
    mu=[0]*nSimultaneous; cov = [[0]*nSimultaneous for j in xrange(nSimultaneous)]; 
    for j in xrange(nSimultaneous):
        cov[j][j]=variance 
    distribution=multivariate_normal(mu,cov) 

The utility function is defined below

    #Define Utility function that take design parameters as arguments after passing in 
    KL = KLUtility(forwardModelClass,distribution,genPriorSamples,nSamples,nSimultaneous,designDimension)
    
and the relevant inputs to the particle swarm optimizer are defined in the following block of code

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

Finally the optimization is run in the last part of the code

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
