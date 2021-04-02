'''

Supervised pretraining:

s: simulator state

repeat in minibatches:
  s <- sampleState()
  sample a sequence of H actions
  simulate to get a sequence of H observations
  Adam update to predict actions from observations

Training with REINFORCE and baseline reduction:

k_b: how many samples used to compute baseline

repeat in minibatches
  s <- sampleState()
  O_target=sampleTargetObsSeq(s)  #see below
  for k in 0...k_b-1:
    sample actions A^(k) ~ pi(a_1,...a_H | s,O)
    O_sim^(k) = simulate(A^(k))
    error^(k)=||O_target-O_sim^(k)||
  baseline=mean_k(error^(k))
  for k in 0...k_b-1:
    advantage^(k)=-(error^(k)-baseline)
  Adam update with loss=mean_batch(advantage*log pi(a_1,...a_H | s,O))


sampleTargetObsSeq(s):
  with probability p_feasible:
    sample random actions
    O=simulate to get a sequence of observations
  with probability p_interp
    O=lerp(s0,sampleInitialState())
  with probability p_diverged:
    O=lerp(sampleInitialState(),sampleInitialState())
  noiseScale~Uniform(0,maxnoise)
  O+=noiseScale+random noise ~N(mu_obs,sigma_obs)
  return O,w

sampleState:
  with probabity p_contact:
    sample random movement state in contact with ground
  with probabity p_near_contact
    sample random movement state close to ground
  otherwise:
    sample random movement state at random height
  

Rationale:

For infeasible targets, the cost could be very high. Thus, advantage estimation with value function prediction is unstable.
Since the simulated sequences are short, Monte Carlo estimation of advantage is not prohibitively expensive, and is unbiased.

Contextual bandit.

'''


import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" #disable Tensorflow GPU usage, these simple graphs run faster on CPU
import numpy as np
import tensorflow as tf
import MLP
# from stable_baselines import logger

#VALUE NET -RELATED HYPERPARAMS
#Whether to use value network for advantage estimation.
useValueNetForAdvantages=False
#If the above is False, the baseline for advantages is computed by repeating episodes this many times from the same initial obs,
#and averaging. Since our episode length is 1, this is not terribly expensive.
nValueEstimationSamples=4
#Whether to use value net recursively, when querying the value from child LLC:s (i.e., value from the state resulting from taking an action)
#If this is False, we estimate the value by actually running the agent on the child LLC actions.
useValueNetForValue=False

#REGULARIZAION PARAMETERS
useAdvantageNormalization=True
gradientClippingMaxNorm=0.5
negativeAdvantageClippingThreshold=1.0      #Clip negative advantages beyond this many standard deviations
klDivergencePenaltyWeight=2.0               #Weight for the PPO KL-divergence penalty in the policy loss
useNegativeAdvantages=False                 #Whether to utilize negative advantages at all
useNegativeAdvantageMirroring=False         #If neg. adv. are used, whether to use the the PPO-CMA negative advantage mirroring trick
negativeAdvantageMirroringKernelWidth=0.5   #Kernel width parameter for the above


def gaussianLogp(mean,target,var,logVar):
    return -0.5*tf.reduce_sum(tf.square(mean-target)/var+logVar,axis=1,keepdims=True)

#P=1,Q=2, for shapes[nBatch,nVariables]
def batchGaussianKLDivergence(mu1,var1,logVar1,mu2,var2,logVar2):
    return tf.reduce_mean(tf.reduce_sum(0.5*(logVar2-logVar1)+0.5*(var1+(mu1-mu2)**2)/var2 - 0.5,axis=1))

#Helper, eq 5 in http://papers.nips.cc/paper/7141-what-uncertainties-do-we-need-in-bayesian-deep-learning-for-computer-vision.pdf,
#assuming shapes [nBatch,nVariables]
def weightedGaussianLoss(mean,target,var,logVar,weights):
    if weights==None:
        weights=1
    return -tf.reduce_mean(weights*gaussianLogp(mean,target,var,logVar))


class LLC:
    def __init__(self,sess:tf.Session,env,H:int,dataFile,singleTargetObs=False,trainIterBudget=8192,nTrainIter=100
                 ,nNeuronsPerLayer=128,nHidden=3,forceRetrain=False,activation="swish",excludedFirstObs=0
                 ,valueSimulationSteps=5,createChildren=True
                 , _klDivergencePenaltyWeight=10,_negativeAdvantageMirroringKernelWidth=0.1,_useNegativeAdvantages=False):

        global klDivergencePenaltyWeight
        global useNegativeAdvantageMirroring
        global negativeAdvantageMirroringKernelWidth
        global useNegativeAdvantages
        klDivergencePenaltyWeight = _klDivergencePenaltyWeight
        useNegativeAdvantageMirroring = _negativeAdvantageMirroringKernelWidth > 0
        negativeAdvantageMirroringKernelWidth = _negativeAdvantageMirroringKernelWidth
        useNegativeAdvantages = _useNegativeAdvantages

        self.H=H
        self.sess=sess
        self.rootObs=env.getRootObsIndices()
        self.actionMin=env.action_space.low.copy()
        self.actionMax=env.action_space.high.copy()
        useSkips=True
        if H>1 and createChildren:
            #create 
            childLLC=LLC(sess,env,H=H-1,dataFile=dataFile,singleTargetObs=singleTargetObs,trainIterBudget=trainIterBudget,nNeuronsPerLayer=nNeuronsPerLayer,nHidden=nHidden,activation=activation,excludedFirstObs=excludedFirstObs)
        else:
            childLLC=None
        self.childLLC=childLLC
        obsDim=env.observation_space.shape[0]
        actionDim=env.action_space.shape[0]
        oldIgnoreDone=env.ignoreDone
        env.setIgnoreDone(True)
        targetObsDim=obsDim if singleTargetObs else H*obsDim
        Heff=1 if singleTargetObs else H

        #Create the trajectory follower policy. TODO: refactor into a PPO class, which handles the nets, and takes in batches of observations, actions, and rewards
        tfBatchSize=tf.placeholder(dtype=tf.int32,name="tfBatchSize_placeholder")
        currObsIn=tf.placeholder(dtype=tf.float32,shape=[None,obsDim],name="currObsIn_placeholder")
        self.currObsIn=currObsIn
        targetObsIn=tf.placeholder(dtype=tf.float32,shape=[None,Heff,obsDim],name="targetObsIn_placeholder")
        targetObsIn_flat=tf.reshape(targetObsIn,[-1,targetObsDim])
        actionsIn=tf.placeholder(dtype=tf.float32,shape=[None,actionDim],name="actionsIn_placeholder")
        actionsIn_flat=tf.reshape(actionsIn,[-1,actionDim])
        concatenated=tf.concat([currObsIn,targetObsIn_flat],axis=1)
        nOutputs=actionDim
        policy=MLP.MLP(input=concatenated,nLayers=nHidden,nUnitsPerLayer=nNeuronsPerLayer,nOutputUnits=nOutputs
                       ,activation=activation,o_activation="tanh",useSkips=useSkips)
        actionMeanOut,initOps=policy.output,policy.initOutput
        sdScale=1.0/3.0 
        initialSd=sdScale*np.reshape(0.5*(env.action_space.high-env.action_space.low),[1,-1])
        actionOutLogVar=tf.Variable(initial_value=np.log(np.square(initialSd)),dtype=tf.float32)
        actionOutVar=tf.exp(actionOutLogVar)
        actionOutSd=tf.sqrt(actionOutVar)
        actionSamplesOut=tf.truncated_normal([tfBatchSize,actionMeanOut.shape[1].value],mean=actionMeanOut,stddev=actionOutSd)

        #Create model for predicting the mean and variance of future observations (after H steps) based on current observations.
        #For now, we use simple linear model for mean, and a constant model for variance.
        #We init the mean to identity matrix + a little noise (to break symmetries in optimization), as the initial guess is that 
        #the observations stay unchanged
        epsilon=1e-3
        initialW=np.random.uniform(-epsilon,epsilon,size=[obsDim,targetObsDim])
        for h in range(Heff):
            for i in range(obsDim):
                initialW[i,i+h*obsDim]=1     
        W=tf.Variable(initial_value=initialW,dtype=tf.float32,name='W')  
        futureObsMean=tf.matmul(currObsIn,W)
        futureObsLogVar=tf.Variable(initial_value=np.zeros([1,targetObsDim]),dtype=tf.float32,name='W')
        futureObsVar=tf.exp(futureObsLogVar)
        futureLoss=weightedGaussianLoss(futureObsMean,targetObsIn_flat,futureObsVar,futureObsLogVar,weights=None)
        self.futureObsMean=tf.reshape(futureObsMean,[-1,Heff,obsDim])       #reshape for easier access later
        self.futureObsSd=tf.reshape(tf.sqrt(futureObsVar),[-1,Heff,obsDim]) #reshape for easier access later

        #L1 pretraining loss and optimizer
        meanInitLoss=tf.reduce_mean(tf.reduce_sum(tf.abs(actionMeanOut-actionsIn_flat),axis=0))
        pretrainingLoss=meanInitLoss+futureLoss
        optimizer_pretrain=tf.train.AdamOptimizer()
        optimize_pretrain=optimizer_pretrain.minimize(pretrainingLoss)
 

        #Inputs needed for losses
        advantagesIn=tf.placeholder(dtype=tf.float32,shape=[None,1],name="advantagesIn_placeholder")
        logPiOldIn=tf.placeholder(dtype=tf.float32,shape=[None,1],name="logPiOldIn_placeholder")

        #Mirror negative advantage samples and damp their advantages for samples far from the mean using Gaussian kernel, like in PPO-CMA.
        #This avoids divergence due to negative advantage samples pushing the policy mean away over the multiple Adam steps of PPO
        oldActionMeanIn=tf.placeholder(dtype=tf.float32,shape=[None,actionDim],name="oldActionMean_placeholder")
        posAdvantages=tf.nn.relu(advantagesIn)
        if not useNegativeAdvantages:
            #Default PPO/PG loss, but only positive advantages
            policyLoss=weightedGaussianLoss(actionMeanOut,actionsIn_flat,actionOutVar,actionOutLogVar,posAdvantages)
        elif not useNegativeAdvantageMirroring:
            #Default PPO/PG advantage-weighted loss
            policyLoss=weightedGaussianLoss(actionMeanOut,actionsIn_flat,actionOutVar,actionOutLogVar,advantagesIn)
        else:
            #Loss with PPO-CMA negative advantage mirroring trick
            negAdvantages=tf.nn.relu(-advantagesIn)
            if useNegativeAdvantageMirroring:
                mirroredActions=oldActionMeanIn-(actionsIn_flat-oldActionMeanIn)  #mirror negative advantage actions around old policy mean (convert them to positive advantage actions assuming linearity)
                kernelSqWidth=actionOutVar * negativeAdvantageMirroringKernelWidth**2
                avoidanceKernel=tf.reduce_mean(tf.exp(-0.5*tf.square(actionsIn_flat-oldActionMeanIn)/kernelSqWidth),axis=1)
                negAdvantages*=avoidanceKernel
                negAdvantages=tf.stop_gradient(negAdvantages)

            #TODO: try implementing the PPO-CMA separate mean and variance adaptation (simple here, as variance is a variable, not a network)
            #    policyMeanLoss-=tf.reduce_mean((negAdvantages*avoidanceKernel)*logpNoVarGradMirrored)
            #policySigmaLoss=-tf.reduce_mean(posAdvantages*logpNoMeanGrad)
            #policyMeanLoss=-tf.reduce_mean(posAdvantages*logpNoVarGrad)
            #    #Separate mean and sigma adaptation losses
            #    policyNoGrad=tf.stop_gradient(policyMean)
            #    policyVarNoGrad=tf.stop_gradient(policyVar)
            #    policyLogVarNoGrad=tf.stop_gradient(policyLogVar)
            #    logpNoMeanGrad=-tf.reduce_sum(0.5*tf.square(actionIn-policyNoGrad)/policyVar+0.5*policyLogVar,axis=1)
            #    logpNoVarGrad=-tf.reduce_sum(0.5*tf.square(actionIn-policyMean)/policyVarNoGrad+0.5*policyLogVarNoGrad,axis=1) 

            policyLoss=weightedGaussianLoss(actionMeanOut,actionsIn_flat,actionOutVar,actionOutLogVar,posAdvantages)
            policyLoss+=weightedGaussianLoss(actionMeanOut,mirroredActions,actionOutVar,actionOutLogVar,negAdvantages)
        oldActionVarIn=tf.placeholder(dtype=tf.float32,shape=[None,actionDim],name="oldActionVar_placeholder")

        #We don't use the PPO clipped surrogate objective, as it was recently shown to not enforce the trust region (https://openreview.net/forum?id=r1etN1rtPB),
        #we instead add a KL-divergence penalty, which was another option presented in the original Schulman 2017 paper
        policyLoss+=klDivergencePenaltyWeight*batchGaussianKLDivergence(actionMeanOut,actionOutVar,actionOutLogVar,oldActionMeanIn,oldActionVarIn,tf.log(oldActionVarIn))

        #Optimizer, with gradient clipping
        learningRate=tf.Variable(initial_value=0.001,dtype=tf.float32,trainable=False)
        optimizer=tf.train.AdamOptimizer(learning_rate=learningRate)
        gradients, variables = zip(*optimizer.compute_gradients(policyLoss))
        gradients, _ = tf.clip_by_global_norm(gradients, gradientClippingMaxNorm)
        optimizePolicy=optimizer.apply_gradients(zip(gradients, variables))

        #Create value function predictor network. Only used for training
        valueNet=MLP.MLP(input=concatenated,nLayers=nHidden,nUnitsPerLayer=nNeuronsPerLayer,nOutputUnits=1,activation=activation,useSkips=useSkips)
        valuesOut,vfpredInitOps=valueNet.output,valueNet.initOutput
        valuesIn=tf.placeholder(dtype=tf.float32,shape=[None,1],name="valuesIn_placeholder")
        vfpredLoss=tf.reduce_mean(tf.abs(valuesOut-valuesIn))
        valueOptimizer=tf.train.AdamOptimizer()
        optimizeValue=valueOptimizer.minimize(vfpredLoss)

        #reshape tf model outputs for easier use
        actionSamplesOut=tf.reshape(actionSamplesOut,[-1,actionDim])
        actionMeanOut_flat=actionMeanOut
        actionMeanOut=tf.reshape(actionMeanOut,[-1,actionDim])

        #prepare save/load
        var_list=policy.getAllVariables()
        valueWeights=valueNet.getAllVariables()
        for w in valueWeights:
            var_list.append(w)
        var_list.append(W)
        var_list.append(futureObsLogVar)
        var_list.append(actionOutLogVar)
        saver = tf.train.Saver(var_list=var_list)
        modelFileName=dataFile.split('ExplorationData/')[1].split('.')[0]
        modelFileName+="_H={}_nUnits={}_nHidden={}_useSkips={}".format(H,nNeuronsPerLayer,nHidden,useSkips)
        modelFileName+="_"+activation
        modelFileName+="_ppo_2"
        if singleTargetObs and H>1:     #for H=1, the value of singleTargetObs makes no difference, but for H>1, we need to train separate models
            modelFileName+="_singleTargetObs"

        modelFileName += '_KlPenWt=%d' % klDivergencePenaltyWeight
        if useNegativeAdvantages:
            modelFileName += '_WithNegAdv'
        if useNegativeAdvantageMirroring:
            modelFileName += '_MrrKrnlWdth=%.2f' % negativeAdvantageMirroringKernelWidth

        directory = os.path.join("ResultsLLC", modelFileName)
        # logger.configure(directory, ['stdout', 'csv'])

        modelFileName = os.path.join('Models', modelFileName)
        if not os.path.exists(modelFileName):
            os.makedirs(modelFileName)
        modelFileName = os.path.join(modelFileName, 'model')

        #save the variables and tensors we will need later
        self.actionMeanOut=actionMeanOut
        self.currObsIn=currObsIn
        self.targetObsIn=targetObsIn
        self.obsDim=obsDim
        self.tfBatchSize=tfBatchSize
        self.valuesOut=valuesOut
        self.saver=saver
        self.modelFileName=modelFileName
        self.singleTargetObs=singleTargetObs
        self.actionSamplesOut=actionSamplesOut

        #if trained network exists, load it
        if (not forceRetrain) and os.path.isfile(modelFileName+".meta"):
            print("Loading pretrained policy for H={}".format(H))
            saver.restore(sess,modelFileName)
            normData=np.load(modelFileName+"_normalization.npy",allow_pickle=True)
            self.targetObsMean=normData.item()["mean"]
            self.targetObsSd=normData.item()["sd"]
            #self.targetObsMin=normData.item()["min"]
            #self.targetObsMax=normData.item()["max"]
        else:
            #load exploration data
            data=np.load(dataFile,allow_pickle=True)
            parentIndices=data.item()["parentIndices"]
            observations=data.item()["observations"]
            actions=data.item()["actions"]
            nextObservations=data.item()["nextObservations"]
            nData=data.item()["N"]

            #parse dynamics training data from the loaded data and train the models
            #Only those observations that have enough history are included
            print("Generating LLC training data, prediction horizon: ",H)
            idXpInitialObs=np.zeros([nData,obsDim])
            idXpNextObs=np.zeros([nData,Heff,obsDim])
            idXpActions=np.zeros([nData,actionDim])
            minIdObs=np.zeros(obsDim)
            maxIdObs=np.zeros(obsDim)
            nIdXp=0
            for targetIdx in range(nData):
                #Find the initial observation idx by backtracking
                origIdx=targetIdx
                for j in range(H-1):
                    origIdx=parentIndices[origIdx]
                    if origIdx<0:
                        break
                if origIdx>=0:
                    #Store initial observation, next observations, and actions
                    idXpInitialObs[nIdXp]=observations[origIdx][excludedFirstObs:]
                    if singleTargetObs:
                        idXpNextObs[nIdXp,0]=nextObservations[origIdx+H-1][excludedFirstObs:]
                    else:
                        for h in range(H):
                            idXpNextObs[nIdXp,h]=nextObservations[origIdx+h][excludedFirstObs:]
                    idXpActions[nIdXp]=actions[origIdx] 
                    nIdXp+=1
            minIdObs=np.min(idXpInitialObs,axis=0)
            maxIdObs=np.max(idXpInitialObs,axis=0)
            idObsMean=np.mean(idXpInitialObs,axis=0)
            idObsSd=np.std(idXpInitialObs,axis=0)
            #print("minIdObs",minIdObs)
            #print("maxIdObs",maxIdObs)
            print("Loaded a total of {} data points".format(nIdXp))

            #save stats needed for normalizing
            self.targetObsMean=np.repeat(np.reshape(idObsMean,[1,-1]),Heff,axis=0)
            self.targetObsSd=np.repeat(np.reshape(idObsSd,[1,-1]),Heff,axis=0)
            self.targetObsMean=np.reshape(self.targetObsMean,[1,Heff,self.obsDim])
            self.targetObsSd=np.reshape(self.targetObsSd,[1,Heff,self.obsDim])
            #self.targetObsMin=np.repeat(minIdObs,H,axis=0)
            #self.targetObsMax=np.repeat(maxIdObs,H,axis=0)
            np.save(modelFileName+"_normalization",{"mean":self.targetObsMean,"sd":self.targetObsSd}) #,"min":self.targetObsMin,"max":self.targetObsMax})

            #Init model with a large random batch.
            #Note: we avoid using global variables initializer to avoid reinitializing the loaded child LLC
            print("Data-dependent network init...")
            sess.run(tf.variables_initializer(var_list))
            sess.run(tf.variables_initializer(optimizer_pretrain.variables()))
            sess.run(tf.variables_initializer(optimizer.variables()))
            sess.run(tf.variables_initializer(valueOptimizer.variables()))
            sess.run(tf.variables_initializer([learningRate]))
            batchIndices=np.random.randint(0,nIdXp,size=min([nIdXp,10000]))
            sess.run(initOps,feed_dict={currObsIn:idXpInitialObs[batchIndices,:],targetObsIn:idXpNextObs[batchIndices,:],actionsIn:idXpActions[batchIndices,:]})

            #pretrain in supervised manner
            print("Pretraining...")
            nTrainingIter=20000
            for iter in range(nTrainingIter):
                #Minibatch training
                nMinibatch=256
                batchIndices=np.random.randint(0,nIdXp,size=nMinibatch)
                currLoss,currFutureLoss,_=sess.run([meanInitLoss,futureLoss,optimize_pretrain],feed_dict={currObsIn:idXpInitialObs[batchIndices,:],targetObsIn:idXpNextObs[batchIndices,:],actionsIn:idXpActions[batchIndices,:]})
                if iter % 1000==0:
                    print("Iter {}/{}, mean init loss {:.2f}, futureLoss {:.2f}".format(iter,nTrainingIter,currLoss,currFutureLoss))
            obsSd_fetch=sess.run(self.futureObsSd,feed_dict={currObsIn:idXpInitialObs[batchIndices,:]})
            if singleTargetObs:
                print("Future obs at {} steps: {}".format(H,np.mean(obsSd_fetch[:,0,:])))
            else:
                for h in range(H):
                    print("Future obs sd at step {}: {}".format(h,np.mean(obsSd_fetch[:,h,:])))

            #we will be taking snapshots of the network whenever we find a better one
            snapshotValue=-np.inf
            snapshotVariables=None

            #train
            print("Training...")
            initialValueNetworkIter=-1
            nTrainingIter=500

            best_mean_q = -np.inf
            for iter in range(nTrainingIter):
                targetDataPerIter=15000#8192
                if iter==0 and useValueNetForAdvantages:
                    #Collect more data on first iteration to allow more robust initialization of the value network without overfitting
                    targetDataPerIter*=4
                nBatch=env.N   #batch size, for now we use the number of parallel simulations available3
                nBatches=targetDataPerIter//nBatch
                nDataPerIter=nBatches*nBatch
                if iter==0 or iter==1:
                    #Have to reallocate at iter==1 because we use more data at iter==0
                    initialObs=np.zeros([nDataPerIter,obsDim])
                    targetObs=np.zeros([nDataPerIter,Heff,obsDim])
                    actions=np.zeros([nDataPerIter,actionDim])
                    Q=np.zeros([nDataPerIter,1])
                #Collect experience
                print("Collecting experience...")
                meanSimError=0
                for batchIdx in range(nBatches):
                    batchInitialObs=np.zeros([nBatch,obsDim])
                    batchTargetObs=np.zeros([nBatch,Heff,obsDim])
                    batchActions=np.zeros([nBatch,actionDim])

                    #Prepare batch initial and target observations
                    for sampleIdx in range(nBatch):
                        #sample initial state and target observation
                        if useValueNetForAdvantages or (sampleIdx % nValueEstimationSamples==0):
                            r_init=np.random.randint(0,nIdXp)
                            batchInitialObs[sampleIdx]=idXpInitialObs[r_init]

                            #Sample target obs sequence: 
                            r=np.random.uniform(0,1)
                            if r<0.8:
                                #Use the sequence from the exploration data
                                batchTargetObs[sampleIdx]=idXpNextObs[r_init]
                            elif r<0.9 and (not singleTargetObs):
                                #Blend between the exploration sequence and a sequence starting from random other state
                                r_target=np.random.randint(0,nIdXp)
                                for h in range(H):
                                    targetWeight=(h+1)/H
                                    batchTargetObs[sampleIdx,h]=(1.0-targetWeight)*idXpNextObs[r_init][h]+targetWeight*idXpNextObs[r_target][h]
                                batchTargetObs=self.clampTargetObs(batchInitialObs,batchTargetObs)
                            else:
                                #Constant randomly selected target state
                                r_target=np.random.randint(0,nIdXp)
                                batchTargetObs[sampleIdx,:]=idXpInitialObs[r_target]
                                batchTargetObs=self.clampTargetObs(batchInitialObs,batchTargetObs)
                        else:
                            #if not using the value net, we duplicate the initial state and target observations for many samples,
                            #to allow value (and advantage) estimation through averaging rollout returns 
                            batchInitialObs[sampleIdx]=batchInitialObs[sampleIdx-1]
                            batchTargetObs[sampleIdx]=batchTargetObs[sampleIdx-1]
                            assert(nBatch % nValueEstimationSamples==0)


                    #Sample actions from policy. No target observation clamping because already clamped above
                    batchActions=self.act(batchInitialObs,batchTargetObs,actionMode="sample",clamp=False)

                    #Simulate
                    env.setStateFromObs(batchInitialObs)
                    simulatedObs,_,_,_=env.step(batchActions)
                
                    #compute action values.  
                    batchValues=self.rewards(simulatedObs,np.reshape(batchTargetObs[:,0,:],[-1,obsDim]))
                    if childLLC is not None:
                        childTargetObs=batchTargetObs if singleTargetObs else batchTargetObs[:,1:,:]
                        batchValues+=childLLC.value(simulatedObs,env,childTargetObs,valueSimulationSteps=valueSimulationSteps)

                    #Store
                    base=batchIdx*nBatch
                    Q[base:base+nBatch]=batchValues
                    actions[base:base+nBatch]=batchActions
                    initialObs[base:base+nBatch]=batchInitialObs
                    targetObs[base:base+nBatch]=batchTargetObs

                #EXPERIENCE COLLECTED, FINISH ITERATION:
                #Update value function predictor
                if useValueNetForValue or useValueNetForAdvantages:
                    print("Training value network")
                    if iter==0:
                        print("Data-dependent network init...")
                        sess.run(vfpredInitOps,feed_dict={currObsIn:initialObs,targetObsIn:targetObs,valuesIn:Q})
                        nUpdates=20000
                    else:
                        nUpdates=1000
                    nMinibatch=256
                    for ptIter in range(nUpdates):
                        batchIndices=np.random.randint(0,nDataPerIter,size=nMinibatch)
                        currLoss,_=sess.run([vfpredLoss,optimizeValue],feed_dict={
                            currObsIn:initialObs[batchIndices],
                            targetObsIn:targetObs[batchIndices],
                            valuesIn:Q[batchIndices]})
                        if iter==0 and (ptIter % 1000==0):
                            print("Iter {}/{}, Value network loss {}".format(ptIter,nUpdates,currLoss))

                    print("Value network loss",currLoss)
                if useValueNetForAdvantages:
                    values=sess.run(valuesOut,feed_dict={currObsIn:initialObs,targetObsIn:targetObs})
                else:
                    #if not using value net, average the sample returns for same initial and target obs
                    values=np.zeros([nDataPerIter,1])
                    for i in range(0,nDataPerIter,nValueEstimationSamples):
                        values[i:i+nValueEstimationSamples]=np.mean(Q[i:i+nValueEstimationSamples])


                #Compute advantages
                advantages=Q-values
                advantageSd=np.std(advantages,axis=0,keepdims=True)
                advantageMean=np.mean(advantages,axis=0,keepdims=True)
                advantages=np.clip(advantages,advantageMean-advantageSd*negativeAdvantageClippingThreshold,np.inf)
                if useAdvantageNormalization:
                    advantages/=advantageSd

                #Update policy network
                print("Training policy network")
                oldActionMean,oldActionVar=sess.run([actionMeanOut_flat,actionOutVar],feed_dict={currObsIn:initialObs,targetObsIn:targetObs})
                oldActionVar=np.repeat(oldActionVar,nDataPerIter,axis=0)
                nUpdates=1000
                for _ in range(nUpdates):
                    batchIndices=np.random.randint(0,nDataPerIter,size=nMinibatch)
                    currLoss,currSd,_=sess.run([policyLoss,actionOutSd,optimizePolicy],feed_dict={
                        currObsIn:initialObs[batchIndices],
                        targetObsIn:targetObs[batchIndices],
                        actionsIn:actions[batchIndices],
                        oldActionMeanIn:oldActionMean[batchIndices],
                        oldActionVarIn:oldActionVar[batchIndices],
                        advantagesIn:advantages[batchIndices]})
                meanQ=np.mean(Q)
                if meanQ > best_mean_q:
                    best_mean_q = meanQ
                print("Iteration %d/%d, policy loss %.2f, policy sd %.2f, mean action value %.2f (best value %.2f)"
                      % (iter,nTrainingIter,currLoss,np.mean(currSd),meanQ,best_mean_q))
                '''
                logger.logkv("Iteration", iter)
                logger.logkv("policy loss", currLoss)
                logger.logkv("policy sd", np.mean(currSd))
                logger.logkv("last mean action value", meanQ)
                logger.logkv("best mean action value", best_mean_q)
                logger.dumpkvs()
                '''
                print("Iteration %d/%d finished" % (iter, nTrainingIter))

                #Take a snapshot if a new best network found
                #Although the mean action value from each iteration has some randomness, 
                #using the best snapshot is probably safer, as there's often some fluctuation in PPO's final convergence
                if meanQ>snapshotValue:
                    snapshotValue=meanQ
                    print("New best policy found, taking snapshot...")
                    snapshot=sess.run(var_list)

            #Finalize
            print("Done, saving network...")

            #First, load the snapshot values back to the TensorFlow graph
            for i in range(len(var_list)):
                var_list[i].load(snapshot[i],sess)

            #Now we can save the graph
            saver.save(sess,modelFileName)


        #need to remember to reset the ignoreDone of the env
        env.setIgnoreDone(oldIgnoreDone)
    def reload(self):
        self.saver.restore(self.sess,self.modelFileName)
        if self.childLLC is not None:
            self.childLLC.reload()
    def clampTargetObs(self,obs,targetObs):
        return targetObs
        mean,sd=self.sess.run([self.futureObsMean,self.futureObsSd],feed_dict={self.currObsIn:obs})
        sd*=2.0
        #return mean+sd*np.tanh((targetObs-mean)/sd)
        return np.clip(targetObs,mean-sd,mean+sd)
    def act(self,obs,targetObs,actionMode,clamp=True):
        if len(obs.shape) == 1:
            obs = np.reshape(obs, (1, -1))
            # targetObs = np.reshape(targetObs, (1, -1))
        if clamp:
            targetObs=self.clampTargetObs(obs,targetObs) 
        if actionMode=="sample":
            actions=self.sess.run(self.actionSamplesOut,feed_dict={self.currObsIn:obs,self.targetObsIn:targetObs,self.tfBatchSize:obs.shape[0]})
        elif actionMode=="expectation":
            actions=self.sess.run(self.actionMeanOut,feed_dict={self.currObsIn:obs,self.targetObsIn:targetObs})
        else:
            raise Exception("Invalid action mode: {}".format(actionMode))
        #Clamp actions to range
        return np.clip(actions,np.reshape(self.actionMin,[1,-1]),np.reshape(self.actionMax,[1,-1]))

    def rewards(self,obs,targetObs):
        if self.singleTargetObs and self.H!=1:
            #If using a single target observation, only the LLC with H=1 actually computes rewards
            return 0
        diff=np.abs(obs-targetObs)
        rootObsWeight=5.0
        diff[self.rootObs]*=rootObsWeight
        diff/=rootObsWeight
        return -np.mean(diff,axis=1,keepdims=True)

    def value(self,obs,env,targetObs,valueSimulationSteps):
        if valueSimulationSteps>0 or (not useValueNetForValue):
            obs,_,_,_=env.step(self.act(obs,targetObs,actionMode="expectation")) 
            value=self.rewards(obs,np.reshape(targetObs[:,0,:],[-1,self.obsDim]))
            valueSimulationSteps-=1
            if self.H>1:
                childTargetObs=targetObs if self.singleTargetObs else targetObs[:,1:,:]
                value+=self.childLLC.value(obs,env,childTargetObs,valueSimulationSteps)
            return value
        else:
            targetObs=self.clampTargetObs(obs,targetObs) 
            return self.sess.run(self.valuesOut,feed_dict={self.currObsIn:obs,self.targetObsIn:targetObs})

    '''
    Returns a batch of action vectors that drive agents towards reference trajectories
    
    Arguments:
    obs:        current agent state observations, shape: [nBatch,obsDim]
    targetObs:  Normalized reference trajectory observations, shape [nBatch,H,obsDim]. 
    targetObsMean, targetObsSd: Optional mean and sd for denormalizing targetObs, shape: [nBatch,H,obsDim]. If not specified, 
                targetObs is normalized using the mean and sd of the training data.
    '''
    def driveTowards(self,obs,targetObs,targetObsMean=None,targetObsSd=None,normalizedTargetObs=True):
        if normalizedTargetObs:
            if targetObsMean is None:
                targetObsMean=self.targetObsMean
            if targetObsSd is None:
                targetObsSd=self.targetObsSd
            targetObs=targetObsMean+targetObs*targetObsSd
        return self.act(obs,targetObs,actionMode="expectation")
