'''

Fully connected network, with optional dense (canonical correlation network -style) skip-connections.

Implements a modified version of the Mishkin & Matas LSUV initialization (https://arxiv.org/pdf/1511.06422.pdf), 
which ensures unit-variance outputs independent of activation and skip-connections.

Modifications:
- approximation: when orthonormal basis not possible (more units than inputs), each weight vector is independently sampled and normalized
- addition: also set the biases such that E[w'x+b]=0, i.e., zero-mean output before nonlinearity. Thus, b=-w'E[x], where E[x] \approx mean(x) over the initialization batch.
  For zero-mean input, b=0, but input is not zero-mean if and when the previous layer has a nonlinearity. 
  If all x are from same distribution (e.g., previous layer with standard normal input, relu activation), E[b]=-E[w]'E[x]=-kE[w],
  where k is the expectation of a single x variable. Thus, in expectation, b is still 0 if E[w]=0. However, in practice, the sum of elements
  of w is not zero. TODO: test He initialization with weight vectors normalized as w'=sqrt(2)*normalize(w-mean(w)), i.e., ensuring that the weight sampling mean and variance are correct in the realized weights.

The bias initialization improves function approximation, allowing more input data splits in the deep layer activations.

To run the initialization, use sess.run() to query the initialization operation returned by the network constructors.

'''

import numpy as np
import matplotlib.pyplot as pp
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #disable Tensorflow GPU usage, these simple graphs run faster on CPU
import tensorflow as tf
from scipy.stats import ortho_group


LSUV_iter=3             #number of LSUV iterations
LSUV_init_biases=True #set to false to disable the data-dependent bias initialization
LSUV_init_weights=True  #set to false to disable the data-dependent weight initialization
class Layer:
    def __init__(self,input:tf.Tensor,initInput:tf.Tensor,nUnits:int,useSkips=True,activation=None,copyWeightsFrom=None,addNormalizer=False):
        xDim=input.shape[1].value
        #Optional input normalizer, usually used for the first layer. This is initialized based on the initialization batch, see later
        if addNormalizer:
            self.shift=tf.Variable(initial_value=np.zeros([1,xDim]),dtype=tf.float32,name='W',trainable=False)
            self.scale=tf.Variable(initial_value=np.ones([1,xDim]),dtype=tf.float32,name='W',trainable=False)
            input=self.scale*(input+self.shift)
            self.useNormalizer=True
        else:
            self.shift=None
            self.scale=None
            self.useNormalizer=False
        
        if copyWeightsFrom is None:
            #The first step of LSUV initialization is to initialize weights as an orthonormal matrix
            #This is possible, however, only if the number of units is equal or less than the number of inputs.
            #Otherwise, we just select random vectors and normalize their lengths. 
            if nUnits<=xDim:
                ortho=ortho_group.rvs(xDim)
                initialW=ortho[:nUnits,:]
            else:
                initialW=np.random.normal(0,1,size=[nUnits,xDim])
                initialW=initialW/np.linalg.norm(initialW,axis=1,keepdims=True)
            self.W=tf.Variable(initial_value=initialW,dtype=tf.float32,name='W')
            self.b=tf.Variable(initial_value=np.zeros([1,nUnits]),dtype=tf.float32,name='b')
        else:
            self.W=copyWeightsFrom.W
            self.b=copyWeightsFrom.b
            self.shift=copyWeightsFrom.shift
            self.scale=copyWeightsFrom.scale
            if self.shift is not None:
                input=self.scale*(input+self.shift)

        h=self.computeOutput(input,self.W,self.b,activation)
        if useSkips:
            self.output=tf.concat([input,h],axis=1)     #Note: input is now normalized, if normalizer used
        else:
            self.output=h

        #We skip the data-dependent initialization below, if copying weights from another network instance
        if copyWeightsFrom is not None:
            self.initOutput=None
            return

        #Data-dependent initialization of normalizer
        if addNormalizer:
            shift=tf.assign(self.shift,-tf.reduce_mean(initInput,axis=0,keepdims=True))
            scale=tf.assign(self.scale,1.0/tf.linalg.norm(initInput+shift,axis=0,keepdims=True))
            initInput=scale*(initInput+shift)

        #LSUV initialization
        initW=self.W
        initInputMean=tf.reduce_mean(initInput,axis=0,keepdims=True)
        for iter in range(LSUV_iter):
            #Data-dependent initialization of bias, such that the output is zero-mean before the nonlinearity
            if LSUV_init_biases:
                b=tf.assign(self.b,-tf.matmul(initInputMean,tf.transpose(initW)))
            else:
                b=self.b

            #Data-dependent initialization of weights
            if LSUV_init_weights:
                h=self.computeOutput(initInput,initW,b,activation)

                #Compute output mean and sd, divide weights by sd
                hMean=tf.reduce_mean(h,axis=0,keepdims=True)
                hSd=tf.sqrt(tf.reduce_mean(tf.square(h-hMean),axis=0,keepdims=True))
                hSd=tf.maximum(hSd,1e-4)
                initW=tf.assign(self.W,initW/tf.reshape(hSd,[nUnits,1]))

        #Compute output again
        h=self.computeOutput(initInput,initW,b,activation)

        if useSkips:
            self.initOutput=tf.concat([initInput,h],axis=1)  #Note: initInput is now normalized, if normalizer used
        else:
            self.initOutput=h

    def computeOutput(self,X,W,b,activation):
        h=tf.matmul(X,tf.transpose(W))+b
        if activation=="relu":
            h=tf.nn.relu(h)
        elif activation=="selu":
            h=tf.nn.selu(h)
        elif activation=="sigmoid":
            h=tf.nn.sigmoid(h)
        elif activation=="lrelu":
            h=tf.nn.leaky_relu(h,alpha=0.1)
        elif activation=="tanh":
            h=tf.nn.tanh(h)
        elif activation=="swish":
            h=tf.nn.swish(h)
        elif activation is not None:
            raise NameError("Invalid activation type ({}) for Layer".format(activation))
        return h

        
class MLP:
    def __init__(self,input:tf.Tensor,nLayers:int,nUnitsPerLayer:int, nOutputUnits:int, activation="lrelu", o_activation=None, firstLinearLayerUnits:int=0, useSkips:bool=False,copyWeightsFrom=None):
        self.layers=[]
        self.nLayers=nLayers
        self.nUnitsPerLayer=nUnitsPerLayer
        self.nOutputUnits=nOutputUnits
        self.activation=activation
        self.firstLinearLayerUnits=firstLinearLayerUnits
        self.useSkips=useSkips
        X=input
        initX=input

        #add optional first linear layer (useful, e.g., for reducing the dimensionality of high-dimensional outputs
        #which reduces the parameter count of all subsequent layers if using dense skip-connections)
        baseIdx=0
        if firstLinearLayerUnits!=0:
            layer=Layer(X,initX,firstLinearLayerUnits,useSkips=False,activation=None,addNormalizer=True,copyWeightsFrom=None if copyWeightsFrom is None else copyWeightsFrom.layers[0])
            self.layers.append(layer)
            X,initX=layer.output,layer.initOutput
            baseIdx=1

        #add hidden layers
        for layerIdx in range(nLayers):
            addNormalizer=layerIdx==0 and firstLinearLayerUnits==0 #input normalizer only needed for first layer, if not added above
            layer=Layer(X,initX,nUnitsPerLayer,useSkips=useSkips,activation=activation,addNormalizer=addNormalizer,copyWeightsFrom=None if copyWeightsFrom is None else copyWeightsFrom.layers[layerIdx+baseIdx])
            self.layers.append(layer)
            X,initX=layer.output,layer.initOutput

        #add output layer
        layer=Layer(X,initX,nOutputUnits,useSkips=False,activation=o_activation
                    ,copyWeightsFrom=None if copyWeightsFrom is None else copyWeightsFrom.layers[nLayers+baseIdx])
        self.layers.append(layer)
        self.output,self.initOutput=layer.output,layer.initOutput



    #This method returns a list of assign ops that can be used with a sess.run() call to copy
    #all network parameters from a source network. This is useful, e.g., for implementing a slowly updated
    #target network in Reinforcement Learning
    def copyFromOps(self,src):
        result=[]
        for layerIdx in range(len(self.layers)):
            result.append(tf.assign(self.layers[layerIdx].W,src.layers[layerIdx].W))
            result.append(tf.assign(self.layers[layerIdx].b,src.layers[layerIdx].b))
            if self.layers[layerIdx].useNormalizer:
                result.append(tf.assign(self.layers[layerIdx].scale,src.layers[layerIdx].scale))
                result.append(tf.assign(self.layers[layerIdx].shift,src.layers[layerIdx].shift))
        return result
    def getAllVariables(self):
        result=[]
        for layerIdx in range(len(self.layers)):
            result.append(self.layers[layerIdx].W)
            result.append(self.layers[layerIdx].b)
            if self.layers[layerIdx].useNormalizer:
                result.append(self.layers[layerIdx].scale)
                result.append(self.layers[layerIdx].shift)
        return result

    def loadAllVariables(self,variableValues,sess):
        vars=self.getAllVariables()
        assert(len(vars)==len(variableValues))
        for i in range(len(vars)):
            vars[i].load(variableValues[i],sess)

#functional interface
def mlp(input:tf.Tensor,nLayers:int,nUnitsPerLayer:int, nOutputUnits:int, activation="relu", firstLinearLayerUnits:int=0,useSkips:bool=True,copyWeightsFrom:MLP=None):
    instance=MLP(input,nLayers,nUnitsPerLayer,nOutputUnits,activation,firstLinearLayerUnits,useSkips,copyWeightsFrom)
    return instance.output,instance.initOutput

def mlpCopy(input:tf.Tensor,copyFrom:MLP):
    instance=MLP(input,copyFrom.nLayers,copyFrom.nUnitsPerLayer,copyFrom.nOutputUnits,copyFrom.activation,copyFrom.firstLinearLayerUnits,copyFrom.useSkips)
    return instance.output


#simple test: 
if __name__ == "__main__":
    print("Generating toy data")
    x=[]
    y=[]
    maxAngle=5*np.pi
    discontinuousTest=True
    if discontinuousTest:
        maxAngle=np.pi
        for angle in np.arange(0,maxAngle,0.01):
            x.append(angle)
            if angle>maxAngle*0.8:
                y.append(0.0)
            else:
                y.append(np.sin(angle)*np.sign(np.sin(angle*10)))
    else:
        for angle in np.arange(0,maxAngle,0.1):
            r=angle*0.15
            x.append(angle)
            if angle>maxAngle*0.8:
                y.append(0.0)
            else:
                y.append(r*np.sin(angle))

    x=np.array(x)
    y=np.array(y)
    x=np.reshape(x,[x.shape[0],1])
    y=np.reshape(y,[y.shape[0],1])
    interpRange=0.2
    xtest=np.arange(-interpRange+np.min(x),np.max(x)+interpRange,0.001)
    xtest=np.reshape(xtest,[xtest.shape[0],1])
    
    print("Initializing matplotlib")
    pp.figure(1)
    pp.subplot(3,1,1)
    pp.scatter(x[:,0],y[:,0])
    pp.ylabel("data")

    print("Creating model")
    sess=tf.InteractiveSession()
    tfX=tf.placeholder(dtype=tf.float32,shape=[None,1])
    tfY=tf.placeholder(dtype=tf.float32,shape=[None,1])
    #IMPORTANT: deep networks benefit immensely from data-dependent initialization.
    #This is why the constructor returns the initial predictions separately - to initialize, fetch this tensor in a sess.run with 
    #the first minibatch. See the sess.run below
    predictions,initialPredictions=mlp(input=tfX,nLayers=8,nUnitsPerLayer=16,nOutputUnits=1,activation="relu",useSkips=False)
    optimizer=tf.train.AdamOptimizer()
    loss=tf.losses.mean_squared_error(tfY,predictions)
    optimize=optimizer.minimize(loss)
  
    print("Initializing model")
    tf.global_variables_initializer().run(session=sess)
    #This sess.run() initializes the network biases based on x, and also returns the initial predictions.
    #It is noteworthy that with this initialization, even a deep network has zero-mean output with variance similar to input.
    networkOut=sess.run(initialPredictions,feed_dict={tfX:x})
    pp.subplot(3,1,2)
    pp.scatter(x[:,0],networkOut[:,0])
    pp.ylabel("initialization")
    pp.draw()
    pp.pause(0.001)


    print("Optimizing")
    nIter=16000
    for iter in range(nIter):
        temp,currLoss=sess.run([optimize,loss],feed_dict={tfX:x,tfY:y})
        if iter % 100 == 0:
            print("Iter {}/{}, loss {}".format(iter,nIter,currLoss))
    networkOut=sess.run(predictions,feed_dict={tfX:x})
    pp.subplot(3,1,3)
    pp.scatter(x[:,0],networkOut[:,0])
    pp.ylabel("trained")

    pp.show()