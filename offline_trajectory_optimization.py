"""
CMA-ES offline trajectory optimization
"""
import sys
import numpy as np
import tensorflow as tf
import time
import os
import cma
import argparse
import logger
from EnvWrapper import MultiEnvWrapper
np.set_printoptions(threshold=sys.maxsize)  # Tell numpy not to summarize big arrays!

nIter=200
T_seconds=4.0           #optimized trajectory length in seconds
N=32
renderInterval=-10       #set to negative for no rendering
keyFrameSpacing=0.1999  #seconds
nKeyFrames=int(T_seconds/keyFrameSpacing)+1 #3 keyframes per second, +1 because the last keyframe only defines tangent and is not reached by the agent
noTermination=False


def TrajectoryOptimization(env_name, task, action_mode, h, exploration_mode, singleTargetObs):
	global N
	global T_seconds
	global nKeyFrames
	render = False
	if render:
		N = 100
		T_seconds = 10
		nKeyFrames = int(T_seconds / keyFrameSpacing) + 1
	if task == "standup":
		T_seconds = 5
	model_name = '%s,%s,%s%s%s%s' % (
		env_name
		, task
		, action_mode
		, ("[%d]" % h) if action_mode == "LLC" else ""
		, ("[%s]" % singleTargetObs) if action_mode == "LLC" and h > 1 else ""
		, ("[%s]" % exploration_mode) if action_mode == "LLC" else ""
	)

	Heff = 1 if singleTargetObs else h  # helper
	# Init tensorflow
	tf.reset_default_graph()
	sess = tf.Session()

	#Setup environment and query some of it's properties
	env = MultiEnvWrapper(envName=env_name, taskName="" if task == "mjc_task" else task, actionMode=action_mode
						  , singleTargetObs=singleTargetObs, llcData=exploration_mode, N=N, sess=sess, H=h
						  , ignoreDone=noTermination, render_online=not render)
	T = int(T_seconds / env.dt)
	print("Optimized trajectory length: {} seconds, {} steps".format(T_seconds,T))
	actionDim=env.action_space.low.shape[0] if action_mode=="default" else env.action_space.low.shape[0]//Heff
	actionMin=env.action_space.low
	actionMax=env.action_space.high
	actionMean=0.5*(actionMin+actionMax)
	actionMean=np.reshape(actionMean,[1,-1])
	actionSd=0.5*(actionMax-actionMin)
	actionSd=np.reshape(actionSd,[1,-1])

	#Init optimization
	totalActions = 0
	cmaes_options = {'popsize': N,'CMA_diagonal': True}
	optDim = T * actionDim
	optimizer = cma.CMAEvolutionStrategy(np.zeros(optDim), 1, inopts=cmaes_options)

	#init logging
	base_dir = os.path.join(os.getcwd(), 'ResultsOfflineTrajOpt')
	suffix = model_name
	logDir = os.path.join(base_dir, suffix)
	if not os.path.exists(logDir):
		os.makedirs(logDir)
	print('Starting optimization run, output logged to',logDir)
	logger.configure(dir=logDir)

	#timing
	startupTime=time.perf_counter()

	#Optimization loop
	for iter in range(nIter):
		#Get the actions for all trajectories
		xBatch = optimizer.ask()

		#Trajectory batch simulation as a function, as we will also use the same code later for rendering
		#the best trajectory
		def evaluateBatch(xBatch, render):
			fVals=np.zeros(N)
			totalBatchActions=0

			#reset simulators, get initial observations
			obs = env.reset(mode="fixed")      #need to use fixed initial state for trajectory optimization

			unpackedBatch=np.zeros([N,T*actionDim])
			for n in range(N):
				unpackedBatch[n]=xBatch[n]

			#for each decision
			cmaesActions=np.zeros([N,actionDim if action_mode=="default" else actionDim*Heff])
			for t in range(T):
				#extract actions from the CMA-ES batch
				if action_mode=="default":
					cmaesActions[:]=unpackedBatch[:,t*actionDim:(t+1)*actionDim]
				else:
					for h in range(Heff):
						tClamped=min([t+h,T-1])
						cmaesActions[:,h*actionDim:(h+1)*actionDim]=unpackedBatch[:,tClamped*actionDim:(tClamped+1)*actionDim]

				#scale and shift
				actions = actionMean + 1 * cmaesActions * actionSd

				#Simulate and render
				obs, rewards, dones, info = env.step(actions, render)
				nSimulated = info['nSimulated']
				totalBatchActions+=nSimulated
				fVals -= rewards
			return fVals, totalBatchActions

		#Simulate and evaluate all trajectories
		fVals, batchActions = evaluateBatch(xBatch, render=False)
		totalActions += batchActions
		meanReturn =- np.mean(fVals)

		#All trajectories simulated
		optimizer.tell(xBatch,fVals)
		logger.record_tabular("Iteration", iter)
		logger.record_tabular("Total experience",totalActions)
		logger.record_tabular("Best trajectory return",-optimizer.result.fbest)
		logger.record_tabular("Mean return",meanReturn)
		logger.record_tabular("Time since startup", (time.perf_counter()-startupTime))
		logger.dump_tabular()

		if renderInterval>0 and ((iter % renderInterval==0) or (iter==nIter-1)):
			renderBatch=[]
			for _ in range(N):
				renderBatch.append(optimizer.result.xbest)
			evaluateBatch(renderBatch,render=True)

	data = {"fbest": optimizer.result.fbest, "xbest": optimizer.result.xbest}
	np.save(os.path.join(base_dir, suffix, 'output.npy'), data)

	if render:
		for b in range(len(xBatch)):
			xBatch[b] = optimizer.result.xbest
		fVals, batchActions = evaluateBatch(xBatch, render=True)
		# Save the video
		env.save_video("offline_traj_opt-%s (%.2f)" % (model_name, -fVals[0]))
	print('All done.')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default="Hopper-v2")  # ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Humanoid-v2"]
	parser.add_argument('--task', type=str, default="")  # ["", "balance", "walk", "run", "standup", "walkbwd"]
	parser.add_argument('--action_mode', type=str, default="LLC")  # ["default", "LLC"]
	parser.add_argument('--h', type=int, default=1)  # [1, 2, 3, 4, 5]
	parser.add_argument('--exploration_mode', type=str, default="ContactExplorer2")  # ["NaiveExplorer", "ContactExplorer2"]
	parser.add_argument('--singleTargetObs', default=False, action='store_true')  # [False, True]

	args = parser.parse_args()

	TrajectoryOptimization(env_name=args.env_name, task=args.task, action_mode=args.action_mode, h=args.h
						   , exploration_mode=args.exploration_mode, singleTargetObs=args.singleTargetObs)
