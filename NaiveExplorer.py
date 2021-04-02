"""
Baseline naive explorer that runs episodes of length T with random actions
"""
import numpy as np
import os
import sys
from EnvWrapper import MultiEnvWrapper
import argparse


def generate_data(env_name):
	if not os.path.exists('ExplorationData'):
		os.makedirs('ExplorationData')
	#Algorithm parameters
	T=100 #rollout length

	#Other parameters
	capacity = 100_000
	render = False
	render_online = False

	if render:
		capacity = 1000

	#Create environment and query some of its properties and define helpers
	env = MultiEnvWrapper(env_name, 1, ignoreDone=True, render_online=render_online)
	print(env_name)

	obsDim=env.observe().shape[0]
	actionDim=env.action_space.low.shape[0]

	#Data buffer for all observations, actions, and states
	observations=np.zeros([capacity,obsDim])
	nextObservations=np.zeros([capacity,obsDim])
	actions=np.zeros([capacity,actionDim])
	parentIndices=np.zeros([capacity],dtype=int)    #for maintaining tree linkage, needed for inverse dynamics training

	#Phase 2: Explore contact and actuation dynamics
	N=0
	scriptPath=os.path.dirname(os.path.realpath(sys.argv[0]))
	while N<capacity:
		#initialize character with random mid-air initial state
		env.reset()

		#sample random actions and simulate for H steps
		obs=env.observe()
		parentIndex=-1
		for stepIdx in range(T):
			action=env.action_space.sample()
			env.step(action,render)
			nextObs=env.observe()
			observations[N,:]=obs
			nextObservations[N,:]=nextObs
			obs=nextObs
			actions[N,:]=action
			parentIndices[N]=parentIndex
			parentIndex=N
			N+=1
			if N>=capacity:
				break

	# Save all
	print("Collected {} [obs,action,nextObs] tuples, saving...".format(N))
	if not render:
		data={"observations":observations,"nextObservations":nextObservations,"actions":actions,"parentIndices":parentIndices,"N":N}
		np.save("{}/ExplorationData/{}_NaiveExplorer.npy".format(scriptPath,env_name),data)
	if render and not render_online:
		env.save_video("{}_NaiveExplorer".format(env_name))
	print("Done.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default="Hopper-v2")  # ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Humanoid-v2"]

	args = parser.parse_args()

	generate_data(**vars(args))
