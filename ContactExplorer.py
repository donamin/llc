import numpy as np
import os
import sys
from EnvWrapper import MultiEnvWrapper
import argparse

def generate_data(env_name):
	if not os.path.exists('ExplorationData'):
		os.makedirs('ExplorationData')
	#Algorithm parameters
	y_max=1.0
	H=5    #contact exploration rollout length
	p_free=0.1
	p_nearContact=0.4
	h_nearContact=0.05
	p_newAction=1.0

	#Other parameters
	capacity = 100_000
	render = False
	render_online = False

	if render:
		capacity = 1000

	#Create environment and query some of its properties. For now, we only use a single env, but still use the MultiEnvWrapper
	#class to support the upcoming UnityEnvWrapper with minimal changes.
	#Note: to allow diverse state exploration, we must ignore the terminal states signaled by Done=True by the wrapped environments.
	env=MultiEnvWrapper(env_name,1,ignoreDone=True, render_online=render_online)
	print("Using environment: ",env_name,"with time step",env.dt)
	obsDim=env.observation_space.shape[0]
	actionDim=env.action_space.shape[0]

	#Data buffer for all observations, actions, and states
	observations=np.zeros([capacity,obsDim])
	nextObservations=np.zeros([capacity,obsDim])
	actions=np.zeros([capacity,actionDim])
	parentIndices=np.zeros([capacity],dtype=int)    #for maintaining tree linkage, needed for inverse dynamics training

	N=0
	scriptPath=os.path.dirname(os.path.realpath(sys.argv[0]))

	#Explore contact and actuation dynamics
	print("Starting the contact exploration phase")
	while N<capacity:
		#initialize the character in random movement state
		r=np.random.uniform()
		if r<p_free:
			target_start_height=np.random.uniform()*y_max
		elif r<p_free+p_nearContact:
			target_start_height=np.random.uniform()*h_nearContact
		else:
			target_start_height=0


		#V2 exploration: everything is random, but we bias
		#initial root and joint rotations towards the default state, and root and joint angular velocities towards static movements.
		#This rougly represents how we spend most of our lives upright, and in relatively slow movement.
		#Even in practicing unusual poses (climbing, handstands), we often practice at slow movement speeds
		start_rot_scale=np.random.uniform(0,1)
		start_vel_scale=np.random.uniform(0,1)

		obs=env.reset(mode="random",
					  target_start_height=target_start_height,
					  start_vel_scale=start_vel_scale,
					  start_rot_scale=start_rot_scale)

		#sample random actions and simulate for H steps
		parentIndex=-1
		for stepIdx in range(H):
			observations[N,:]=obs
			if stepIdx==0 or np.random.uniform()<=p_newAction:
				action=env.action_space.sample()
			actions[N,:]=action
			parentIndices[N]=parentIndex
			obs,_,_,_=env.step(action,render)
			nextObservations[N,:]=obs
			parentIndex=N
			N+=1
			if N>=capacity:
				break

	#Save all
	print("Collected {} [obs,action,nextObs] tuples, saving...".format(N))
	data={"observations":observations,"nextObservations":nextObservations,"actions":actions,"parentIndices":parentIndices,"N":N}
	if not render:
		np.save("{}/ExplorationData/{}_ContactExplorer2.npy".format(scriptPath,env_name),data)
	elif render and not render_online:
		env.save_video("{}_ContactExplorer2".format(env_name))
	print("Done.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default="Hopper-v2")  # ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Humanoid-v2"]
	args = parser.parse_args()
	generate_data(**vars(args))
