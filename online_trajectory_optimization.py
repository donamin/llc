"""
Online optimization code
"""
import numpy as np
import tensorflow as tf
import os
from EnvWrapper import MultiEnvWrapper
import argparse
import logger
import time


def run_online_optimization(env_name, task, action_mode, h, exploration_mode, singleTargetObs, max_steps):
	model_name = '%s,%s,%s%s%s%s' % (
		env_name
		, "mjc_task" if task == "" else task
		, action_mode
		, ("[%d]" % h) if action_mode == "LLC" else ""
		, ("[%s]" % singleTargetObs) if action_mode == "LLC" and h > 1 else ""
		, ("[%s]" % exploration_mode) if action_mode == "LLC" else ""
	)

	print('Run Title: %s' % model_name)

	T_seconds = 5.0  # optimized trajectory length in seconds
	N = 250
	render = False
	r_use_best_traj = 0.5

	tf.reset_default_graph()
	sess = tf.Session()
	env = MultiEnvWrapper(
		envName=env_name, taskName="" if task == "mjc_task" else task, actionMode=action_mode, llcData=exploration_mode
		, singleTargetObs=singleTargetObs, N=N, sess=sess, H=h, ignoreDone=False, render_online=not render
	)

	T = int(T_seconds / env.dt)
	print("Optimized trajectory length: {} seconds, {} steps".format(T_seconds, T))
	action_dim = env.action_space.low.shape[0]
	action_min = env.action_space.low
	action_max = env.action_space.high
	action_mean = 0.5 * (action_min + action_max)
	action_sd = 1.0 * (action_max - action_min)

	# init logging
	base_dir = os.path.join(os.getcwd(), 'ResultsOnlineTrajOpt')
	logDir = os.path.join(base_dir, model_name)
	print('Starting optimization run, output logged to', logDir)
	if not os.path.exists(logDir):
		os.makedirs(logDir)
	logger.configure(dir=logDir)

	last_best_traj = []
	startupTime = time.perf_counter()

	n_vid = 0
	ep_len = 0

	for step in range(max_steps):
		# Init optimization
		master_state = env.envs[0].getState()
		threads_actions = []
		threads_parents = -np.ones((N, T))
		threads_rewards = np.zeros(N)
		step_limit = T
		# Generate and simulate random walks
		for t in range(T):
			# Generate actions
			actions = np.zeros((N, action_dim))
			for n in range(N):
				if (t + 1) < len(last_best_traj) and n <= r_use_best_traj * N:
					# Sample using the last best trajectory, shifted by one timestep
					mean = last_best_traj[t + 1]
					# Add exploration noise proportional to trajectory index
					sd = n / (N - 1) / r_use_best_traj * action_sd
					actions[n, :] = np.random.normal(mean, sd)
				else:
					# Sample randomly
					if np.random.uniform(0, 1) < 0.5:
						actions[n, :] = np.random.normal(action_mean, action_sd)
					else:
						actions[n, :] = np.random.uniform(action_min, action_max)
			threads_actions.append(actions)

			# Simulate all threads
			if t == 0:
				env.setStateForAll(master_state)
			obs, rewards, dones, info = env.step(actions)
			threads_rewards += rewards

			if t + 1 < T:
				# Fork terminated trajectories
				fork_pool = np.argwhere(np.array(dones) == 0)[:, 0].tolist()
				if len(fork_pool) == 0:
					print('All trajectories were terminated at t = %d' % t)
					step_limit = t + 1
					break
				for n in range(N):
					if dones[n]:
						env.dones[n] = False
						idx = np.argmax(threads_rewards)
						if dones[idx]:
							idx = np.random.choice(fork_pool)
						threads_parents[n, t + 1] = idx
						threads_rewards[n] = threads_rewards[idx]
						env.envs[n].setState(env.envs[idx].getState())
					else:
						threads_parents[n, t + 1] = n

		# Simulate the master context using best trajectory
		idx = np.argmax(threads_rewards)
		last_best_traj = []
		for t in range(step_limit, 0, -1):
			last_best_traj = [threads_actions[t - 1][idx, :]] + last_best_traj
			idx = int(threads_parents[idx, t - 1])
		assert idx == -1

		env.reset()
		env.envs[0].setState(master_state)
		ob, reward, done, info = env.step_single_env(last_best_traj[0], 0, render)
		ep_len += 1
		if done or (render and ep_len == 100):
			last_best_traj.clear()
			env.reset()
			ep_len = 0
			if render:
				# Save the video
				env.save_video("online_traj_opt-%s_%d (%.2f)" % (model_name, n_vid, np.max(threads_rewards)))
				n_vid += 1
				if n_vid == 1:
					break

		# Logs bookkeeping
		logger.record_tabular("Timestep", step)
		logger.record_tabular("Done", int(done))
		logger.record_tabular("Reward", reward)
		logger.record_tabular("Best return", np.max(threads_rewards))
		logger.record_tabular("Mean return", np.mean(threads_rewards))
		logger.record_tabular("Time", (time.perf_counter() - startupTime))
		logger.dump_tabular()

	print('All done.')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default="Hopper-v2")  # ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Humanoid-v2"]
	parser.add_argument('--task', type=str, default="")  # ["", "balance", "walk", "run", "standup", "walkbwd"]
	parser.add_argument('--action_mode', type=str, default="LLC")  # ["default", "LLC"]
	parser.add_argument('--h', type=int, default=1)  # [1, 2, 3, 4, 5]
	parser.add_argument('--exploration_mode', type=str, default="ContactExplorer2")  # ["NaiveExplorer", "ContactExplorer2"]
	parser.add_argument('--singleTargetObs', default=False, action='store_true')  # [False, True]
	parser.add_argument('--max_steps', type=int, default=100)

	args = parser.parse_args()

	run_online_optimization(env_name=args.env_name, task=args.task, action_mode=args.action_mode, h=args.h
							, exploration_mode=args.exploration_mode, singleTargetObs=args.singleTargetObs
							, max_steps=args.max_steps)