import tensorflow as tf
import os
from EnvWrapper import MultiEnvWrapper
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.logger import configure
from stable_baselines.sac.policies import MlpPolicy as SacMlpPolicy
from stable_baselines import SAC
import argparse


def rl(alg, env_name, task, action_mode, h, exploration_mode, singleTargetObs, max_steps):
	model_name = '%s,%s,%s%s%s%s' % (
		env_name
		, "mjc_task" if task == "" else task
		, action_mode
		, ("[%d]" % h) if action_mode == "LLC" else ""
		, ("[%s]" % singleTargetObs) if action_mode == "LLC" and h > 1 else ""
		, ("[%s]" % exploration_mode) if action_mode == "LLC" else ""
	)

	iter_budget = 2000 if "Humanoid" in env_name else 500

	dir_name = "NA"
	if alg == "PPO2":
		dir_name = "ResultsPPO"
	elif alg == "SAC":
		dir_name = "ResultsSAC"
	directory = os.path.join(dir_name, model_name)

	print('Run Title: %s' % model_name)

	configure(directory, ['stdout', 'csv'])

	tf.reset_default_graph()
	sess = tf.Session()
	env = MultiEnvWrapper(
		envName=env_name, taskName="" if task == "mjc_task" else task, actionMode=action_mode, llcData=exploration_mode
		, singleTargetObs=singleTargetObs, N=1, sess=sess, H=h, ignoreDone=False
	)

	model = None
	if alg == "PPO2":
		model = PPO2(MlpPolicy, env, verbose=1, n_steps=iter_budget, learning_rate=1e-4, nminibatches=20, noptepochs=25, cliprange=0.2)
	elif alg == "SAC":
		model = SAC(SacMlpPolicy, env, verbose=1, buffer_size=50000, learning_starts=100, train_freq=1, batch_size=64
					, tau=0.005, ent_coef='auto', target_update_interval=1, gradient_steps=1, target_entropy='auto'
					, action_noise=None, random_exploration=0.0, seed=None)
	assert model is not None
	model.learn(total_timesteps=max_steps)

	if not os.path.exists(directory):
		os.makedirs(directory)
	model.save(os.path.join(directory, "model"))
	env.close()
	print('Finished training for %s' % model_name)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default="Hopper-v2")  # ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Humanoid-v2"]
	parser.add_argument('--task', type=str, default="")  # ["", "balance", "walk", "run", "standup", "walkbwd"]
	parser.add_argument('--action_mode', type=str, default="LLC")  # ["default", "LLC"]
	parser.add_argument('--h', type=int, default=1)  # [1, 2, 3, 4, 5]
	parser.add_argument('--exploration_mode', type=str, default="ContactExplorer2")  # ["NaiveExplorer", "ContactExplorer2"]
	parser.add_argument('--singleTargetObs', default=False, action='store_true')  # [False, True]
	parser.add_argument('--alg', type=str, default="PPO2")  # ["PPO2", "SAC"]
	parser.add_argument('--max_steps', type=int, default=int(1e6))
	args = parser.parse_args()

	rl(alg=args.alg, env_name=args.env_name, task=args.task, action_mode=args.action_mode, h=args.h
	   , exploration_mode=args.exploration_mode, singleTargetObs=args.singleTargetObs, max_steps=args.max_steps)
