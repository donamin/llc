import tensorflow as tf
import os
from EnvWrapper import MultiEnvWrapper
from stable_baselines import PPO2
from stable_baselines import SAC
import argparse


def render(alg, env_name, task, action_mode, exploration_mode, singleTargetObs):
	h = 1 if alg == "PPO2" else 1

	model_name = '%s,%s,%s%s%s%s' % (
		env_name
		, "mjc_task" if task == "" else task
		, action_mode
		, ("[%d]" % h) if action_mode == "LLC" else ""
		, ("[%s]" % singleTargetObs) if action_mode == "LLC" and h > 1 else ""
		, ("[%s]" % exploration_mode) if action_mode == "LLC" else ""
	)

	dir_name = "NA"
	if alg == "PPO2":
		dir_name = "ResultsPPO"
	elif alg == "SAC":
		dir_name = "ResultsSAC"
	directory = os.path.join(dir_name, model_name)

	print('Run Title: %s' % model_name)

	tf.reset_default_graph()
	sess = tf.Session()
	env = MultiEnvWrapper(
		envName=env_name, taskName="" if task == "mjc_task" else task, actionMode=action_mode, llcData=exploration_mode
		, singleTargetObs=singleTargetObs, N=1, sess=sess, H=h, ignoreDone=True, render_online=False
	)

	model = PPO2.load(os.path.join(directory, "model")) if alg == "PPO2" else SAC.load(os.path.join(directory, "model"))

	# Enjoy trained agent
	ob = env.reset()
	eps = 0
	ep_len = 0
	r = 0
	while True:
		action, _states = model.predict(ob)
		ob, reward, done, info = env.step(action, render=True)
		r += reward
		ep_len += 1
		if ep_len == 100 or done:
			env.save_video(name="rl-%s-%s-v%d (%.2f)" % (alg, model_name, eps, r))
			ob = env.reset()
			r = 0
			ep_len = 0
			eps += 1
			if eps == 5:
				break
	env.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default="Hopper-v2")  # ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Humanoid-v2"]
	parser.add_argument('--task', type=str, default="")  # ["", "balance", "walk", "run", "standup", "walkbwd"]
	parser.add_argument('--action_mode', type=str, default="LLC")  # ["default", "LLC"]
	parser.add_argument('--h', type=int, default=1)  # [1, 2, 3, 4, 5]
	parser.add_argument('--exploration_mode', type=str, default="ContactExplorer2")  # ["NaiveExplorer", "ContactExplorer2"]
	parser.add_argument('--singleTargetObs', default=False, action='store_true')  # [False, True]
	parser.add_argument('--alg', type=str, default="PPO2")  # ["PPO2", "SAC"]
	args = parser.parse_args()

	render(alg=args.alg, env_name=args.env_name, task=args.task, action_mode=args.action_mode
		, exploration_mode=args.exploration_mode, singleTargetObs=args.singleTargetObs)
