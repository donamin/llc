"""
This script is used for training the LLC networks.
"""
import tensorflow as tf
from EnvWrapper import MultiEnvWrapper
import argparse

def produce_llc(env_name, h, exploration_mode, singleTargetObs, **kwargs):
	# Init tensorflow
	tf.reset_default_graph()
	sess = tf.Session()
	# Setup environment and query some of it's properties
	print('Environment: %s' % env_name)
	env = MultiEnvWrapper(envName=env_name, taskName="", actionMode="LLC", singleTargetObs=singleTargetObs
						  , llcData=exploration_mode, N=1, sess=sess, H=h, ignoreDone=True, **kwargs)
	del env

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default="Hopper-v2")  # ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Humanoid-v2"]
	parser.add_argument('--h', type=int, default=1)  # [1, 2, 3, 4, 5]
	parser.add_argument('--exploration_mode', type=str, default="ContactExplorer2")  # ["NaiveExplorer", "ContactExplorer2"]
	parser.add_argument('--singleTargetObs', default=False, action='store_true')  # [False, True]

	args = parser.parse_args()

	produce_llc(**vars(args))
