import argparse
import json
import os
import pickle
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
import time
from train import TrainingOptions, TrainingState, train
from typing import List, Optional


def init_training(dir_out):
	strategy = tf.distribute.MirroredStrategy()

	with open(f'{dir_out}/options.json', 'r') as f:
		opt = json.load(f)

	options = TrainingOptions(
		opt['dataset_file_pattern'],
		opt['resolution'],
		opt['replica_batch_size'],
		opt['epoch_sample_count'],
		opt['total_sample_count'],
		opt['learning_rate'],
		opt['latent_size'],
		opt['beta_1'],
		opt['beta_2'])

	training_state = TrainingState(options)

	train(
		strategy,
		dir_out,
		training_state)


def resume_training(dir_out, checkpoint_i):
	strategy = tf.distribute.MirroredStrategy()

	with strategy.scope():
		filepath = f'{dir_out}/{checkpoint_i}.checkpoint'
		with open(filepath, 'rb') as f:
			state = pickle.loads(f.read())
		training_state = TrainingState(
			state.options,
			state.epoch_i,
			state.generator,
			state.discriminator)

	train(
		strategy,
		dir_out,
		training_state)


def main(dir_out):
	files = os.listdir(dir_out)
	checkpoints = [f for f in files if f.endswith('checkpoint')]

	if not checkpoints:
		init_training(dir_out)
	else:
		checkpoint_i = max([int(c.split('.')[0]) for c in checkpoints])
		resume_training(dir_out, checkpoint_i)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'dir_out',
		type=str,
		help='folder to save checkpoints. must contain options.json')
	args = parser.parse_args()
	main(args.dir_out)















