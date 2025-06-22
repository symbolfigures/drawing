import argparse
from generate import generate, load_generator, get_dir_out_path
import ffmpeg
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random as r
import tensorflow as tf
import time


def random(generator, dir_out, count):
	# set of random images, anti-path
	noise_shape = generator.input_shape[-1]
	vectors = tf.random.normal((count, noise_shape))
	generate(generator, vectors, dir_out)


def bezier_interpolation(p0, p1, p2, p3, p4, frames):
	segment = []
	t_intervals = tf.linspace(0.0, 1.0, frames)
	for t in t_intervals:
		B_t = (
		    1 * (1 - t)**4 * t**0 * p0 +
		    4 * (1 - t)**3 * t**1 * p1 +
		    6 * (1 - t)**2 * t**2 * p2 +
		    4 * (1 - t)**1 * t**3 * p3 +
		    1 * (1 - t)**0 * t**4 * p4
		)
		segment.append(B_t)
		# print(f'segment shape: ({len(segment)}, {len(segment[0])})')
	return segment


def bezier(generator, dir_out, segments, frames):
	# looping bezier curve
	# each segment of the curve has 5 points from end to end
	noise_shape = generator.input_shape[-1]
	noises = tf.random.normal((segments * 3, noise_shape))

	vectors = []
	prog_bar = tf.keras.utils.Progbar(segments)
	for i in range(segments):
		p0 = noises[3*i-1]
		p1 = p0 + (p0 - noises[3*i-2])
		p2 = noises[3*i]
		p3 = noises[3*i+1]
		p4 = noises[3*i+2]
		vectors.extend(bezier_interpolation(p0, p1, p2, p3, p4, frames))
		prog_bar.add(1)
	#print(f'vectors shape: ({len(vectors)}, {len(vectors[0])})')
	generate(generator, vectors, dir_out)


def plot(waves, im_path):
	plt.rcParams.update({'font.size': 4})
	s = len(waves)
	w = int(math.sqrt(s))
	while s % w != 0:
		w += 1
	h = s // w
	fig, axes = plt.subplots(w, h)
	for i, ax in enumerate(axes.flatten()):
		ax.plot(waves[i], label=f'dim_{i+1}')
		ax.set_title(f'dim_{i+1}')
	plt.tight_layout()
	plt.savefig(f'{im_path}.png', dpi=300)


def sinewave(seed, frames, periods, a=1.0):
	wave = []
	thetas = np.linspace(0.0, 2 * np.pi, frames)
	for _ in range(periods):
		wave.extend(np.sin(thetas) * a + seed)
	return wave


def bitloop(generator, dir_out, bs, frames=512):
	# input a bitloop having n bits
	# each dimension in the model's latent space is mapped to a bit
	# - bit = 1, the dimension modulates over a sine wave
	# - bit = 0, the dimension is constant
	# waves are plotted on a chart
	dims = generator.input_shape[-1]
	seeds = np.random.normal(size=(1, dims))

	waves = []
	for i in range(len(seeds[0])):
		bit = bs[i%len(bs)] # bitloop loops up to number of dimensions
		if bool(int(bit)): # if dimension should have motion
			wave_i = sinewave(seeds[0][i], frames, periods=1) # make sine waves
		else:
			wave_i = np.tile(seeds[0][i], frames) # make constant
		waves.append(wave_i)

	plot(waves, dir_out)
	generate(generator, np.transpose(waves), dir_out)


def period(generator, dir_out, sec=60, fps=30):
	# each dimension is assigned a period
	# the period is a random factor of 60
	# they all have the same phase
	dims = generator.input_shape[-1]
	seeds = np.random.normal(size=(1, dims))

	dims = len(seeds[0])
	waves = []
	for dim in range(dims):
		periods = r.choice([1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60])
		period_len = sec // periods
		frames = period_len * fps
		wave = sinewave(seeds[0][dim], frames, periods)
		waves.append(wave)

	plot(waves, dir_out)
	generate(generator, np.transpose(waves), dir_out)


def phase(generator, dir_out, sec=60, fps=30):
	# each dimension is assigned a phase
	# they all have the same period
	# the period is the duration of the path
	dims = generator.input_shape[-1]
	seeds = np.random.normal(size=(1, dims))

	dims = len(seeds[0])
	waves = []
	for dim in range(dims):
		wave = sinewave(seeds[0][dim], sec*fps, 1)
		wave = np.roll(wave, r.randint(1, sec*fps))
		waves.append(wave)

	plot(waves, dir_out)
	generate(generator, np.transpose(waves), dir_out)


def main(model, dir_out, style, count, segments, frames, bs, seconds):
	generator = load_generator(model)
	dir_out = get_dir_out_path(dir_out, style, count, segments, frames, bs, seconds)
	os.makedirs(dir_out, exist_ok=True)
	if style == 'random':
		random(generator, dir_out, count)
	if style == 'bezier':
		bezier(generator, dir_out, segments, frames)
	if style == 'bitloop':
		bitloop(generator, dir_out, bs)
	if style == 'period':
		period(generator, dir_out, seconds)
	if style == 'phase':
		phase(generator, dir_out, seconds)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'model',
		type=str,
		help='path to model used to generate images')
	parser.add_argument(
		'style',
		type=str,
		choices=['random', 'bezier', 'bitloop', 'period', 'phase'],
		help='choices: random, bezier, bitloop, period, phase')
	parser.add_argument(
		'--dir_out',
		type=str,
		default='out',
		help='output folder')
	parser.add_argument(
		'--count',
		type=int,
		default=1,
		help='random: number of random images to generate')
	parser.add_argument(
		'--segments',
		type=int,
		default=32,
		help='bezier: number of points in the vector space for bezier curve to pass through')
	parser.add_argument(
		'--frames',
		type=int,
		default=256,
		help='bezier: number of frames per segment or period')
	parser.add_argument(
		'--bitloop',
		type=str,
		default='1100',
		help='for example, 00110011')
	parser.add_argument(
		'--seconds',
		type=int,
		default=60,
		help='period and phase: how long an animation would last at 30 fps')

	args = parser.parse_args()
	main(args.model, args.dir_out, args.style, args.count, args.segments, args.frames, args.bitloop, args.seconds)














