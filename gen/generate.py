import argparse
import math
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import time
import uuid
from train import TrainingState


def get_dir_out_path(
	dir_out,
	style,
	count=None,
	segments=None,
	frames=None,
	bs=None):
	instance = f'{int(time.time())}'
	if style == 'random':
		subdir = f'{style}_{count}'
	if style == 'bezier':
		subdir = f'{style}_s{segments}_f{frames}'
	if style == 'bitloop':
		subdir = f'{style}_{bs}'
	if style == 'period':
		subdir = f'{style}'
	if style == 'phase':
		subdir = f'{style}'
	dir_out = f'{dir_out}/{subdir}/{instance}'
	os.makedirs(dir_out, exist_ok=True)
	return dir_out


def load_generator(model):
	with open(model, 'rb') as f:
		bytes = f.read()
	training_state: TrainingState = pickle.loads(bytes)
	return training_state.generator


def generate(generator, vectors, dir_out, batch_size=16):
	prog_bar = tf.keras.utils.Progbar(max(len(vectors) // batch_size, 1))
	vectors = tf.convert_to_tensor(vectors)
	if len(vectors) == 1:
		pad = 1
	else:
		pad = int(math.log10(len(vectors) - 1) + 1)
	for start in range(0, len(vectors), batch_size):
		images = []
		end = start + batch_size
		batch = vectors[start:end]
		image_batch = generator(batch)
		images.append(image_batch)
		prog_bar.add(1)
		images = tf.concat(images, axis=0)
		for i, image in enumerate(images):
		    img_no = start + i
		    file_path = os.path.join(dir_out, f'{img_no:0{pad}}.png')
		    image = tf.convert_to_tensor(image)
		    image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
		    image = tf.io.encode_png(image).numpy()
		    with open(file_path, 'wb') as f:
		        f.write(image)





















